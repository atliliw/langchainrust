//! MCP tool adapter - wraps MCP Tools as `BaseTool`s

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use super::client::MCPClient;
use super::protocol::MCPError;
use super::sandbox::ServerSandbox;
use super::tool_timeout::{call_tool_with_timeout, ToolSpec};
use super::types::{MCPToolDefinition, MCPToolResult};
use lc_core::tools::ToolError;
use lc_core::BaseTool;

/// Maps an MCP transport-layer error to [`ToolError::McpError`], preserving code/message/data (P1-6).
///
/// No longer degrades everything to `ExecutionFailed`; upper layers can distinguish error categories by `code`.
/// `pub(crate)`: the Gateway (P2-8) unified call entry reuses the same mapping.
pub(crate) fn from_mcp_error(e: MCPError) -> ToolError {
    ToolError::McpError {
        code: e.code,
        message: e.message,
        data: e.data,
    }
}

/// Turns a tool-call result into text; explicitly errors when the server side has `is_error=true` (P1-6).
///
/// In MCP a server tool execution failure is expressed as "a successful JSON-RPC response + the `is_error`
/// flag"; the old implementation swallowed it as a success, here it becomes an explicit `ExecutionFailed`.
/// `pub(crate)`: the Gateway (P2-8) unified call entry reuses the same conversion.
pub(crate) fn result_to_string_or_error(result: &MCPToolResult) -> Result<String, ToolError> {
    if result.is_error {
        Err(ToolError::ExecutionFailed(result.text()))
    } else {
        Ok(result.text())
    }
}

/// MCP tool adapter - wraps the tools a MCP Server exposes as `BaseTool`s
pub struct MCPToolAdapter {
    client: MCPClient,
    definition: MCPToolDefinition,
    /// The externally visible tool name (P2-2): after namespacing it is `server_name:tool_name`;
    /// without namespacing it equals the original tool name. The LLM sees this; the actual call still uses
    /// the original name.
    display_name: String,
    /// per-tool timeout (P2-4): `None` uses the default; `Some(spec)` uses the
    /// progress-reset + hard-cap timed call.
    timeout_spec: Option<ToolSpec>,
    /// per-Server security sandbox (P2-6): when `Some`, `run()` runs parameter least-privilege validation
    /// before sending the request; a block returns an error and records the audit. Multiple adapters of the
    /// same Server share one sandbox.
    sandbox: Option<Arc<ServerSandbox>>,
}

impl MCPToolAdapter {
    /// Creates an adapter from the original tool name (no namespacing).
    pub fn new(client: MCPClient, definition: MCPToolDefinition) -> Self {
        let display_name = definition.name.clone();
        Self {
            client,
            definition,
            display_name,
            timeout_spec: None,
            sandbox: None,
        }
    }

    /// Namespaced adapter (P2-2): the LLM sees the tool name as `server_name:tool_name`; the call strips the
    /// prefix automatically and uses the Server-side original tool name.
    ///
    /// Pairs with [`crate::ToolNamespace::qualify`] for unique routing of same-named tools in the 100+ Server
    /// scenario: when several Servers all have `read_file`, each exposes `fs:read_file` / `db:read_file`, but
    /// the calls all go through their own `read_file`.
    pub fn namespaced(client: MCPClient, server: &str, definition: MCPToolDefinition) -> Self {
        let display_name = format!("{server}:{}", definition.name);
        Self {
            client,
            definition,
            display_name,
            timeout_spec: None,
            sandbox: None,
        }
    }

    /// Attaches a per-tool timeout (P2-4): the timeout/progress-reset/hard-cap semantics are in
    /// [`ToolSpec`] and [`call_tool_with_timeout`].
    pub fn with_timeout(mut self, spec: ToolSpec) -> Self {
        self.timeout_spec = Some(spec);
        self
    }

    /// Attaches a per-Server security sandbox (P2-6): `run()` runs parameter-level least-privilege validation
    /// before sending the request; a block returns [`ToolError::InvalidInput`] and records the audit; only
    /// allowed calls actually reach the Server.
    pub fn with_sandbox(mut self, sandbox: Arc<ServerSandbox>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// The externally visible tool name (P2-2): `server:tool` after namespacing, otherwise the original name.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }
}

#[async_trait]
impl BaseTool for MCPToolAdapter {
    fn name(&self) -> &str {
        &self.display_name
    }

    fn description(&self) -> &str {
        &self.definition.description
    }

    fn args_schema(&self) -> Option<Value> {
        Some(self.definition.input_schema.clone())
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let args: Value = serde_json::from_str(&input)
            .map_err(|e| ToolError::ExecutionFailed(format!("Invalid JSON input: {}", e)))?;
        // per-Server security sandbox (P2-6): run parameter-level least-privilege validation before sending.
        if let Some(sandbox) = &self.sandbox {
            sandbox
                .check_call(&self.definition.name, &args)
                .map_err(|e| ToolError::InvalidInput(e.to_string()))?;
        }
        let result = match &self.timeout_spec {
            Some(spec) => {
                call_tool_with_timeout(&self.client, &self.definition.name, args, spec).await
            }
            None => self.client.call_tool(&self.definition.name, args).await,
        }
        .map_err(from_mcp_error)?;
        result_to_string_or_error(&result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::ParamRule;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::types::MCPContent;
    use crate::MCPConfig;
    use serde_json::json;

    fn sample_definition() -> MCPToolDefinition {
        MCPToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn test_from_mcp_error_preserves_fields() {
        // P1-6: code/message/data are preserved as-is, not degraded to a structureless ExecutionFailed.
        let e = MCPError {
            code: -32001,
            message: "timeout".to_string(),
            data: Some(json!({"k": 1})),
        };
        match from_mcp_error(e) {
            ToolError::McpError {
                code,
                message,
                data,
            } => {
                assert_eq!(code, -32001);
                assert_eq!(message, "timeout");
                assert_eq!(data, Some(json!({"k": 1})));
            }
            other => panic!("expected McpError, actual: {:?}", other),
        }
    }

    #[test]
    fn test_from_mcp_error_display() {
        let e = MCPError::new(-32602, "invalid params");
        let err = from_mcp_error(e);
        assert_eq!(err.to_string(), "MCP error [-32602]: invalid params");
    }

    #[test]
    fn test_result_is_error_returns_execution_failed() {
        // A server-side tool failure (is_error=true) → explicit Err, not swallowed as a success.
        let result = MCPToolResult {
            content: vec![MCPContent::Text {
                text: "server exploded".to_string(),
            }],
            is_error: true,
        };
        let err = result_to_string_or_error(&result).unwrap_err();
        assert!(matches!(
            err,
            ToolError::ExecutionFailed(ref m) if m.contains("server exploded")
        ));
    }

    #[test]
    fn test_result_ok_returns_joined_text() {
        let result = MCPToolResult {
            content: vec![
                MCPContent::Text {
                    text: "a".to_string(),
                },
                MCPContent::Text {
                    text: "b".to_string(),
                },
            ],
            is_error: false,
        };
        assert_eq!(result_to_string_or_error(&result).unwrap(), "a\nb");
    }

    /// Sandbox block: violating parameters are intercepted before the request is sent (P2-6), never reaching
    /// the Server.
    #[tokio::test]
    async fn test_adapter_sandbox_blocks_before_call() {
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        let sandbox = Arc::new(ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        }));
        let adapter = MCPToolAdapter::new(client, sample_definition()).with_sandbox(sandbox);
        let err = adapter
            .run(r#"{"path": "file:///etc/passwd"}"#.to_string())
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref m) if m.contains("least-privilege")),
            "{}",
            err
        );
    }

    /// Sandbox allow: compliant parameters really reach the Server (P2-6).
    #[tokio::test]
    async fn test_adapter_sandbox_allows_and_reaches_server() {
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        let sandbox = Arc::new(ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        }));
        let adapter = MCPToolAdapter::new(client, sample_definition()).with_sandbox(sandbox);
        let out = adapter
            .run(r#"{"path": "file:///tmp/a.txt"}"#.to_string())
            .await;
        assert!(
            matches!(out.as_deref(), Ok("read_file")),
            "should reach the server and echo the tool name after allow, actual: {:?}",
            out.as_deref()
        );
    }
}
