//! MCP tool adapter - wraps MCP Tools as `BaseTool`s
//!
//! # Execution is fail-closed (0.22.4 A16)
//!
//! A freshly constructed adapter does NOT execute: calls are refused with
//! [`ToolError::PermissionDenied`] until the embedding application declares
//! how unattended execution is authorized. Pick exactly one:
//!
//! - [`MCPToolAdapter::with_sandbox`] — parameter least-privilege checks via a
//!   shared [`ServerSandbox`], unattended calls allowed when the arguments pass;
//! - [`MCPToolAdapter::allow_unattended_execution`] — explicit opt-in for a
//!   fully trusted server (local sidecar / tests);
//! - [`MCPToolAdapter::with_approver`] — runtime per-call confirmation through
//!   a [`crate::ToolCallApprover`] (human-in-the-loop / policy engine).
//!
//! The gate always runs before the request is sent, so a denial never reaches
//! the MCP server.

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use super::client_stateless::StatelessMcpClient;
use super::execution::{authorize_call, ToolCallApprover, ToolExecutionPolicy};
use super::protocol::MCPError;
use super::sandbox::ServerSandbox;
use super::tool_client::McpToolClient;
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
///
/// Transport-agnostic since 0.22.4: the client behind an adapter is any
/// [`McpToolClient`] — stateless HTTP, official stdio, or (slated) Streamable
/// HTTP. Build stateless-track adapters with [`MCPToolAdapter::new`] /
/// [`MCPToolAdapter::namespaced`], and other tracks with
/// [`MCPToolAdapter::from_client`] / [`MCPToolAdapter::namespaced_from_client`].
pub struct MCPToolAdapter {
    client: Arc<dyn McpToolClient>,
    definition: MCPToolDefinition,
    /// The externally visible tool name (P2-2): after namespacing it is `server_name:tool_name`;
    /// without namespacing it equals the original tool name. The LLM sees this; the actual call still uses
    /// the original name.
    display_name: String,
    /// per-tool timeout (P2-4): `None` uses the default; `Some(spec)` uses the
    /// progress-reset + hard-cap timed call.
    timeout_spec: Option<ToolSpec>,
    /// Server namespace (P2-2): `Some(name)` for a namespaced adapter (the
    /// display name is `server:tool`); empty for a plain adapter. Passed to the
    /// approval gate for context.
    server: String,
    /// Execution authorization gate (A16): fail-closed
    /// ([`ToolExecutionPolicy::RequireApproval`]) until explicitly declared.
    policy: ToolExecutionPolicy,
    /// Runtime per-call approval gate, consulted under the default
    /// [`ToolExecutionPolicy::RequireApproval`] policy.
    approver: Option<Arc<dyn ToolCallApprover>>,
}

impl MCPToolAdapter {
    /// Creates an adapter from the original tool name (no namespacing).
    ///
    /// The adapter is fail-closed (A16): attach a sandbox
    /// ([`MCPToolAdapter::with_sandbox`]), an approver
    /// ([`MCPToolAdapter::with_approver`]) or explicitly opt into unattended
    /// execution ([`MCPToolAdapter::allow_unattended_execution`]) before
    /// calling `run`.
    pub fn new(client: StatelessMcpClient, definition: MCPToolDefinition) -> Self {
        Self::from_client(Arc::new(client), definition)
    }

    /// Creates an adapter over any MCP track (stdio, stateless HTTP, or a
    /// custom [`McpToolClient`]) from the original tool name (no namespacing).
    ///
    /// Share one connected client across tools by cloning the `Arc`.
    ///
    /// The adapter is fail-closed (A16), like [`MCPToolAdapter::new`]: attach a
    /// sandbox, an approver or [`MCPToolAdapter::allow_unattended_execution`]
    /// before calling `run`.
    pub fn from_client(client: Arc<dyn McpToolClient>, definition: MCPToolDefinition) -> Self {
        let display_name = definition.name.clone();
        Self {
            client,
            definition,
            display_name,
            timeout_spec: None,
            server: String::new(),
            policy: ToolExecutionPolicy::default(),
            approver: None,
        }
    }

    /// Namespaced adapter (P2-2): the LLM sees the tool name as `server_name:tool_name`; the call strips the
    /// prefix automatically and uses the Server-side original tool name.
    ///
    /// Pairs with [`crate::ToolNamespace::qualify`] for unique routing of same-named tools in the 100+ Server
    /// scenario: when several Servers all have `read_file`, each exposes `fs:read_file` / `db:read_file`, but
    /// the calls all go through their own `read_file`.
    pub fn namespaced(
        client: StatelessMcpClient,
        server: &str,
        definition: MCPToolDefinition,
    ) -> Self {
        Self::namespaced_from_client(Arc::new(client), server, definition)
    }

    /// Namespaced adapter over any MCP track (B1): like
    /// [`MCPToolAdapter::from_client`], the LLM sees `server:tool` and the call
    /// strips the prefix and uses the server-side original tool name.
    pub fn namespaced_from_client(
        client: Arc<dyn McpToolClient>,
        server: &str,
        definition: MCPToolDefinition,
    ) -> Self {
        let display_name = format!("{server}:{}", definition.name);
        Self {
            client,
            definition,
            display_name,
            timeout_spec: None,
            server: server.to_string(),
            policy: ToolExecutionPolicy::default(),
            approver: None,
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
    ///
    /// This is one of the three ways to satisfy the A16 fail-closed gate.
    pub fn with_sandbox(mut self, sandbox: Arc<ServerSandbox>) -> Self {
        self.policy = ToolExecutionPolicy::Sandboxed(sandbox);
        self
    }

    /// Explicitly opts into unattended, unrestricted execution (A16): every
    /// call is dispatched without a sandbox or approval.
    ///
    /// Use only for a fully trusted server (local sidecar, tests). It is the
    /// deliberate escape hatch from the fail-closed default.
    pub fn allow_unattended_execution(mut self) -> Self {
        self.policy = ToolExecutionPolicy::AllowUnattended;
        self
    }

    /// Attaches a runtime approval gate (A16): under the default
    /// [`ToolExecutionPolicy::RequireApproval`] policy every `run` asks the
    /// approver before the request is sent; a denial surfaces as
    /// [`ToolError::PermissionDenied`] and never reaches the server.
    pub fn with_approver(mut self, approver: Arc<dyn ToolCallApprover>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Replaces the execution policy outright (used by the Gateway so the
    /// adapters it builds carry the same gate as the unified `call` entry).
    pub(crate) fn with_execution_policy(mut self, policy: ToolExecutionPolicy) -> Self {
        self.policy = policy;
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
        // A16 fail-closed gate: sandbox / explicit unattended opt-in / runtime
        // approval. Runs before any network I/O, so a denial never reaches the server.
        authorize_call(
            &self.policy,
            self.approver.as_ref(),
            &self.display_name,
            &self.server,
            &self.definition.name,
            &args,
        )
        .await?;
        let result = match &self.timeout_spec {
            Some(spec) => {
                call_tool_with_timeout(&*self.client, &self.definition.name, args, spec).await
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
    use crate::test_support::{start_fake_stateless_server, StatelessMode};
    use crate::types::MCPContent;
    use serde_json::json;

    fn sample_definition() -> MCPToolDefinition {
        MCPToolDefinition {
            name: "echo".to_string(),
            description: "Echo a message".to_string(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn sample_client(server_url: &str) -> StatelessMcpClient {
        StatelessMcpClient::connect(server_url)
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
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client = sample_client(&server.url);
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
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client = sample_client(&server.url);
        let sandbox = Arc::new(ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        }));
        let adapter = MCPToolAdapter::new(client, sample_definition()).with_sandbox(sandbox);
        let out = adapter
            .run(r#"{"path": "file:///tmp/a.txt"}"#.to_string())
            .await;
        assert!(
            matches!(out.as_deref(), Ok(text) if text.contains("echo")),
            "should reach the server after allow, actual: {:?}",
            out.as_deref()
        );
    }

    /// A16 fail-closed default: an adapter with no sandbox / approver /
    /// unattended opt-in refuses before dispatch, and the server sees nothing.
    #[tokio::test]
    async fn test_adapter_denies_by_default_without_policy() {
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client = sample_client(&server.url);
        let adapter = MCPToolAdapter::new(client, sample_definition());
        let err = adapter
            .run(r#"{"path": "file:///etc/passwd"}"#.to_string())
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::PermissionDenied(ref m) if m.contains("no unattended-execution policy")),
            "default policy must deny, actual: {err}"
        );
        assert_eq!(
            server
                .request_count
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a denied call must never reach the server"
        );
    }

    /// A16: the explicit unattended opt-in dispatches without further checks.
    #[tokio::test]
    async fn test_adapter_allow_unattended_reaches_server() {
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client = sample_client(&server.url);
        let adapter = MCPToolAdapter::new(client, sample_definition()).allow_unattended_execution();
        let out = adapter
            .run(r#"{"path": "anything"}"#.to_string())
            .await
            .expect("explicit unattended opt-in should dispatch");
        assert!(out.contains("echo"), "actual: {out}");
    }

    /// A16 approver: a denying gate blocks before dispatch; flipping the shared
    /// decision lets the next call through the same adapter.
    #[tokio::test]
    async fn test_adapter_approver_denies_then_allows() {
        use std::sync::atomic::AtomicBool;

        struct FlagApprover {
            allow: Arc<AtomicBool>,
        }
        #[async_trait::async_trait]
        impl ToolCallApprover for FlagApprover {
            async fn approve(
                &self,
                _server: &str,
                tool: &str,
                _arguments: &Value,
            ) -> Result<(), String> {
                if self.allow.load(std::sync::atomic::Ordering::SeqCst) {
                    Ok(())
                } else {
                    Err(format!("human declined {tool}"))
                }
            }
        }

        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let flag = Arc::new(AtomicBool::new(false));
        let approver: Arc<dyn ToolCallApprover> = Arc::new(FlagApprover {
            allow: flag.clone(),
        });
        let adapter = MCPToolAdapter::new(sample_client(&server.url), sample_definition())
            .with_approver(approver);

        let err = adapter.run("{}".into()).await.unwrap_err();
        assert!(
            matches!(err, ToolError::PermissionDenied(ref m) if m.contains("human declined echo")),
            "denial must carry the approver reason, actual: {err}"
        );
        assert_eq!(
            server
                .request_count
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "denied call must not reach the server"
        );

        // The user confirms: the same adapter now dispatches.
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        let out = adapter
            .run("{}".into())
            .await
            .expect("after approval the call should dispatch");
        assert!(out.contains("echo"), "actual: {out}");
    }
}
