//! MCP 工具适配器 - 将 MCP Tool 转为 BaseTool

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

/// 把 MCP 传输层错误映射为 [`ToolError::McpError`],保留 code/message/data(P1-6)。
///
/// 不再一律降级为 `ExecutionFailed`,上层可据 `code` 区分错误类别。
/// `pub(crate)`:Gateway(P2-8)统一调用入口复用同一映射。
pub(crate) fn from_mcp_error(e: MCPError) -> ToolError {
    ToolError::McpError {
        code: e.code,
        message: e.message,
        data: e.data,
    }
}

/// 把工具调用结果转为文本;服务器侧 `is_error=true` 时显式报错(P1-6)。
///
/// MCP 协议中服务器工具执行失败以"成功的 JSON-RPC 响应 + `is_error` 标记"
/// 表达,旧实现会把它当成功吞掉,这里转为显式 `ExecutionFailed`。
/// `pub(crate)`:Gateway(P2-8)统一调用入口复用同一转换。
pub(crate) fn result_to_string_or_error(result: &MCPToolResult) -> Result<String, ToolError> {
    if result.is_error {
        Err(ToolError::ExecutionFailed(result.text()))
    } else {
        Ok(result.text())
    }
}

/// MCP 工具适配器 - 把 MCP Server 暴露的工具包装为 `BaseTool`
pub struct MCPToolAdapter {
    client: MCPClient,
    definition: MCPToolDefinition,
    /// 对外工具名(P2-2):命名空间化后为 `server_name:tool_name`;
    /// 未命名空间时等于原始工具名。LLM 看到的是它,实际调用仍走原始名。
    display_name: String,
    /// per-tool 超时(P2-4):`None` 用默认;`Some(spec)` 走
    /// progress 重置 + 硬上限的计时调用。
    timeout_spec: Option<ToolSpec>,
    /// per-Server 安全沙箱(P2-6):`Some` 时 `run()` 发请求前先过参数最小权限
    /// 校验,拦截则返回错误并记审计。同一 Server 的多个适配器共享一份沙箱。
    sandbox: Option<Arc<ServerSandbox>>,
}

impl MCPToolAdapter {
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

    /// 命名空间化适配器(P2-2):LLM 看到的工具名为 `server_name:tool_name`,
    /// 调用时自动剥掉前缀走 Server 侧原始工具名。
    ///
    /// 与 [`crate::ToolNamespace::qualify`] 配套,用于 100+ Server 场景下同名
    /// 工具的唯一化路由:多个 Server 都有 `read_file` 时,各自对外名
    /// `fs:read_file` / `db:read_file`,但调用都走各自的 `read_file`。
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

    /// 给该工具挂 per-tool 超时(P2-4):超时/进度重置/硬上限语义见
    /// [`ToolSpec`] 与 [`call_tool_with_timeout`]。
    pub fn with_timeout(mut self, spec: ToolSpec) -> Self {
        self.timeout_spec = Some(spec);
        self
    }

    /// 挂 per-Server 安全沙箱(P2-6):`run()` 发请求前先过参数级最小权限校验,
    /// 拦截返回 [`ToolError::InvalidInput`] 并记审计;放行才真正调用 Server。
    pub fn with_sandbox(mut self, sandbox: Arc<ServerSandbox>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// 对外展示的工具名(P2-2):命名空间化后为 `server:tool`,否则等于原始名。
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
        // per-Server 安全沙箱(P2-6):发请求前先做参数级最小权限校验。
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

    #[tokio::test]
    #[ignore = "需要本地 MCP SSE Server"]
    async fn test_adapter_metadata() {
        let client = MCPClient::connect(MCPConfig::sse("http://localhost:3001/sse"))
            .await
            .unwrap();
        let adapter = MCPToolAdapter::new(client, sample_definition());
        assert_eq!(adapter.name(), "read_file");
        assert_eq!(adapter.description(), "Read a file");
        assert!(adapter.args_schema().is_some());
    }

    #[test]
    fn test_from_mcp_error_preserves_fields() {
        // P1-6:code/message/data 原样保留,不降级为无结构 ExecutionFailed。
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
            other => panic!("应为 McpError, 实际: {:?}", other),
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
        // 服务器侧工具失败(is_error=true)→ 显式 Err,而非被当成功吞掉。
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

    /// 沙箱拦截:违规参数在发请求前被拦截(P2-6),不进 Server。
    #[tokio::test]
    async fn test_adapter_sandbox_blocks_before_call() {
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");
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
            matches!(err, ToolError::InvalidInput(ref m) if m.contains("最小权限")),
            "{}",
            err
        );
    }

    /// 沙箱放行:合规参数真正到达 Server(P2-6)。
    #[tokio::test]
    async fn test_adapter_sandbox_allows_and_reaches_server() {
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");
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
            "放行后应到达 Server 并回显工具名, 实际: {:?}",
            out.as_deref()
        );
    }
}
