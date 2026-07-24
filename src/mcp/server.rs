//! MCP Server - 把本地 `BaseTool` 暴露为 MCP Server,供其他 Host(Claude Desktop/Cursor 等)调用
//!
//! 与 `MCPClient` 对称:Client 连别人的 Server 用工具,Server 把自己的工具暴露给别人。
//! 支持 `initialize` 握手、`tools/list`、`tools/call`;`resources`/`prompts` 暂留 method_not_found。

use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::protocol::{MCPError, MCPRequest, MCPResponse, MCP_VERSION};
use super::types::{MCPContent, MCPToolDefinition, MCPToolResult};
use crate::BaseTool;

/// MCP Server - 暴露一组 `BaseTool` 为 MCP 工具
pub struct MCPServer {
    tools: Vec<Arc<dyn BaseTool>>,
    server_name: String,
    server_version: String,
}

impl MCPServer {
    /// 创建空 server
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            server_name: "langchainrust-mcp-server".to_string(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// 注册一个工具
    pub fn with_tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// 设置 serverInfo(名称/版本)
    pub fn with_server_info(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.server_name = name.into();
        self.server_version = version.into();
        self
    }

    /// 从 BaseTool 构造 MCP 工具定义
    fn tool_definition(tool: &dyn BaseTool) -> MCPToolDefinition {
        MCPToolDefinition {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool
                .args_schema()
                .unwrap_or_else(|| json!({"type":"object"})),
        }
    }

    /// 处理一条 JSON-RPC 请求,返回响应
    ///
    /// 供单元测试直接调用;`serve_stdio` 内部也用它处理每行请求。
    pub async fn handle_request(&self, req: MCPRequest) -> MCPResponse {
        match req.method.as_str() {
            "initialize" => MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(req.id),
                result: Some(json!({
                    "protocolVersion": MCP_VERSION,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": self.server_name, "version": self.server_version }
                })),
                error: None,
            },
            "tools/list" => {
                let tools: Vec<MCPToolDefinition> = self
                    .tools
                    .iter()
                    .map(|t| Self::tool_definition(t.as_ref()))
                    .collect();
                let tools_val =
                    serde_json::to_value(&tools).unwrap_or_else(|_| Value::Array(vec![]));
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(json!({ "tools": tools_val })),
                    error: None,
                }
            }
            "tools/call" => self.handle_tools_call(req).await,
            _ => MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(req.id),
                result: None,
                error: Some(MCPError::method_not_found()),
            },
        }
    }

    async fn handle_tools_call(&self, req: MCPRequest) -> MCPResponse {
        let params = req.params.clone().unwrap_or(Value::Null);
        let name = params.get("name").and_then(|v| v.as_str());
        let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

        let name = match name {
            Some(n) => n,
            None => {
                return MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: None,
                    error: Some(MCPError::invalid_params("缺少 name 参数")),
                }
            }
        };

        let tool = self.tools.iter().find(|t| t.name() == name);
        match tool {
            Some(t) => {
                let input_str = serde_json::to_string(&arguments).unwrap_or_else(|_| "null".into());
                let result = t.run(input_str).await;
                let mcp_result = match result {
                    Ok(text) => MCPToolResult {
                        content: vec![MCPContent::Text { text }],
                        is_error: false,
                    },
                    Err(e) => MCPToolResult {
                        content: vec![MCPContent::Text {
                            text: e.to_string(),
                        }],
                        is_error: true,
                    },
                };
                let result_val = serde_json::to_value(&mcp_result).unwrap_or(Value::Null);
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(result_val),
                    error: None,
                }
            }
            None => MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(req.id),
                result: None,
                error: Some(MCPError::invalid_params(format!("未知工具: {}", name))),
            },
        }
    }

    /// 在 stdio 上运行 server:从 stdin 读 JSON-RPC,处理后写回 stdout
    ///
    /// 通知(无 id 的消息,如 `notifications/initialized`)被忽略;请求(有 id)返回响应。
    pub async fn serve_stdio(&self) -> Result<(), MCPError> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut stdout = tokio::io::stdout();

        loop {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .await
                .map_err(|e| MCPError::new(-1, format!("读 stdin 失败: {}", e)))?;
            if n == 0 {
                break; // EOF
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // 用宽松结构解析:通知无 id,请求有 id
            let msg: ServerMessage = match serde_json::from_str(trimmed) {
                Ok(m) => m,
                Err(e) => {
                    // Per JSON-RPC 2.0 spec: if the request could not be parsed,
                    // the response id MUST be null.
                    let resp = MCPResponse {
                        jsonrpc: "2.0".to_string(),
                        id: None,
                        result: None,
                        error: Some(MCPError::invalid_params(format!("解析请求失败: {}", e))),
                    };
                    let json = serde_json::to_string(&resp)
                        .map_err(|e| MCPError::new(-1, format!("序列化失败: {}", e)))?;
                    let _ = write_line(&mut stdout, &json).await;
                    continue;
                }
            };

            // 通知(无 id)忽略
            let id = match msg.id {
                Some(id) => id,
                None => continue,
            };

            let req = MCPRequest {
                jsonrpc: "2.0".to_string(),
                id,
                method: msg.method,
                params: msg.params,
            };
            let resp = self.handle_request(req).await;
            let json = serde_json::to_string(&resp)
                .map_err(|e| MCPError::new(-1, format!("序列化失败: {}", e)))?;
            write_line(&mut stdout, &json)
                .await
                .map_err(|e| MCPError::new(-1, format!("写 stdout 失败: {}", e)))?;
        }
        Ok(())
    }
}

impl Default for MCPServer {
    fn default() -> Self {
        Self::new()
    }
}

/// 宽松的入站消息:通知无 id
#[derive(Deserialize)]
struct ServerMessage {
    #[serde(default)]
    id: Option<u64>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

async fn write_line<W: AsyncWriteExt + Unpin>(w: &mut W, json: &str) -> Result<(), std::io::Error> {
    w.write_all(json.as_bytes()).await?;
    w.write_all(b"\n").await?;
    w.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tools::ToolError;

    /// 测试用工具:回显输入
    struct EchoTool;
    #[async_trait::async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "回显输入"
        }
        fn args_schema(&self) -> Option<Value> {
            Some(json!({"type":"object","properties":{"text":{"type":"string"}}}))
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(input)
        }
    }

    fn server_with_echo() -> MCPServer {
        MCPServer::new().with_tool(Arc::new(EchoTool))
    }

    #[tokio::test]
    async fn test_initialize() {
        let server = server_with_echo();
        let resp = server
            .handle_request(MCPRequest::new(1, "initialize", None))
            .await;
        assert!(!resp.is_error());
        let result = resp.result.unwrap();
        assert!(result.get("protocolVersion").is_some());
        assert!(result.get("capabilities").is_some());
        assert!(result.get("serverInfo").is_some());
    }

    #[tokio::test]
    async fn test_tools_list() {
        let server = server_with_echo();
        let resp = server
            .handle_request(MCPRequest::new(2, "tools/list", None))
            .await;
        assert!(!resp.is_error());
        let result = resp.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
        assert_eq!(tools[0]["inputSchema"]["type"], "object");
    }

    #[tokio::test]
    async fn test_tools_call_success() {
        let server = server_with_echo();
        let params = json!({"name":"echo","arguments":{"text":"hello"}});
        let resp = server
            .handle_request(MCPRequest::new(3, "tools/call", Some(params)))
            .await;
        assert!(!resp.is_error());
        let mcp_result: MCPToolResult = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert!(!mcp_result.is_error);
        // echo 返回输入(arguments 的 JSON 串)
        assert_eq!(mcp_result.text(), r#"{"text":"hello"}"#);
    }

    #[tokio::test]
    async fn test_tools_call_unknown_tool() {
        let server = server_with_echo();
        let params = json!({"name":"nonexistent","arguments":{}});
        let resp = server
            .handle_request(MCPRequest::new(4, "tools/call", Some(params)))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32602); // invalid_params
    }

    #[tokio::test]
    async fn test_tools_call_missing_name() {
        let server = server_with_echo();
        let params = json!({"arguments":{}});
        let resp = server
            .handle_request(MCPRequest::new(5, "tools/call", Some(params)))
            .await;
        assert!(resp.is_error());
    }

    #[tokio::test]
    async fn test_unknown_method() {
        let server = MCPServer::new();
        let resp = server
            .handle_request(MCPRequest::new(6, "foo/bar", None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601); // method_not_found
    }

    #[test]
    fn test_server_message_notification_has_no_id() {
        // 通知(无 id)应解析为 id=None
        let json = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        assert!(msg.id.is_none());
        assert_eq!(msg.method, "notifications/initialized");
    }

    #[test]
    fn test_server_message_request_has_id() {
        let json = r#"{"jsonrpc":"2.0","id":42,"method":"tools/list"}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.id, Some(42));
    }
}
