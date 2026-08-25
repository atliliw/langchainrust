//! MCP Server - 把本地 `BaseTool` 暴露为 MCP Server,供其他 Host(Claude Desktop/Cursor 等)调用
//!
//! 与 `MCPClient` 对称:Client 连别人的 Server 用工具,Server 把自己的工具暴露给别人。
//! 支持 `initialize` 握手、`tools/list`、`tools/call`;`resources`/`prompts` 暂留 method_not_found。

use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::broadcast;

use super::protocol::{
    MCPError, MCPRequest, MCPResponse, MCP_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
};
use super::stream::PartialContent;
use super::types::{MCPContent, MCPToolDefinition, MCPToolResult};
use lc_core::BaseTool;

/// MCP Server - 暴露一组 `BaseTool` 为 MCP 工具
pub struct MCPServer {
    tools: Vec<Arc<dyn BaseTool>>,
    server_name: String,
    server_version: String,
    /// 流式工具输出广播(P2-9):`publish_partial` 推送增量片段,
    /// `InMemoryTransport` 等传输层订阅后转发给客户端。
    partial_tx: broadcast::Sender<PartialContent>,
}

impl MCPServer {
    /// 创建空 server
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            server_name: "langchainrust-mcp-server".to_string(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
            partial_tx: broadcast::channel(64).0,
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

    /// 推送一个流式工具输出增量片段(P2-9)。
    ///
    /// 长任务工具"边跑边推":执行期间把部分结果拆成多个片段,逐个
    /// `publish_partial` 推给已连接的 Host(`InMemoryTransport` 等传输层
    /// 订阅后经 `notifications/tool_partial` 转发给客户端)。无订阅者时
    /// 静默丢弃——增量是预览,最终结果仍由 `tools/call` 响应承载。
    pub fn publish_partial(&self, partial: PartialContent) {
        let _ = self.partial_tx.send(partial);
    }

    /// 订阅本 server 推送的流式增量片段(P2-9)。
    ///
    /// 供传输层(如 [`InMemoryTransport`](crate::InMemoryTransport))把
    /// server 侧的 `publish_partial` 转成客户端可见的推送事件。
    pub fn subscribe_partials(&self) -> broadcast::Receiver<PartialContent> {
        self.partial_tx.subscribe()
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
            // P2-10 版本协商:客户端请求的版本在支持列表内则回显,否则降级到
            // 本实现版本(Server 永远只回复自己支持的版本)。
            "initialize" => {
                let requested = req
                    .params
                    .as_ref()
                    .and_then(|p| p.get("protocolVersion"))
                    .and_then(Value::as_str);
                let protocol_version = match requested {
                    Some(v) if SUPPORTED_PROTOCOL_VERSIONS.contains(&v) => v.to_string(),
                    _ => MCP_VERSION.to_string(),
                };
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(json!({
                        "protocolVersion": protocol_version,
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": self.server_name, "version": self.server_version }
                    })),
                    error: None,
                }
            }
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
                    error: Some(MCPError::invalid_params("missing name parameter")),
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
                error: Some(MCPError::invalid_params(format!("unknown tool: {}", name))),
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
                .map_err(|e| MCPError::new(-1, format!("failed to read stdin: {}", e)))?;
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
                        error: Some(MCPError::invalid_params(format!(
                            "failed to parse request: {}",
                            e
                        ))),
                    };
                    let json = serde_json::to_string(&resp)
                        .map_err(|e| MCPError::new(-1, format!("serialization failed: {}", e)))?;
                    let _ = write_line(&mut stdout, &json).await;
                    continue;
                }
            };

            // 通知(无 id):P0-4 分发给 handle_notification,不再直接丢弃
            let id = match msg.id {
                Some(id) => id,
                None => {
                    self.handle_notification(&msg.method, msg.params).await;
                    continue;
                }
            };

            let req = MCPRequest {
                jsonrpc: "2.0".to_string(),
                id,
                method: msg.method,
                params: msg.params,
            };
            let resp = self.handle_request(req).await;
            let json = serde_json::to_string(&resp)
                .map_err(|e| MCPError::new(-1, format!("serialization failed: {}", e)))?;
            write_line(&mut stdout, &json)
                .await
                .map_err(|e| MCPError::new(-1, format!("failed to write stdout: {}", e)))?;
        }
        Ok(())
    }

    /// 在已绑定的 TCP listener 上提供 MCP SSE 网络服务,返回客户端要连的 SSE 入口 URL。
    ///
    /// 这是"可部署的 MCP server"的入口:把本 server 暴露为 HTTP/SSE 服务,
    /// 任何 MCP 客户端(`MCPClient::connect(MCPConfig::sse(...))` / Cursor /
    /// Claude Desktop 等)都能连上来用注册的工具。SSE 帧格式与
    /// `MCPClient`(SseTransport)客户端行为对齐。
    ///
    /// - `listener`:已绑定好地址的 `TcpListener`。本地联调绑 `127.0.0.1:0`,
    ///   部署到远程服务器绑 `0.0.0.0:PORT`。
    /// - `public_base`:客户端访问本服务器的基地址(如 `http://your-server-ip:8788`)。
    ///   服务端发给客户端的 POST 地址由它拼出,部署在远程时必须是客户端真能访问的
    ///   地址(不能用 `0.0.0.0`)。
    ///
    /// 启动后立即返回 SSE URL,接收循环在后台任务运行直到进程退出。
    pub fn serve_sse(
        self: Arc<Self>,
        listener: tokio::net::TcpListener,
        public_base: impl Into<String>,
    ) -> String {
        crate::sse::serve(self, listener, public_base.into())
    }

    /// 处理服务器收到的通知(无 id 的消息)。
    ///
    /// P0-4: 对 MCP 标准通知显式分发处理,而不是直接丢弃:
    /// - `notifications/cancelled` —— 客户端请求取消某个工具调用
    /// - `notifications/progress` —— 客户端上报工具执行进度
    /// - `notifications/roots/list_changed` —— 根目录列表变化
    /// - `notifications/initialized` —— 客户端完成握手
    ///
    /// 当前实现记录日志并预留扩展点;后续可在派生类型中覆盖以接入
    /// 取消/进度回调。
    pub async fn handle_notification(&self, method: &str, params: Option<Value>) {
        match method {
            "notifications/cancelled" => {
                // 携带 requestId,指向要取消的请求
                log::info!("MCP received cancelled notification: {:?}", params);
            }
            "notifications/progress" => {
                // 携带 token + progress/estimatedTotal
                log::info!("MCP received progress notification: {:?}", params);
            }
            "notifications/roots/list_changed" => {
                log::info!("MCP received roots/list_changed notification: {:?}", params);
            }
            "notifications/initialized" => {
                log::debug!("MCP received initialized notification");
            }
            _ => {
                log::debug!("ignoring unknown notification: {}", method);
            }
        }
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
    use lc_core::tools::ToolError;

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

    /// P2-10 版本协商:请求受支持版本时回显,无请求版本时回复本实现版本。
    #[tokio::test]
    async fn test_initialize_echoes_supported_version() {
        let server = server_with_echo();
        let params = serde_json::json!({ "protocolVersion": MCP_VERSION });
        let resp = server
            .handle_request(MCPRequest::new(1, "initialize", Some(params)))
            .await;
        let version = resp
            .result
            .as_ref()
            .and_then(|r| r.get("protocolVersion"))
            .and_then(Value::as_str)
            .map(str::to_string);
        assert_eq!(version.as_deref(), Some(MCP_VERSION));
    }

    /// P2-10 版本协商:请求不受支持的版本时降级到本实现版本(Server 只回复
    /// 自己支持的版本,不回显未知版本)。
    #[tokio::test]
    async fn test_initialize_degrades_unsupported_version() {
        let server = server_with_echo();
        let params = serde_json::json!({ "protocolVersion": "2099-01-01" });
        let resp = server
            .handle_request(MCPRequest::new(1, "initialize", Some(params)))
            .await;
        let version = resp
            .result
            .as_ref()
            .and_then(|r| r.get("protocolVersion"))
            .and_then(Value::as_str)
            .map(str::to_string);
        assert_eq!(
            version.as_deref(),
            Some(MCP_VERSION),
            "unknown version should degrade to the current implementation version"
        );
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

    #[tokio::test]
    async fn test_handle_notification_known_and_unknown() {
        let server = server_with_echo();
        // 标准通知应被处理(不 panic)
        server
            .handle_notification("notifications/cancelled", Some(json!({"requestId": 1})))
            .await;
        server
            .handle_notification(
                "notifications/progress",
                Some(json!({"token": 1, "progress": 0.5})),
            )
            .await;
        server
            .handle_notification("notifications/roots/list_changed", None)
            .await;
        server
            .handle_notification("notifications/initialized", None)
            .await;
        // 未知通知应被忽略
        server.handle_notification("foo/bar", None).await;
    }
}
