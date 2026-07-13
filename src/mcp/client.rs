//! MCP Client - 连接 MCP Server,获取并调用工具

use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

use super::protocol::{MCPError, MCPRequest};
use super::tool_adapter::MCPToolAdapter;
use super::transport::{MCPTransport, SseTransport, StdioTransport};
use super::types::{MCPConfig, MCPToolDefinition, MCPToolResult};
use crate::BaseTool;

struct MCPClientInner {
    transport: Box<dyn MCPTransport + Send + Sync>,
    tools: Mutex<Vec<MCPToolDefinition>>,
    request_id: AtomicU64,
}

/// MCP Client - 连接 MCP Server 获取工具能力
pub struct MCPClient {
    inner: Arc<MCPClientInner>,
}

impl Clone for MCPClient {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl MCPClient {
    /// 连接 MCP Server
    ///
    /// 建立传输层连接后,发送 MCP `initialize` 握手请求,
    /// 再发送 `notifications/initialized` 通知,完成协议握手。
    pub async fn connect(config: MCPConfig) -> Result<Self, MCPError> {
        let transport: Box<dyn MCPTransport + Send + Sync> = match &config {
            MCPConfig::Stdio { .. } => Box::new(StdioTransport::new(&config).await?),
            MCPConfig::Sse { .. } => Box::new(SseTransport::new(&config)?),
        };

        let client = Self {
            inner: Arc::new(MCPClientInner {
                transport,
                tools: Mutex::new(Vec::new()),
                request_id: AtomicU64::new(1),
            }),
        };

        // MCP 协议握手: initialize + initialized 通知
        let init_params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "langchainrust-mcp-client",
                "version": "0.3.0"
            }
        });
        client.send_request("initialize", Some(init_params)).await?;

        // 发送 initialized 通知(无 id,不等响应)
        client.inner.transport.notify("notifications/initialized", None).await?;

        Ok(client)
    }

    fn next_id(&self) -> u64 {
        self.inner.request_id.fetch_add(1, Ordering::SeqCst)
    }

    async fn send_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, MCPError> {
        let req = MCPRequest::new(self.next_id(), method, params);
        let resp = self.inner.transport.request(req).await?;
        resp.into_result()
    }

    /// 获取可用工具列表(`tools/list`)
    pub async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        let result = self.send_request("tools/list", None).await?;
        let tools: Vec<MCPToolDefinition> = result
            .get("tools")
            .and_then(|t| serde_json::from_value(t.clone()).ok())
            .unwrap_or_default();
        *self.inner.tools.lock().await = tools.clone();
        Ok(tools)
    }

    /// 调用工具(`tools/call`)
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<MCPToolResult, MCPError> {
        let params = json!({"name": name, "arguments": arguments});
        let result = self.send_request("tools/call", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("解析工具结果失败: {}", e)))
    }

    /// 关闭连接
    pub async fn close(&self) -> Result<(), MCPError> {
        self.inner.transport.close().await
    }

    /// 转换为 `BaseTool` 列表(供 Agent 使用)
    ///
    /// 需先调用 `list_tools` 填充工具列表。
    pub async fn as_tools(&self) -> Vec<Arc<dyn BaseTool>> {
        let tools = self.inner.tools.lock().await;
        tools
            .iter()
            .map(|def| {
                Arc::new(MCPToolAdapter::new(self.clone(), def.clone())) as Arc<dyn BaseTool>
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_invalid_stdio_command() {
        let config = MCPConfig::stdio("nonexistent_command_xyz_zzz", vec![]);
        let result = MCPClient::connect(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    #[ignore = "需要本地 MCP SSE Server"]
    async fn test_connect_sse_creates_client() {
        // SSE transport 惰性创建(不立即连接),应成功
        let config = MCPConfig::sse("http://localhost:3001/sse");
        let client = MCPClient::connect(config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    #[ignore = "需要本地 MCP SSE Server"]
    async fn test_next_id_increments() {
        let client = MCPClient::connect(MCPConfig::sse("http://localhost:3001/sse"))
            .await
            .unwrap();
        assert_eq!(client.next_id(), 1);
        assert_eq!(client.next_id(), 2);
        assert_eq!(client.next_id(), 3);
    }
}
