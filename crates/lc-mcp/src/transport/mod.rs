//! MCP 传输层:Stdio + SSE
//!
//! P0 修复:
//! - P0-1: SSE 由一次性 `text()` 读 body 改为长连接 + 后台逐行流式读取,
//!   持续消费服务器推送事件(progress/logging 等)。
//! - P0-2: Stdio 子进程崩溃后后台监控 + 指数退避自动重连,连接状态可查询。

mod sse;
mod stdio;

pub use sse::SseTransport;
pub use stdio::StdioTransport;

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

use crate::protocol::{MCPError, MCPRequest, MCPResponse};
use async_trait::async_trait;

/// Default timeout for SSE endpoint discovery (30 seconds).
const SSE_DISCOVER_TIMEOUT: Duration = Duration::from_secs(30);

/// SSE 心跳间隔:读循环在此时长内未收到任何数据(含 `: keep-alive` 注释行)
/// 即判定连接断开,触发重连(P1-2)。
const SSE_HEARTBEAT: Duration = Duration::from_secs(30);

/// 子进程重连退避上限(秒)。
const MAX_RECONNECT_BACKOFF_MS: u64 = 30_000;

/// MCP 传输层事件(服务器推送 / 连接状态变化)。
#[derive(Debug, Clone)]
pub enum MCPEvent {
    /// 连接已建立。
    Connected,
    /// 连接已断开(子进程退出 / SSE 中断)。
    Disconnected,
    /// 服务器主动推送的消息(SSE `event:`/`data:` 行)。
    Message {
        /// SSE 事件名,如 `logging`、`progress`。
        method: String,
        /// 解析后的 data(若可解析为 JSON)。
        params: Option<serde_json::Value>,
    },
}

/// MCP 传输层抽象
#[async_trait]
pub trait MCPTransport: Send + Sync {
    /// 发送请求并等待响应
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError>;
    /// 发送通知(不等响应)
    async fn notify(&self, method: &str, params: Option<serde_json::Value>)
        -> Result<(), MCPError>;
    /// 关闭连接
    async fn close(&self) -> Result<(), MCPError>;
    /// 连接是否存活(子进程存活 / SSE 长连接保持)。
    fn is_connected(&self) -> bool;
    /// 重连并等待恢复(断连后由上层触发)。
    async fn reconnect(&self) -> Result<(), MCPError>;
    /// 订阅服务器推送事件。
    fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent>;
}

/// 子进程重连指数退避:0.5s → 1s → 2s → 4s → ... 上限 30s。
fn backoff_delay(attempt: u32) -> Duration {
    let ms = 500u64
        .checked_shl(attempt.min(6))
        .unwrap_or(MAX_RECONNECT_BACKOFF_MS);
    Duration::from_millis(ms.min(MAX_RECONNECT_BACKOFF_MS))
}

/// 进程内传输:把 [`MCPClient`](crate::client::MCPClient) 直接接到一个
/// [`MCPServer`](crate::MCPServer) 上,
/// 不走子进程 / 网络,便于嵌入式集成与测试(P2-6)。
///
/// 请求经 [`MCPServer::handle_request`](crate::MCPServer::handle_request) 原地处理;通知(`notifications/initialized`
/// 等)无需响应,直接忽略;事件通道仅广播一次 `Connected`(没有服务器推送)。
pub struct InMemoryTransport {
    server: Arc<crate::MCPServer>,
    event_tx: broadcast::Sender<MCPEvent>,
}

impl InMemoryTransport {
    /// 包住一个进程内的 MCP Server。
    pub fn new(server: Arc<crate::MCPServer>) -> Self {
        let (event_tx, _) = broadcast::channel(64);
        let _ = event_tx.send(MCPEvent::Connected);
        // P2-9 流式工具输出:订阅 server 的 `publish_partial`,把每个增量
        // 片段转成 `notifications/tool_partial` 推送事件,客户端事件监听
        // 路由给 `subscribe_tool_stream`。无客户端监听时静默丢弃。
        let mut partial_rx = server.subscribe_partials();
        let fwd = event_tx.clone();
        tokio::spawn(async move {
            loop {
                match partial_rx.recv().await {
                    Ok(partial) => {
                        let params = serde_json::to_value(&partial).ok();
                        let evt = MCPEvent::Message {
                            method: "notifications/tool_partial".to_string(),
                            params,
                        };
                        if fwd.send(evt).is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
        Self { server, event_tx }
    }
}

#[async_trait]
impl MCPTransport for InMemoryTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
        Ok(self.server.handle_request(req).await)
    }

    async fn notify(
        &self,
        _method: &str,
        _params: Option<serde_json::Value>,
    ) -> Result<(), MCPError> {
        Ok(())
    }

    async fn close(&self) -> Result<(), MCPError> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        true
    }

    async fn reconnect(&self) -> Result<(), MCPError> {
        Ok(())
    }

    fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent> {
        self.event_tx.subscribe()
    }
}

#[cfg(test)]
mod tests;
