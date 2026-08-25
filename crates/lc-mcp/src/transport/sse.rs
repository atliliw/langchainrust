//! SSE 传输:长连接 + 后台逐行流式读取,持续消费服务器推送事件。

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::TryStreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{broadcast, watch};
use tokio::time::{timeout, Duration};
use tokio_util::io::StreamReader;

use super::{MCPEvent, MCPTransport, SSE_DISCOVER_TIMEOUT, SSE_HEARTBEAT};
use crate::protocol::{MCPError, MCPRequest, MCPResponse};
use crate::types::MCPConfig;

/// SSE transport for MCP.
///
/// P0-1: 建立 SSE 长连接后由后台 task 持续逐行读取事件流,消费
/// `endpoint`/`progress`/`logging` 等事件;不再用 `text()` 一次性读 body。
pub struct SseTransport {
    /// SSE endpoint URL (for receiving events).
    sse_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// POST endpoint URL (for sending messages). Filled by the reader loop.
    ///
    /// P1-3: 用 `watch` 通道替代 `Mutex<Option<>>` —— `borrow()` 无锁读,
    /// 并发 discovery 互不阻塞;失效时 `send(None)` 清空缓存,重连后读循环
    /// `send(Some(新地址))` 直接覆盖。
    ///
    /// 为何不用 OnceCell:std 的 sync 变体实为 [`std::sync::OnceLock`],
    /// 无 `take()` 且 set 一次后不可覆盖(重连无法刷新);once_cell 的
    /// `OnceCell::take` 需要 `&mut self`,`Arc` 共享下不可得。watch 在语义上
    /// 等价(无锁读 + 可失效)且是 tokio 原生通道。
    post_url_tx: watch::Sender<Option<String>>,
    post_url_rx: watch::Receiver<Option<String>>,
    /// 长连接是否保持。
    connected: Arc<AtomicBool>,
    /// 手动关闭标志。
    closed: Arc<AtomicBool>,
    /// 重连信号(后台读循环收到即断开重连)。
    reconnect_signal: watch::Sender<u64>,
    /// 读循环只启动一次。
    reader_started: Arc<AtomicBool>,
    event_tx: broadcast::Sender<MCPEvent>,
}

impl SseTransport {
    /// 创建 SSE 传输(要求配置为 `MCPConfig::Sse`)。
    pub fn new(config: &MCPConfig) -> Result<Self, MCPError> {
        let sse_url = match config {
            MCPConfig::Sse { url } => url.clone(),
            _ => return Err(MCPError::new(-1, "SseTransport requires SSE config")),
        };
        let (event_tx, _) = broadcast::channel(64);
        let (reconnect_signal, _) = watch::channel(0u64);
        let (post_url_tx, post_url_rx) = watch::channel(None);
        Ok(Self {
            sse_url,
            client: reqwest::Client::new(),
            post_url_tx,
            post_url_rx,
            connected: Arc::new(AtomicBool::new(false)),
            closed: Arc::new(AtomicBool::new(false)),
            reconnect_signal,
            reader_started: Arc::new(AtomicBool::new(false)),
            event_tx,
        })
    }

    /// 确保后台读循环已启动(惰性、只启动一次)。
    pub(crate) fn ensure_reader(&self) {
        if self.reader_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let sse_url = self.sse_url.clone();
        let client = self.client.clone();
        let post_url = self.post_url_tx.clone();
        let connected = self.connected.clone();
        let closed = self.closed.clone();
        let reconnect_signal = self.reconnect_signal.clone();
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            while !closed.load(Ordering::SeqCst) {
                // 每次(重)连接前重新订阅重连信号。
                let mut reconnect_rx = reconnect_signal.subscribe();

                let response = match client
                    .get(&sse_url)
                    .header("Accept", "text/event-stream")
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        connected.store(false, Ordering::SeqCst);
                        let _ = event_tx.send(MCPEvent::Disconnected);
                        log::warn!("SSE connection failed: {}, will retry", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };

                let status = response.status();
                if !status.is_success() {
                    connected.store(false, Ordering::SeqCst);
                    let _ = event_tx.send(MCPEvent::Disconnected);
                    log::warn!("SSE endpoint returned HTTP {}", status);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }

                connected.store(true, Ordering::SeqCst);
                let _ = event_tx.send(MCPEvent::Connected);

                // 逐行流式读取(长连接保持,持续消费服务器推送事件)
                let stream = response
                    .bytes_stream()
                    .map_err(|e| io::Error::other(e.to_string()));
                let reader = BufReader::new(StreamReader::new(stream));
                let mut lines = reader.lines();
                let mut current_event = String::new();

                loop {
                    tokio::select! {
                        // 收到重连信号 → 主动断开当前连接
                        _ = reconnect_rx.changed() => {
                            log::debug!("SSE received reconnect signal, closing current connection");
                            break;
                        }
                        // P1-2 心跳:超过 SSE_HEARTBEAT 未收到任何数据(含
                        // `: keep-alive` 注释行)→ 判定连接断开,触发重连。
                        line = timeout(SSE_HEARTBEAT, lines.next_line()) => {
                            match line {
                                Ok(Ok(Some(l))) => {
                                    if let Some((evt, data)) = parse_sse_line(&l, &mut current_event) {
                                        if evt == "endpoint" {
                                            let _ = post_url.send(Some(data));
                                        } else if !data.is_empty() {
                                            let params = serde_json::from_str::<serde_json::Value>(&data).ok();
                                            let _ = event_tx.send(MCPEvent::Message { method: evt, params });
                                        }
                                    }
                                }
                                Ok(Ok(None)) => {
                                    // EOF → 连接断开
                                    connected.store(false, Ordering::SeqCst);
                                    let _ = event_tx.send(MCPEvent::Disconnected);
                                    log::warn!("SSE connection ended, will retry");
                                    break;
                                }
                                Ok(Err(e)) => {
                                    connected.store(false, Ordering::SeqCst);
                                    let _ = event_tx.send(MCPEvent::Disconnected);
                                    log::warn!("SSE read error: {}, will retry", e);
                                    break;
                                }
                                Err(_elapsed) => {
                                    connected.store(false, Ordering::SeqCst);
                                    let _ = event_tx.send(MCPEvent::Disconnected);
                                    log::warn!("SSE heartbeat timeout ({:?}) with no data, treating connection as lost and reconnecting", SSE_HEARTBEAT);
                                    break;
                                }
                            }
                        }
                    }
                }

                // 断开后稍等再重连,避免忙循环
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });
    }

    /// Discover the POST endpoint from the SSE stream.
    ///
    /// 由后台读循环负责建立长连接并填充 `post_url`;此处等待其就绪。
    /// P1-3: `watch::Receiver::borrow()` 无锁读,不再抢 `Mutex`,并发调用互不阻塞。
    async fn discover_post_url(&self) -> Result<String, MCPError> {
        self.ensure_reader();
        let rx = self.post_url_rx.clone();
        timeout(SSE_DISCOVER_TIMEOUT, async {
            loop {
                if let Some(url) = rx.borrow().clone() {
                    return url;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .map_err(|_| MCPError::new(-1, "SSE discover timed out: no endpoint event"))
    }

    /// 清空 post_url 缓存并触发后台读循环重连、重新发现 endpoint(P1-1)。
    ///
    /// POST 失败或连接断开时调用:失效的缓存不能继续被后续请求复用。
    fn invalidate_endpoint(&self) {
        let _ = self.post_url_tx.send(None);
        self.connected.store(false, Ordering::SeqCst);
        let _ = self.reconnect_signal.send_if_modified(|n| {
            *n = n.wrapping_add(1);
            true
        });
    }

    /// 发送一次 POST 请求并解析 MCP 响应(直接 JSON 或 SSE `data:` 行)。
    async fn post_request(
        &self,
        post_url: &str,
        body: &serde_json::Value,
    ) -> Result<MCPResponse, MCPError> {
        let resp = self
            .client
            .post(post_url)
            .json(body)
            .send()
            .await
            .map_err(|e| MCPError::new(-1, format!("HTTP POST failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(MCPError::new(-1, format!("HTTP error: {}", status)));
        }

        // The server may respond directly or send the response via SSE.
        // For compatibility, try parsing the direct response first.
        let body = resp
            .text()
            .await
            .map_err(|e| MCPError::new(-1, format!("Failed to read response: {}", e)))?;

        // Try parsing as MCPResponse
        if let Ok(mcp_resp) = serde_json::from_str::<MCPResponse>(&body) {
            return Ok(mcp_resp);
        }

        // If not a direct response, try SSE event format
        // Parse SSE "data:" lines and extract JSON
        for line in body.lines() {
            if let Some(data) = line.strip_prefix("data:") {
                let trimmed = data.trim();
                if !trimmed.is_empty() {
                    if let Ok(mcp_resp) = serde_json::from_str::<MCPResponse>(trimmed) {
                        return Ok(mcp_resp);
                    }
                }
            }
        }

        Err(MCPError::new(
            -1,
            format!("Could not parse MCP response from: {}", body),
        ))
    }
}

/// 解析一行 SSE 文本。
///
/// 返回 `Some((event, data))` 表示这是 `data:` 行;`None` 表示
/// 事件名行(`event:`)或其他行。事件名状态在 `current_event` 中维护。
pub(crate) fn parse_sse_line(line: &str, current_event: &mut String) -> Option<(String, String)> {
    let trimmed = line.trim_end();
    if let Some(stripped) = trimmed.strip_prefix("event:") {
        *current_event = stripped.trim().to_string();
        return None;
    }
    trimmed
        .strip_prefix("data:")
        .map(|data| (current_event.clone(), data.trim().to_string()))
}

#[async_trait]
impl MCPTransport for SseTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
        self.ensure_reader();
        if !self.connected.load(Ordering::SeqCst) {
            return Err(MCPError::connection_lost());
        }
        let post_url = self.discover_post_url().await?;
        let body = serde_json::to_value(&req)
            .map_err(|e| MCPError::new(-1, format!("failed to serialize request: {}", e)))?;

        match self.post_request(&post_url, &body).await {
            Ok(resp) => Ok(resp),
            Err(first) => {
                // P1-1: POST 失败(网络错误 / HTTP 非 2xx / 响应不可解析)→
                // 清空失效缓存 + 触发重连重新发现 endpoint,再重试一次。
                // 仍失败则返回(可能携带的)首个错误,不让上层重复重试。
                log::warn!(
                    "SSE POST failed ({}), clearing post_url cache and rediscovering before one retry",
                    first
                );
                self.invalidate_endpoint();
                let post_url = self.discover_post_url().await?;
                match self.post_request(&post_url, &body).await {
                    Ok(resp) => Ok(resp),
                    Err(_second) => Err(first),
                }
            }
        }
    }

    async fn close(&self) -> Result<(), MCPError> {
        self.closed.store(true, Ordering::SeqCst);
        self.connected.store(false, Ordering::SeqCst);
        // Clear cached endpoint
        let _ = self.post_url_tx.send(None);
        Ok(())
    }

    async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), MCPError> {
        self.ensure_reader();
        if !self.connected.load(Ordering::SeqCst) {
            return Err(MCPError::connection_lost());
        }
        let post_url = self.discover_post_url().await?;

        let mut payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(p) = params {
            // M11 fix: use defensive check instead of unwrap
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("params".to_string(), p);
            }
        }
        self.client
            .post(&post_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| MCPError::new(-1, format!("Failed to send notification: {}", e)))?;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn reconnect(&self) -> Result<(), MCPError> {
        // 确保后台读循环存在(惰性:首次调用才启动),否则 invalidate 后
        // 无人拉长连接,connected 永远不会恢复(P1-1 初始连接也用本方法)。
        self.ensure_reader();
        // 清空缓存 + 触发后台读循环断开当前连接并重连
        self.invalidate_endpoint();
        timeout(Duration::from_secs(30), async {
            while !self.connected.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .map_err(|_| MCPError::connection_lost())
    }

    fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent> {
        self.event_tx.subscribe()
    }
}
