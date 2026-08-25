//! SSE 传输:长连接 + 后台逐行流式读取,持续消费服务器推送事件。

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::TryStreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{broadcast, oneshot, watch};
use tokio::time::{timeout, Duration};
use tokio_util::io::StreamReader;

use super::{MCPEvent, MCPTransport, SSE_DISCOVER_TIMEOUT, SSE_HEARTBEAT};
use crate::protocol::{MCPError, MCPRequest, MCPResponse};
use crate::types::MCPConfig;

/// MCP 单次请求总超时(POST 发送 + 读响应体)。
///
/// 服务器"连上了但吞响应不回复"时,在此时间后返回清晰错误并走现有的
/// "失效缓存 → 重连 → 重试一次"路径,避免调用方永久挂起。
const MCP_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// SSE transport for MCP.
///
/// P0-1: 建立 SSE 长连接后由后台 task 持续逐行读取事件流,消费
/// `endpoint`/`progress`/`logging` 等事件;不再用 `text()` 一次性读 body。
pub struct SseTransport {
    /// SSE endpoint URL (for receiving events).
    sse_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// 单次 POST 请求超时(发送 + 读响应体)。默认 [`MCP_REQUEST_TIMEOUT`];
    /// 测试可用 [`SseTransport::with_request_timeout`] 缩短。
    request_timeout: Duration,
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
    /// 挂起请求登记(F4):POST 发出后若服务器"先回 202、响应经 SSE 推送",
    /// 后台读循环按 JSON-RPC `id` 关联到这里,用 oneshot 把推送的响应投递回
    /// `post_request`。连接断开时清空登记,让等待方以"推送通道关闭"失败退出,
    /// 走 `request` 的失效缓存 → 重连 → 重试路径。
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<MCPResponse>>>>,
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
        // F2: client 只配 connect_timeout(只约束建连,不影响 SSE 长连接);
        // 请求总超时按 POST 路径在 post_request / notify 里用 timeout 包裹,
        // 避免总超时误杀长连接的 GET。
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| MCPError::new(-1, format!("failed to build HTTP client: {}", e)))?;
        Ok(Self {
            sse_url,
            client,
            request_timeout: MCP_REQUEST_TIMEOUT,
            post_url_tx,
            post_url_rx,
            connected: Arc::new(AtomicBool::new(false)),
            closed: Arc::new(AtomicBool::new(false)),
            reconnect_signal,
            reader_started: Arc::new(AtomicBool::new(false)),
            event_tx,
            pending: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// 设置单次请求超时(默认 30s)。主要用于测试缩短超时窗口;
    /// 仅测试构建存在,避免 `-D warnings` 下非测试构建报 dead-code。
    #[cfg(test)]
    pub(crate) fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
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
        let pending = self.pending.clone();

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
                                        } else if evt == "message" {
                                            // F4:SSE 推送的 JSON-RPC 响应,按 `id` 投递给挂起的 POST。
                                            if let Ok(resp) = serde_json::from_str::<MCPResponse>(&data) {
                                                if let Some(id) = resp.id {
                                                    if let Some(tx) = pending.lock().unwrap().remove(&id) {
                                                        let _ = tx.send(resp);
                                                        continue;
                                                    }
                                                }
                                            }
                                            // 无匹配挂起请求 → 当作普通消息事件广播。
                                            let params = serde_json::from_str::<serde_json::Value>(&data).ok();
                                            let _ = event_tx.send(MCPEvent::Message { method: evt, params });
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

                // F4:连接断开,清空挂起登记——等待中的请求收不到推送,
                // 对应 oneshot 被 drop,以"推送通道关闭"失败退出,由
                // `request` 走"失效缓存 → 重连 → 重试一次"路径。
                pending.lock().unwrap().clear();

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

    /// 发送一次 POST 请求并解析 MCP 响应。
    ///
    /// 先按 JSON-RPC `id` 登记挂起请求(F4):若服务器"先回 202、响应经 SSE
    /// 推送",后台读循环会把推送的响应投递到这个 oneshot;POST 直接回 JSON
    /// 时这个登记在收尾阶段移除,不留泄漏。无论成败都清登记,多个请求并发
    /// 时互不干扰。
    async fn post_request(
        &self,
        post_url: &str,
        body: &serde_json::Value,
    ) -> Result<MCPResponse, MCPError> {
        let req_id = body.get("id").and_then(serde_json::Value::as_u64);
        let (pending_tx, pending_rx) = oneshot::channel();
        if let Some(id) = req_id {
            self.pending.lock().unwrap().insert(id, pending_tx);
        }

        let result = self.post_and_wait(post_url, body, pending_rx).await;

        // 收尾清理:读循环可能已 remove(把 tx 拿走投递),此时 remove 返回
        // None 无害;迟到的推送因登记已空而自然被忽略。
        if let Some(id) = req_id {
            self.pending.lock().unwrap().remove(&id);
        }
        result
    }

    /// POST 并等待响应(由 [`SseTransport::post_request`] 调用,便于收尾清理)。
    ///
    /// 解析顺序:先试 POST 响应体直接解析(兼容自家与直接响应型服务器);
    /// 解析不到则等 SSE 长连接按 `id` 推送的响应(F4,202 + 推送型服务器)。
    ///
    /// F2:发送与读响应体均受 `request_timeout` 约束——服务器吞响应不回复时,
    /// 返回带"timed out"的错误,而不是永久挂起。超时错误走 `request` 里现有的
    /// "失效缓存 → 重连 → 重试一次"路径。
    async fn post_and_wait(
        &self,
        post_url: &str,
        body: &serde_json::Value,
        pending_rx: oneshot::Receiver<MCPResponse>,
    ) -> Result<MCPResponse, MCPError> {
        let resp = timeout(
            self.request_timeout,
            self.client.post(post_url).json(body).send(),
        )
        .await
        .map_err(|_| {
            MCPError::new(
                -1,
                format!("HTTP POST timed out after {:?}", self.request_timeout),
            )
        })?
        .map_err(|e| MCPError::new(-1, format!("HTTP POST failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(MCPError::new(-1, format!("HTTP error: {}", status)));
        }

        // The server may respond directly or send the response via SSE.
        // For compatibility, try parsing the direct response first.
        let body = timeout(self.request_timeout, resp.text())
            .await
            .map_err(|_| {
                MCPError::new(
                    -1,
                    format!(
                        "Reading MCP response timed out after {:?}",
                        self.request_timeout
                    ),
                )
            })?
            .map_err(|e| MCPError::new(-1, format!("Failed to read response: {}", e)))?;

        // Try parsing as MCPResponse (direct-response servers).
        if let Ok(mcp_resp) = serde_json::from_str::<MCPResponse>(&body) {
            return Ok(mcp_resp);
        }

        // If not a direct response, try SSE `data:` lines inside the POST body
        // (some servers put the response there).
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

        // F4:POST 已发出(可能回了 202 Accepted / 空 body),响应经 SSE 长连接
        // 推送——等后台读循环按 `id` 投递。若连接断开,读循环清空登记表,
        // 本端的 oneshot 被 drop(RecvError)→ 以清晰错误返回,由 request 重连重试。
        timeout(self.request_timeout, pending_rx)
            .await
            .map_err(|_| {
                MCPError::new(
                    -1,
                    format!(
                        "waiting for SSE-pushed response timed out after {:?}",
                        self.request_timeout
                    ),
                )
            })?
            .map_err(|_| MCPError::new(-1, "SSE connection closed before response was pushed"))
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
        timeout(
            self.request_timeout,
            self.client.post(&post_url).json(&payload).send(),
        )
        .await
        .map_err(|_| {
            MCPError::new(
                -1,
                format!(
                    "Sending notification timed out after {:?}",
                    self.request_timeout
                ),
            )
        })?
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
