//! SSE transport: long connection + background line-by-line streaming read, continuously consuming
//! server-pushed events.

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

/// Total timeout for one MCP request (POST send + reading the response body).
///
/// When the server "connected but swallows the response without replying", a clear error is returned after
/// this duration and the existing "invalidate cache → reconnect → retry once" path is taken, avoiding a
/// permanent hang for the caller.
const MCP_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// SSE transport for MCP.
///
/// P0-1: after establishing the SSE long connection, a background task continuously reads the event stream
/// line by line, consuming `endpoint`/`progress`/`logging` etc. events; no longer a one-shot `text()` body
/// read.
pub struct SseTransport {
    /// SSE endpoint URL (for receiving events).
    sse_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// Timeout for a single POST request (send + read response body). Defaults to [`MCP_REQUEST_TIMEOUT`];
    /// tests can shorten it with [`SseTransport::with_request_timeout`].
    request_timeout: Duration,
    /// POST endpoint URL (for sending messages). Filled by the reader loop.
    ///
    /// P1-3: a `watch` channel replaces `Mutex<Option<>>` — `borrow()` reads without locking, so concurrent
    /// discovery does not block each other; `send(None)` clears the cache when invalidated, and after a
    /// reconnect the read loop's `send(Some(new address))` overwrites directly.
    ///
    /// Why not OnceCell: std's sync variant is really [`std::sync::OnceLock`] — no `take()`, and once set it
    /// cannot be overwritten (a reconnect cannot refresh it); once_cell's `OnceCell::take` needs `&mut self`,
    /// unavailable under shared `Arc`. watch is semantically equivalent (lock-free read + invalidatable) and
    /// is a native tokio channel.
    post_url_tx: watch::Sender<Option<String>>,
    post_url_rx: watch::Receiver<Option<String>>,
    /// Whether the long connection is held.
    connected: Arc<AtomicBool>,
    /// Manual close flag.
    closed: Arc<AtomicBool>,
    /// Reconnect signal (the background read loop disconnects and reconnects when it receives this).
    reconnect_signal: watch::Sender<u64>,
    /// The read loop only starts once.
    reader_started: Arc<AtomicBool>,
    event_tx: broadcast::Sender<MCPEvent>,
    /// Pending request registry (F4): once a POST is sent, if the server "replies 202 first and pushes the
    /// response over SSE", the background read loop correlates it here by JSON-RPC `id` and delivers the pushed
    /// response back to `post_request` via oneshot. When the connection drops, the registry is cleared so
    /// waiters fail out with "push channel closed", taking the `request` invalidate-cache → reconnect → retry
    /// path.
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<MCPResponse>>>>,
}

impl SseTransport {
    /// Creates an SSE transport (requires the config to be `MCPConfig::Sse`).
    pub fn new(config: &MCPConfig) -> Result<Self, MCPError> {
        let sse_url = match config {
            MCPConfig::Sse { url } => url.clone(),
            _ => return Err(MCPError::new(-1, "SseTransport requires SSE config")),
        };
        let (event_tx, _) = broadcast::channel(64);
        let (reconnect_signal, _) = watch::channel(0u64);
        let (post_url_tx, post_url_rx) = watch::channel(None);
        // F2: the client only sets connect_timeout (bounds only the handshake, not the SSE long connection);
        // the total request timeout is applied per-POST via `timeout` wrapping in post_request / notify, so a
        // total timeout never kills the long-lived GET.
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

    /// Sets the per-request timeout (default 30s). Mainly used by tests to shorten the timeout window;
    /// only exists in test builds, to avoid a dead-code warning in non-test builds under `-D warnings`.
    #[cfg(test)]
    pub(crate) fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Ensures the background read loop has started (lazy, only once).
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
                // Re-subscribe to the reconnect signal before each (re)connection.
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

                // Line-by-line streaming read (long connection held, continuously consuming server pushes)
                let stream = response
                    .bytes_stream()
                    .map_err(|e| io::Error::other(e.to_string()));
                let reader = BufReader::new(StreamReader::new(stream));
                let mut lines = reader.lines();
                let mut current_event = String::new();

                loop {
                    tokio::select! {
                        // Received a reconnect signal → proactively disconnect the current connection
                        _ = reconnect_rx.changed() => {
                            log::debug!("SSE received reconnect signal, closing current connection");
                            break;
                        }
                        // P1-2 heartbeat: no data (including `: keep-alive` comment lines) received within
                        // SSE_HEARTBEAT → judge the connection broken and trigger a reconnect.
                        line = timeout(SSE_HEARTBEAT, lines.next_line()) => {
                            match line {
                                Ok(Ok(Some(l))) => {
                                    if let Some((evt, data)) = parse_sse_line(&l, &mut current_event) {
                                        if evt == "endpoint" {
                                            let _ = post_url.send(Some(data));
                                        } else if evt == "message" {
                                            // F4: a JSON-RPC response pushed over SSE, delivered to the pending
                                            // POST by `id`.
                                            if let Ok(resp) = serde_json::from_str::<MCPResponse>(&data) {
                                                if let Some(id) = resp.id {
                                                    if let Some(tx) = pending.lock().unwrap().remove(&id) {
                                                        let _ = tx.send(resp);
                                                        continue;
                                                    }
                                                }
                                            }
                                            // No matching pending request → broadcast as an ordinary message
                                            // event.
                                            let params = serde_json::from_str::<serde_json::Value>(&data).ok();
                                            let _ = event_tx.send(MCPEvent::Message { method: evt, params });
                                        } else if !data.is_empty() {
                                            let params = serde_json::from_str::<serde_json::Value>(&data).ok();
                                            let _ = event_tx.send(MCPEvent::Message { method: evt, params });
                                        }
                                    }
                                }
                                Ok(Ok(None)) => {
                                    // EOF → connection dropped
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

                // F4: connection dropped, clear the pending registry — waiting requests receive no push, their
                // oneshots are dropped and fail out with "push channel closed", taking the `request`
                // invalidate-cache → reconnect → retry-once path.
                pending.lock().unwrap().clear();

                // Wait a moment before reconnecting to avoid a busy loop
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });
    }

    /// Discover the POST endpoint from the SSE stream.
    ///
    /// The background read loop is responsible for establishing the long connection and filling `post_url`;
    /// this waits for it to be ready. P1-3: `watch::Receiver::borrow()` reads without locking — no more
    /// contending for the `Mutex`, concurrent calls do not block each other.
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

    /// Clears the post_url cache and triggers the background read loop to reconnect and re-discover the
    /// endpoint (P1-1).
    ///
    /// Called on a POST failure or a dropped connection: a stale cache must not be reused by later requests.
    fn invalidate_endpoint(&self) {
        let _ = self.post_url_tx.send(None);
        self.connected.store(false, Ordering::SeqCst);
        let _ = self.reconnect_signal.send_if_modified(|n| {
            *n = n.wrapping_add(1);
            true
        });
    }

    /// Sends one POST request and parses the MCP response.
    ///
    /// First registers the pending request by JSON-RPC `id` (F4): if the server "replies 202 first and pushes
    /// the response over SSE", the background read loop delivers the pushed response to this oneshot; when the
    /// POST returns JSON directly, this registration is removed in the cleanup stage, leaving no leak. The
    /// registry is cleared whether it succeeded or not, and concurrent requests do not interfere.
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

        // Cleanup: the read loop may have already removed (taking the tx to deliver), in which case remove
        // returning None is harmless; a late push is naturally ignored because the registry is empty.
        if let Some(id) = req_id {
            self.pending.lock().unwrap().remove(&id);
        }
        result
    }

    /// POSTs and waits for a response (called by [`SseTransport::post_request`], so cleanup can be shared).
    ///
    /// Parse order: first try parsing the POST response body directly (compatible with our own and
    /// direct-response servers); if it does not parse, wait for the response pushed over the SSE long
    /// connection by `id` (F4, 202 + push-style servers).
    ///
    /// F2: both the send and the response-body read are bounded by `request_timeout` — when the server
    /// swallows the response without replying, an error carrying "timed out" is returned instead of a
    /// permanent hang. A timeout error takes the existing "invalidate cache → reconnect → retry once" path in
    /// `request`.
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

        // F4: the POST is already sent (may have returned 202 Accepted / empty body); the response is pushed
        // over the SSE long connection — wait for the background read loop to deliver by `id`. If the
        // connection drops, the read loop clears the registry, this side's oneshot is dropped (RecvError) →
        // returns with a clear error, and request reconnects and retries.
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

/// Parses one line of SSE text.
///
/// Returns `Some((event, data))` when it is a `data:` line; `None` for an event-name line (`event:`) or other
/// lines. The event-name state is maintained in `current_event`.
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
                // P1-1: POST failed (network error / non-2xx HTTP / unparsable response) → clear the stale
                // cache + trigger a reconnect to re-discover the endpoint, then retry once. Still failing
                // returns the (possible) first error, without letting the upper layer retry repeatedly.
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
        // Ensure the background read loop exists (lazy: started on first call), otherwise after invalidate
        // nobody holds the long connection and connected would never recover (P1-1 initial connection also
        // uses this method).
        self.ensure_reader();
        // Clear the cache + trigger the background read loop to disconnect the current connection and
        // reconnect
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
