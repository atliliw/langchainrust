//! MCP 传输层:Stdio + SSE
//!
//! P0 修复:
//! - P0-1: SSE 由一次性 `text()` 读 body 改为长连接 + 后台逐行流式读取,
//!   持续消费服务器推送事件(progress/logging 等)。
//! - P0-2: Stdio 子进程崩溃后后台监控 + 指数退避自动重连,连接状态可查询。

use async_trait::async_trait;
use futures_util::TryStreamExt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{broadcast, watch, Mutex};
use tokio::time::{timeout, Duration};
use tokio_util::io::StreamReader;

use super::protocol::{MCPError, MCPRequest, MCPResponse};
use super::types::MCPConfig;

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

/// Stdio 传输:启动子进程,通过 stdin/stdout 以换行分隔的 JSON 通信
///
/// P0-2: 后台监控 task 检测子进程退出后,以指数退避自动重新 spawn。
/// `is_connected()` 暴露当前连接状态,断连期间请求返回
/// `MCPError::connection_lost()` 供上层触发重连。
pub struct StdioTransport {
    /// 保存原始配置以便重连时重新 spawn 子进程。
    config: MCPConfig,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
    child: Arc<Mutex<Option<Child>>>,
    /// Per-request lock: ensures write + read is atomic for each request.
    request_lock: Arc<Mutex<()>>,
    /// 连接状态(true = 子进程存活)。
    connected: Arc<AtomicBool>,
    /// 手动关闭标志:close() 后不再自动重连。
    closed: Arc<AtomicBool>,
    event_tx: broadcast::Sender<MCPEvent>,
}

/// 子进程 spawn 结果(临时结构,new 后拆入字段)。
struct SpawnedProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

fn spawn_process(config: &MCPConfig) -> Result<SpawnedProcess, MCPError> {
    let (command, args, env) = match config {
        MCPConfig::Stdio { command, args, env } => (command, args, env),
        _ => return Err(MCPError::new(-1, "StdioTransport 需要 Stdio 配置")),
    };

    let mut cmd = Command::new(command);
    cmd.args(args);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| MCPError::new(-1, format!("启动子进程失败: {}", e)))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| MCPError::new(-1, "子进程无 stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| MCPError::new(-1, "子进程无 stdout"))?;

    // Capture stderr in a background task so it doesn't block the process
    //
    // P1-4: 用 `tracing`(开 `log` feature,事件同时进 log facade,与工作区
    // `env_logger` 兼容)。按内容粗分级别:error/panic → error!,warn → warn!,
    // 其余 → debug!,并按 target 过滤。
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let lower = line.to_ascii_lowercase();
                if lower.contains("error") || lower.contains("panic") {
                    tracing::error!(target: "mcp::stdio::stderr", "{}", line);
                } else if lower.contains("warn") {
                    tracing::warn!(target: "mcp::stdio::stderr", "{}", line);
                } else {
                    tracing::debug!(target: "mcp::stdio::stderr", "{}", line);
                }
            }
        });
    }

    Ok(SpawnedProcess {
        child,
        stdin,
        stdout,
    })
}

impl StdioTransport {
    pub async fn new(config: &MCPConfig) -> Result<Self, MCPError> {
        let spawned = spawn_process(config)?;
        let (event_tx, _) = broadcast::channel(64);
        let connected = Arc::new(AtomicBool::new(true));
        let closed = Arc::new(AtomicBool::new(false));
        let transport = Self {
            config: config.clone(),
            stdin: Arc::new(Mutex::new(spawned.stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(spawned.stdout))),
            child: Arc::new(Mutex::new(Some(spawned.child))),
            request_lock: Arc::new(Mutex::new(())),
            connected: connected.clone(),
            closed: closed.clone(),
            event_tx: event_tx.clone(),
        };
        transport.spawn_monitor();
        Ok(transport)
    }

    /// 后台监控 task:等待子进程退出 → 标记断连 → 指数退避自动重连。
    fn spawn_monitor(&self) {
        let child = self.child.clone();
        let stdin = self.stdin.clone();
        let stdout = self.stdout.clone();
        let connected = self.connected.clone();
        let closed = self.closed.clone();
        let event_tx = self.event_tx.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                // 取出当前子进程等待退出(避免长时间持锁阻塞 request/close)
                let child_status = {
                    let mut guard = child.lock().await;
                    match guard.take() {
                        Some(mut c) => c.wait().await,
                        None => return, // 已被 close() 取走
                    }
                };
                log::warn!("MCP 子进程退出: {:?}, 开始自动重连", child_status);
                connected.store(false, Ordering::SeqCst);
                let _ = event_tx.send(MCPEvent::Disconnected);

                // 指数退避重连
                let mut attempt = 0u32;
                loop {
                    if closed.load(Ordering::SeqCst) {
                        return;
                    }
                    let delay = backoff_delay(attempt);
                    tokio::time::sleep(delay).await;
                    match spawn_process(&config) {
                        Ok(spawned) => {
                            *child.lock().await = Some(spawned.child);
                            *stdin.lock().await = spawned.stdin;
                            *stdout.lock().await = BufReader::new(spawned.stdout);
                            connected.store(true, Ordering::SeqCst);
                            let _ = event_tx.send(MCPEvent::Connected);
                            log::info!("MCP 子进程重连成功");
                            break;
                        }
                        Err(e) => {
                            log::warn!("MCP 子进程重连失败(attempt {}): {}", attempt, e);
                            attempt += 1;
                        }
                    }
                }
            }
        });
    }
}

#[async_trait]
impl MCPTransport for StdioTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
        if !self.connected.load(Ordering::SeqCst) {
            return Err(MCPError::connection_lost());
        }
        // Acquire per-request lock so write+read is atomic
        let _guard = self.request_lock.lock().await;

        let json = serde_json::to_string(&req)
            .map_err(|e| MCPError::new(-1, format!("序列化请求失败: {}", e)))?;

        {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(json.as_bytes())
                .await
                .map_err(|e| MCPError::new(-1, format!("写 stdin 失败: {}", e)))?;
            stdin
                .write_all(b"\n")
                .await
                .map_err(|e| MCPError::new(-1, format!("写换行失败: {}", e)))?;
            stdin
                .flush()
                .await
                .map_err(|e| MCPError::new(-1, format!("flush stdin 失败: {}", e)))?;
        }

        let mut line = String::new();
        {
            let mut stdout = self.stdout.lock().await;
            // 跳过空行,直到读到非空行
            loop {
                line.clear();
                let n = stdout
                    .read_line(&mut line)
                    .await
                    .map_err(|e| MCPError::new(-1, format!("读 stdout 失败: {}", e)))?;
                if n == 0 {
                    // 子进程退出,连接断开 → 后台 monitor 会自动重连
                    return Err(MCPError::connection_lost());
                }
                if !line.trim().is_empty() {
                    break;
                }
            }
        }

        serde_json::from_str::<MCPResponse>(line.trim())
            .map_err(|e| MCPError::new(-1, format!("解析响应失败: {} | 原文: {}", e, line)))
    }

    async fn close(&self) -> Result<(), MCPError> {
        self.closed.store(true, Ordering::SeqCst);
        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill().await;
        }
        Ok(())
    }

    async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), MCPError> {
        if !self.connected.load(Ordering::SeqCst) {
            return Err(MCPError::connection_lost());
        }
        // JSON-RPC 2.0 notification: 无 id 字段
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        let mut payload = notif;
        if let Some(p) = params {
            // M11 fix: use defensive check instead of unwrap
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("params".to_string(), p);
            }
        }
        let json = serde_json::to_string(&payload)
            .map_err(|e| MCPError::new(-1, format!("序列化通知失败: {}", e)))?;
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(json.as_bytes())
            .await
            .map_err(|e| MCPError::new(-1, format!("写通知到 stdin 失败: {}", e)))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| MCPError::new(-1, format!("写换行失败: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| MCPError::new(-1, format!("flush stdin 失败: {}", e)))?;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn reconnect(&self) -> Result<(), MCPError> {
        if self.connected.load(Ordering::SeqCst) {
            return Ok(());
        }
        // 后台 monitor 已在自动重连;此处等待其恢复。
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
    fn ensure_reader(&self) {
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
                        log::warn!("SSE 连接失败: {}, 稍后重连", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };

                let status = response.status();
                if !status.is_success() {
                    connected.store(false, Ordering::SeqCst);
                    let _ = event_tx.send(MCPEvent::Disconnected);
                    log::warn!("SSE endpoint 返回 HTTP {}", status);
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
                            log::debug!("SSE 收到重连信号,断开当前连接");
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
                                    log::warn!("SSE 连接已结束,稍后重连");
                                    break;
                                }
                                Ok(Err(e)) => {
                                    connected.store(false, Ordering::SeqCst);
                                    let _ = event_tx.send(MCPEvent::Disconnected);
                                    log::warn!("SSE 读取错误: {}, 稍后重连", e);
                                    break;
                                }
                                Err(_elapsed) => {
                                    connected.store(false, Ordering::SeqCst);
                                    let _ = event_tx.send(MCPEvent::Disconnected);
                                    log::warn!("SSE 心跳超时({:?})未收到数据,判定连接断开,重连", SSE_HEARTBEAT);
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
fn parse_sse_line(line: &str, current_event: &mut String) -> Option<(String, String)> {
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
            .map_err(|e| MCPError::new(-1, format!("序列化请求失败: {}", e)))?;

        match self.post_request(&post_url, &body).await {
            Ok(resp) => Ok(resp),
            Err(first) => {
                // P1-1: POST 失败(网络错误 / HTTP 非 2xx / 响应不可解析)→
                // 清空失效缓存 + 触发重连重新发现 endpoint,再重试一次。
                // 仍失败则返回(可能携带的)首个错误,不让上层重复重试。
                log::warn!(
                    "SSE POST 失败({}),清空 post_url 缓存并重新发现后重试一次",
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

/// 进程内传输:把 [`MCPClient`] 直接接到一个 [`MCPServer`](crate::MCPServer) 上,
/// 不走子进程 / 网络,便于嵌入式集成与测试(P2-6)。
///
/// 请求经 [`MCPServer::handle_request`] 原地处理;通知(`notifications/initialized`
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
#[allow(unused_imports)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stdio_transport_new_invalid_command() {
        let config = MCPConfig::stdio("nonexistent_command_xyz_zzz", vec![]);
        let result = StdioTransport::new(&config).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_sse_transport_new_wrong_config() {
        let config = MCPConfig::stdio("npx", vec![]);
        let result = SseTransport::new(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sse_transport_new_ok() {
        let config = MCPConfig::sse("http://localhost:3001/sse");
        let transport = SseTransport::new(&config);
        assert!(transport.is_ok());
    }

    #[test]
    fn test_backoff_delay_starts_small() {
        assert_eq!(backoff_delay(0), Duration::from_millis(500));
        assert_eq!(backoff_delay(1), Duration::from_millis(1000));
        assert_eq!(backoff_delay(2), Duration::from_millis(2000));
    }

    #[test]
    fn test_backoff_delay_capped() {
        // attempt=6 → 0.5 * 2^6 = 32s → 上限 30s
        assert_eq!(backoff_delay(6), Duration::from_millis(30_000));
        assert_eq!(backoff_delay(100), Duration::from_millis(30_000));
    }

    #[test]
    fn test_parse_sse_line_event_name() {
        let mut current = String::new();
        let result = parse_sse_line("event: endpoint", &mut current);
        assert!(result.is_none());
        assert_eq!(current, "endpoint");
    }

    #[test]
    fn test_parse_sse_line_data() {
        let mut current = "endpoint".to_string();
        let result = parse_sse_line("data: http://localhost:3001/message", &mut current);
        let (evt, data) = result.unwrap();
        assert_eq!(evt, "endpoint");
        assert_eq!(data, "http://localhost:3001/message");
    }

    #[test]
    fn test_parse_sse_line_other_line_ignored() {
        let mut current = String::new();
        let result = parse_sse_line(": keep-alive comment", &mut current);
        assert!(result.is_none());
        assert!(current.is_empty());
    }

    #[test]
    fn test_connection_lost_error() {
        let err = MCPError::connection_lost();
        assert!(err.is_connection_lost());
        let other = MCPError::new(-1, "boom");
        assert!(!other.is_connection_lost());
    }

    /// 等待 SSE 后台读循环建立连接(request 的早退检查要求 connected=true)。
    async fn wait_connected(transport: &SseTransport) {
        transport.ensure_reader();
        timeout(Duration::from_secs(5), async {
            while !transport.is_connected() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("SSE 连接应在 5s 内建立");
    }

    #[tokio::test]
    async fn test_sse_request_success() {
        // 正常流程:发现 endpoint → POST 成功。
        let server =
            crate::test_support::start_fake_sse_server(crate::test_support::PostMode::Quiet).await;
        let config = MCPConfig::sse(&server.sse_url);
        let transport = SseTransport::new(&config).unwrap();
        wait_connected(&transport).await;

        // 用未知方法(测试服务器对未识别方法回 {"ok": true})
        let req = MCPRequest::new(1, "ping", None);
        let resp = transport
            .request(req)
            .await
            .expect("request should succeed");
        assert!(!resp.is_error());
        assert_eq!(resp.result, Some(serde_json::json!({ "ok": true })));
    }

    #[tokio::test]
    async fn test_sse_request_retries_after_post_failure() {
        // P1-1:第一次 POST 返回 500 → 清空缓存 + 重连重发现 + 重试一次 → 成功。
        let server = crate::test_support::start_fake_sse_server(
            crate::test_support::PostMode::FailFirstPost,
        )
        .await;
        let config = MCPConfig::sse(&server.sse_url);
        let transport = SseTransport::new(&config).unwrap();
        wait_connected(&transport).await;

        // 用未知方法(测试服务器对未识别方法回 {"ok": true})
        let req = MCPRequest::new(1, "ping", None);
        let resp = transport.request(req).await.expect("retry should succeed");
        assert_eq!(resp.result, Some(serde_json::json!({ "ok": true })));
        // 至少发生了 2 次 POST(首次失败 + 重试成功),证明缓存确实被清掉重试了
        assert!(
            server.post_count.load(Ordering::SeqCst) >= 2,
            "expected >=2 POSTs after failure+retry, got {}",
            server.post_count.load(Ordering::SeqCst)
        );
    }

    /// P2-6: 进程内传输 + 真实 `MCPServer` 打通 Client↔Server 协议链路。
    ///
    /// 走 `MCPClient::with_transport`(握手) → `list_tools` → `call_tool`,
    /// 全程无子进程 / 网络,验证 `tools/call` 经 `BaseTool::run` 被真实执行。
    #[tokio::test]
    async fn test_in_memory_transport_round_trip() {
        use crate::MCPClient;
        use lc_core::tools::ToolError;
        use lc_core::BaseTool;

        struct EchoTool;
        #[async_trait]
        impl BaseTool for EchoTool {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "echo back input"
            }
            async fn run(&self, input: String) -> Result<String, ToolError> {
                Ok(input)
            }
        }

        let server =
            Arc::new(crate::MCPServer::new().with_tool(Arc::new(EchoTool) as Arc<dyn BaseTool>));
        let client = MCPClient::with_transport(Box::new(InMemoryTransport::new(server)))
            .await
            .unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let result = client
            .call_tool("echo", serde_json::json!({"msg": "hi"}))
            .await
            .unwrap();
        assert!(!result.is_error, "服务器工具不应报错");
        assert_eq!(result.text(), r#"{"msg":"hi"}"#);
    }
}
