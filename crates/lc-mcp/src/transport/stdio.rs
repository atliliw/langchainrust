//! Stdio transport: spawns a child process, communicating over stdin/stdout with newline-delimited JSON.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{broadcast, Mutex};
use tokio::time::{timeout, Duration};

use super::backoff_delay;
use super::{MCPEvent, MCPTransport};
use crate::protocol::{MCPError, MCPRequest, MCPResponse};
use crate::types::MCPConfig;

/// Stdio transport: spawns a child process, communicating over stdin/stdout with newline-delimited JSON
///
/// P0-2: a background monitor task detects the child-process exit, then auto-respawns with exponential
/// backoff. `is_connected()` exposes the current connection state; requests during a disconnect return
/// `MCPError::connection_lost()` for the upper layer to trigger a reconnect.
pub struct StdioTransport {
    /// Keeps the original config so the child process can be respawned on reconnect.
    config: MCPConfig,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
    child: Arc<Mutex<Option<Child>>>,
    /// Per-request lock: ensures write + read is atomic for each request.
    request_lock: Arc<Mutex<()>>,
    /// Per-request timeout (M1): a child that swallows the request cannot make
    /// the caller wait forever. Tests shorten it with [`Self::with_request_timeout`].
    request_timeout: Duration,
    /// Connection state (true = the child process is alive).
    connected: Arc<AtomicBool>,
    /// Manual close flag: after close() no more auto-reconnect.
    closed: Arc<AtomicBool>,
    event_tx: broadcast::Sender<MCPEvent>,
}

/// Default per-request timeout (M1).
const STDIO_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// A child-process spawn result (a temporary struct, split into fields after new).
struct SpawnedProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

fn spawn_process(config: &MCPConfig) -> Result<SpawnedProcess, MCPError> {
    let (command, args, env) = match config {
        MCPConfig::Stdio { command, args, env } => (command, args, env),
        _ => return Err(MCPError::new(-1, "StdioTransport requires Stdio config")),
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
        .map_err(|e| MCPError::new(-1, format!("failed to spawn child process: {}", e)))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| MCPError::new(-1, "child process has no stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| MCPError::new(-1, "child process has no stdout"))?;

    // Capture stderr in a background task so it doesn't block the process
    //
    // P1-4: uses `tracing` (with the `log` feature, events also go through the log facade, compatible with the
    // workspace `env_logger`). Coarse level split by content: error/panic → error!, warn → warn!, the rest →
    // debug!, filtered by target.
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

/// Read from `reader` until the JSON-RPC response matching `req_id` arrives.
///
/// The child may interleave notifications (no `id`) or a stale response to a
/// previously timed-out request (M1), so the stdout buffer must not be consumed
/// as a single line. Only a message whose `id` equals `req_id` — and that
/// actually carries a `result` or `error` (a JSON-RPC response has exactly one)
/// — is returned; everything else is skipped (M2, 0.20.0). EOF means the child
/// exited; the background monitor will auto-reconnect.
pub(super) async fn read_response_for<R>(
    reader: &mut R,
    req_id: u64,
) -> Result<MCPResponse, MCPError>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| MCPError::new(-1, format!("failed to read stdout: {e}")))?;
        if n == 0 {
            return Err(MCPError::connection_lost());
        }
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<MCPResponse>(line.trim()) {
            Ok(resp)
                if resp.id == Some(req_id) && (resp.result.is_some() || resp.error.is_some()) =>
            {
                return Ok(resp);
            }
            _ => continue,
        }
    }
}

impl StdioTransport {
    /// Creates a Stdio transport: spawns the child process and establishes the stdin/stdout channels.
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
            request_timeout: STDIO_REQUEST_TIMEOUT,
            connected: connected.clone(),
            closed: closed.clone(),
            event_tx: event_tx.clone(),
        };
        transport.spawn_monitor();
        Ok(transport)
    }

    /// Sets the per-request timeout (M1, default 30s). Mainly used by tests to shorten
    /// the timeout window; only exists in test builds, to avoid a dead-code warning in
    /// non-test builds under `-D warnings`.
    #[cfg(test)]
    pub(crate) fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Background monitor task: waits for the child process to exit → marks disconnected → auto-reconnects
    /// with exponential backoff.
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
                // Take out the current child process and wait for its exit (avoid holding the lock for a long
                // time and blocking request/close)
                let child_status = {
                    let mut guard = child.lock().await;
                    match guard.take() {
                        Some(mut c) => c.wait().await,
                        None => return, // already taken by close()
                    }
                };
                log::warn!(
                    "MCP child process exited: {:?}, starting auto-reconnect",
                    child_status
                );
                connected.store(false, Ordering::SeqCst);
                let _ = event_tx.send(MCPEvent::Disconnected);

                // Exponential-backoff reconnect
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
                            log::info!("MCP child process reconnected successfully");
                            break;
                        }
                        Err(e) => {
                            log::warn!(
                                "MCP child process reconnect failed (attempt {}): {}",
                                attempt,
                                e
                            );
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
            .map_err(|e| MCPError::new(-1, format!("failed to serialize request: {}", e)))?;

        // M1 (0.20.0): the whole write + read is bounded by `request_timeout`,
        // so a child that swallows the request cannot block the caller forever.
        let req_id = req.id;
        let result = timeout(self.request_timeout, async {
            {
                let mut stdin = self.stdin.lock().await;
                stdin
                    .write_all(json.as_bytes())
                    .await
                    .map_err(|e| MCPError::new(-1, format!("failed to write to stdin: {e}")))?;
                stdin
                    .write_all(b"\n")
                    .await
                    .map_err(|e| MCPError::new(-1, format!("failed to write newline: {e}")))?;
                stdin
                    .flush()
                    .await
                    .map_err(|e| MCPError::new(-1, format!("failed to flush stdin: {e}")))?;
            }
            // M2 (0.20.0): correlate by request id — skip interleaved
            // notifications and stale responses to other (e.g. previously
            // timed-out) requests.
            let mut stdout = self.stdout.lock().await;
            read_response_for(&mut *stdout, req_id).await
        })
        .await;

        match result {
            Ok(resp) => resp,
            Err(_) => Err(MCPError::new(
                -1,
                format!("stdio request timed out after {:?}", self.request_timeout),
            )),
        }
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
        // JSON-RPC 2.0 notification: no id field
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
            .map_err(|e| MCPError::new(-1, format!("failed to serialize notification: {}", e)))?;
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(json.as_bytes()).await.map_err(|e| {
            MCPError::new(-1, format!("failed to write notification to stdin: {}", e))
        })?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| MCPError::new(-1, format!("failed to write newline: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| MCPError::new(-1, format!("failed to flush stdin: {}", e)))?;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn reconnect(&self) -> Result<(), MCPError> {
        if self.connected.load(Ordering::SeqCst) {
            return Ok(());
        }
        // The background monitor is already auto-reconnecting; wait here for it to recover.
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
