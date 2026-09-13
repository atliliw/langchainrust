//! Official MCP stdio transport (B1, 0.22.4): JSON-RPC 2.0 over a local
//! subprocess's stdin/stdout pipes.
//!
//! This is the transport Claude Desktop and the official MCP SDKs use for
//! local servers: the client spawns the server process, one newline-delimited
//! JSON-RPC message per line (messages MUST NOT contain embedded newlines),
//! the server writes logs to stderr, and lifecycle follows the
//! `initialize` → `notifications/initialized` handshake performed by
//! [`crate::StdioMcpClient`].
//!
//! Design notes:
//! - a single writer task owns the child stdin and serializes outgoing frames;
//! - a reader task owns stdout and routes responses to pending request oneshot
//!   channels, answers server-initiated `ping`, and refuses other
//!   server-initiated methods (the client advertises no sampling/roots
//!   capabilities, so a compliant server never sends them);
//! - stderr is drained by a pump task into the `log` facade, so a chatty
//!   server can never block its own stdout pipe;
//! - process exit / pipe EOF fails every pending request with
//!   [`MCPError::connection_lost`] (code -32000).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::{oneshot, Mutex};

use crate::protocol::{
    notification_message, MCPError, MCPRequest, MCPResponse, MCP_ERROR_REQUEST_TIMEOUT,
};

/// Grace period for the child to exit after its stdin gets EOF, before kill.
const STDIO_SHUTDOWN_GRACE: Duration = Duration::from_millis(800);

/// Spawn specification for an MCP stdio server: program, arguments,
/// extra environment variables and optional working directory.
#[derive(Debug, Clone)]
pub struct StdioCommand {
    /// Executable (looked up on `PATH` when not an absolute path).
    pub program: PathBuf,
    /// Arguments passed after the program name.
    pub args: Vec<String>,
    /// Extra environment variables for the child (merged onto the parent
    /// environment; bearer tokens for hosted backends typically travel here).
    pub envs: Vec<(String, String)>,
    /// Working directory for the child (`None` inherits the client's).
    pub working_dir: Option<PathBuf>,
}

impl StdioCommand {
    /// Creates a spawn spec for `program` (a bare command name resolved on
    /// `PATH`, or any path accepted by [`tokio::process::Command::new`]).
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            envs: Vec::new(),
            working_dir: None,
        }
    }

    /// Appends one argument (chainable).
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Appends multiple arguments (chainable).
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Adds/overrides one environment variable for the child (chainable).
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }

    /// Sets the child's working directory (chainable).
    pub fn working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }
}

/// Shared transport state (held behind [`Arc`] so the reader/writer tasks and
/// the public handle share one session).
#[derive(Debug)]
struct Inner {
    /// Outgoing-frame channel sender; `None` after shutdown (channel closed).
    out_tx: Mutex<Option<tokio::sync::mpsc::UnboundedSender<String>>>,
    /// Pending requests keyed by JSON-RPC id.
    pending: Mutex<HashMap<u64, oneshot::Sender<MCPResponse>>>,
    next_id: AtomicU64,
    child: Mutex<Option<Child>>,
    closed: AtomicBool,
}

/// MCP stdio transport: one spawned subprocess, newline-delimited JSON-RPC.
#[derive(Debug)]
pub struct StdioTransport {
    inner: Arc<Inner>,
}

impl StdioTransport {
    /// Spawns the server process and starts the reader/writer/stderr tasks.
    ///
    /// Failure to spawn (missing executable, etc.) returns an
    /// [`MCPError::connection_lost`]-class error without any task running.
    pub fn spawn(command: StdioCommand) -> Result<Self, MCPError> {
        let mut cmd = Command::new(&command.program);
        cmd.args(&command.args)
            .envs(command.envs.iter().cloned())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Safety net: if the handle is dropped without `shutdown`, the
            // reaped child must not outlive the client as an orphan process.
            .kill_on_drop(true);
        if let Some(dir) = &command.working_dir {
            cmd.current_dir(dir);
        }

        let mut child = cmd.spawn().map_err(|e| {
            MCPError::new(
                -32000,
                format!(
                    "failed to spawn MCP stdio server '{}': {e}",
                    command.program.display()
                ),
            )
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| MCPError::new(-32000, "child stdin was not captured"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| MCPError::new(-32000, "child stdout was not captured"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| MCPError::new(-32000, "child stderr was not captured"))?;

        let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let inner = Arc::new(Inner {
            out_tx: Mutex::new(Some(out_tx)),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            child: Mutex::new(Some(child)),
            closed: AtomicBool::new(false),
        });

        tokio::spawn(stdout_reader(inner.clone(), stdout));
        tokio::spawn(stdin_writer(inner.clone(), stdin, out_rx));
        tokio::spawn(stderr_pump(stderr));

        Ok(Self { inner })
    }

    /// Sends one JSON-RPC request and waits for the matching response.
    ///
    /// A process exit / pipe EOF fails with [`MCPError::connection_lost`];
    /// `timeout` elapsing without a response fails with
    /// [`MCP_ERROR_REQUEST_TIMEOUT`] (-32004).
    pub async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> Result<Value, MCPError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        // Register before sending, and re-check `closed` while holding the
        // pending lock: `fail_all_pending` stores `closed` and THEN takes this
        // lock to drain, so either it has not run yet (and will drain this
        // entry on EOF) or it has run (and we fail here) — no entry can be
        // inserted after the drain and abandoned to a timeout.
        {
            let mut pending = self.inner.pending.lock().await;
            if self.inner.closed.load(Ordering::Acquire) {
                return Err(MCPError::connection_lost());
            }
            pending.insert(id, tx);
        }

        let frame = serde_json::to_string(&MCPRequest::new(id, method, params))
            .map_err(|e| MCPError::new(-32603, format!("failed to encode request: {e}")))?;
        if self.send_frame(frame).await.is_err() {
            self.inner.pending.lock().await.remove(&id);
            return Err(MCPError::connection_lost());
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(response)) => response.into_result(),
            Ok(Err(_)) => {
                // The reader task dropped the sender after EOF / parse shutdown.
                self.inner.pending.lock().await.remove(&id);
                Err(MCPError::connection_lost())
            }
            Err(_) => {
                self.inner.pending.lock().await.remove(&id);
                Err(MCPError::new(
                    MCP_ERROR_REQUEST_TIMEOUT,
                    format!("MCP stdio request '{method}' timed out after {timeout:?}"),
                ))
            }
        }
    }

    /// Sends a JSON-RPC notification (no id, no response expected).
    ///
    /// Delivery is best-effort, per the JSON-RPC notification semantics: a
    /// server dying around handshake time must not turn the post-`initialize`
    /// `notifications/initialized` write into a flaky connect failure. The
    /// only hard error is an already-shut-down session (frame channel gone).
    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), MCPError> {
        let message = notification_message(method, params);
        let frame = serde_json::to_string(&message)
            .map_err(|e| MCPError::new(-32603, format!("failed to encode notification: {e}")))?;
        self.send_frame(frame).await
    }

    /// Enqueues one newline-terminated frame; `Err` means the writer task is
    /// gone (process shutting down / dead).
    async fn send_frame(&self, mut frame: String) -> Result<(), MCPError> {
        frame.push('\n');
        let guard = self.inner.out_tx.lock().await;
        match guard.as_ref() {
            Some(tx) => tx.send(frame).map_err(|_| MCPError::connection_lost()),
            None => Err(MCPError::connection_lost()),
        }
    }

    /// Gracefully shuts the session down: closes stdin (EOF), waits briefly
    /// for the child to exit, then kills it. Idempotent.
    pub async fn shutdown(&self) {
        if self.inner.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        // Drop the outgoing channel: the writer task finishes and closes the
        // child's stdin, giving the server an EOF to exit on.
        let tx = self.inner.out_tx.lock().await.take();
        drop(tx);

        let mut guard = self.inner.child.lock().await;
        if let Some(child) = guard.as_mut() {
            match tokio::time::timeout(STDIO_SHUTDOWN_GRACE, child.wait()).await {
                Ok(_) => {}
                Err(_) => {
                    // Grace period elapsed: force termination.
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
            }
        }
        *guard = None;
        drop(guard);

        // Nothing can answer now; unblock any in-flight requests.
        fail_all_pending(&self.inner).await;
    }
}

/// Reads stdout lines and routes JSON-RPC messages for the session's lifetime.
async fn stdout_reader(inner: Arc<Inner>, stdout: ChildStdout) {
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF: process exited or closed stdout
            Ok(_) => handle_line(&inner, line.trim_end()).await,
            Err(_) => break,
        }
    }
    fail_all_pending(&inner).await;
}

/// Parses and dispatches one stdout frame.
async fn handle_line(inner: &Arc<Inner>, line: &str) {
    if line.is_empty() {
        return;
    }
    let message: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            // Per JSON-RPC 2.0: reply with a parse error carrying null id.
            let frame = json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": {"code": -32700, "message": "Parse error"},
            })
            .to_string();
            enqueue_best_effort(inner, frame).await;
            return;
        }
    };

    let is_response = message.get("id").is_some()
        && (message.get("result").is_some() || message.get("error").is_some());
    let is_server_request = message.get("id").is_some() && message.get("method").is_some();

    if is_response {
        if let Ok(response) = serde_json::from_value::<MCPResponse>(message) {
            if let Some(id) = response.id {
                if let Some(tx) = inner.pending.lock().await.remove(&id) {
                    let _ = tx.send(response);
                }
            }
        }
        return;
    }

    if is_server_request {
        // The client advertises no sampling/roots/elicitation capabilities.
        // Answer server pings (liveness) and refuse anything else.
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let method = message.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let frame = if method == "ping" {
            json!({"jsonrpc": "2.0", "id": id, "result": {}}).to_string()
        } else {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": "Method not found (client exposes no server-request capabilities)"},
            })
            .to_string()
        };
        enqueue_best_effort(inner, frame).await;
        return;
    }

    // Server notification (no id): nothing to answer; trace-level diagnostics
    // only — log facade avoids a hard dependency on a subscriber.
    log::trace!(
        target: "lc_mcp::transport::stdio",
        "server notification: {}",
        message
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("?")
    );
}

/// Enqueues an outgoing frame without failing the reader if the session is
/// already closing.
async fn enqueue_best_effort(inner: &Arc<Inner>, frame: String) {
    let guard = inner.out_tx.lock().await;
    if let Some(tx) = guard.as_ref() {
        let _ = tx.send(format!("{frame}\n"));
    }
}

/// Fails every pending request with a connection-lost response.
async fn fail_all_pending(inner: &Arc<Inner>) {
    inner.closed.store(true, Ordering::Release);
    let mut pending = inner.pending.lock().await;
    for (id, tx) in pending.drain() {
        let response = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            result: None,
            error: Some(MCPError::connection_lost()),
        };
        let _ = tx.send(response);
    }
}

/// Owns the child stdin and serializes all outgoing frames.
async fn stdin_writer(
    inner: Arc<Inner>,
    mut stdin: ChildStdin,
    mut out_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
) {
    while let Some(frame) = out_rx.recv().await {
        if stdin.write_all(frame.as_bytes()).await.is_err() {
            break;
        }
        let _ = stdin.flush().await;
    }
    // Channel closed (shutdown) or write failed: drop stdin for EOF.
    let _ = stdin.shutdown().await;
    // A broken write generally means the child is dead; the stdout reader
    // observes EOF and drains pending — nothing more to do here.
    drop(inner);
}

/// Drains the child's stderr into the `log` facade (debug target) so server
/// logging can never fill the pipe and deadlock the server.
async fn stderr_pump(stderr: ChildStderr) {
    let mut reader = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        log::debug!(target: "lc_mcp::transport::stdio::stderr", "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawning a missing executable fails fast with the connection-lost
    /// class, without panicking or leaving tasks behind.
    #[tokio::test]
    async fn spawn_missing_program_errors() {
        let command = StdioCommand::new("definitely-no-such-mcp-binary-0x22f4")
            .arg("--nope")
            .env("K", "V");
        let err = StdioTransport::spawn(command).unwrap_err();
        assert_eq!(err.code, -32000, "{err}");
        assert!(err.to_string().contains("failed to spawn"), "{err}");
    }

    /// A request after explicit shutdown fails immediately (connection lost),
    /// without hanging on a timeout.
    #[tokio::test]
    async fn request_after_shutdown_is_connection_lost() {
        // An instantly-exiting child (`true` / `cmd /C exit 0`): spawn
        // succeeds, shutdown is a fast wait, and the closed flag is set.
        let command = if cfg!(windows) {
            StdioCommand::new("cmd").args(["/C", "exit 0"])
        } else {
            StdioCommand::new("true")
        };
        let transport = StdioTransport::spawn(command).expect("spawn");
        transport.shutdown().await;
        transport.shutdown().await; // idempotent

        let err = transport
            .request("ping", None, Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(err.is_connection_lost(), "{err}");
    }

    /// StdioCommand builders compose.
    #[test]
    fn command_builder_shape() {
        let cmd = StdioCommand::new("srv")
            .arg("--x")
            .args(["a", "b"])
            .env("TOKEN", "t")
            .working_dir("/tmp");
        assert_eq!(cmd.program, PathBuf::from("srv"));
        assert_eq!(cmd.args, vec!["--x", "a", "b"]);
        assert_eq!(cmd.envs, vec![("TOKEN".to_string(), "t".to_string())]);
        assert_eq!(cmd.working_dir, Some(PathBuf::from("/tmp")));
    }
}
