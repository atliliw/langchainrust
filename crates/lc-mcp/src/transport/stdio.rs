//! Stdio 传输:启动子进程,通过 stdin/stdout 以换行分隔的 JSON 通信。

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
    /// 创建 Stdio 传输:启动子进程并建立 stdin/stdout 通道。
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
                log::warn!(
                    "MCP child process exited: {:?}, starting auto-reconnect",
                    child_status
                );
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

        {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(json.as_bytes())
                .await
                .map_err(|e| MCPError::new(-1, format!("failed to write to stdin: {}", e)))?;
            stdin
                .write_all(b"\n")
                .await
                .map_err(|e| MCPError::new(-1, format!("failed to write newline: {}", e)))?;
            stdin
                .flush()
                .await
                .map_err(|e| MCPError::new(-1, format!("failed to flush stdin: {}", e)))?;
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
                    .map_err(|e| MCPError::new(-1, format!("failed to read stdout: {}", e)))?;
                if n == 0 {
                    // 子进程退出,连接断开 → 后台 monitor 会自动重连
                    return Err(MCPError::connection_lost());
                }
                if !line.trim().is_empty() {
                    break;
                }
            }
        }

        serde_json::from_str::<MCPResponse>(line.trim()).map_err(|e| {
            MCPError::new(
                -1,
                format!("failed to parse response: {} | raw: {}", e, line),
            )
        })
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
