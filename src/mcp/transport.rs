//! MCP 传输层:Stdio + SSE

use async_trait::async_trait;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::{timeout, Duration};

use super::protocol::{MCPError, MCPRequest, MCPResponse};
use super::types::MCPConfig;

/// Default timeout for SSE endpoint discovery (30 seconds).
const SSE_DISCOVER_TIMEOUT: Duration = Duration::from_secs(30);

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
}

/// Stdio 传输:启动子进程,通过 stdin/stdout 以换行分隔的 JSON 通信
///
/// Uses a per-request lock to ensure that a write-then-read cycle is atomic,
/// preventing interleaved reads/writes from concurrent callers.
pub struct StdioTransport {
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
    child: Arc<Mutex<Child>>,
    /// Per-request lock: ensures write + read is atomic for each request.
    request_lock: Arc<Mutex<()>>,
}

impl StdioTransport {
    pub async fn new(config: &MCPConfig) -> Result<Self, MCPError> {
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
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    eprintln!("[mcp-stdio-stderr] {}", line);
                }
            });
        }

        Ok(Self {
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
            child: Arc::new(Mutex::new(child)),
            request_lock: Arc::new(Mutex::new(())),
        })
    }
}

#[async_trait]
impl MCPTransport for StdioTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
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
                    return Err(MCPError::new(-1, "子进程 stdout 已关闭"));
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
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        Ok(())
    }

    async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), MCPError> {
        // JSON-RPC 2.0 notification: 无 id 字段
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        let mut payload = notif;
        if let Some(p) = params {
            payload
                .as_object_mut()
                .unwrap()
                .insert("params".to_string(), p);
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
}

/// SSE transport for MCP.
///
/// Connects to an MCP server using the SSE protocol:
/// - Connects to the SSE endpoint to receive server-push events
/// - Sends requests via HTTP POST to the server's message endpoint
///
/// The SSE endpoint URL format: `http://host:port/sse`
/// The server sends an `endpoint` event with the POST URL for sending messages.
pub struct SseTransport {
    /// SSE endpoint URL (for receiving events).
    sse_url: String,
    /// POST endpoint URL (for sending messages). Discovered from SSE `endpoint` event.
    post_url: Arc<Mutex<Option<String>>>,
    /// HTTP client.
    client: reqwest::Client,
}

impl SseTransport {
    pub fn new(config: &MCPConfig) -> Result<Self, MCPError> {
        let sse_url = match config {
            MCPConfig::Sse { url } => url.clone(),
            _ => return Err(MCPError::new(-1, "SseTransport requires SSE config")),
        };
        Ok(Self {
            sse_url,
            post_url: Arc::new(Mutex::new(None)),
            client: reqwest::Client::new(),
        })
    }

    /// Discover the POST endpoint from the SSE stream.
    ///
    /// The MCP server sends an `endpoint` event with the URL to use for
    /// posting messages. This method connects to the SSE stream and waits
    /// for that event, with a timeout to prevent hanging indefinitely.
    async fn discover_post_url(&self) -> Result<String, MCPError> {
        // Check if already discovered
        {
            let post_url = self.post_url.lock().await;
            if let Some(url) = &*post_url {
                return Ok(url.clone());
            }
        }

        // Connect to SSE endpoint and read the `endpoint` event, with timeout
        let response = timeout(SSE_DISCOVER_TIMEOUT, async {
            self.client
                .get(&self.sse_url)
                .header("Accept", "text/event-stream")
                .send()
                .await
                .map_err(|e| MCPError::new(-1, format!("SSE connection failed: {}", e)))
        })
        .await
        .map_err(|_| {
            MCPError::new(
                -1,
                "SSE discover timed out: server did not respond within 30s",
            )
        })??;

        let status = response.status();
        if !status.is_success() {
            return Err(MCPError::new(
                -1,
                format!("SSE endpoint returned HTTP {}", status),
            ));
        }

        // Parse SSE events to find the `endpoint` event, with timeout
        let text = timeout(SSE_DISCOVER_TIMEOUT, async {
            response
                .text()
                .await
                .map_err(|e| MCPError::new(-1, format!("Failed to read SSE response: {}", e)))
        })
        .await
        .map_err(|_| {
            MCPError::new(
                -1,
                "SSE discover timed out: reading response body took longer than 30s",
            )
        })??;

        // Parse SSE format: "event: endpoint\ndata: <url>\n\n"
        let mut found_url = None;
        let mut current_event = String::new();
        for line in text.lines() {
            if let Some(stripped) = line.strip_prefix("event:") {
                current_event = stripped.trim().to_string();
            } else if line.starts_with("data:") && current_event == "endpoint" {
                found_url = Some(line[5..].trim().to_string());
                break;
            }
        }

        let url = found_url
            .ok_or_else(|| MCPError::new(-1, "SSE stream did not send 'endpoint' event"))?;

        // Cache the discovered URL
        {
            let mut post_url = self.post_url.lock().await;
            *post_url = Some(url.clone());
        }

        Ok(url)
    }
}

#[async_trait]
impl MCPTransport for SseTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
        let post_url = self.discover_post_url().await?;

        let resp = self
            .client
            .post(&post_url)
            .json(&req)
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

    async fn close(&self) -> Result<(), MCPError> {
        // Clear cached endpoint
        let mut post_url = self.post_url.lock().await;
        *post_url = None;
        Ok(())
    }

    async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), MCPError> {
        let post_url = self.discover_post_url().await?;

        let mut payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(p) = params {
            payload
                .as_object_mut()
                .unwrap()
                .insert("params".to_string(), p);
        }
        self.client
            .post(&post_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| MCPError::new(-1, format!("Failed to send notification: {}", e)))?;
        Ok(())
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
}
