//! MCP 传输层:Stdio + SSE

use async_trait::async_trait;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use super::protocol::{MCPError, MCPRequest, MCPResponse};
use super::types::MCPConfig;

/// MCP 传输层抽象
#[async_trait]
pub trait MCPTransport: Send + Sync {
    /// 发送请求并等待响应
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError>;
    /// 发送通知(不等响应)
    async fn notify(&self, method: &str, params: Option<serde_json::Value>) -> Result<(), MCPError>;
    /// 关闭连接
    async fn close(&self) -> Result<(), MCPError>;
}

/// Stdio 传输:启动子进程,通过 stdin/stdout 以换行分隔的 JSON 通信
pub struct StdioTransport {
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
    child: Arc<Mutex<Child>>,
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
            .stderr(std::process::Stdio::null());

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

        Ok(Self {
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
            child: Arc::new(Mutex::new(child)),
        })
    }
}

#[async_trait]
impl MCPTransport for StdioTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
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

    async fn notify(&self, method: &str, params: Option<serde_json::Value>) -> Result<(), MCPError> {
        // JSON-RPC 2.0 notification: 无 id 字段
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        let mut payload = notif;
        if let Some(p) = params {
            payload.as_object_mut().unwrap().insert("params".to_string(), p);
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

/// SSE 传输(简化版):通过 HTTP POST 发送请求,读取 JSON 响应
///
/// 注:完整 MCP SSE 需建立长连接事件流;此处为兼容性简化实现,
/// 适用于返回单次 JSON 响应的 HTTP 端点。
pub struct SseTransport {
    url: String,
    client: reqwest::Client,
}

impl SseTransport {
    pub fn new(config: &MCPConfig) -> Result<Self, MCPError> {
        let url = match config {
            MCPConfig::Sse { url } => url.clone(),
            _ => return Err(MCPError::new(-1, "SseTransport 需要 Sse 配置")),
        };
        Ok(Self {
            url,
            client: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl MCPTransport for SseTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
        let resp = self
            .client
            .post(&self.url)
            .json(&req)
            .send()
            .await
            .map_err(|e| MCPError::new(-1, format!("HTTP 请求失败: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(MCPError::new(-1, format!("HTTP 错误: {}", status)));
        }

        resp.json::<MCPResponse>()
            .await
            .map_err(|e| MCPError::new(-1, format!("解析响应失败: {}", e)))
    }

    async fn close(&self) -> Result<(), MCPError> {
        Ok(())
    }

    async fn notify(&self, method: &str, params: Option<serde_json::Value>) -> Result<(), MCPError> {
        let mut payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(p) = params {
            payload.as_object_mut().unwrap().insert("params".to_string(), p);
        }
        self.client
            .post(&self.url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| MCPError::new(-1, format!("发送通知失败: {}", e)))?;
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
