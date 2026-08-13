//! per-tool 超时 + Progress 重置计时器(P2-4)。
//!
//! 长任务工具可能远超默认超时;直接 `timeout` 会误杀仍在正常推进的工具。
//! 本模块:
//!
//! - **`ToolSpec{default_timeout}`**:按工具声明默认超时,超时即终止;
//! - **Progress 重置**:调用期间收到 `notifications/progress` 把计时器重置回
//!   `default_timeout`(工具还活着,继续给时间);
//! - **硬上限兜底**:总时长不得超过 `max_timeout`,防止"半死不活但一直报进度"
//!   的工具无限占用连接。

use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::broadcast;
use tokio::time::sleep;

use crate::client::MCPClient;
use crate::protocol::MCPError;
use crate::transport::MCPEvent;
use crate::types::MCPToolResult;

/// MCP 进度通知方法名。
const PROGRESS_METHOD: &str = "notifications/progress";

/// 单个工具的超时声明(P2-4)。
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// 工具名(诊断信息)。
    pub name: String,
    /// 默认超时:超时即终止;收到 `notifications/progress` 重置回该值。
    pub default_timeout: Duration,
    /// 硬上限:无论 progress 如何,超过该时长必终止。
    pub max_timeout: Duration,
}

impl ToolSpec {
    /// 创建一个工具超时声明,硬上限默认 = `default_timeout * 3`。
    pub fn new(name: impl Into<String>, default_timeout: Duration) -> Self {
        Self {
            name: name.into(),
            default_timeout,
            max_timeout: default_timeout.saturating_mul(3),
        }
    }

    /// 显式设置硬上限(至少不小于默认超时)。
    pub fn with_max_timeout(mut self, max_timeout: Duration) -> Self {
        self.max_timeout = max_timeout.max(self.default_timeout);
        self
    }
}

/// 带 per-tool 超时的工具调用(P2-4)。
///
/// 收到 `notifications/progress` 重置默认超时计时器;总时长超过
/// `spec.max_timeout` 硬上限则终止。调用 future 只构造一次,`select!` 用
/// `&mut call` 轮询——事件分支抢先时取消的是借用而非 future 本身,重新轮询
/// 会续上在途请求,不会重复发送 `tools/call`。
pub async fn call_tool_with_timeout(
    client: &MCPClient,
    name: &str,
    arguments: Value,
    spec: &ToolSpec,
) -> Result<MCPToolResult, MCPError> {
    let default = spec.default_timeout;
    let hard_deadline = Instant::now() + spec.max_timeout;
    let mut deadline = Instant::now() + default;

    // 提前订阅 progress,避免漏掉调用期间的推送。
    let mut events: Option<broadcast::Receiver<MCPEvent>> = Some(client.subscribe_events());
    let is_progress = |ev: &MCPEvent| {
        matches!(
            ev,
            MCPEvent::Message { method, .. } if method == PROGRESS_METHOD
        )
    };

    let mut call = Box::pin(client.call_tool(name, arguments));

    loop {
        let now = Instant::now();
        if now >= hard_deadline {
            return Err(MCPError::new(
                -1,
                format!(
                    "工具 '{name}' 调用超过硬上限 {:?},终止(progress 未豁免)",
                    spec.max_timeout
                ),
            ));
        }
        let remain = deadline.saturating_duration_since(now);
        if remain.is_zero() {
            return Err(MCPError::new(
                -1,
                format!(
                    "工具 '{name}' 调用超时:{} 内无响应且无 progress",
                    spec.default_timeout.as_millis()
                ),
            ));
        }
        tokio::select! {
            result = &mut call => {
                return result;
            }
            _ = sleep(remain) => {
                // 计时器到点:下一轮循环顶部判定超时(progress 重置后不会触发)。
            }
            ev = async {
                match &mut events {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match ev {
                    Ok(e) if is_progress(&e) => {
                        // 工具仍在推进:重置默认超时,但不越过硬上限。
                        deadline = (Instant::now() + default).min(hard_deadline);
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Closed) => {
                        // 事件源关闭:停止监听,只靠计时器判定。
                        events = None;
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::MCPConfig;
    use serde_json::json;

    #[tokio::test]
    async fn test_fast_tool_returns_immediately() {
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");
        let spec = ToolSpec::new("echo", Duration::from_secs(5));
        let r = call_tool_with_timeout(&client, "echo", json!({}), &spec).await;
        assert!(r.is_ok(), "快速工具应立即返回");
    }

    /// 无 progress:默认超时到点即终止(不等慢服务器的最终响应)。
    #[tokio::test]
    async fn test_timeout_without_progress() {
        let server = start_fake_sse_server(PostMode::SlowCall(Duration::from_secs(5))).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");
        let spec = ToolSpec::new("echo", Duration::from_millis(100))
            .with_max_timeout(Duration::from_secs(2));
        let err = call_tool_with_timeout(&client, "echo", json!({}), &spec)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("超时"), "{}", err);
    }

    /// progress 持续重置计时器:慢工具(默认超时内完不成)最终正常完成。
    #[tokio::test]
    async fn test_progress_resets_deadline_and_completes() {
        let server =
            start_fake_sse_server(PostMode::ProgressSlowCall(Duration::from_millis(900))).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");
        let spec = ToolSpec::new("echo", Duration::from_millis(400))
            .with_max_timeout(Duration::from_secs(3));
        let r = call_tool_with_timeout(&client, "echo", json!({}), &spec).await;
        assert!(r.is_ok(), "progress 应持续重置计时器并最终完成");
    }

    /// 硬上限:即使 progress 一直刷新,总时长到硬上限仍终止(防"半死不活")。
    #[tokio::test]
    async fn test_hard_cap_bounds_despite_progress() {
        let server =
            start_fake_sse_server(PostMode::ProgressSlowCall(Duration::from_millis(800))).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");
        let spec = ToolSpec::new("echo", Duration::from_millis(300))
            .with_max_timeout(Duration::from_millis(400));
        let err = call_tool_with_timeout(&client, "echo", json!({}), &spec)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("硬上限"), "{}", err);
    }

    #[test]
    fn test_spec_max_timeout_at_least_default() {
        let spec =
            ToolSpec::new("t", Duration::from_secs(2)).with_max_timeout(Duration::from_millis(1));
        assert!(spec.max_timeout >= spec.default_timeout);
        // 默认:max = default * 3
        let spec2 = ToolSpec::new("t", Duration::from_secs(2));
        assert_eq!(spec2.max_timeout, Duration::from_secs(6));
    }
}
