//! per-tool timeout + Progress reset timer (P2-4).
//!
//! Long-running tools can far exceed the default timeout; a plain `timeout` would wrongly kill a tool still
//! making normal progress. This module:
//!
//! - **`ToolSpec{default_timeout}`**: per-tool declared default timeout, aborting when it expires;
//! - **Progress reset**: a `notifications/progress` received during the call resets the timer back to
//!   `default_timeout` (the tool is still alive, keep giving it time);
//! - **Hard cap backstop**: the total duration must not exceed `max_timeout`, preventing a
//!   "half-dead but always reporting progress" tool from occupying the connection indefinitely.

use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::broadcast;
use tokio::time::sleep;

use crate::client::MCPClient;
use crate::protocol::MCPError;
use crate::transport::MCPEvent;
use crate::types::MCPToolResult;

/// MCP progress notification method name.
const PROGRESS_METHOD: &str = "notifications/progress";

/// A single tool's timeout declaration (P2-4).
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// The tool name (for diagnostics).
    pub name: String,
    /// The default timeout: aborting when it expires; a `notifications/progress` resets back to this value.
    pub default_timeout: Duration,
    /// The hard cap: regardless of progress, it must abort beyond this duration.
    pub max_timeout: Duration,
}

impl ToolSpec {
    /// Creates a tool timeout declaration; the hard cap defaults to `default_timeout * 3`.
    pub fn new(name: impl Into<String>, default_timeout: Duration) -> Self {
        Self {
            name: name.into(),
            default_timeout,
            max_timeout: default_timeout.saturating_mul(3),
        }
    }

    /// Sets the hard cap explicitly (at least not less than the default timeout).
    pub fn with_max_timeout(mut self, max_timeout: Duration) -> Self {
        self.max_timeout = max_timeout.max(self.default_timeout);
        self
    }
}

/// A tool call with a per-tool timeout (P2-4).
///
/// A `notifications/progress` resets the default-timeout timer; the total duration past the `spec.max_timeout`
/// hard cap aborts. The call future is constructed once, `select!` polls it via `&mut call` — when an event
/// branch wins, it cancels the borrow, not the future itself; re-polling resumes the in-flight request without
/// re-sending `tools/call`.
pub async fn call_tool_with_timeout(
    client: &MCPClient,
    name: &str,
    arguments: Value,
    spec: &ToolSpec,
) -> Result<MCPToolResult, MCPError> {
    let default = spec.default_timeout;
    let hard_deadline = Instant::now() + spec.max_timeout;
    let mut deadline = Instant::now() + default;

    // Subscribe to progress early, so pushes during the call are not missed.
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
                    "tool '{name}' call exceeded hard cap {:?}, aborting (progress did not exempt)",
                    spec.max_timeout
                ),
            ));
        }
        let remain = deadline.saturating_duration_since(now);
        if remain.is_zero() {
            return Err(MCPError::new(
                -1,
                format!(
                    "tool '{name}' call timed out: no response and no progress within {} ms",
                    spec.default_timeout.as_millis()
                ),
            ));
        }
        tokio::select! {
            result = &mut call => {
                return result;
            }
            _ = sleep(remain) => {
                // Timer expired: the timeout is judged at the top of the next loop iteration (not triggered
                // after a progress reset).
            }
            ev = async {
                match &mut events {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match ev {
                    Ok(e) if is_progress(&e) => {
                        // The tool is still making progress: reset the default timeout, but do not cross the
                        // hard cap.
                        deadline = (Instant::now() + default).min(hard_deadline);
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Closed) => {
                        // The event source closed: stop listening, rely only on the timer.
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
            .expect("connecting to fake SSE server should succeed");
        let spec = ToolSpec::new("echo", Duration::from_secs(5));
        let r = call_tool_with_timeout(&client, "echo", json!({}), &spec).await;
        assert!(r.is_ok(), "fast tool should return immediately");
    }

    /// Without progress: the default timeout expires and aborts (does not wait for the slow server's final
    /// response).
    #[tokio::test]
    async fn test_timeout_without_progress() {
        let server = start_fake_sse_server(PostMode::SlowCall(Duration::from_secs(5))).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        let spec = ToolSpec::new("echo", Duration::from_millis(100))
            .with_max_timeout(Duration::from_secs(2));
        let err = call_tool_with_timeout(&client, "echo", json!({}), &spec)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{}", err);
    }

    /// Progress keeps resetting the timer: a slow tool (not finishing within the default timeout) completes
    /// normally in the end.
    #[tokio::test]
    async fn test_progress_resets_deadline_and_completes() {
        let server =
            start_fake_sse_server(PostMode::ProgressSlowCall(Duration::from_millis(900))).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        let spec = ToolSpec::new("echo", Duration::from_millis(400))
            .with_max_timeout(Duration::from_secs(3));
        let r = call_tool_with_timeout(&client, "echo", json!({}), &spec).await;
        assert!(
            r.is_ok(),
            "progress should keep resetting the timer and eventually complete"
        );
    }

    /// Hard cap: even with progress refreshing continuously, the total duration still aborts at the hard cap
    /// (preventing "half-dead" tools).
    #[tokio::test]
    async fn test_hard_cap_bounds_despite_progress() {
        let server =
            start_fake_sse_server(PostMode::ProgressSlowCall(Duration::from_millis(800))).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        let spec = ToolSpec::new("echo", Duration::from_millis(300))
            .with_max_timeout(Duration::from_millis(400));
        let err = call_tool_with_timeout(&client, "echo", json!({}), &spec)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("hard cap"), "{}", err);
    }

    #[test]
    fn test_spec_max_timeout_at_least_default() {
        let spec =
            ToolSpec::new("t", Duration::from_secs(2)).with_max_timeout(Duration::from_millis(1));
        assert!(spec.max_timeout >= spec.default_timeout);
        // default: max = default * 3
        let spec2 = ToolSpec::new("t", Duration::from_secs(2));
        assert_eq!(spec2.max_timeout, Duration::from_secs(6));
    }
}
