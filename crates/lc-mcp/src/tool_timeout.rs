//! Per-tool timeout (P2-4, stateless-track form).
//!
//! On the stateless track there is no server-push channel, so the old
//! "progress notification resets the deadline" semantics are gone. What
//! remains is the bounded call: a default deadline with a hard-cap backstop
//! (`max_timeout >= default_timeout`), preventing a hung tool from blocking
//! the caller indefinitely.

use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::time::sleep;

use crate::client_stateless::StatelessMcpClient;
use crate::protocol::MCPError;
use crate::types::MCPToolResult;

/// A single tool's timeout declaration (P2-4).
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// The tool name (for diagnostics).
    pub name: String,
    /// The default timeout: aborts when it expires.
    pub default_timeout: Duration,
    /// The hard cap: aborts regardless once passed.
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

/// A tool call with a bounded deadline (P2-4, stateless form).
///
/// Hard cap semantics preserved from the push-era implementation: past
/// `spec.max_timeout` the call aborts even if it might still finish.
pub async fn call_tool_with_timeout(
    client: &StatelessMcpClient,
    name: &str,
    arguments: Value,
    spec: &ToolSpec,
) -> Result<MCPToolResult, MCPError> {
    let hard_deadline = Instant::now() + spec.max_timeout;
    let deadline = Instant::now() + spec.default_timeout;

    let mut call = Box::pin(client.call_tool(name, arguments));

    loop {
        let now = Instant::now();
        if now >= hard_deadline {
            return Err(MCPError::new(
                -1,
                format!(
                    "tool '{name}' call exceeded hard cap {:?}, aborting",
                    spec.max_timeout
                ),
            ));
        }
        let remain = deadline.saturating_duration_since(now);
        if remain.is_zero() {
            return Err(MCPError::new(
                -1,
                format!(
                    "tool '{name}' call timed out after {} ms",
                    spec.default_timeout.as_millis()
                ),
            ));
        }
        tokio::select! {
            result = &mut call => {
                return result;
            }
            _ = sleep(remain) => {
                // Deadline expired: judged at the top of the next loop iteration.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_stateless::MrtrConfig;
    use crate::test_support::{start_fake_stateless_server, StatelessMode};
    use serde_json::json;

    #[tokio::test]
    async fn test_fast_tool_returns_immediately() {
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client =
            StatelessMcpClient::connect(&server.url).with_mrtr(MrtrConfig { max_round_trips: 0 });
        let spec = ToolSpec::new("echo", Duration::from_secs(5));
        let r = call_tool_with_timeout(&client, "echo", json!({}), &spec).await;
        assert!(r.is_ok(), "fast tool should return immediately");
    }

    /// Without a response the default deadline aborts (slow server).
    #[tokio::test]
    async fn test_timeout_without_response() {
        let server =
            start_fake_stateless_server(StatelessMode::SlowCall(Duration::from_secs(5))).await;
        let client = StatelessMcpClient::connect(&server.url);
        let spec = ToolSpec::new("echo", Duration::from_millis(100))
            .with_max_timeout(Duration::from_secs(2));
        let err = call_tool_with_timeout(&client, "echo", json!({}), &spec)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{}", err);
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
