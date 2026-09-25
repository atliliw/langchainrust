//! Per-tool timeout (P2-4, stateless-track form).
//!
//! On the stateless track there is no server-push channel, so the old
//! "progress notification resets the deadline" semantics are gone. What
//! remains is the bounded call: a single `default_timeout` deadline aborts
//! a hung tool so it cannot block the caller indefinitely. `ToolSpec` keeps
//! an optional validated `max_timeout` (>= default) for API compatibility,
//! but it is never the limiter — the default deadline always fires first.

use std::time::Duration;

use serde_json::Value;
use tokio::time::timeout;

use crate::protocol::MCPError;
use crate::tool_client::McpToolClient;
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
/// The effective bound is `default_timeout`; `max_timeout` (>= default, enforced by
/// `ToolSpec::with_max_timeout`) can never fire first, so a separate hard-cap deadline
/// would be unreachable — a single `timeout` around the call is the whole story.
pub async fn call_tool_with_timeout(
    client: &dyn McpToolClient,
    name: &str,
    arguments: Value,
    spec: &ToolSpec,
) -> Result<MCPToolResult, MCPError> {
    match timeout(spec.default_timeout, client.call_tool(name, arguments)).await {
        Ok(result) => result,
        Err(_) => Err(MCPError::new(
            -1,
            format!(
                "tool '{name}' call timed out after {} ms",
                spec.default_timeout.as_millis()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_stateless::{MrtrConfig, StatelessMcpClient};
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
