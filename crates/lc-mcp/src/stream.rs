//! Streaming tool output (P2-9): incremental chunks for long-running tools "streaming while they run".
//!
//! The server splits a long task's partial results into chunks, pushed via `notifications/tool_partial`;
//! the client subscribes with [`subscribe_tool_stream`](crate::MCPClient::subscribe_tool_stream) and receives
//! the increments in order until a chunk with `final: true` arrives.
//!
//! Works with P1-7 multi-type content: each chunk carries its own [`MCPContent`]
//! (text / image / resource), `render_text` renders uniformly, non-text content is represented by a placeholder.
//!
//! ## Push format (`notifications/tool_partial` params)
//!
//! ```json
//! {
//!   "tool": "read_file",
//!   "seq": 0,
//!   "progress": 0.33,
//!   "content": { "type": "text", "text": "第一段" },
//!   "final": false
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use tokio::sync::broadcast;
use tokio::time::{timeout, Duration};

use super::types::MCPContent;

/// One incremental chunk of streaming tool output (P2-9).
///
/// Fields map to the `notifications/tool_partial` push params; `seq` is monotonically increasing, for
/// sorting / dedup / resume; `is_final` marks the last chunk and `collect` terminates on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialContent {
    /// The owning tool name.
    pub tool: String,
    /// Chunk sequence number (monotonically increasing, starting at 0).
    pub seq: u64,
    /// Chunk content (P1-7 multi-type: text / image / resource).
    pub content: MCPContent,
    /// Optional progress (0.0~1.0).
    pub progress: Option<f32>,
    /// Whether this is the last chunk (the server pushes `final: true`).
    #[serde(rename = "final", default)]
    pub is_final: bool,
}

impl PartialContent {
    /// Renders this chunk as text (works with P1-7): text as-is; images / resources are represented by a
    /// placeholder description.
    pub fn render_text(&self) -> String {
        self.content.render_text()
    }
}

/// Error kinds for streaming subscription (P2-9).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolStreamError {
    /// Broadcast buffer backlog dropped frames (pushed too fast for the consumer).
    Lagged,
    /// `collect` did not receive a `final` chunk within the deadline.
    Timeout,
}

impl fmt::Display for ToolStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolStreamError::Lagged => write!(
                f,
                "tool stream lagged: chunks were pushed too fast and buffered increments were dropped"
            ),
            ToolStreamError::Timeout => write!(f, "tool stream timed out: no final chunk received within the deadline"),
        }
    }
}

impl std::error::Error for ToolStreamError {}

/// Streaming incremental subscription for one tool (P2-9).
///
/// Created by [`MCPClient::subscribe_tool_stream`](crate::MCPClient::subscribe_tool_stream); only delivers
/// increments belonging to this tool name, other tools' pushes are filtered out.
///
/// A broadcast channel only delivers pushes that arrive "after the subscription moment", so one should
/// **subscribe first, then call the tool**.
pub struct ToolStream {
    rx: broadcast::Receiver<PartialContent>,
    tool: String,
}

impl ToolStream {
    pub(crate) fn new(rx: broadcast::Receiver<PartialContent>, tool: String) -> Self {
        Self { rx, tool }
    }

    /// Waits for the next incremental chunk belonging to this tool.
    ///
    /// - `Ok(Some(chunk))` — one increment received;
    /// - `Ok(None)` — the channel closed (connection closed), the stream ended;
    /// - `Err(Lagged)` — pushed too fast, frames dropped, increments are discontinuous.
    pub async fn next(&mut self) -> Result<Option<PartialContent>, ToolStreamError> {
        loop {
            match self.rx.recv().await {
                Ok(c) if c.tool == self.tool => return Ok(Some(c)),
                Ok(_) => continue, // another tool's push, filtered
                Err(broadcast::error::RecvError::Lagged(_)) => return Err(ToolStreamError::Lagged),
                Err(broadcast::error::RecvError::Closed) => return Ok(None),
            }
        }
    }

    /// Collects all increments up to and including the `final` chunk; the returned last chunk has
    /// `is_final == true`.
    ///
    /// If the stream closes before receiving final, returns the collected chunks (no error); if no final arrives
    /// within `deadline`, returns [`ToolStreamError::Timeout`].
    pub async fn collect(
        &mut self,
        deadline: Duration,
    ) -> Result<Vec<PartialContent>, ToolStreamError> {
        timeout(deadline, async {
            let mut out = Vec::new();
            while let Some(c) = self.next().await? {
                let is_final = c.is_final;
                out.push(c);
                if is_final {
                    break;
                }
            }
            Ok(out)
        })
        .await
        .map_err(|_| ToolStreamError::Timeout)?
    }
}

/// Parses `notifications/tool_partial` params into a [`PartialContent`].
///
/// Reused by the client event listener; missing fields / unparsable content return `None` (silently dropped,
/// a malformed chunk must not break the stream).
pub(crate) fn parse_partial_notification(params: Option<Value>) -> Option<PartialContent> {
    let p = params?;
    let tool = p.get("tool")?.as_str()?.to_string();
    let seq = p.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let progress = p.get("progress").and_then(|v| v.as_f64()).map(|f| f as f32);
    let content: MCPContent = serde_json::from_value(p.get("content")?.clone()).ok()?;
    let is_final = p.get("final").and_then(|v| v.as_bool()).unwrap_or(false);
    Some(PartialContent {
        tool,
        seq,
        content,
        progress,
        is_final,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PostMode;
    use crate::{MCPClient, MCPConfig, MCPServer};
    use lc_core::tools::ToolError;
    use lc_core::BaseTool;
    use std::sync::Arc;

    /// A test tool that echoes its input (for in-memory end-to-end streaming tests).
    struct EchoTool;
    #[async_trait::async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "回显输入"
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(input)
        }
    }

    fn chunk(tool: &str, seq: u64, text: &str, is_final: bool) -> PartialContent {
        PartialContent {
            tool: tool.to_string(),
            seq,
            content: MCPContent::Text {
                text: text.to_string(),
            },
            progress: Some(seq as f32 + 1.0),
            is_final,
        }
    }

    /// params serialize in wire format then parse back to the original value (field ↔ JSON `final` roundtrip).
    #[test]
    fn test_parse_partial_notification_roundtrip() {
        let c = PartialContent {
            tool: "read_file".to_string(),
            seq: 2,
            content: MCPContent::Text {
                text: "第二段".to_string(),
            },
            progress: Some(0.66),
            is_final: true,
        };
        let json = serde_json::to_value(&c).unwrap();
        let parsed = parse_partial_notification(Some(json))
            .expect("should parse back to the original value");
        assert_eq!(parsed.tool, "read_file");
        assert_eq!(parsed.seq, 2);
        assert_eq!(parsed.render_text(), "第二段");
        assert_eq!(parsed.progress, Some(0.66));
        assert!(parsed.is_final);
    }

    /// Malformed params (missing fields / unparsable content) → None, no panic.
    #[test]
    fn test_parse_partial_notification_malformed_is_none() {
        assert!(parse_partial_notification(None).is_none());
        assert!(parse_partial_notification(Some(serde_json::json!({}))).is_none());
        assert!(parse_partial_notification(Some(serde_json::json!({
            "tool": "x", "seq": 0, "content": "not-an-object"
        })))
        .is_none());
    }

    /// Filtering: only delivers increments for this tool name, other tools' pushes are skipped.
    #[tokio::test]
    async fn test_stream_filters_by_tool() {
        let server = Arc::new(MCPServer::new().with_tool(Arc::new(EchoTool)));
        let client =
            MCPClient::with_transport(Box::new(crate::InMemoryTransport::new(server.clone())))
                .await
                .expect("in-memory connection should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        // First push a chunk for "another tool": it should be filtered, collect unaffected.
        server.publish_partial(chunk("other", 0, "无关片段", false));
        server.publish_partial(chunk("echo", 0, "第一段", false));
        server.publish_partial(chunk("echo", 1, "第二段", true));

        let chunks = stream
            .collect(Duration::from_secs(2))
            .await
            .expect("should receive increments");
        assert_eq!(
            chunks.len(),
            2,
            "chunks from other tools should be filtered"
        );
        assert_eq!(chunks[0].render_text(), "第一段");
        assert_eq!(chunks[0].progress, Some(1.0));
        assert!(!chunks[0].is_final);
        assert!(chunks[1].is_final, "collect should end with a final chunk");
    }

    /// Multi-type content (P1-7): when a chunk carries image content, render_text represents it with a
    /// placeholder description.
    #[tokio::test]
    async fn test_partial_multi_type_content_renders_placeholder() {
        let server = Arc::new(MCPServer::new().with_tool(Arc::new(EchoTool)));
        let client =
            MCPClient::with_transport(Box::new(crate::InMemoryTransport::new(server.clone())))
                .await
                .expect("in-memory connection should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        server.publish_partial(PartialContent {
            tool: "echo".to_string(),
            seq: 0,
            content: MCPContent::Image {
                data: "base64...".to_string(),
                mime_type: "image/png".to_string(),
            },
            progress: None,
            is_final: true,
        });
        let chunks = stream
            .collect(Duration::from_secs(2))
            .await
            .expect("should receive increments");
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].render_text().contains("[image: image/png"),
            "{}",
            chunks[0].render_text()
        );
    }

    /// collect timeout: no final chunk received within the deadline → ToolStreamError::Timeout.
    #[tokio::test]
    async fn test_collect_times_out_without_final() {
        let server = Arc::new(MCPServer::new().with_tool(Arc::new(EchoTool)));
        let client =
            MCPClient::with_transport(Box::new(crate::InMemoryTransport::new(server.clone())))
                .await
                .expect("in-memory connection should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        // Push only non-final chunks, no ending ever arrives → timeout.
        server.publish_partial(chunk("echo", 0, "卡住", false));
        let err = stream
            .collect(Duration::from_millis(100))
            .await
            .expect_err("should time out");
        assert_eq!(err, ToolStreamError::Timeout);
    }

    /// SSE path end-to-end: the StreamingCall fake server pushes 3 incremental chunks over the SSE long
    /// connection after the first tools/call, and the client's subscribe receives all of them.
    #[tokio::test]
    async fn test_subscribe_collects_partials_via_sse() {
        let fake = crate::test_support::start_fake_sse_server(PostMode::StreamingCall).await;
        let client = MCPClient::connect(MCPConfig::sse(&fake.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        // call_seen gate: triggers the server to start streaming over SSE (see test_support).
        let out = client
            .call_tool("echo", serde_json::json!({"msg": "hi"}))
            .await;
        assert!(out.is_ok(), "normal call should still succeed");

        let chunks = stream
            .collect(Duration::from_secs(5))
            .await
            .expect("should receive streaming increments");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].render_text(), "chunk0");
        assert_eq!(chunks[0].progress, Some(1.0 / 3.0));
        assert_eq!(chunks[1].render_text(), "chunk1");
        assert_eq!(chunks[2].render_text(), "chunk2");
        assert!(chunks[2].is_final, "last chunk should be marked final");
    }
}
