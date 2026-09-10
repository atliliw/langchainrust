use std::collections::VecDeque;

use crate::protocol::TaskPushNotification;

use super::A2AError;

/// A live SSE connection to an A2A server (P2-1).
///
/// Consume it with [`A2ASseStream::next`], which yields one
/// [`TaskPushNotification`] per complete SSE event until the server closes the
/// stream.
pub struct A2ASseStream {
    response: reqwest::Response,
    parser: A2aSseParser,
    pending: VecDeque<TaskPushNotification>,
}

impl A2ASseStream {
    pub(crate) fn new(response: reqwest::Response) -> Self {
        Self {
            response,
            parser: A2aSseParser::new(),
            pending: VecDeque::new(),
        }
    }

    /// Wait for the next event, or `None` when the stream ends.
    ///
    /// Returns an error if the connection breaks; a malformed event is logged
    /// and skipped so one bad payload does not kill the whole stream
    /// (0.22.0 audit fix).
    pub async fn next(&mut self) -> Option<Result<TaskPushNotification, A2AError>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            match self.response.chunk().await {
                Ok(Some(chunk)) => {
                    let text = String::from_utf8_lossy(&chunk);
                    self.pending.extend(self.parser.feed(&text));
                    // Loop so queued events are returned even if a chunk
                    // carried none (or we keep reading on an empty chunk).
                }
                Ok(None) => return None,
                Err(e) => return Some(Err(A2AError::from(e))),
            }
        }
    }
}

/// Incremental parser for A2A SSE event frames.
///
/// Mirrors the SSE parsing in `lc-providers/src/openai/sse.rs`: events are
/// terminated by a blank line (`\n\n`), and the payload is the `data:` field
/// (multi-line `data:` fields are joined with newlines per the SSE spec).
struct A2aSseParser {
    buffer: String,
}

impl A2aSseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Feed a chunk of the response body; returns any complete notifications.
    ///
    /// 0.22.0 audit fix: an event whose JSON payload fails to parse is logged
    /// and skipped; only transport-level errors surface to the caller.
    fn feed(&mut self, chunk: &str) -> Vec<TaskPushNotification> {
        self.buffer.push_str(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_text: String = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos + 1);

            let data = event_text
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.trim_start().trim_end())
                .collect::<Vec<_>>()
                .join("\n");
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<TaskPushNotification>(&data) {
                Ok(notification) => out.push(notification),
                Err(e) => {
                    log::warn!("skipping malformed SSE event `{data}`: {e}");
                }
            }
        }
        out
    }
}
