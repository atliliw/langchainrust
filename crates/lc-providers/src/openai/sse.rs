// src/language_models/openai/sse.rs
//! SSE (Server-Sent Events) parser
//!
//! Used to parse OpenAI streaming responses

use lc_core::tools::ToolCall;
use serde::Deserialize;

/// Cap on incomplete bytes the framer will hold while waiting for an event
/// terminator (0.26.0 M5).
///
/// A healthy OpenAI-compatible stream delivers complete events per chunk; at
/// most the trailing partial event sits in the buffer (a few KB). Reaching
/// this cap therefore means the peer is emitting data with **no** `\n\n`
/// separator — a malformed or runaway stream — and is the memory-exhaustion
/// guard: without it `SseByteFramer` would grow its `Vec` without bound and an
/// attacker streaming unterminated bytes could OOM the process.
const DEFAULT_MAX_BUFFERED_BYTES: usize = 16 * 1024 * 1024;

/// Byte-level SSE framer (0.22.0 C1 fix).
///
/// Accumulates **raw bytes** and only decodes complete events to UTF-8.
/// Event boundaries (`\n\n` or `\r\n\r\n`) are pure ASCII, so they are safe
/// to search at the byte layer; a multi-byte character (CJK / emoji) split
/// across TCP chunks is therefore never lossy-converted mid-character. Feed
/// the returned event texts into [`SSEParser`] (or any event parser) instead
/// of feeding raw chunks into `String::from_utf8_lossy` first.
///
/// 0.26.0 M5: `push` is O(n) for the pushed bytes (a single left-to-right scan
/// that retains only the incomplete trailer, replacing the old per-event
/// re-scan + `drain` that was quadratic in the number of complete events) and
/// returns an error instead of growing without bound if the buffered partial
/// exceeds `DEFAULT_MAX_BUFFERED_BYTES`.
///
/// ```rust
/// use lc_providers::openai::sse::{SseByteFramer, SSEParser};
///
/// let mut framer = SseByteFramer::new();
/// let mut parser = SSEParser::new();
/// // Split a Chinese chunk in the middle of a 3-byte character:
/// let full = b"data: {\"x\": \"\xe4\xbd\xa0\xe5\xa5\xbd\"}\n\n";
/// let (a, b) = full.split_at(full.len() - 1);
/// let mut events = Vec::new();
/// for part in [a, b] {
///     for text in framer.push(part).unwrap() {
///         events.extend(parser.parse(&text));
///     }
/// }
/// assert_eq!(events.len(), 1);
/// assert!(events[0].data.contains("你好")); // no U+FFFD
/// ```
pub struct SseByteFramer {
    buf: Vec<u8>,
    /// Fail-fast cap on buffered incomplete bytes (M5 bounded-memory guard).
    max_buffered: usize,
}

/// Framing error (0.26.0 M5).
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SseError {
    /// The incomplete-event buffer exceeded its cap — the peer sent bytes with
    /// no `\n\n` terminator. Reported as a stream error (never silently
    /// truncated) and the framer clears its buffer, so a hostile/corrupt peer
    /// cannot keep a partial buffer alive by poking one byte at a time.
    #[error(
        "SSE stream exceeded buffered-event cap of {max} bytes (buffered {buffered}); \
         no event terminator received — malformed or runaway stream"
    )]
    BufferOverflow {
        /// How many incomplete bytes had buffered when the cap was hit.
        buffered: usize,
        /// The configured cap that was exceeded (`DEFAULT_MAX_BUFFERED_BYTES`
        /// unless [`SseByteFramer::with_max_buffered`] was used).
        max: usize,
    },
}

impl SseByteFramer {
    /// Creates an empty framer with the default `DEFAULT_MAX_BUFFERED_BYTES`
    /// cap.
    pub fn new() -> Self {
        Self::with_max_buffered(DEFAULT_MAX_BUFFERED_BYTES)
    }

    /// Creates an empty framer with an explicit incomplete-buffer cap
    /// (primarily for tests / low-memory consumers).
    pub fn with_max_buffered(max_buffered: usize) -> Self {
        Self {
            buf: Vec::new(),
            max_buffered,
        }
    }

    /// Pushes one raw chunk; returns the text of each **complete** SSE event
    /// (decoded to UTF-8, terminator normalized to `\n\n` so downstream
    /// string-level parsers see a finished event).
    ///
    /// Returns [`Err(SseError::BufferOverflow)`](SseError::BufferOverflow) if
    /// the incomplete buffer would exceed its cap — see [`SseByteFramer`].
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>, SseError> {
        self.buf.extend_from_slice(chunk);
        if self.buf.len() > self.max_buffered {
            let buffered = self.buf.len();
            let max = self.max_buffered;
            // Fail-closed: drop the partial so a follow-up single-byte poke
            // after the overflow cannot grow memory again.
            self.buf.clear();
            return Err(SseError::BufferOverflow { buffered, max });
        }
        // Single left-to-right pass. Each `find_event_end(&self.buf[start..])`
        // scans only from where the last event ended, so every byte is examined
        // at most once per push (O(n)); the old loop re-scanned from index 0 and
        // `drain`ed the consumed prefix per event (quadratic in event count).
        // At the end, one `drain(..start)` compact keeps only the incomplete
        // trailer (a single O(trailer) move per push).
        let mut events = Vec::new();
        let mut start = 0usize;
        loop {
            let Some((sep_len, end)) = find_event_end(&self.buf[start..]) else {
                break;
            };
            let end_abs = start + end;
            events.push(format!(
                "{}\n\n",
                String::from_utf8_lossy(&self.buf[start..end_abs])
            ));
            start = end_abs + sep_len;
        }
        if start > 0 {
            self.buf.drain(..start);
        }
        Ok(events)
    }

    /// Bytes still buffered (incomplete event); for tests / diagnostics.
    pub fn pending(&self) -> usize {
        self.buf.len()
    }
}

impl Default for SseByteFramer {
    fn default() -> Self {
        Self::new()
    }
}

/// Finds the first event terminator in `buf`.
/// Returns `(separator_len, event_text_end)` where `event_text_end` is the
/// index just past the event text (exclusive of the separator bytes).
fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < buf.len() {
        if buf[i] == b'\n' {
            // "\n\n"
            if i + 1 < buf.len() && buf[i + 1] == b'\n' {
                return Some((2, i));
            }
            // "\n\r\n" can only be the tail of "\r\n\r\n"; require the
            // leading \r so we never mis-frame "\n\r\n" alone.
            if i + 2 < buf.len()
                && buf[i + 1] == b'\r'
                && buf[i + 2] == b'\n'
                && i >= 1
                && buf[i - 1] == b'\r'
            {
                return Some((4, i - 1));
            }
        }
        i += 1;
    }
    None
}

/// SSE parser
pub struct SSEParser {
    buffer: String,
}

impl SSEParser {
    /// Creates a new SSE parser
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Parses an SSE data chunk
    ///
    /// # Arguments
    /// * `chunk` - The received data chunk
    ///
    /// # Returns
    /// The list of complete events
    pub fn parse(&mut self, chunk: &str) -> Vec<SSEEvent> {
        // 0.22.0 audit fix (Medium): normalize CRLF line endings to LF so both
        // "\n\n" and "\r\n\r\n" event terminators are recognized (previously
        // CRLF-framed events were never split and buffered forever). A "\r\n"
        // split across two chunks is still handled by `lines()` / per-line
        // `trim()` in `parse_event`.
        self.buffer.push_str(&chunk.replace("\r\n", "\n"));

        let mut events = Vec::new();

        // SSE events are separated by a double newline
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_text = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos + 1);

            if let Some(event) = self.parse_event(&event_text) {
                events.push(event);
            }
        }

        events
    }

    /// Parses a single SSE event
    fn parse_event(&self, text: &str) -> Option<SSEEvent> {
        let mut event_type = None;
        // 0.22.0 audit fix (Medium): multi-line `data:` fields are JOINED with
        // "\n" per the SSE spec (previously each line overwrote the last, so
        // only the final line survived).
        let mut data_lines: Vec<String> = Vec::new();

        for line in text.lines() {
            if let Some(value) = line.strip_prefix("event:") {
                event_type = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("data:") {
                // `strip_prefix("data:")` + `trim` handles both "data: x" and
                // the space-less "data:x" form.
                data_lines.push(value.trim().to_string());
            }
        }

        // an event with only a data field still counts as valid
        if data_lines.is_empty() {
            None
        } else {
            Some(SSEEvent {
                event: event_type,
                data: data_lines.join("\n"),
            })
        }
    }
}

impl Default for SSEParser {
    fn default() -> Self {
        Self::new()
    }
}

/// SSE event
#[derive(Debug, Clone)]
pub struct SSEEvent {
    /// Event type
    pub event: Option<String>,

    /// Event data
    pub data: String,
}

impl SSEEvent {
    /// Checks whether this is the end event
    pub fn is_done(&self) -> bool {
        self.data == "[DONE]"
    }

    /// Parses OpenAI streaming response data
    pub fn parse_openai_chunk(&self) -> Result<Option<OpenAIStreamChunk>, serde_json::Error> {
        if self.is_done() {
            return Ok(None);
        }

        let chunk: OpenAIStreamChunk = serde_json::from_str(&self.data)?;
        Ok(Some(chunk))
    }
}

/// OpenAI streaming response chunk
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    /// Chunk ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model name
    pub model: String,
    /// Streaming choices
    pub choices: Vec<StreamChoice>,
    /// Total token usage for the whole call. OpenAI carries it at the end of the
    /// stream (usually in the last chunk before `[DONE]`); intermediate chunks omit it.
    #[serde(default)]
    pub usage: Option<StreamUsage>,
}

/// Token usage carried by a streaming response (OpenAI-compatible interface).
#[derive(Debug, Deserialize, Clone)]
pub struct StreamUsage {
    /// Input token count
    pub prompt_tokens: usize,
    /// Output token count
    pub completion_tokens: usize,
    /// Total token count
    pub total_tokens: usize,
    /// T5 (v0.23.0): reasoning-token accounting (OpenAI-compatible, both nesting styles).
    #[serde(default)]
    pub reasoning_tokens: Option<usize>,
    /// OpenAI-style `completion_tokens_details.reasoning_tokens` (nested reporting).
    #[serde(default)]
    pub completion_tokens_details: Option<StreamCompletionTokensDetails>,
}

/// T5 (v0.23.0): OpenAI-style `completion_tokens_details` on a streaming usage chunk.
#[derive(Debug, Deserialize, Clone)]
pub struct StreamCompletionTokensDetails {
    /// Reasoning-token count reported under the nested form.
    #[serde(default)]
    pub reasoning_tokens: Option<usize>,
}

impl StreamUsage {
    /// T5 (v0.23.0): the reasoning-token count across nesting styles, if reported.
    pub fn reasoning_tokens(&self) -> Option<usize> {
        self.reasoning_tokens.or_else(|| {
            self.completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)
        })
    }
}

/// Choice in a streaming response
#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    /// Choice index
    pub index: i32,
    /// Incremental content
    pub delta: Delta,
    /// Finish reason
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Streaming incremental content
#[derive(Debug, Deserialize)]
pub struct Delta {
    /// Role
    #[serde(default)]
    pub role: Option<String>,
    /// Content
    #[serde(default)]
    pub content: Option<String>,
    /// Reasoning/thinking delta from reasoning models (DeepSeek-R1, GLM and
    /// other OpenAI-compatible vendors expose it as `delta.reasoning_content`).
    /// 0.25.0: previously parsed nowhere, so the reasoning stream was lost.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Tool-call fragments (OpenAI streaming format). Each tool call is split
    /// across fragments: `id`/`name` arrive once (usually on the first
    /// fragment), `arguments` is a string concatenated across fragments.
    /// 0.20.0 S3.2: accumulated by [`StreamToolCallAccumulator`] so the
    /// terminal stream chunk carries complete tool calls. Absent for streams
    /// without tool calls.
    #[serde(default)]
    pub tool_calls: Option<Vec<StreamToolCallDelta>>,
}

/// One fragment of a streaming tool call (OpenAI `delta.tool_calls` element).
///
/// Fields are optional because OpenAI only sends `id`/`type` on the first
/// fragment and appends `arguments` on subsequent fragments. 0.20.0 S3.2.
#[derive(Debug, Deserialize)]
pub struct StreamToolCallDelta {
    /// Index of the tool call within the response. Always present in real
    /// OpenAI output; defaults to 0 for tolerant parsing of compatible vendors.
    #[serde(default)]
    pub index: usize,
    /// Tool call id (first fragment only).
    #[serde(default)]
    pub id: Option<String>,
    /// Tool type (`"function"`), first fragment only.
    #[serde(default, rename = "type")]
    pub tool_type: Option<String>,
    /// Function name / argument fragments.
    #[serde(default)]
    pub function: Option<StreamFunctionDelta>,
}

/// Function fragments of a streaming tool call.
#[derive(Debug, Deserialize)]
pub struct StreamFunctionDelta {
    /// Function name (first fragment only).
    #[serde(default)]
    pub name: Option<String>,
    /// Arguments delta — concatenated across fragments.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Streaming tool-call accumulator (0.20.0 S3.2).
///
/// OpenAI-family providers split each tool call into `delta.tool_calls`
/// fragments: `id`/`name` arrive once, `arguments` is a string concatenated
/// across fragments. Accumulates per `index` and flattens into complete
/// [`ToolCall`]s once the stream ends (or when the terminal usage chunk
/// arrives). Shared by the OpenAI and Azure streaming loops.
#[derive(Default)]
pub struct StreamToolCallAccumulator {
    calls: Vec<(usize, AccumulatedToolCall)>,
}

#[derive(Default)]
struct AccumulatedToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl StreamToolCallAccumulator {
    /// Fold one `delta.tool_calls` fragment into the accumulator.
    pub fn push(&mut self, delta: &StreamToolCallDelta) {
        let slot = match self.calls.iter_mut().find(|(i, _)| *i == delta.index) {
            Some(slot) => slot,
            None => {
                self.calls
                    .push((delta.index, AccumulatedToolCall::default()));
                self.calls.last_mut().expect("just pushed")
            }
        };
        // First fragment carries the id and name; later fragments only append
        // argument text. Guard so a provider echoing them twice does not clobber.
        if let Some(id) = &delta.id {
            if slot.1.id.is_empty() {
                slot.1.id.clone_from(id);
            }
        }
        if let Some(function) = &delta.function {
            if let Some(name) = &function.name {
                if slot.1.name.is_empty() {
                    slot.1.name.clone_from(name);
                }
            }
            if let Some(arguments) = &function.arguments {
                slot.1.arguments.push_str(arguments);
            }
        }
    }

    /// Flatten accumulated fragments into complete [`ToolCall`]s.
    ///
    /// Entries that never received a name (a tool call cancelled mid-stream)
    /// are dropped.
    pub fn build(&self) -> Vec<ToolCall> {
        self.calls
            .iter()
            .filter(|(_, c)| !c.name.is_empty())
            .map(|(_, c)| {
                ToolCall::builder(&c.id)
                    .name(&c.name)
                    .arguments(&c.arguments)
                    .build()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parser() {
        let mut parser = SSEParser::new();

        let chunk = "data: {\"test\": \"value\"}\n\n";
        let events = parser.parse(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"test\": \"value\"}");
    }

    #[test]
    fn test_sse_done_event() {
        let mut parser = SSEParser::new();

        let chunk = "data: [DONE]\n\n";
        let events = parser.parse(chunk);

        assert_eq!(events.len(), 1);
        assert!(events[0].is_done());
    }

    // 0.22.0 audit fix (Medium): CRLF-framed events must parse.

    #[test]
    fn test_sse_parser_crlf_event_separator() {
        let mut parser = SSEParser::new();
        let events = parser.parse("data: {\"a\":1}\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"a\":1}");
    }

    #[test]
    fn test_sse_parser_crlf_multiple_events_across_chunks() {
        let mut parser = SSEParser::new();
        let events = parser.parse("data: {\"a\":1}\r\n\r\ndata: {\"b\":2}\r\n\r\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "{\"a\":1}");
        assert_eq!(events[1].data, "{\"b\":2}");
    }

    #[test]
    fn test_sse_parser_crlf_split_across_chunks() {
        // The "\r\n" line ending is split across two pushed chunks.
        let mut parser = SSEParser::new();
        assert_eq!(parser.parse("data: {\"a\":1}\r\n").len(), 0);
        let events = parser.parse("\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"a\":1}");
    }

    // 0.22.0 audit fix (Medium): multi-line data fields must be joined.

    #[test]
    fn test_sse_parser_multiline_data_joined_with_newline() {
        let mut parser = SSEParser::new();
        let events = parser.parse("data: line1\ndata: line2\ndata: line3\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn test_sse_parser_data_without_space() {
        let mut parser = SSEParser::new();
        let events = parser.parse("data:{\"no\":\"space\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"no\":\"space\"}");
    }

    #[test]
    fn test_openai_chunk_parsing() {
        let event = SSEEvent {
            event: None,
            data: r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#.to_string(),
        };

        let chunk = event.parse_openai_chunk().unwrap().unwrap();
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_openai_chunk_parsing_reasoning_content_delta() {
        // 0.25.0: reasoning models stream CoT on `delta.reasoning_content`;
        // the Delta struct must capture it so the provider can forward it as
        // StreamChunk::thinking_content.
        let event = SSEEvent {
            event: None,
            data: r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"step one"},"finish_reason":null}]}"#.to_string(),
        };

        let chunk = event.parse_openai_chunk().unwrap().unwrap();
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("step one")
        );
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_openai_chunk_parsing_tool_calls_delta() {
        // 0.20.0 S3.2: the SSE Delta must deserialize a real OpenAI tool-calls
        // fragment (id/type on the first fragment, arguments appended later).
        let event = SSEEvent {
            event: None,
            data: r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#.to_string(),
        };

        let chunk = event.parse_openai_chunk().unwrap().unwrap();
        let deltas = chunk.choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool_calls parsed");
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            deltas[0].tool_type.as_deref(),
            Some("function"),
            "serde(rename=\"type\") maps the wire field"
        );
        assert_eq!(
            deltas[0].function.as_ref().and_then(|f| f.name.as_deref()),
            Some("get_weather")
        );
    }

    #[test]
    fn test_stream_tool_call_accumulator_reconstructs_fragmented_calls() {
        // 0.20.0 S3.2: OpenAI splits each tool call into delta.tool_calls
        // fragments; the accumulator re-assembles id/name (first fragment) and
        // concatenated arguments across fragments, keyed by index.
        let mut acc = StreamToolCallAccumulator::default();
        acc.push(
            &serde_json::from_str::<StreamToolCallDelta>(
                r#"{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"beij"}}"#,
            )
            .unwrap(),
        );
        acc.push(
            &serde_json::from_str::<StreamToolCallDelta>(
                r#"{"index":0,"function":{"arguments":"ing\"}"}}"#,
            )
            .unwrap(),
        );
        acc.push(
            &serde_json::from_str::<StreamToolCallDelta>(
                r#"{"index":1,"id":"call_2","type":"function","function":{"name":"add","arguments":"{\"a\":1}"}}"#,
            )
            .unwrap(),
        );

        let calls = acc.build();
        assert_eq!(calls.len(), 2, "two distinct tool calls by index");
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name(), "get_weather");
        assert_eq!(calls[0].arguments(), r#"{"city":"beijing"}"#);
        assert_eq!(calls[1].id, "call_2");
        assert_eq!(calls[1].name(), "add");
        assert_eq!(calls[1].arguments(), r#"{"a":1}"#);
    }

    // 0.26.0 M5: framer bounded buffer + O(n) scan regression.

    /// Runaway stream (bytes but never a terminator) must error, not grow
    /// memory without bound, and must fail-closed (buffer cleared) so a
    /// one-byte-after-overflow poke cannot regrow it.
    #[test]
    fn test_framer_rejects_runaway_buffer() {
        let mut framer = SseByteFramer::with_max_buffered(1024);
        let runaway = vec![b'x'; 4096];
        let err = framer.push(&runaway).unwrap_err();
        match err {
            SseError::BufferOverflow { buffered, max } => {
                assert_eq!(max, 1024);
                assert!(buffered >= 1024, "reports how much had buffered");
            }
        }
        // Fail-closed: buffer cleared, so a subsequent small poke stays small.
        assert_eq!(framer.pending(), 0);
        assert_eq!(framer.push(b"y").unwrap(), Vec::<String>::new());
        assert!(framer.pending() <= 1);
    }

    /// The O(n) scan must emit every complete event in one push — including a
    /// boundary-crossing chunk — and retain only the incomplete trailer.
    #[test]
    fn test_framer_emits_many_complete_events_and_keeps_trailer() {
        let mut framer = SseByteFramer::new();
        // Three complete events + an unterminated trailer, all in one chunk.
        let chunk = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: {\"c\":3}\n\ndata: {\"d\":4}";
        let events = framer.push(chunk.as_bytes()).unwrap();
        assert_eq!(events.len(), 3, "three complete events framed in one pass");
        assert_eq!(events[0], "data: {\"a\":1}\n\n");
        assert_eq!(events[2], "data: {\"c\":3}\n\n");
        assert_eq!(framer.pending(), "data: {\"d\":4}".len(), "trailer retained");

        // Completing the trailer next push produces the final event.
        let events = framer.push(b"\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "data: {\"d\":4}\n\n");
        assert_eq!(framer.pending(), 0);
    }

    /// O(n) regression anchor: a large number of complete events must frame in
    /// one push (the old per-event rescale was quadratic). This is a loose but
    /// measurable guard against the algorithmic regression, not a benchmark.
    #[test]
    fn test_framer_handles_many_events_linearly_in_one_push() {
        let mut framer = SseByteFramer::new();
        let n = 20_000;
        let chunk = "data: {}\n\n".repeat(n);
        let events = framer.push(chunk.as_bytes()).unwrap();
        assert_eq!(events.len(), n);
        assert_eq!(framer.pending(), 0);
    }

    #[test]
    fn test_stream_tool_call_accumulator_drops_unfinished_call() {
        // A tool call that started (id + name) but never received arguments is
        // still emitted; a fragment that never got a name is dropped.
        let mut acc = StreamToolCallAccumulator::default();
        acc.push(
            &serde_json::from_str::<StreamToolCallDelta>(
                r#"{"index":0,"id":"call_1","function":{"name":"get_weather"}}"#,
            )
            .unwrap(),
        );
        acc.push(
            &serde_json::from_str::<StreamToolCallDelta>(
                r#"{"index":1,"function":{"arguments":"{\"x\":1}"}}"#,
            )
            .unwrap(),
        );

        let calls = acc.build();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name(), "get_weather");
        assert_eq!(calls[0].arguments(), "");
    }
}
