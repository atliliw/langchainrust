// src/language_models/openai/sse.rs
//! SSE (Server-Sent Events) parser
//!
//! Used to parse OpenAI streaming responses

use lc_core::tools::ToolCall;
use serde::Deserialize;

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
        self.buffer.push_str(chunk);

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
        let mut data = None;

        for line in text.lines() {
            if let Some(value) = line.strip_prefix("event:") {
                event_type = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("data:") {
                data = Some(value.trim().to_string());
            }
        }

        // an event with only a data field still counts as valid
        if data.is_some() {
            Some(SSEEvent {
                event: event_type,
                data: data?,
            })
        } else {
            None
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
