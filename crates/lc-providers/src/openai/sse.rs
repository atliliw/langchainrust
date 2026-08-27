// src/language_models/openai/sse.rs
//! SSE (Server-Sent Events) parser
//!
//! Used to parse OpenAI streaming responses

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
}
