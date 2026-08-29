// lc-providers/src/providers/cohere/types.rs
//! Private response types for the Cohere v2 API.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereChatResponse {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) message: Option<CohereMessage>,
    pub(crate) usage: Option<CohereUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereMessage {
    pub(crate) role: String,
    pub(crate) content: Vec<CohereContentPart>,
    #[serde(default)]
    pub(crate) tool_calls: Vec<CohereToolCall>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereContentPart {
    pub(crate) r#type: String,
    pub(crate) text: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereToolCall {
    pub(crate) id: String,
    pub(crate) r#type: String,
    pub(crate) function: CohereFunctionCall,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereFunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereUsage {
    pub(crate) tokens: CohereTokenUsage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereTokenUsage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
}

/// Cohere v2 streaming event — the `data:` payload of one SSE message.
///
/// 0.20.0 P4: Cohere v2 SSE is **not** OpenAI-compatible. There is no
/// `choices[0].delta.content`; the wire `type` field discriminates the event and
/// the payload differs per type:
///
/// - `message-start` / `content-start` / `content-end` — framing, no text
/// - `content-delta` (once per token) — text at `delta.message.content.text`
/// - `message-end` — `finish_reason` and usage at `delta.usage.tokens`
/// - `tool-plan-delta` / `tool-call-start` / `tool-call-delta` / `tool-call-end`
///   — streaming tool calls (not consumed by this parser yet)
///
/// Only the fields this crate reads are declared; serde ignores the rest.
#[derive(Debug, Deserialize)]
pub(crate) struct CohereStreamEvent {
    #[serde(rename = "type", default)]
    pub(crate) event_type: String,
    #[serde(default)]
    pub(crate) delta: Option<CohereStreamDelta>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereStreamDelta {
    #[serde(default)]
    pub(crate) message: Option<CohereStreamMessage>,
    #[serde(default)]
    pub(crate) usage: Option<CohereUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereStreamMessage {
    #[serde(default)]
    pub(crate) content: Option<CohereStreamContent>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereStreamContent {
    #[serde(default)]
    pub(crate) text: Option<String>,
}
