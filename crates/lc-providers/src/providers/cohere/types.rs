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
    // 0.25.0: optional — `tool` content parts (tool-result history) carry
    // `tool_call_id`/`content` and no top-level `text`; a required String
    // made every such response fail to deserialize.
    #[serde(default)]
    pub(crate) text: Option<String>,
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
    // 0.25.0: the non-streaming v2 wire shape carries `arguments` as a JSON
    // **object**, not as a string (only streaming `tool-call-delta` fragments
    // are strings). Accept both and normalize to a string at the boundary.
    pub(crate) arguments: serde_json::Value,
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
/// - `tool-plan-delta` — planning text at `delta.message.tool_plan`, forwarded
///   as `thinking_content` (0.25.0)
/// - `message-end` — `finish_reason` and usage at `delta.usage.tokens`; also
///   flushes the accumulated tool calls
/// - `tool-call-start` / `tool-call-delta` / `tool-call-end` — streaming tool
///   calls fragmented by `index`, accumulated into a terminal chunk (0.25.0)
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
    /// `tool-call-end` carries the finished call index at `delta.index`.
    /// Framing marker only — the accumulator correlates via the per-call index.
    #[allow(dead_code)]
    #[serde(default)]
    pub(crate) index: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereStreamMessage {
    #[serde(default)]
    pub(crate) content: Option<CohereStreamContent>,
    /// Fragmented tool call deltas, correlated by `index`.
    #[serde(default)]
    pub(crate) tool_calls: Vec<CohereStreamToolCallDelta>,
    /// `tool-plan-delta` fragments: the model's chain-of-thought plan.
    #[serde(default)]
    pub(crate) tool_plan: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereStreamContent {
    #[serde(default)]
    pub(crate) text: Option<String>,
}

/// One fragment of a streaming tool call. `tool-call-start` carries the
/// id/name (with empty `arguments`), subsequent `tool-call-delta` events carry
/// only `index` + a fragment of the JSON arguments string.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereStreamToolCallDelta {
    pub(crate) index: usize,
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(rename = "type", default)]
    pub(crate) tool_type: Option<String>,
    #[serde(default)]
    pub(crate) function: Option<CohereStreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereStreamFunctionDelta {
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) arguments: Option<String>,
}
