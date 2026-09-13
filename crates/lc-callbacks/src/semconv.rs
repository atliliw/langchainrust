//! Stabilized OTel GenAI semantic-convention names shared by every OTel
//! emission surface ([`crate::handlers::OtelHandler`] and
//! [`crate::tracing::OtelTracingBackend`]) so the two cannot drift.
//!
//! References the OTel GenAI semconv stable release (March 2025).

#![allow(dead_code)] // Not every surface uses every constant.

/// `gen_ai.provider.name` (stabilized rename of experimental `gen_ai.system`).
pub(crate) const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
/// `gen_ai.operation.name` (chat / execute_tool / retrieve …).
pub(crate) const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
/// `gen_ai.request.model`.
pub(crate) const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
/// `gen_ai.response.model`.
pub(crate) const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
/// `gen_ai.usage.input_tokens`.
pub(crate) const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
/// `gen_ai.usage.output_tokens`.
pub(crate) const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
/// `gen_ai.response.finish_reasons` (string array, plural).
pub(crate) const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
/// `gen_ai.request.max_tokens`.
pub(crate) const GEN_AI_REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
/// `gen_ai.request.temperature`.
pub(crate) const GEN_AI_REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
/// `gen_ai.tool.name`.
pub(crate) const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
/// `gen_ai.message.content`.
pub(crate) const GEN_AI_MESSAGE_CONTENT: &str = "gen_ai.message.content";
/// `gen_ai.choice.index`.
pub(crate) const GEN_AI_CHOICE_INDEX: &str = "gen_ai.choice.index";

/// Standard error attribute (OTel general semconv).
pub(crate) const ERROR_TYPE: &str = "error.type";

/// Framework join key: the run id (also carried on evaluation reports).
pub(crate) const RUN_ID_ATTR: &str = "langchainrust.run_id";
/// Framework join key: the trace id a run belongs to.
pub(crate) const TRACE_ID_ATTR: &str = "langchainrust.trace_id";

// Provider-extension attributes still in registry development.
pub(crate) const GEN_AI_USAGE_CACHE_READ: &str = "gen_ai.usage.cache_read.input_tokens";
pub(crate) const GEN_AI_USAGE_CACHE_CREATION: &str = "gen_ai.usage.cache_creation.input_tokens";
pub(crate) const GEN_AI_USAGE_REASONING: &str = "gen_ai.usage.reasoning.output_tokens";

/// Cap for `error.type` values (it is a type, not a message).
const MAX_ERROR_TYPE_CHARS: usize = 64;

/// Derives an `error.type`-style short type from an error message: the first
/// line, cut at the first colon. Falls back to `"error"`.
pub(crate) fn error_type(message: &str) -> String {
    let first = message.lines().next().unwrap_or("").trim();
    let head = first.split(':').next().unwrap_or(first).trim();
    if head.is_empty() {
        "error".to_string()
    } else {
        truncate(head, MAX_ERROR_TYPE_CHARS)
    }
}

/// Unicode-safe truncation (chars, not bytes), appends an ellipsis when cut.
pub(crate) fn truncate(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        None => s.to_string(),
        Some((idx, _)) => format!("{}…", &s[..idx]),
    }
}
