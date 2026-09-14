//! OTel GenAI / MCP semantic-convention names shared by every OTel emission
//! surface (`OtelHandler` and `OtelTracingBackend`) so the two cannot drift.
//! Other crates (e.g. `lc-agents`) may also reference the `pub` constants
//! here instead of hardcoding attribute strings.
//!
//! # Stability (T10, v0.23 — 2026 alignment)
//!
//! The GenAI semconv is still published at **Development** stability: in 2026
//! the working group moved the docs to a dedicated repository
//! (`open-telemetry/semantic-conventions-genai`) and renames still happen
//! (notably `gen_ai.usage.cache_creation.input_tokens` →
//! [`GEN_AI_USAGE_CACHE_WRITE`]). Centralizing every name in this module is
//! deliberate: when the registry moves a name, one constant changes and both
//! export surfaces follow. The provider-extension and agent/tool attributes
//! below follow the 2026 draft (`gen-ai-agent-spans.md`, `gen-ai-events.md`,
//! `mcp.md`); the core chat attributes (`provider.name`, `operation.name`,
//! request/response model, usage, finish_reasons) follow the March 2025 stable
//! release.
//!
//! MCP client spans live in `lc-mcp`, which cannot depend on this crate;
//! their `mcp.*` constants are mirrored in `lc_mcp::instrument` with a
//! cross-reference comment.

#![allow(dead_code)] // Not every surface uses every constant.

/// `gen_ai.provider.name` (stabilized rename of experimental `gen_ai.system`).
pub const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
/// `gen_ai.operation.name` (chat / execute_tool / invoke_agent / plan / retrieve …).
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
/// `gen_ai.request.model`.
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
/// `gen_ai.response.model`.
pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
/// `gen_ai.usage.input_tokens`.
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
/// `gen_ai.usage.output_tokens`.
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
/// `gen_ai.response.finish_reasons` (string array, plural).
pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
/// `gen_ai.request.max_tokens`.
pub const GEN_AI_REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
/// `gen_ai.request.temperature`.
pub const GEN_AI_REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
/// `gen_ai.tool.name` (tool spans; conditionally required on MCP tool calls).
pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
/// `gen_ai.tool.call.id` — the provider tool-call id (Recommended, 2026 draft).
pub const GEN_AI_TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
/// `gen_ai.tool.description` (Recommended, 2026 draft).
pub const GEN_AI_TOOL_DESCRIPTION: &str = "gen_ai.tool.description";
/// `gen_ai.tool.call.arguments` — Opt-In: recorded only when explicitly enabled.
pub const GEN_AI_TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
/// `gen_ai.tool.call.result` — Opt-In: recorded only when explicitly enabled.
pub const GEN_AI_TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
/// `gen_ai.agent.name` — agent identity for `invoke_agent` / `plan` spans.
pub const GEN_AI_AGENT_NAME: &str = "gen_ai.agent.name";

/// `gen_ai.message.content` — payload key of the legacy per-message span
/// events (see [`EVENT_INFERENCE_DETAILS`] doc for why those are retained).
pub const GEN_AI_MESSAGE_CONTENT: &str = "gen_ai.message.content";
/// `gen_ai.choice.index`.
pub const GEN_AI_CHOICE_INDEX: &str = "gen_ai.choice.index";

/// `gen_ai.client.inference.operation.details` (2026 events model, Opt-In).
///
/// The only GenAI LLM event the current registry defines: a single opt-in
/// snapshot event on the chat span carrying the operation attributes plus
/// structured `gen_ai.input.messages` / `gen_ai.output.messages`. It is
/// emitted solely when the consumer opts in
/// (`OtelHandler::with_operation_details_event`); the legacy
/// `gen_ai.{system,user,assistant,tool}.message` / `gen_ai.choice.message`
/// span events (removed from the registry in the 2025 events redesign) stay
/// on by default because shipping backends (Langfuse, Datadog, New Relic)
/// still key off them.
pub const EVENT_INFERENCE_DETAILS: &str = "gen_ai.client.inference.operation.details";
/// Structured input messages JSON on [`EVENT_INFERENCE_DETAILS`].
pub const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
/// Structured output messages JSON on [`EVENT_INFERENCE_DETAILS`].
pub const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";

/// Standard error attribute (OTel general semconv).
pub const ERROR_TYPE: &str = "error.type";

/// Framework join key: the run id (also carried on evaluation reports).
pub const RUN_ID_ATTR: &str = "langchainrust.run_id";
/// Framework join key: the trace id a run belongs to.
pub const TRACE_ID_ATTR: &str = "langchainrust.trace_id";

// Provider-extension usage attributes (Development stability). Values SHOULD
// be included in `gen_ai.usage.input_tokens` totals by backend convention.
/// `gen_ai.usage.cache_read.input_tokens` (Anthropic prompt-cache hits).
pub const GEN_AI_USAGE_CACHE_READ: &str = "gen_ai.usage.cache_read.input_tokens";
/// `gen_ai.usage.cache_write.input_tokens` — the 2026-draft rename of
/// `gen_ai.usage.cache_creation.input_tokens` (still the published name in
/// registry v1.40). Provider payloads use *both* spellings
/// (`cache_write_input_tokens` and the Anthropic legacy
/// `cache_creation_input_tokens`); the handler accepts either key.
pub const GEN_AI_USAGE_CACHE_WRITE: &str = "gen_ai.usage.cache_write.input_tokens";
/// `gen_ai.usage.reasoning.output_tokens` (reasoning/thinking tokens).
pub const GEN_AI_USAGE_REASONING: &str = "gen_ai.usage.reasoning.output_tokens";

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
