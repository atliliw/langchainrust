//! OpenTelemetry callback handler (feature = "opentelemetry")
//!
//! Converts framework execution events (LLM/Chain/Tool/Retriever start/end/error)
//! into OTel spans.
//!
//! A global tracer provider must be configured first (see the `otlp` feature's
//! [`crate::otlp`] pipeline); otherwise a noop tracer is used.
//!
//! # Example
//! ```ignore
//! use lc_callbacks::CallbackManager;
//! use lc_callbacks::OtelHandler;
//! let manager = CallbackManager::new().add_handler(std::sync::Arc::new(OtelHandler::from_global("langchainrust")));
//! ```
//!
//! # GenAI semantic conventions (B9 v0.22.4, T10 v0.23)
//!
//! Core attributes follow the stabilized OTel GenAI semantic conventions
//! (March 2025 stable release); the agent / tool-call / usage-extension and
//! events surfaces follow the 2026 draft (`gen-ai-agent-spans.md`,
//! `gen-ai-events.md`, `mcp.md`, all still Development stability — see
//! [`crate::semconv`]):
//!
//! - `gen_ai.provider.name` (the stabilized rename of experimental
//!   `gen_ai.system`), `gen_ai.operation.name`, `gen_ai.request.model`,
//!   `gen_ai.response.model`
//! - token usage as `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
//!   plus the Development extensions `gen_ai.usage.cache_read.input_tokens`,
//!   `gen_ai.usage.cache_write.input_tokens` (2026 rename of
//!   `cache_creation`; both provider payload spellings are accepted) and
//!   `gen_ai.usage.reasoning.output_tokens`
//! - `gen_ai.response.finish_reasons` as a **string array**
//! - `gen_ai.request.temperature` / `gen_ai.request.max_tokens`
//! - tool spans carry `gen_ai.tool.name` (always) and, when the executor
//!   supplied them, `gen_ai.tool.call.id` / `gen_ai.tool.description`;
//!   `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` are Opt-In and
//!   emitted only after [`OtelHandler::with_tool_payloads`]
//! - agent roots are emitted as `invoke_agent` spans (the executor stamps
//!   the run metadata; a run carrying `gen_ai.operation.name = "plan"`
//!   becomes a `plan` span); retriever spans use
//!   `gen_ai.operation.name = "retrieve"` (RAG retrieval is still under
//!   development upstream)
//! - chat span names follow `chat {model}` and tool spans
//!   `execute_tool {tool}`
//! - input messages are recorded as `gen_ai.system.message` /
//!   `gen_ai.user.message` / `gen_ai.assistant.message` /
//!   `gen_ai.tool.message` events with `gen_ai.message.content`; the model
//!   answer is recorded as a `gen_ai.choice.message` event. These per-message
//!   events are the **pre-2025 experimental form** (removed from the current
//!   registry) and are kept because shipping backends still key off them. The
//!   registry's replacement, a single opt-in
//!   `gen_ai.client.inference.operation.details` event with structured
//!   `gen_ai.input.messages` / `gen_ai.output.messages`, is emitted only
//!   after [`OtelHandler::with_operation_details_event`]. Message bodies are
//!   truncated to [`OtelHandler::with_max_message_chars`] (2048 default).
//! - errors set the span status to error plus `error.type` and an
//!   `exception` event carrying `exception.message`.
//!
//! # Data-source fallback
//!
//! Providers in this workspace record the model on the run's `inputs` and
//! token usage/response model on `outputs` (not every integration copies them
//! into `metadata`). The handler therefore resolves each attribute from
//! `metadata` first (explicit instrumentation always wins) and then falls
//! back to `run.inputs` / `run.outputs`.
//!
//! Every span also carries `langchainrust.run_id` and
//! `langchainrust.trace_id`, the join keys against evaluation reports
//! (`lc_evaluation::Report::run_id`, propagated via
//! `RunnableConfig.metadata["trace_id"]`).

use async_trait::async_trait;
use opentelemetry::global::{self, BoxedSpan, BoxedTracer};
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::{Array, Context, KeyValue, StringValue, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::base::CallbackHandler;
use crate::run_tree::RunTree;
use lc_schema::{Message, MessageType};

use crate::semconv::{
    error_type, truncate, ERROR_TYPE, EVENT_INFERENCE_DETAILS, GEN_AI_AGENT_NAME,
    GEN_AI_CHOICE_INDEX, GEN_AI_INPUT_MESSAGES, GEN_AI_MESSAGE_CONTENT, GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES, GEN_AI_PROVIDER_NAME, GEN_AI_REQUEST_MAX_TOKENS, GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE, GEN_AI_RESPONSE_FINISH_REASONS, GEN_AI_RESPONSE_MODEL,
    GEN_AI_TOOL_CALL_ARGUMENTS, GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_DESCRIPTION, GEN_AI_TOOL_NAME, GEN_AI_USAGE_CACHE_READ, GEN_AI_USAGE_CACHE_WRITE,
    GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS, GEN_AI_USAGE_REASONING, RUN_ID_ATTR,
    TRACE_ID_ATTR,
};

/// Default cap (chars) for message bodies recorded as span events.
const DEFAULT_MAX_MESSAGE_CHARS: usize = 2048;

/// OpenTelemetry callback handler: converts execution events into OTel spans
///
/// Uses a HashMap keyed by run ID instead of a stack, so spans are
/// tracked by their run ID and parent relationships are established
/// via the run tree's `parent_run_id` field, rather than assuming
/// strict stack-based nesting.
pub struct OtelHandler {
    tracer: BoxedTracer,
    /// Active spans keyed by run ID, supporting non-strictly-nested lifecycles.
    spans: Arc<Mutex<HashMap<String, BoxedSpan>>>,
    /// Maximum message-body length recorded in span events.
    max_message_chars: usize,
    /// Opt-in (2026 semconv): record `gen_ai.tool.call.arguments` /
    /// `gen_ai.tool.call.result` on tool spans. Default off because payloads
    /// may carry secrets/PII.
    record_tool_payloads: bool,
    /// Opt-in (2026 events model): emit the
    /// `gen_ai.client.inference.operation.details` snapshot event on chat
    /// spans. Default off; the legacy per-message events stay on regardless.
    record_details_event: bool,
    /// Chat-run id → serialized `gen_ai.input.messages` snapshot, captured at
    /// `on_llm_start` so the details event (emitted at end) can carry inputs.
    details_inputs: Arc<Mutex<HashMap<String, String>>>,
}

impl OtelHandler {
    /// Construct with the given tracer
    pub fn new(tracer: BoxedTracer) -> Self {
        Self {
            tracer,
            spans: Arc::new(Mutex::new(HashMap::new())),
            max_message_chars: DEFAULT_MAX_MESSAGE_CHARS,
            record_tool_payloads: false,
            record_details_event: false,
            details_inputs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Construct with the global tracer (requires `global::set_tracer_provider` first)
    pub fn from_global(name: &str) -> Self {
        Self::new(global::tracer(name.to_string()))
    }

    /// Caps how many characters of message/choice bodies are recorded as
    /// span events (default 2048). Set to a large value to capture full bodies.
    pub fn with_max_message_chars(mut self, max: usize) -> Self {
        self.max_message_chars = max;
        self
    }

    /// Opt in to `gen_ai.tool.call.arguments` (on start) and
    /// `gen_ai.tool.call.result` (on end) attributes on tool spans. Both are
    /// Opt-In in the 2026 semconv and may contain secrets or PII; default off.
    /// Bodies are truncated to the [`OtelHandler::with_max_message_chars`] cap.
    pub fn with_tool_payloads(mut self, enabled: bool) -> Self {
        self.record_tool_payloads = enabled;
        self
    }

    /// Opt in to the 2026 events-model snapshot event
    /// `gen_ai.client.inference.operation.details` on chat spans, carrying
    /// structured `gen_ai.input.messages` / `gen_ai.output.messages` plus the
    /// operation/usage attributes. Default off; the legacy
    /// `gen_ai.*.message` events are emitted either way.
    pub fn with_operation_details_event(mut self, enabled: bool) -> Self {
        self.record_details_event = enabled;
        self
    }

    /// Start a new span, setting parent context from the run tree if available.
    ///
    /// Every span gets the framework run/trace ids so exported spans join back
    /// to evaluation reports and trace trees.
    async fn start_span(&self, name: &str, run: &RunTree) {
        // If the run has a parent run with an active span, build an OTel Context
        // carrying that span's SpanContext so the new span is a true child in the
        // trace topology. The SpanContext is cloned while the lock is held, then
        // released before `start_with_context` runs.
        let parent_cx = {
            let spans = self.spans.lock().await;
            run.parent_run_id
                .and_then(|parent_id| {
                    spans
                        .get(&parent_id.to_string())
                        .map(|parent_span| parent_span.span_context().clone())
                })
                .map(|parent_ctx| Context::new().with_remote_span_context(parent_ctx))
        };

        let mut span = match parent_cx {
            Some(cx) => self.tracer.start_with_context(name.to_string(), &cx),
            None => self.tracer.start(name.to_string()),
        };
        span.set_attribute(KeyValue::new(RUN_ID_ATTR, run.id.to_string()));
        span.set_attribute(KeyValue::new(
            TRACE_ID_ATTR,
            run.trace_id.unwrap_or(run.id).to_string(),
        ));
        self.spans.lock().await.insert(run.id.to_string(), span);
    }

    /// End the span associated with the given run ID.
    async fn end_span(&self, run: &RunTree) {
        if let Some(mut span) = self.spans.lock().await.remove(&run.id.to_string()) {
            span.end();
        }
    }

    /// Add an event to the span associated with the given run ID.
    async fn add_event(&self, name: &str, run: &RunTree) {
        let mut spans = self.spans.lock().await;
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            span.add_event(name.to_string(), Vec::new());
        }
    }

    /// Record an error on the span (error status + `error.type` + exception
    /// event) and end it.
    async fn fail_span(&self, run: &RunTree, error: &str) {
        // Drop any pending details-event input snapshot for this run.
        self.details_inputs.lock().await.remove(&run.id.to_string());
        let mut spans = self.spans.lock().await;
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            let err_type = error_type(error);
            span.set_status(opentelemetry::trace::Status::error(error.to_string()));
            span.set_attribute(KeyValue::new(ERROR_TYPE, err_type.clone()));
            span.add_event(
                "exception".to_string(),
                vec![
                    KeyValue::new("exception.message", truncate(error, 1024)),
                    KeyValue::new("exception.type", err_type),
                ],
            );
        }
        drop(spans);
        self.end_span(run).await;
    }

    /// Number of currently active spans (for tests)
    pub async fn active_span_count(&self) -> usize {
        self.spans.lock().await.len()
    }
}

#[async_trait]
impl CallbackHandler for OtelHandler {
    async fn on_run_start(&self, run: &RunTree) {
        self.start_span("run", run).await;
    }
    async fn on_run_end(&self, run: &RunTree) {
        self.end_span(run).await;
    }
    async fn on_run_error(&self, run: &RunTree, error: &str) {
        self.fail_span(run, error).await;
    }

    async fn on_llm_start(&self, run: &RunTree, messages: &[Message]) {
        // 2026 events model: stash the structured input snapshot for the
        // `…operation.details` event emitted at end (Opt-In).
        if self.record_details_event {
            let input_json = messages_json(messages, self.max_message_chars);
            self.details_inputs
                .lock()
                .await
                .insert(run.id.to_string(), input_json);
        }

        // Semconv span name: `chat {model}` when the requested model is known.
        let model = resolve_request_model(run);
        let span_name = match &model {
            Some(m) => format!("chat {m}"),
            None => "chat".to_string(),
        };
        self.start_span(&span_name, run).await;

        let mut spans = self.spans.lock().await;
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            span.set_attribute(KeyValue::new(GEN_AI_OPERATION_NAME, "chat".to_string()));
            if let Some(system) = resolve_provider(run, model.as_deref()) {
                span.set_attribute(KeyValue::new(GEN_AI_PROVIDER_NAME, system));
            }
            if let Some(m) = &model {
                span.set_attribute(KeyValue::new(GEN_AI_REQUEST_MODEL, m.clone()));
            }
            request_sampling_attrs(span, run);
        }
        // Message events must be recorded while the span is active; the SDK
        // ignores events added after end(). Lock is still held, so record on
        // the same mutable borrow before it drops.
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            for message in messages {
                let event = message_event_name(&message.message_type);
                span.add_event(
                    event.to_string(),
                    vec![KeyValue::new(
                        GEN_AI_MESSAGE_CONTENT,
                        truncate(&message.content, self.max_message_chars),
                    )],
                );
            }
        }
    }

    async fn on_llm_end(&self, run: &RunTree, response: &str) {
        // Take the Opt-In details-event input snapshot before the span lock
        // (lock ordering: details_inputs is never held across spans).
        let details_input = if self.record_details_event {
            self.details_inputs.lock().await.remove(&run.id.to_string())
        } else {
            None
        };
        // Response-level semconv attributes (resolved from metadata with
        // fallback to run.outputs, where providers actually put them).
        let mut spans = self.spans.lock().await;
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            if let Some(model) =
                metadata_str(run, "response_model").or_else(|| outputs_str(run, "model"))
            {
                span.set_attribute(KeyValue::new(GEN_AI_RESPONSE_MODEL, model));
            }

            if let Some(reason) =
                metadata_str(run, "finish_reason").or_else(|| outputs_str(run, "finish_reason"))
            {
                span.set_attribute(KeyValue::new(
                    GEN_AI_RESPONSE_FINISH_REASONS,
                    Value::Array(Array::String(vec![StringValue::from(reason)])),
                ));
            }

            let usage = run
                .metadata
                .get("token_usage")
                .and_then(|v| v.as_object())
                .or_else(|| {
                    run.outputs
                        .as_ref()
                        .and_then(|o| o.get("token_usage"))
                        .and_then(|v| v.as_object())
                });
            if let Some(tokens) = usage {
                if let Some(p) = tokens.get("prompt_tokens").and_then(|v| v.as_u64()) {
                    span.set_attribute(KeyValue::new(GEN_AI_USAGE_INPUT_TOKENS, p as i64));
                }
                if let Some(c) = tokens.get("completion_tokens").and_then(|v| v.as_u64()) {
                    span.set_attribute(KeyValue::new(GEN_AI_USAGE_OUTPUT_TOKENS, c as i64));
                }
                // 0.21.0 S6.4 / T10: cache / reasoning token attribution
                // (Development-stability extension names — providers report
                // these only on some models). Cache *write* accepts both the
                // 2026 provider spelling and the Anthropic legacy key, emitting
                // the 2026-draft `gen_ai.usage.cache_write.input_tokens`.
                for (key, attr) in [
                    ("cache_read_input_tokens", GEN_AI_USAGE_CACHE_READ),
                    ("cache_write_input_tokens", GEN_AI_USAGE_CACHE_WRITE),
                    ("cache_creation_input_tokens", GEN_AI_USAGE_CACHE_WRITE),
                    ("reasoning_output_tokens", GEN_AI_USAGE_REASONING),
                ] {
                    if let Some(n) = tokens.get(key).and_then(|v| v.as_u64()) {
                        span.set_attribute(KeyValue::new(attr, n as i64));
                    }
                }
            }

            // Opt-In (2026 events model): one snapshot event carrying the
            // operation attributes plus structured input/output messages.
            if self.record_details_event {
                let output_json = serde_json::to_string(&serde_json::json!([{
                    "role": "assistant",
                    "content": truncate(response, self.max_message_chars),
                }]))
                .unwrap_or_else(|_| "[]".to_string());
                let mut attrs = vec![
                    KeyValue::new(GEN_AI_OPERATION_NAME, "chat".to_string()),
                    KeyValue::new(
                        GEN_AI_INPUT_MESSAGES,
                        details_input.unwrap_or_else(|| "[]".to_string()),
                    ),
                    KeyValue::new(GEN_AI_OUTPUT_MESSAGES, output_json),
                ];
                let request_model = resolve_request_model(run);
                if let Some(provider) = resolve_provider(run, request_model.as_deref()) {
                    attrs.push(KeyValue::new(GEN_AI_PROVIDER_NAME, provider));
                }
                if let Some(m) = request_model {
                    attrs.push(KeyValue::new(GEN_AI_REQUEST_MODEL, m));
                }
                if let Some(m) =
                    metadata_str(run, "response_model").or_else(|| outputs_str(run, "model"))
                {
                    attrs.push(KeyValue::new(GEN_AI_RESPONSE_MODEL, m));
                }
                if let Some(reason) =
                    metadata_str(run, "finish_reason").or_else(|| outputs_str(run, "finish_reason"))
                {
                    attrs.push(KeyValue::new(
                        GEN_AI_RESPONSE_FINISH_REASONS,
                        Value::Array(Array::String(vec![StringValue::from(reason)])),
                    ));
                }
                if let Some(tokens) = run
                    .metadata
                    .get("token_usage")
                    .and_then(|v| v.as_object())
                    .or_else(|| {
                        run.outputs
                            .as_ref()
                            .and_then(|o| o.get("token_usage"))
                            .and_then(|v| v.as_object())
                    })
                {
                    for (key, attr) in [
                        ("prompt_tokens", GEN_AI_USAGE_INPUT_TOKENS),
                        ("completion_tokens", GEN_AI_USAGE_OUTPUT_TOKENS),
                        ("cache_read_input_tokens", GEN_AI_USAGE_CACHE_READ),
                        ("cache_write_input_tokens", GEN_AI_USAGE_CACHE_WRITE),
                        ("cache_creation_input_tokens", GEN_AI_USAGE_CACHE_WRITE),
                        ("reasoning_output_tokens", GEN_AI_USAGE_REASONING),
                    ] {
                        if let Some(n) = tokens.get(key).and_then(|v| v.as_u64()) {
                            attrs.push(KeyValue::new(attr, n as i64));
                        }
                    }
                }
                span.add_event(EVENT_INFERENCE_DETAILS.to_string(), attrs);
            }

            // gen_ai.choice.message: the produced answer.
            span.add_event(
                "gen_ai.choice.message".to_string(),
                vec![
                    KeyValue::new(GEN_AI_CHOICE_INDEX, 0_i64),
                    KeyValue::new(
                        GEN_AI_MESSAGE_CONTENT,
                        truncate(response, self.max_message_chars),
                    ),
                ],
            );
        }
        drop(spans);
        self.end_span(run).await;
    }

    async fn on_llm_new_token(&self, run: &RunTree, _token: &str) {
        // Streaming token; semconv has no per-token event, keep the legacy one.
        self.add_event("token", run).await;
    }
    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        self.fail_span(run, error).await;
    }

    async fn on_chain_start(&self, run: &RunTree, _inputs: &serde_json::Value) {
        // T10: explicit instrumentation may reclassify a chain run as a GenAI
        // agent operation (`invoke_agent` for the executor root, `plan` for
        // planner components). When stamped, span naming follows the 2026
        // convention `{operation} {agent.name}` (name suffix only when known).
        let operation = metadata_str(run, GEN_AI_OPERATION_NAME);
        let agent_name = metadata_str(run, GEN_AI_AGENT_NAME);
        let (span_name, op_value) = match (&operation, &agent_name) {
            (Some(op), Some(name)) => (format!("{op} {name}"), op.clone()),
            (Some(op), None) => (op.clone(), op.clone()),
            (None, _) => ("chain".to_string(), "chain".to_string()),
        };
        self.start_span(&span_name, run).await;
        let mut spans = self.spans.lock().await;
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            span.set_attribute(KeyValue::new(GEN_AI_OPERATION_NAME, op_value));
            if let Some(name) = agent_name {
                span.set_attribute(KeyValue::new(GEN_AI_AGENT_NAME, name));
            }
        }
    }
    async fn on_chain_end(&self, run: &RunTree, _outputs: &serde_json::Value) {
        self.end_span(run).await;
    }
    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.fail_span(run, error).await;
    }

    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        // Semconv span name: `execute_tool {tool}`.
        self.start_span(&format!("execute_tool {tool_name}"), run)
            .await;
        let mut spans = self.spans.lock().await;
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            span.set_attribute(KeyValue::new(
                GEN_AI_OPERATION_NAME,
                "execute_tool".to_string(),
            ));
            span.set_attribute(KeyValue::new(GEN_AI_TOOL_NAME, tool_name.to_string()));
            // T10: Recommended tool attributes when the executor stamped them
            // (provider tool-call id + the tool's own description).
            if let Some(call_id) = metadata_str(run, GEN_AI_TOOL_CALL_ID) {
                span.set_attribute(KeyValue::new(GEN_AI_TOOL_CALL_ID, call_id));
            }
            if let Some(description) = metadata_str(run, GEN_AI_TOOL_DESCRIPTION) {
                span.set_attribute(KeyValue::new(GEN_AI_TOOL_DESCRIPTION, description));
            }
            if self.record_tool_payloads {
                span.set_attribute(KeyValue::new(
                    GEN_AI_TOOL_CALL_ARGUMENTS,
                    truncate(input, self.max_message_chars),
                ));
            }
        }
    }
    async fn on_tool_end(&self, run: &RunTree, output: &str) {
        if self.record_tool_payloads {
            let mut spans = self.spans.lock().await;
            if let Some(span) = spans.get_mut(&run.id.to_string()) {
                span.set_attribute(KeyValue::new(
                    GEN_AI_TOOL_CALL_RESULT,
                    truncate(output, self.max_message_chars),
                ));
            }
        }
        self.end_span(run).await;
    }
    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        self.fail_span(run, error).await;
    }

    async fn on_retriever_start(&self, run: &RunTree, _query: &str) {
        self.start_span("retriever", run).await;
        // Retrieval is not yet standardized — use the operation-name extension
        // (see module docs).
        let mut spans = self.spans.lock().await;
        if let Some(span) = spans.get_mut(&run.id.to_string()) {
            span.set_attribute(KeyValue::new(GEN_AI_OPERATION_NAME, "retrieve".to_string()));
        }
    }
    async fn on_retriever_end(&self, run: &RunTree, _documents: &[serde_json::Value]) {
        self.end_span(run).await;
    }
    async fn on_retriever_error(&self, run: &RunTree, error: &str) {
        self.fail_span(run, error).await;
    }
}

// --- resolution helpers -----------------------------------------------------

fn metadata_str(run: &RunTree, key: &str) -> Option<String> {
    run.metadata
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn inputs_str(run: &RunTree, key: &str) -> Option<String> {
    run.inputs
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn outputs_str(run: &RunTree, key: &str) -> Option<String> {
    run.outputs
        .as_ref()
        .and_then(|o| o.get(key))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Requested model: explicit metadata wins, then `inputs.model` (where all
/// workspace providers record it).
fn resolve_request_model(run: &RunTree) -> Option<String> {
    metadata_str(run, "model")
        .or_else(|| metadata_str(run, "ls_model_name"))
        .or_else(|| inputs_str(run, "model"))
}

/// Provider name: explicit instrumentation (`model_provider` /
/// `gen_ai.provider.name` metadata) wins; otherwise infer from the model id.
fn resolve_provider(run: &RunTree, model: Option<&str>) -> Option<String> {
    if let Some(p) = metadata_str(run, "model_provider")
        .or_else(|| metadata_str(run, "provider"))
        .or_else(|| metadata_str(run, GEN_AI_PROVIDER_NAME))
    {
        return Some(p);
    }
    model.and_then(infer_provider).map(str::to_string)
}

/// Best-effort provider inference from common model-id prefixes.
fn infer_provider(model: &str) -> Option<&'static str> {
    let m = model.trim().to_ascii_lowercase();
    let known: &[(&str, &str)] = &[
        ("gpt-", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
        ("chatgpt", "openai"),
        ("claude", "anthropic"),
        ("gemini", "gemini"),
        ("deepseek", "deepseek"),
        ("qwen", "dashscope"),
        ("glm", "zhipu"),
        ("command", "cohere"),
        ("mistral", "mistralai"),
        ("codestral", "mistralai"),
    ];
    known
        .iter()
        .find(|(prefix, _)| m.starts_with(prefix))
        .map(|(_, provider)| *provider)
}

/// Sets `gen_ai.request.temperature` / `gen_ai.request.max_tokens` from
/// metadata (when explicit instrumentation supplied them).
fn request_sampling_attrs(span: &mut BoxedSpan, run: &RunTree) {
    if let Some(max) = run.metadata.get("max_tokens").and_then(|v| v.as_u64()) {
        span.set_attribute(KeyValue::new(GEN_AI_REQUEST_MAX_TOKENS, max as i64));
    }
    if let Some(temp) = run.metadata.get("temperature").and_then(|v| v.as_f64()) {
        span.set_attribute(KeyValue::new(GEN_AI_REQUEST_TEMPERATURE, temp));
    }
}

/// Maps a framework message role to its GenAI message event name.
fn message_event_name(message_type: &MessageType) -> &'static str {
    match message_type {
        MessageType::System => "gen_ai.system.message",
        MessageType::Human => "gen_ai.user.message",
        MessageType::AI => "gen_ai.assistant.message",
        // Tool variant carries the tool_call_id; the id is not part of the
        // stable event attributes, content is what gets exported.
        MessageType::Tool { .. } => "gen_ai.tool.message",
    }
}

/// 2026 events-model role spelling for structured message arrays.
fn message_role(message_type: &MessageType) -> &'static str {
    match message_type {
        MessageType::System => "system",
        MessageType::Human => "user",
        MessageType::AI => "assistant",
        MessageType::Tool { .. } => "tool",
    }
}

/// Serializes messages as the `[{"role": …, "content": …}]` structured form
/// used by `gen_ai.input.messages` on the Opt-In details event. Each body is
/// truncated independently; serialization of plain string objects never fails,
/// but the `unwrap_or` keeps that contract honest.
fn messages_json(messages: &[Message], max_chars: usize) -> String {
    let payload: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": message_role(&m.message_type),
                "content": truncate(&m.content, max_chars),
            })
        })
        .collect();
    serde_json::to_string(&payload).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::testing::trace::InMemorySpanExporterBuilder;
    use opentelemetry_sdk::trace::{SimpleSpanProcessor, TracerProvider};
    use serde_json::json;

    /// Builds a handler backed by an in-memory exporter (no collector needed).
    fn handler_with_exporter() -> (
        OtelHandler,
        opentelemetry_sdk::testing::trace::InMemorySpanExporter,
    ) {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = TracerProvider::builder()
            .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter.clone())))
            .build();
        let tracer: BoxedTracer = BoxedTracer::new(Box::new(provider.tracer("test")));
        (OtelHandler::new(tracer), exporter)
    }

    fn llm_run() -> RunTree {
        // Mirrors what OpenAIChat records: model in inputs, usage in outputs.
        let mut run = RunTree::new(
            "gpt-4o-mini:chat",
            crate::RunType::Llm,
            json!({
                "messages": ["2+2?"],
                "model": "gpt-4o-mini",
            }),
        );
        run.trace_id = Some(run.id);
        run
    }

    fn attr_str(span: &opentelemetry_sdk::export::trace::SpanData, key: &str) -> Option<String> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.as_str().into_owned())
    }

    fn attr_i64(span: &opentelemetry_sdk::export::trace::SpanData, key: &str) -> Option<i64> {
        span.attributes.iter().find_map(|kv| {
            if kv.key.as_str() == key {
                match &kv.value {
                    Value::I64(v) => Some(*v),
                    _ => None,
                }
            } else {
                None
            }
        })
    }

    fn finish_reasons(span: &opentelemetry_sdk::export::trace::SpanData) -> Vec<String> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == GEN_AI_RESPONSE_FINISH_REASONS)
            .map(|kv| match &kv.value {
                Value::Array(Array::String(values)) => {
                    values.iter().map(|v| v.as_str().to_string()).collect()
                }
                _ => vec![],
            })
            .unwrap_or_default()
    }

    fn event_names(span: &opentelemetry_sdk::export::trace::SpanData) -> Vec<String> {
        span.events.iter().map(|e| e.name.to_string()).collect()
    }

    #[tokio::test]
    async fn llm_span_uses_stable_semconv_from_inputs_and_outputs() {
        let (h, exporter) = handler_with_exporter();
        let run = llm_run();
        h.on_llm_start(&run, &[Message::system("be brief"), Message::human("2+2?")])
            .await;

        // Provider metadata is what on_llm_end observes: mutate via RunTree end.
        let mut ended = run.clone();
        ended.end(json!({
            "content": "4",
            "model": "gpt-4o-mini-2024-07-18",
            "token_usage": {"prompt_tokens": 11u64, "completion_tokens": 3u64, "total_tokens": 14u64},
            "finish_reason": "stop",
        }));
        h.on_llm_end(&ended, "4").await;

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name, "chat gpt-4o-mini");
        assert_eq!(
            attr_str(span, GEN_AI_PROVIDER_NAME).as_deref(),
            Some("openai")
        );
        assert_eq!(
            attr_str(span, GEN_AI_OPERATION_NAME).as_deref(),
            Some("chat")
        );
        assert_eq!(
            attr_str(span, GEN_AI_REQUEST_MODEL).as_deref(),
            Some("gpt-4o-mini")
        );
        assert_eq!(
            attr_str(span, GEN_AI_RESPONSE_MODEL).as_deref(),
            Some("gpt-4o-mini-2024-07-18")
        );
        assert_eq!(attr_i64(span, GEN_AI_USAGE_INPUT_TOKENS), Some(11));
        assert_eq!(attr_i64(span, GEN_AI_USAGE_OUTPUT_TOKENS), Some(3));
        assert_eq!(finish_reasons(span), vec!["stop".to_string()]);
        // join keys
        assert_eq!(
            attr_str(span, RUN_ID_ATTR).as_deref(),
            Some(run.id.to_string().as_str())
        );
        assert_eq!(
            attr_str(span, TRACE_ID_ATTR).as_deref(),
            Some(run.id.to_string().as_str())
        );
        // message + choice events
        let names = event_names(span);
        assert!(names.contains(&"gen_ai.system.message".to_string()));
        assert!(names.contains(&"gen_ai.user.message".to_string()));
        assert!(names.contains(&"gen_ai.choice.message".to_string()));
    }

    #[tokio::test]
    async fn llm_span_records_choice_content_truncated() {
        // Handler with a tiny message cap via the public builder.
        let exporter2 = InMemorySpanExporterBuilder::new().build();
        let provider = TracerProvider::builder()
            .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter2.clone())))
            .build();
        let h = OtelHandler::new(BoxedTracer::new(Box::new(provider.tracer("t"))))
            .with_max_message_chars(5);

        let run = llm_run();
        h.on_llm_start(&run, &[Message::human("abcdefghij")]).await;
        h.on_llm_end(&run, "0123456789").await;
        let spans = exporter2.get_finished_spans().unwrap();
        let choice = spans[0]
            .events
            .iter()
            .find(|e| e.name == "gen_ai.choice.message")
            .unwrap();
        let content = choice
            .attributes
            .iter()
            .find(|kv| kv.key.as_str() == GEN_AI_MESSAGE_CONTENT)
            .unwrap();
        assert_eq!(content.value.as_str().as_ref(), "01234…");
    }

    #[tokio::test]
    async fn error_sets_status_error_type_and_exception_event() {
        let (h, exporter) = handler_with_exporter();
        let run = llm_run();
        h.on_llm_start(&run, &[Message::human("hi")]).await;
        h.on_llm_error(&run, "http error: upstream 500").await;

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        assert!(matches!(
            span.status,
            opentelemetry::trace::Status::Error { .. }
        ));
        assert_eq!(attr_str(span, ERROR_TYPE).as_deref(), Some("http error"));
        let exception = span.events.iter().find(|e| e.name == "exception").unwrap();
        assert!(exception
            .attributes
            .iter()
            .any(|kv| kv.key.as_str() == "exception.message"));
    }

    #[tokio::test]
    async fn tool_span_named_execute_tool_with_tool_name() {
        let (h, exporter) = handler_with_exporter();
        let run = RunTree::new("calc", crate::RunType::Tool, json!({}));
        h.on_tool_start(&run, "calculator", "1+1").await;
        h.on_tool_end(&run, "2").await;

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        assert_eq!(span.name, "execute_tool calculator");
        assert_eq!(
            attr_str(span, GEN_AI_OPERATION_NAME).as_deref(),
            Some("execute_tool")
        );
        assert_eq!(
            attr_str(span, GEN_AI_TOOL_NAME).as_deref(),
            Some("calculator")
        );
    }

    #[tokio::test]
    async fn metadata_overrides_inputs_and_enables_cache_attribution() {
        let (h, exporter) = handler_with_exporter();
        let mut run = llm_run();
        run.metadata
            .insert("model_provider".into(), json!("custom-gateway"));
        run.metadata.insert("temperature".into(), json!(0.2));
        run.metadata.insert("max_tokens".into(), json!(512));

        h.on_llm_start(&run, &[]).await;
        let mut ended = run.clone();
        ended.end(json!({
            "token_usage": {
                "prompt_tokens": 100u64,
                "completion_tokens": 20u64,
                "cache_read_input_tokens": 80u64,
                "reasoning_output_tokens": 5u64,
            }
        }));
        h.on_llm_end(&ended, "").await;

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        assert_eq!(
            attr_str(span, GEN_AI_PROVIDER_NAME).as_deref(),
            Some("custom-gateway"),
            "explicit metadata provider beats model-prefix inference"
        );
        assert_eq!(attr_i64(span, GEN_AI_REQUEST_MAX_TOKENS), Some(512));
        assert_eq!(attr_i64(span, GEN_AI_USAGE_CACHE_READ), Some(80));
        assert_eq!(attr_i64(span, GEN_AI_USAGE_REASONING), Some(5));
    }

    #[tokio::test]
    async fn cache_write_accepts_new_and_legacy_provider_keys() {
        // 2026 spelling.
        let (h, exporter) = handler_with_exporter();
        let run = llm_run();
        h.on_llm_start(&run, &[]).await;
        let mut ended = run.clone();
        ended.end(json!({
            "token_usage": {
                "prompt_tokens": 100u64,
                "completion_tokens": 10u64,
                "cache_write_input_tokens": 40u64,
            }
        }));
        h.on_llm_end(&ended, "").await;
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(attr_i64(&spans[0], GEN_AI_USAGE_CACHE_WRITE), Some(40));

        // Anthropic legacy spelling maps to the renamed attribute.
        let (h, exporter) = handler_with_exporter();
        h.on_llm_start(&run, &[]).await;
        let mut ended_legacy = run.clone();
        ended_legacy.end(json!({
            "token_usage": {
                "prompt_tokens": 100u64,
                "completion_tokens": 10u64,
                "cache_creation_input_tokens": 7u64,
            }
        }));
        h.on_llm_end(&ended_legacy, "").await;
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(attr_i64(&spans[0], GEN_AI_USAGE_CACHE_WRITE), Some(7));
    }

    #[tokio::test]
    async fn chain_run_uses_stamped_agent_operation_and_span_name() {
        let (h, exporter) = handler_with_exporter();

        // Executor root: stamped `invoke_agent`, no agent name.
        let mut root = RunTree::new("AgentExecutor", crate::RunType::Chain, json!({}));
        root.metadata
            .insert(GEN_AI_OPERATION_NAME.into(), json!("invoke_agent"));
        h.on_chain_start(&root, &json!({})).await;
        h.on_chain_end(&root, &json!({})).await;

        // Planner component: stamped `plan` with an agent name.
        let mut planner = root.create_child("planner", crate::RunType::Chain, json!({}));
        planner
            .metadata
            .insert(GEN_AI_OPERATION_NAME.into(), json!("plan"));
        planner
            .metadata
            .insert(GEN_AI_AGENT_NAME.into(), json!("researcher"));
        h.on_chain_start(&planner, &json!({})).await;
        h.on_chain_end(&planner, &json!({})).await;

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 2);
        let agent = spans.iter().find(|s| s.name == "invoke_agent").unwrap();
        assert_eq!(
            attr_str(agent, GEN_AI_OPERATION_NAME).as_deref(),
            Some("invoke_agent")
        );
        assert_eq!(attr_str(agent, GEN_AI_AGENT_NAME), None);
        let plan = spans.iter().find(|s| s.name == "plan researcher").unwrap();
        assert_eq!(
            attr_str(plan, GEN_AI_OPERATION_NAME).as_deref(),
            Some("plan")
        );
        assert_eq!(
            attr_str(plan, GEN_AI_AGENT_NAME).as_deref(),
            Some("researcher")
        );
    }

    #[tokio::test]
    async fn unstamped_chain_run_keeps_legacy_chain_span() {
        let (h, exporter) = handler_with_exporter();
        let run = RunTree::new("ordinary-chain", crate::RunType::Chain, json!({}));
        h.on_chain_start(&run, &json!({})).await;
        h.on_chain_end(&run, &json!({})).await;
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans[0].name, "chain");
        assert_eq!(
            attr_str(&spans[0], GEN_AI_OPERATION_NAME).as_deref(),
            Some("chain")
        );
    }

    #[tokio::test]
    async fn tool_span_carries_call_id_and_description_from_metadata() {
        let (h, exporter) = handler_with_exporter();
        let mut run = RunTree::new("calc", crate::RunType::Tool, json!({}));
        run.metadata
            .insert(GEN_AI_TOOL_CALL_ID.into(), json!("call_42"));
        run.metadata
            .insert(GEN_AI_TOOL_DESCRIPTION.into(), json!("adds numbers"));
        h.on_tool_start(&run, "calculator", "1+1").await;
        h.on_tool_end(&run, "2").await;

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        assert_eq!(
            attr_str(span, GEN_AI_TOOL_CALL_ID).as_deref(),
            Some("call_42")
        );
        assert_eq!(
            attr_str(span, GEN_AI_TOOL_DESCRIPTION).as_deref(),
            Some("adds numbers")
        );
        // Opt-In payloads stay off by default.
        assert_eq!(attr_str(span, GEN_AI_TOOL_CALL_ARGUMENTS), None);
        assert_eq!(attr_str(span, GEN_AI_TOOL_CALL_RESULT), None);
    }

    #[tokio::test]
    async fn tool_payloads_are_opt_in() {
        let exporter2 = InMemorySpanExporterBuilder::new().build();
        let provider = TracerProvider::builder()
            .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter2.clone())))
            .build();
        let h = OtelHandler::new(BoxedTracer::new(Box::new(provider.tracer("t"))))
            .with_tool_payloads(true)
            .with_max_message_chars(4);

        let run = RunTree::new("calc", crate::RunType::Tool, json!({}));
        h.on_tool_start(&run, "calculator", "123456789").await;
        h.on_tool_end(&run, "987654321").await;

        let spans = exporter2.get_finished_spans().unwrap();
        let span = &spans[0];
        assert_eq!(
            attr_str(span, GEN_AI_TOOL_CALL_ARGUMENTS).as_deref(),
            Some("1234…")
        );
        assert_eq!(
            attr_str(span, GEN_AI_TOOL_CALL_RESULT).as_deref(),
            Some("9876…")
        );
    }

    #[tokio::test]
    async fn operation_details_event_is_opt_in_snapshot() {
        // Default off: no details event on the chat span.
        let (h, exporter) = handler_with_exporter();
        let run = llm_run();
        h.on_llm_start(&run, &[Message::human("hi")]).await;
        h.on_llm_end(&run, "hello").await;
        let spans = exporter.get_finished_spans().unwrap();
        assert!(!spans[0]
            .events
            .iter()
            .any(|e| e.name == EVENT_INFERENCE_DETAILS));

        // Opted in: one details event with structured input/output JSON and
        // the operation attributes.
        let exporter2 = InMemorySpanExporterBuilder::new().build();
        let provider = TracerProvider::builder()
            .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter2.clone())))
            .build();
        let h = OtelHandler::new(BoxedTracer::new(Box::new(provider.tracer("t"))))
            .with_operation_details_event(true);

        let run = llm_run();
        h.on_llm_start(&run, &[Message::human("hi")]).await;
        let mut ended = run.clone();
        ended.end(json!({
            "model": "gpt-4o-mini-2024-07-18",
            "token_usage": {"prompt_tokens": 11u64, "completion_tokens": 3u64,
                            "cache_read_input_tokens": 2u64},
            "finish_reason": "stop",
        }));
        h.on_llm_end(&ended, "hello").await;

        let spans = exporter2.get_finished_spans().unwrap();
        let event = spans[0]
            .events
            .iter()
            .find(|e| e.name == EVENT_INFERENCE_DETAILS)
            .expect("details event emitted when opted in");
        let event_attr = |key: &str| {
            event
                .attributes
                .iter()
                .find(|kv| kv.key.as_str() == key)
                .map(|kv| kv.value.as_str().to_string())
        };
        let inputs = event_attr(GEN_AI_INPUT_MESSAGES).unwrap();
        assert!(inputs.contains("\"role\":\"user\""), "inputs: {inputs}");
        assert!(inputs.contains("\"content\":\"hi\""));
        let outputs = event_attr(GEN_AI_OUTPUT_MESSAGES).unwrap();
        assert!(outputs.contains("\"role\":\"assistant\""));
        assert!(outputs.contains("hello"));
        assert_eq!(event_attr(GEN_AI_OPERATION_NAME).as_deref(), Some("chat"));
        assert_eq!(event_attr(GEN_AI_PROVIDER_NAME).as_deref(), Some("openai"));
        assert_eq!(
            event_attr(GEN_AI_RESPONSE_MODEL).as_deref(),
            Some("gpt-4o-mini-2024-07-18")
        );
        // Legacy per-message events are still emitted (backends consume them).
        assert!(spans[0]
            .events
            .iter()
            .any(|e| e.name == "gen_ai.user.message"));
    }

    #[test]
    fn error_type_takes_first_line_before_colon() {
        assert_eq!(error_type("TimeoutError: deadlined"), "TimeoutError");
        assert_eq!(error_type("boom"), "boom");
        assert_eq!(error_type(""), "error");
        assert_eq!(error_type("line1\nline2: x"), "line1");
    }

    #[test]
    fn provider_inference_known_prefixes() {
        assert_eq!(infer_provider("gpt-4o"), Some("openai"));
        assert_eq!(infer_provider("o4-mini"), Some("openai"));
        assert_eq!(infer_provider("Claude-3.5-Sonnet"), Some("anthropic"));
        assert_eq!(infer_provider("gemini-2.0-flash"), Some("gemini"));
        assert_eq!(infer_provider("qwen-plus"), Some("dashscope"));
        assert_eq!(infer_provider("my-internal-model"), None);
    }

    #[tokio::test]
    async fn from_global_without_provider_does_not_panic() {
        // Without a global provider the noop tracer is returned; must not panic.
        let _h = OtelHandler::from_global("test");
    }

    #[tokio::test]
    async fn span_lifecycle_balance_and_empty_end() {
        let (h, _exporter) = handler_with_exporter();
        assert_eq!(h.active_span_count().await, 0);
        let r1 = RunTree::new("r1", crate::RunType::Llm, json!({}));
        let r2 = RunTree::new("r2", crate::RunType::Tool, json!({}));
        h.start_span("a", &r1).await;
        h.start_span("b", &r2).await;
        assert_eq!(h.active_span_count().await, 2);
        h.end_span(&r2).await;
        assert_eq!(h.active_span_count().await, 1);
        h.end_span(&r1).await;
        assert_eq!(h.active_span_count().await, 0);
        h.end_span(&r1).await;
        assert_eq!(h.active_span_count().await, 0);
    }
}
