use std::sync::Mutex;

#[cfg(feature = "opentelemetry")]
use std::collections::HashMap;

#[cfg(feature = "opentelemetry")]
use crate::semconv;

use super::span::{build_tree, TraceSpan};
use super::TracingBackend;

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// In-memory tracing backend for development and testing.
pub struct InMemoryTracingBackend {
    spans: Mutex<Vec<TraceSpan>>,
}

impl InMemoryTracingBackend {
    /// Create a new empty in-memory backend.
    pub fn new() -> Self {
        Self {
            spans: Mutex::new(Vec::new()),
        }
    }

    /// Return a snapshot of all recorded spans.
    pub fn spans(&self) -> Vec<TraceSpan> {
        self.spans.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Remove all recorded spans.
    pub fn clear(&self) {
        self.spans.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Get the full trace tree rooted at `root_id`.
    ///
    /// Returns `None` if no span with that ID exists.
    pub fn trace_tree(&self, root_id: &str) -> Option<super::TraceNode> {
        let spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        let root = spans.iter().find(|s| s.id == root_id)?;
        Some(build_tree(root, &spans))
    }
}

impl Default for InMemoryTracingBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TracingBackend for InMemoryTracingBackend {
    fn start_span(&self, span: &TraceSpan) {
        self.spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(span.clone());
    }

    fn end_span(&self, span: &TraceSpan) {
        let mut spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = spans.iter_mut().find(|s| s.id == span.id) {
            *existing = span.clone();
        }
    }

    fn flush(&self) {
        // In-memory backend has nothing to flush.
    }
}

// ---------------------------------------------------------------------------
// Console / logging backend
// ---------------------------------------------------------------------------

/// Console logging backend that prints span lifecycle events.
pub struct ConsoleTracingBackend;

impl TracingBackend for ConsoleTracingBackend {
    fn start_span(&self, span: &TraceSpan) {
        println!("[TRACE START] {} ({})", span.name, span.kind);
    }

    fn end_span(&self, span: &TraceSpan) {
        let latency = span.latency_ms.unwrap_or(0);
        let status_str = match &span.status {
            super::SpanStatus::Ok => "OK".to_string(),
            super::SpanStatus::Error(e) => format!("ERROR: {}", e),
        };
        println!(
            "[TRACE END]   {} latency={}ms status={}",
            span.name, latency, status_str
        );
    }

    fn flush(&self) {}
}

// ---------------------------------------------------------------------------
// OpenTelemetry backend (feature-gated)
// ---------------------------------------------------------------------------

/// OpenTelemetry tracing backend (requires `opentelemetry` feature).
///
/// Converts framework trace spans into OTel spans via the global tracer,
/// emitting the stabilized GenAI semantic-convention attributes carried on
/// [`TraceSpan`] (same vocabulary as `OtelHandler`). Parent spans are linked
/// through the framework `parent_id`, so the exported trace topology matches
/// the in-process trace tree.
#[cfg(feature = "opentelemetry")]
pub struct OtelTracingBackend {
    tracer: opentelemetry::global::BoxedTracer,
    /// Active OTel spans keyed by framework span ID.
    spans: Mutex<HashMap<String, opentelemetry::global::BoxedSpan>>,
}

#[cfg(feature = "opentelemetry")]
impl OtelTracingBackend {
    /// Create a new backend with the given tracer.
    pub fn new(tracer: opentelemetry::global::BoxedTracer) -> Self {
        Self {
            tracer,
            spans: Mutex::new(HashMap::new()),
        }
    }

    /// Create a backend using the global tracer provider.
    pub fn from_global(name: &str) -> Self {
        Self::new(opentelemetry::global::tracer(name.to_string()))
    }
}

#[cfg(feature = "opentelemetry")]
impl TracingBackend for OtelTracingBackend {
    fn start_span(&self, span: &TraceSpan) {
        use opentelemetry::trace::{Span as _, TraceContextExt, Tracer as _};
        use opentelemetry::{Context, KeyValue};

        // Establish the OTel parent from the framework parent_id, cloning the
        // SpanContext while the lock is held (matching OtelHandler).
        let parent_cx = {
            let spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
            span.parent_id
                .as_ref()
                .and_then(|pid| spans.get(pid))
                .map(|parent| {
                    Context::new().with_remote_span_context(parent.span_context().clone())
                })
        };

        let name = semconv_span_name(span);
        let mut otel_span = match parent_cx {
            Some(cx) => self.tracer.start_with_context(name, &cx),
            None => self.tracer.start(name),
        };

        if let Some(op) = &span.gen_ai_operation_name {
            otel_span.set_attribute(KeyValue::new(semconv::GEN_AI_OPERATION_NAME, op.clone()));
        }
        if let Some(system) = &span.gen_ai_system {
            otel_span.set_attribute(KeyValue::new(semconv::GEN_AI_PROVIDER_NAME, system.clone()));
        }
        if let Some(model) = &span.gen_ai_request_model {
            otel_span.set_attribute(KeyValue::new(semconv::GEN_AI_REQUEST_MODEL, model.clone()));
        }
        if let Some(max) = span.gen_ai_request_max_tokens {
            otel_span.set_attribute(KeyValue::new(
                semconv::GEN_AI_REQUEST_MAX_TOKENS,
                max as i64,
            ));
        }
        if let Some(temp) = span.gen_ai_request_temperature {
            otel_span.set_attribute(KeyValue::new(semconv::GEN_AI_REQUEST_TEMPERATURE, temp));
        }
        if let Some(tool) = &span.gen_ai_tool_name {
            otel_span.set_attribute(KeyValue::new(semconv::GEN_AI_TOOL_NAME, tool.clone()));
        }
        // Eval/trace join keys when the caller stamped them on metadata.
        if let Some(obj) = span.metadata.as_object() {
            for (key, attr) in [
                ("run_id", semconv::RUN_ID_ATTR),
                ("trace_id", semconv::TRACE_ID_ATTR),
            ] {
                if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                    otel_span.set_attribute(KeyValue::new(attr, v.to_string()));
                }
            }
        }

        self.spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(span.id.clone(), otel_span);
    }

    fn end_span(&self, span: &TraceSpan) {
        use opentelemetry::trace::{Span as _, Status};
        use opentelemetry::{Array, KeyValue, StringValue, Value};

        let mut spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut s) = spans.remove(&span.id) {
            if let Some(model) = &span.gen_ai_response_model {
                s.set_attribute(KeyValue::new(semconv::GEN_AI_RESPONSE_MODEL, model.clone()));
            }
            if let Some(tokens) = &span.tokens {
                s.set_attribute(KeyValue::new(
                    semconv::GEN_AI_USAGE_INPUT_TOKENS,
                    tokens.prompt_tokens as i64,
                ));
                s.set_attribute(KeyValue::new(
                    semconv::GEN_AI_USAGE_OUTPUT_TOKENS,
                    tokens.completion_tokens as i64,
                ));
            }
            if let Some(reason) = &span.gen_ai_finish_reason {
                s.set_attribute(KeyValue::new(
                    semconv::GEN_AI_RESPONSE_FINISH_REASONS,
                    Value::Array(Array::String(vec![StringValue::from(reason.clone())])),
                ));
            }
            if let crate::SpanStatus::Error(error) = &span.status {
                s.set_status(Status::error(error.clone()));
                s.set_attribute(KeyValue::new(
                    semconv::ERROR_TYPE,
                    crate::semconv::error_type(error),
                ));
                s.add_event(
                    "exception".to_string(),
                    vec![KeyValue::new(
                        "exception.message",
                        crate::semconv::truncate(error, 1024),
                    )],
                );
            }
            s.end();
        }
    }

    fn flush(&self) {}
}

/// Picks the semconv span name for a trace span: `chat {model}` for LLM
/// spans, `execute_tool {tool}` for tool spans; falls back to the recorded
/// name.
#[cfg(feature = "opentelemetry")]
fn semconv_span_name(span: &TraceSpan) -> String {
    use super::span::SpanKind;
    match &span.kind {
        SpanKind::Llm => match &span.gen_ai_request_model {
            Some(model) => format!("chat {model}"),
            None => "chat".to_string(),
        },
        crate::SpanKind::Tool => match &span.gen_ai_tool_name {
            Some(tool) => format!("execute_tool {tool}"),
            None => span.name.clone(),
        },
        _ => span.name.clone(),
    }
}

#[cfg(all(test, feature = "opentelemetry"))]
mod otel_backend_tests {
    use super::*;
    use crate::semconv;
    use crate::tracing::span::make_span;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::Value;
    use opentelemetry_sdk::testing::trace::InMemorySpanExporterBuilder;
    use opentelemetry_sdk::trace::{SimpleSpanProcessor, TracerProvider};

    fn backend_with_exporter() -> (
        OtelTracingBackend,
        opentelemetry_sdk::testing::trace::InMemorySpanExporter,
    ) {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = TracerProvider::builder()
            .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter.clone())))
            .build();
        let tracer: opentelemetry::global::BoxedTracer =
            opentelemetry::global::BoxedTracer::new(Box::new(provider.tracer("test")));
        (OtelTracingBackend::new(tracer), exporter)
    }

    fn attr_str(span: &opentelemetry_sdk::export::trace::SpanData, key: &str) -> Option<String> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.as_str().into_owned())
    }

    #[test]
    fn llm_span_emits_semconv_name_and_request_attributes() {
        let (backend, exporter) = backend_with_exporter();
        let mut span = make_span(
            "root".into(),
            None,
            "gpt-4o-mini call",
            crate::SpanKind::Llm,
        );
        span.gen_ai_system = Some("openai".into());
        span.gen_ai_request_model = Some("gpt-4o-mini".into());
        span.gen_ai_operation_name = Some("chat".into());
        span.gen_ai_request_temperature = Some(0.1);
        span.gen_ai_request_max_tokens = Some(256);
        span.metadata = serde_json::json!({"run_id": "run-42", "trace_id": "trace-7"});

        backend.start_span(&span);
        backend.end_span(&span);

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let exported = &spans[0];
        assert_eq!(exported.name, "chat gpt-4o-mini");
        assert_eq!(
            attr_str(exported, semconv::GEN_AI_PROVIDER_NAME).as_deref(),
            Some("openai")
        );
        assert_eq!(
            attr_str(exported, semconv::RUN_ID_ATTR).as_deref(),
            Some("run-42")
        );
        assert_eq!(
            attr_str(exported, semconv::TRACE_ID_ATTR).as_deref(),
            Some("trace-7")
        );
        assert!(exported
            .attributes
            .iter()
            .any(|kv| kv.key.as_str() == semconv::GEN_AI_REQUEST_TEMPERATURE
                && matches!(kv.value, Value::F64(_))));
    }

    #[test]
    fn end_span_emits_usage_finish_reasons_and_error_status() {
        let (backend, exporter) = backend_with_exporter();
        let mut span = make_span("t".into(), None, "tool call", crate::SpanKind::Tool);
        span.gen_ai_tool_name = Some("calculator".into());
        span.gen_ai_response_model = Some("gpt-4o-mini-2024-07-18".into());
        span.gen_ai_finish_reason = Some("tool_calls".into());
        span.tokens = Some(crate::SpanTokenUsage {
            prompt_tokens: 10,
            completion_tokens: 4,
            total_tokens: 14,
        });
        span.status = crate::SpanStatus::Error("TimeoutError: 30s".into());

        backend.start_span(&span);
        backend.end_span(&span);

        let spans = exporter.get_finished_spans().unwrap();
        let exported = &spans[0];
        assert_eq!(exported.name, "execute_tool calculator");
        assert_eq!(
            attr_str(exported, semconv::GEN_AI_RESPONSE_MODEL).as_deref(),
            Some("gpt-4o-mini-2024-07-18")
        );
        let reasons = exported
            .attributes
            .iter()
            .find(|kv| kv.key.as_str() == semconv::GEN_AI_RESPONSE_FINISH_REASONS)
            .map(|kv| match &kv.value {
                opentelemetry::Value::Array(opentelemetry::Array::String(values)) => values
                    .iter()
                    .map(|v| v.as_str().to_string())
                    .collect::<Vec<_>>(),
                _ => vec![],
            })
            .unwrap();
        assert_eq!(reasons, vec!["tool_calls".to_string()]);
        assert!(matches!(
            exported.status,
            opentelemetry::trace::Status::Error { .. }
        ));
        assert_eq!(
            attr_str(exported, semconv::ERROR_TYPE).as_deref(),
            Some("TimeoutError")
        );
        assert!(exported.events.iter().any(|e| e.name == "exception"));
    }

    #[test]
    fn child_span_links_to_parent_context() {
        let (backend, exporter) = backend_with_exporter();
        let root = make_span("root".into(), None, "root", crate::SpanKind::Chain);
        let mut child = make_span(
            "child".into(),
            Some("root".into()),
            "child",
            crate::SpanKind::Llm,
        );
        // LLM spans are renamed `chat {model}`.
        child.gen_ai_request_model = Some("mini".into());

        backend.start_span(&root);
        backend.start_span(&child);
        backend.end_span(&child);
        backend.end_span(&root);

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 2);
        let root_ctx = spans
            .iter()
            .find(|s| s.name == "root")
            .unwrap()
            .span_context
            .clone();
        let child_data = spans.iter().find(|s| s.name == "chat mini").unwrap();
        assert_eq!(child_data.parent_span_id, root_ctx.span_id());
        assert_eq!(child_data.span_context.trace_id(), root_ctx.trace_id());
    }
}
