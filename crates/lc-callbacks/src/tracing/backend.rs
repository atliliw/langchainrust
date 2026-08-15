use std::sync::Mutex;

#[cfg(feature = "opentelemetry")]
use std::collections::HashMap;

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
/// Converts framework trace spans into OTel spans via the global tracer.
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
        use opentelemetry::trace::Tracer as OtelTracer;
        let otel_span = OtelTracer::start(&self.tracer, span.name.clone());
        self.spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(span.id.clone(), otel_span);
    }

    fn end_span(&self, span: &TraceSpan) {
        use opentelemetry::trace::Span as OtelSpan;
        if let Some(mut s) = self
            .spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&span.id)
        {
            OtelSpan::end(&mut s);
        }
    }

    fn flush(&self) {}
}
