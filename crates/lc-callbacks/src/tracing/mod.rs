//! Agent observability / deep tracing system
//!
//! Structured tracing with parent-child span trees, RAII span guards,
//! and pluggable backends (in-memory, console, OpenTelemetry).
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use lc_callbacks::tracing::{
//!     Tracer, InMemoryTracingBackend, SpanKind,
//! };
//! use std::sync::Arc;
//!
//! let backend = Arc::new(InMemoryTracingBackend::new());
//! let tracer = Arc::new(Tracer::new(backend.clone()));
//!
//! // Start a root span
//! let root = tracer.start("my_chain", SpanKind::Chain);
//!
//! // Start a child span (inherits parent from tracer context)
//! let llm = tracer.start_child("llm_call", SpanKind::Llm);
//! drop(llm); // ends the child span
//!
//! drop(root); // ends the root span
//!
//! // Inspect recorded spans
//! let spans = backend.spans();
//! ```

/// Backend implementations for persisting/processing trace spans.
pub mod backend;
/// Span types and the span tree.
pub mod span;
/// The tracer, span guards, and span-stack helpers.
pub mod tracer;

#[cfg(test)]
mod tests;

// Re-export all public types to preserve the public API.
#[cfg(feature = "opentelemetry")]
pub use backend::OtelTracingBackend;
pub use backend::{ConsoleTracingBackend, InMemoryTracingBackend};
pub use span::{SpanId, SpanKind, SpanStatus, SpanTokenUsage, TraceNode, TraceSpan};
pub use tracer::{clear_span_stack, init_task_span_stack, SpanGuard, Tracer};

/// Backend for persisting/processing trace spans.
pub trait TracingBackend: Send + Sync {
    /// Called when a span starts.
    fn start_span(&self, span: &TraceSpan);
    /// Called when a span ends.
    fn end_span(&self, span: &TraceSpan);
    /// Flush any buffered spans to storage.
    fn flush(&self);
}
