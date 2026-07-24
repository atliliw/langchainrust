// src/callbacks/tracing.rs
//! Agent observability / deep tracing system
//!
//! Structured tracing with parent-child span trees, RAII span guards,
//! and pluggable backends (in-memory, console, OpenTelemetry).
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use langchainrust::callbacks::tracing::{
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

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use uuid::Uuid;

#[cfg(feature = "opentelemetry")]
use opentelemetry::trace::Span as OtelSpan;
#[cfg(feature = "opentelemetry")]
use opentelemetry::trace::Tracer as OtelTracer;

/// Unique span identifier.
pub type SpanId = String;

/// Kind of span for categorization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    /// LLM inference call
    Llm,
    /// Chain execution
    Chain,
    /// Tool invocation
    Tool,
    /// Retriever query
    Retriever,
    /// Agent execution
    Agent,
    /// Custom span kind
    Custom(String),
}

impl std::fmt::Display for SpanKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpanKind::Llm => write!(f, "llm"),
            SpanKind::Chain => write!(f, "chain"),
            SpanKind::Tool => write!(f, "tool"),
            SpanKind::Retriever => write!(f, "retriever"),
            SpanKind::Agent => write!(f, "agent"),
            SpanKind::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

/// Token usage recorded in a span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpanTokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Status of a span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    /// Span completed successfully
    Ok,
    /// Span ended with an error
    Error(String),
}

/// A single trace span with parent-child relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Unique span identifier
    pub id: SpanId,
    /// Parent span ID (None for root spans)
    pub parent_id: Option<SpanId>,
    /// Human-readable span name
    pub name: String,
    /// Span category
    pub kind: SpanKind,
    /// ISO 8601 start time
    pub start_time: Option<String>,
    /// ISO 8601 end time
    pub end_time: Option<String>,
    /// Token usage (for LLM spans)
    pub tokens: Option<SpanTokenUsage>,
    /// Estimated cost in USD
    pub cost: Option<f64>,
    /// Measured latency in milliseconds
    pub latency_ms: Option<u64>,
    /// Arbitrary key-value metadata
    pub metadata: serde_json::Value,
    /// Span completion status
    pub status: SpanStatus,
}

/// Backend for persisting/processing trace spans.
pub trait TracingBackend: Send + Sync {
    /// Called when a span starts.
    fn start_span(&self, span: &TraceSpan);
    /// Called when a span ends.
    fn end_span(&self, span: &TraceSpan);
    /// Flush any buffered spans to storage.
    fn flush(&self);
}

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
    pub fn trace_tree(&self, root_id: &str) -> Option<TraceNode> {
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

/// A node in the trace tree (span + children).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceNode {
    pub span: TraceSpan,
    pub children: Vec<TraceNode>,
}

fn build_tree(root: &TraceSpan, all_spans: &[TraceSpan]) -> TraceNode {
    let children: Vec<TraceNode> = all_spans
        .iter()
        .filter(|s| s.parent_id.as_deref() == Some(root.id.as_str()))
        .map(|child| build_tree(child, all_spans))
        .collect();

    TraceNode {
        span: root.clone(),
        children,
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
            SpanStatus::Ok => "OK".to_string(),
            SpanStatus::Error(e) => format!("ERROR: {}", e),
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
    spans: Mutex<Vec<opentelemetry::global::BoxedSpan>>,
}

#[cfg(feature = "opentelemetry")]
impl OtelTracingBackend {
    /// Create a new backend with the given tracer.
    pub fn new(tracer: opentelemetry::global::BoxedTracer) -> Self {
        Self {
            tracer,
            spans: Mutex::new(Vec::new()),
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
        let otel_span = OtelTracer::start(&self.tracer, span.name.clone());
        self.spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(otel_span);
    }

    fn end_span(&self, _span: &TraceSpan) {
        if let Some(mut s) = self.spans.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            OtelSpan::end(&mut s);
        }
    }

    fn flush(&self) {}
}

// ---------------------------------------------------------------------------
// Tracer — the main entry point
// ---------------------------------------------------------------------------

// Thread-local span stack for synchronous contexts.
#[allow(clippy::missing_const_for_thread_local)]
mod span_stack_tls {
    std::thread_local! {
        pub(super) static SPAN_STACK: std::cell::RefCell<Vec<super::SpanId>> = const { std::cell::RefCell::new(Vec::new()) };
    }
}

// Task-local span stack for async contexts (survives thread migration).
tokio::task_local! {
    pub(super) static ASYNC_SPAN_STACK: std::cell::RefCell<Vec<super::SpanId>>;
}

/// Initialize the task-local span stack for the current async task.
///
/// Call this at the top of your async function to enable task-local span tracking.
/// If not initialized, the tracer falls back to thread-local storage.
///
/// # Example
/// ```ignore
/// tokio::spawn(async {
///     init_task_span_stack().await;
///     // ... use tracer
/// });
/// ```
pub async fn init_task_span_stack() {
    let _ = ASYNC_SPAN_STACK
        .scope(std::cell::RefCell::new(Vec::new()), async {})
        .await;
}

/// Try to get the current span ID from task-local, falling back to thread-local.
fn get_current_span_id() -> Option<SpanId> {
    // Try task-local first (works across thread migration in async)
    if let Ok(id) = ASYNC_SPAN_STACK.try_with(|stack| stack.borrow().last().cloned()) {
        return id;
    }
    // Fall back to thread-local for sync contexts
    span_stack_tls::SPAN_STACK.with(|s| s.borrow().last().cloned())
}

/// Push a span ID onto the appropriate stack (task-local if available, else thread-local).
fn push_span_id(id: SpanId) {
    let id_clone = id.clone();
    if ASYNC_SPAN_STACK
        .try_with(|stack| stack.borrow_mut().push(id))
        .is_ok()
    {
        return;
    }
    span_stack_tls::SPAN_STACK.with(|s| s.borrow_mut().push(id_clone));
}

/// Pop a span ID from the appropriate stack if it matches.
fn pop_span_id_if_matches(span_id: &str) {
    if ASYNC_SPAN_STACK
        .try_with(|stack| {
            let mut s = stack.borrow_mut();
            if s.last().map(|id| id.as_str()) == Some(span_id) {
                s.pop();
            }
        })
        .is_ok()
    {
        return;
    }
    span_stack_tls::SPAN_STACK.with(|s| {
        let mut stack = s.borrow_mut();
        if stack.last().map(|id| id.as_str()) == Some(span_id) {
            stack.pop();
        }
    });
}

/// Clear the thread-local span stack.
///
/// This is primarily useful for test isolation when spans from prior tests
/// leak into the thread-local state.
pub fn clear_span_stack() {
    span_stack_tls::SPAN_STACK.with(|s| s.borrow_mut().clear());
    let _ = ASYNC_SPAN_STACK.try_with(|stack| stack.borrow_mut().clear());
}

/// The main tracer that manages span lifecycle.
pub struct Tracer {
    backend: Arc<dyn TracingBackend>,
}

impl Tracer {
    /// Create a new tracer backed by the given backend.
    pub fn new(backend: Arc<dyn TracingBackend>) -> Self {
        Self { backend }
    }

    /// Start a new root span.
    pub fn start(&self, name: &str, kind: SpanKind) -> SpanGuard {
        let id = Uuid::now_v7().to_string();
        let now = Utc::now().to_rfc3339();

        let span = TraceSpan {
            id: id.clone(),
            parent_id: None,
            name: name.to_string(),
            kind,
            start_time: Some(now),
            end_time: None,
            tokens: None,
            cost: None,
            latency_ms: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            status: SpanStatus::Ok,
        };

        self.backend.start_span(&span);

        // Push this span onto the appropriate stack
        push_span_id(id.clone());

        SpanGuard {
            span,
            backend: Arc::clone(&self.backend),
            start_instant: Instant::now(),
            dropped: false,
        }
    }

    /// Start a child span under the current span.
    ///
    /// If no span is active, starts a root span instead.
    pub fn start_child(&self, name: &str, kind: SpanKind) -> SpanGuard {
        let parent_id = get_current_span_id();
        let id = Uuid::now_v7().to_string();
        let now = Utc::now().to_rfc3339();

        let span = TraceSpan {
            id: id.clone(),
            parent_id,
            name: name.to_string(),
            kind,
            start_time: Some(now),
            end_time: None,
            tokens: None,
            cost: None,
            latency_ms: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            status: SpanStatus::Ok,
        };

        self.backend.start_span(&span);

        // Push this span onto the appropriate stack
        push_span_id(id.clone());

        SpanGuard {
            span,
            backend: Arc::clone(&self.backend),
            start_instant: Instant::now(),
            dropped: false,
        }
    }

    /// Start a child span with an explicit parent ID.
    pub fn start_child_with_parent(
        &self,
        name: &str,
        kind: SpanKind,
        parent_id: SpanId,
    ) -> SpanGuard {
        let id = Uuid::now_v7().to_string();
        let now = Utc::now().to_rfc3339();

        let span = TraceSpan {
            id: id.clone(),
            parent_id: Some(parent_id),
            name: name.to_string(),
            kind,
            start_time: Some(now),
            end_time: None,
            tokens: None,
            cost: None,
            latency_ms: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            status: SpanStatus::Ok,
        };

        self.backend.start_span(&span);

        push_span_id(id.clone());

        SpanGuard {
            span,
            backend: Arc::clone(&self.backend),
            start_instant: Instant::now(),
            dropped: false,
        }
    }

    /// Get the current span ID from the active stack.
    pub fn current_span_id(&self) -> Option<SpanId> {
        get_current_span_id()
    }

    /// Flush all pending spans to the backend.
    pub fn flush(&self) {
        self.backend.flush();
    }

    /// End a span (called from SpanGuard::drop).
    fn end_span(backend: &Arc<dyn TracingBackend>, span: &TraceSpan) {
        backend.end_span(span);
        // Pop from the appropriate stack (only if it matches)
        pop_span_id_if_matches(&span.id);
    }
}

impl Clone for Tracer {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
        }
    }
}

// ---------------------------------------------------------------------------
// SpanGuard — RAII guard that ends the span when dropped
// ---------------------------------------------------------------------------

/// RAII guard that ends the span when dropped.
///
/// Use builder-style methods to attach data before the span completes.
pub struct SpanGuard {
    span: TraceSpan,
    backend: Arc<dyn TracingBackend>,
    start_instant: Instant,
    /// If true, the span has been manually ended and Drop should not end it again.
    dropped: bool,
}

impl SpanGuard {
    /// Get the span ID.
    pub fn id(&self) -> &str {
        &self.span.id
    }

    /// Get the parent span ID.
    pub fn parent_id(&self) -> Option<&str> {
        self.span.parent_id.as_deref()
    }

    /// Set token usage on this span.
    pub fn with_tokens(mut self, usage: SpanTokenUsage) -> Self {
        self.span.tokens = Some(usage);
        self
    }

    /// Set cost on this span.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.span.cost = Some(cost);
        self
    }

    /// Add a metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        if let Some(obj) = self.span.metadata.as_object_mut() {
            obj.insert(key.to_string(), value);
        }
        self
    }

    /// Mark this span as having errored.
    pub fn set_error(&mut self, msg: &str) {
        self.span.status = SpanStatus::Error(msg.to_string());
    }

    /// Manually end the span now instead of waiting for Drop.
    pub fn end(mut self) {
        if !self.dropped {
            self.span.end_time = Some(Utc::now().to_rfc3339());
            self.span.latency_ms = Some(self.start_instant.elapsed().as_millis() as u64);
            Tracer::end_span(&self.backend, &self.span);
            self.dropped = true;
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.dropped {
            self.span.end_time = Some(Utc::now().to_rfc3339());
            self.span.latency_ms = Some(self.start_instant.elapsed().as_millis() as u64);
            Tracer::end_span(&self.backend, &self.span);
            self.dropped = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_kind_display() {
        assert_eq!(format!("{}", SpanKind::Llm), "llm");
        assert_eq!(format!("{}", SpanKind::Chain), "chain");
        assert_eq!(format!("{}", SpanKind::Tool), "tool");
        assert_eq!(format!("{}", SpanKind::Retriever), "retriever");
        assert_eq!(format!("{}", SpanKind::Agent), "agent");
        assert_eq!(
            format!("{}", SpanKind::Custom("embedding".into())),
            "custom:embedding"
        );
    }

    #[test]
    fn test_span_creation_lifecycle() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let guard = tracer.start("test_span", SpanKind::Chain);
            assert!(!guard.id().is_empty());
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name, "test_span");
        assert_eq!(span.kind, SpanKind::Chain);
        assert!(span.start_time.is_some());
        assert!(span.end_time.is_some());
        assert!(span.latency_ms.is_some());
        assert_eq!(span.status, SpanStatus::Ok);
    }

    #[test]
    fn test_parent_child_relationship() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let root = tracer.start("root", SpanKind::Chain);
            let root_id = root.id().to_string();

            {
                let child = tracer.start_child("child", SpanKind::Tool);
                assert_eq!(child.parent_id(), Some(root_id.as_str()));
            }
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 2);

        // Find root and child
        let root = spans.iter().find(|s| s.name == "root").unwrap();
        let child = spans.iter().find(|s| s.name == "child").unwrap();

        assert!(root.parent_id.is_none());
        assert_eq!(child.parent_id.as_deref(), Some(root.id.as_str()));
    }

    #[test]
    fn test_start_child_without_parent_creates_root() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Clear any leftover state
        clear_span_stack();

        {
            let child = tracer.start_child("orphan", SpanKind::Tool);
            // No parent on the stack, so parent_id should be None
            assert!(child.parent_id().is_none());
        }
    }

    #[test]
    fn test_span_guard_with_tokens() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let usage = SpanTokenUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            };
            let guard = tracer.start("llm_call", SpanKind::Llm).with_tokens(usage);
            guard.end();
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        let tokens = spans[0].tokens.as_ref().unwrap();
        assert_eq!(tokens.prompt_tokens, 10);
        assert_eq!(tokens.completion_tokens, 20);
        assert_eq!(tokens.total_tokens, 30);
    }

    #[test]
    fn test_span_guard_with_cost() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            tracer
                .start("llm_call", SpanKind::Llm)
                .with_cost(0.003)
                .end();
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        assert!((spans[0].cost.unwrap() - 0.003).abs() < f64::EPSILON);
    }

    #[test]
    fn test_span_guard_with_metadata() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            tracer
                .start("call", SpanKind::Chain)
                .with_metadata("model", serde_json::json!("gpt-4"))
                .with_metadata("temperature", serde_json::json!(0.7))
                .end();
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        let meta = &spans[0].metadata;
        assert_eq!(meta["model"], "gpt-4");
        assert_eq!(meta["temperature"], 0.7);
    }

    #[test]
    fn test_span_guard_set_error() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let mut guard = tracer.start("call", SpanKind::Tool);
            guard.set_error("tool failed");
            guard.end();
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            spans[0].status,
            SpanStatus::Error("tool failed".to_string())
        );
    }

    #[test]
    fn test_span_guard_raii_drop() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Do NOT call .end() — rely on Drop
        {
            let _guard = tracer.start("auto_dropped", SpanKind::Chain);
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        assert!(spans[0].end_time.is_some(), "Drop should end the span");
        assert!(
            spans[0].latency_ms.is_some(),
            "Drop should calculate latency"
        );
    }

    #[test]
    fn test_in_memory_backend_trace_tree() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Build: root -> child1, root -> child2 -> grandchild
        let root_id;

        {
            let root = tracer.start("root", SpanKind::Agent);
            root_id = root.id().to_string();

            let child1 = tracer.start_child("child1", SpanKind::Tool);
            drop(child1);

            let child2 = tracer.start_child("child2", SpanKind::Chain);

            let grandchild = tracer.start_child("grandchild", SpanKind::Llm);
            drop(grandchild);

            drop(child2);
            drop(root);
        }

        let tree = backend.trace_tree(&root_id).unwrap();
        assert_eq!(tree.span.name, "root");
        assert_eq!(tree.children.len(), 2);

        // Find child1 and child2
        let child1_node = tree
            .children
            .iter()
            .find(|c| c.span.name == "child1")
            .unwrap();
        let child2_node = tree
            .children
            .iter()
            .find(|c| c.span.name == "child2")
            .unwrap();

        assert!(child1_node.children.is_empty());
        assert_eq!(child2_node.children.len(), 1);
        assert_eq!(child2_node.children[0].span.name, "grandchild");
    }

    #[test]
    fn test_in_memory_backend_trace_tree_not_found() {
        let backend = InMemoryTracingBackend::new();
        assert!(backend.trace_tree("nonexistent").is_none());
    }

    #[test]
    fn test_in_memory_backend_clear() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let _guard = tracer.start("temp", SpanKind::Chain);
        }

        assert_eq!(backend.spans().len(), 1);
        backend.clear();
        assert!(backend.spans().is_empty());
    }

    #[test]
    fn test_tracer_current_span_id() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Clear any leftover state
        clear_span_stack();

        assert!(tracer.current_span_id().is_none());

        let root = tracer.start("root", SpanKind::Chain);
        let root_id = root.id().to_string();
        assert_eq!(tracer.current_span_id(), Some(root_id.clone()));

        let child = tracer.start_child("child", SpanKind::Tool);
        let child_id = child.id().to_string();
        assert_eq!(tracer.current_span_id(), Some(child_id));

        drop(child);
        assert_eq!(tracer.current_span_id(), Some(root_id));

        drop(root);
        assert!(tracer.current_span_id().is_none());
    }

    #[test]
    fn test_start_child_with_explicit_parent() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Clear any leftover state
        clear_span_stack();

        let root = tracer.start("root", SpanKind::Agent);
        let root_id = root.id().to_string();

        // Use explicit parent ID without relying on thread-local state
        let child = tracer.start_child_with_parent("child", SpanKind::Tool, root_id.clone());
        assert_eq!(child.parent_id(), Some(root_id.as_str()));
    }

    #[test]
    fn test_span_serialization_roundtrip() {
        let span = TraceSpan {
            id: "test-id".to_string(),
            parent_id: Some("parent-id".to_string()),
            name: "test_span".to_string(),
            kind: SpanKind::Llm,
            start_time: Some("2025-01-01T00:00:00Z".to_string()),
            end_time: Some("2025-01-01T00:00:01Z".to_string()),
            tokens: Some(SpanTokenUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
            cost: Some(0.005),
            latency_ms: Some(1000),
            metadata: serde_json::json!({"model": "gpt-4"}),
            status: SpanStatus::Ok,
        };

        let json = serde_json::to_string(&span).unwrap();
        let deserialized: TraceSpan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, span.id);
        assert_eq!(deserialized.name, span.name);
        assert_eq!(deserialized.kind, span.kind);
        assert_eq!(deserialized.tokens, span.tokens);
    }

    #[test]
    fn test_console_backend_does_not_panic() {
        let backend = ConsoleTracingBackend;
        let span = TraceSpan {
            id: "test".to_string(),
            parent_id: None,
            name: "test_span".to_string(),
            kind: SpanKind::Chain,
            start_time: Some("2025-01-01T00:00:00Z".to_string()),
            end_time: None,
            tokens: None,
            cost: None,
            latency_ms: None,
            metadata: serde_json::Value::Null,
            status: SpanStatus::Ok,
        };

        // Should not panic
        backend.start_span(&span);
        backend.flush();
    }

    #[test]
    fn test_tracer_flush() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());
        // flush should not panic
        tracer.flush();
    }

    #[test]
    fn test_span_guard_end_called_twice_is_safe() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let guard = tracer.start("double_end", SpanKind::Chain);
            guard.end();
            // Drop will also fire but should be a no-op because `dropped` is set
        }

        // Should have exactly 1 span, not duplicated
        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
    }
}
