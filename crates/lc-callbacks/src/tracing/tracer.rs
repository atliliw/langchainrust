use std::sync::Arc;
use std::time::Instant;

use uuid::Uuid;

use crate::tracing::span::{make_span, SpanId, SpanKind, SpanStatus, TraceSpan};
use crate::tracing::TracingBackend;

/// Type alias for the span stack stored in thread-local and task-local storage.
type SpanStack = std::cell::RefCell<Vec<SpanId>>;

// Thread-local span stack for synchronous contexts.
#[allow(clippy::missing_const_for_thread_local)]
mod span_stack_tls {
    use super::SpanStack;

    std::thread_local! {
        pub(super) static SPAN_STACK: SpanStack = const { std::cell::RefCell::new(Vec::new()) };
    }
}

// Task-local span stack for async contexts (survives thread migration).
tokio::task_local! {
    pub(super) static ASYNC_SPAN_STACK: SpanStack;
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
    if let Ok(id) = ASYNC_SPAN_STACK.try_with(|stack: &SpanStack| stack.borrow().last().cloned()) {
        return id;
    }
    // Fall back to thread-local for sync contexts
    span_stack_tls::SPAN_STACK.with(|s| s.borrow().last().cloned())
}

/// Push a span ID onto the appropriate stack (task-local if available, else thread-local).
fn push_span_id(id: SpanId) {
    let id_clone = id.clone();
    if ASYNC_SPAN_STACK
        .try_with(|stack: &SpanStack| stack.borrow_mut().push(id))
        .is_ok()
    {
        return;
    }
    span_stack_tls::SPAN_STACK.with(|s| s.borrow_mut().push(id_clone));
}

/// Pop a span ID from the appropriate stack if it matches.
fn pop_span_id_if_matches(span_id: &str) {
    if ASYNC_SPAN_STACK
        .try_with(|stack: &SpanStack| {
            let mut s = stack.borrow_mut();
            if s.last().map(|id: &String| id.as_str()) == Some(span_id) {
                s.pop();
            }
        })
        .is_ok()
    {
        return;
    }
    span_stack_tls::SPAN_STACK.with(|s| {
        let mut stack = s.borrow_mut();
        if stack.last().map(|id: &String| id.as_str()) == Some(span_id) {
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
    let _ = ASYNC_SPAN_STACK.try_with(|stack: &SpanStack| stack.borrow_mut().clear());
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
        let span = make_span(id.clone(), None, name, kind);

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
        let span = make_span(id.clone(), parent_id, name, kind);

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
        let span = make_span(id.clone(), Some(parent_id), name, kind);

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
    pub fn with_tokens(mut self, usage: crate::tracing::SpanTokenUsage) -> Self {
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
            self.span.end_time = Some(chrono::Utc::now().to_rfc3339());
            self.span.latency_ms = Some(self.start_instant.elapsed().as_millis() as u64);
            Tracer::end_span(&self.backend, &self.span);
            self.dropped = true;
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.dropped {
            self.span.end_time = Some(chrono::Utc::now().to_rfc3339());
            self.span.latency_ms = Some(self.start_instant.elapsed().as_millis() as u64);
            Tracer::end_span(&self.backend, &self.span);
            self.dropped = true;
        }
    }
}
