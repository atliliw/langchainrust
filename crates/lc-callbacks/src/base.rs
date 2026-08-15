// lc-callbacks/src/base.rs
//! Base callback handler trait

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::run_tree::RunTree;
use crate::tracing::{SpanGuard, SpanKind, Tracer};
use lc_schema::Message;

/// Callback handler trait for tracing and monitoring
///
/// Implement this trait to receive callbacks during execution.
/// Can be used for logging, tracing, monitoring, etc.
///
/// # Two information layers
///
/// The trait is split into two layers:
///
/// 1. **Generic lifecycle layer** — [`on_run_start`](CallbackHandler::on_run_start),
///    [`on_run_end`](CallbackHandler::on_run_end) and
///    [`on_run_error`](CallbackHandler::on_run_error) carry the run's lifecycle
///    with only the [`RunTree`] (name, type, inputs/outputs, timing). They are
///    type-agnostic: a chain, an LLM call, a tool and a retriever all funnel
///    through the same three hooks.
/// 2. **Typed payload layer** — `on_llm_*`, `on_chain_*`, `on_tool_*` and
///    `on_retriever_*` additionally receive the type-specific payload
///    (messages, tokens, tool name, query, …). Their default implementations
///    delegate to the generic layer, so a handler that only cares about
///    lifecycle may override just the three generic methods, while a handler
///    that needs typed data overrides the typed ones.
///
/// Handlers should prefer overriding the typed methods when they need more than
/// lifecycle signals; the built-in [`FileCallbackHandler`](crate::FileCallbackHandler)
/// and [`GenericHandler`](crate::GenericHandler) do exactly that.
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    // ============ Lifecycle callbacks ============

    /// Called when any run starts
    async fn on_run_start(&self, run: &RunTree);

    /// Called when a run ends successfully
    async fn on_run_end(&self, run: &RunTree);

    /// Called when a run fails
    async fn on_run_error(&self, run: &RunTree, error: &str);

    // ============ LLM callbacks ============

    /// Called when an LLM starts
    async fn on_llm_start(&self, run: &RunTree, _messages: &[Message]) {
        self.on_run_start(run).await;
    }

    /// Called when an LLM ends
    async fn on_llm_end(&self, run: &RunTree, _response: &str) {
        self.on_run_end(run).await;
    }

    /// Called for each new token during streaming
    async fn on_llm_new_token(&self, _run: &RunTree, _token: &str) {
        // Default: do nothing
    }

    /// Called for each thinking token during streaming (extended thinking).
    ///
    /// Anthropic's extended thinking feature emits thinking content blocks
    /// before the final text response. This callback fires for each chunk
    /// of thinking content, allowing consumers to observe the model's
    /// reasoning process in real time.
    async fn on_llm_thinking(&self, _run: &RunTree, _thinking: &str) {
        // Default: do nothing
    }

    /// Called when an LLM errors
    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }

    // ============ Chain callbacks ============

    /// Called when a chain starts
    async fn on_chain_start(&self, run: &RunTree, _inputs: &serde_json::Value) {
        self.on_run_start(run).await;
    }

    /// Called when a chain ends
    async fn on_chain_end(&self, run: &RunTree, _outputs: &serde_json::Value) {
        self.on_run_end(run).await;
    }

    /// Called when a chain errors
    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }

    // ============ Tool callbacks ============

    /// Called when a tool starts
    async fn on_tool_start(&self, run: &RunTree, _tool_name: &str, _input: &str) {
        self.on_run_start(run).await;
    }

    /// Called when a tool ends
    async fn on_tool_end(&self, run: &RunTree, _output: &str) {
        self.on_run_end(run).await;
    }

    /// Called when a tool errors
    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }

    // ============ Retriever callbacks ============

    /// Called when a retriever starts
    async fn on_retriever_start(&self, run: &RunTree, _query: &str) {
        self.on_run_start(run).await;
    }

    /// Called when a retriever ends
    async fn on_retriever_end(&self, run: &RunTree, _documents: &[serde_json::Value]) {
        self.on_run_end(run).await;
    }

    /// Called when a retriever errors
    async fn on_retriever_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }
}

/// Callback manager that handles multiple handlers
///
/// Dispatch events to all registered handlers. Prefer the `dispatch_*` methods
/// over poking `handlers()` directly: the dispatch methods guarantee every
/// handler is invoked in order and (when a [`Tracer`] is attached via
/// [`with_tracer`](CallbackManager::with_tracer)) also publish each run as a
/// tracing span.
pub struct CallbackManager {
    inner: Arc<CallbackManagerInner>,
    tracer: Option<Arc<Tracer>>,
    active_trace_spans: Arc<Mutex<HashMap<String, SpanGuard>>>,
}

struct CallbackManagerInner {
    handlers: Vec<Arc<dyn CallbackHandler>>,
}

impl std::fmt::Debug for CallbackManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackManager")
            .field("handlers_count", &self.inner.handlers.len())
            .field("tracer_attached", &self.tracer.is_some())
            .finish()
    }
}

impl CallbackManager {
    /// Create a new callback manager
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CallbackManagerInner {
                handlers: Vec::new(),
            }),
            tracer: None,
            active_trace_spans: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add a callback handler
    pub fn add_handler(self, handler: Arc<dyn CallbackHandler>) -> Self {
        let mut handlers = self.inner.handlers.clone();
        handlers.push(handler);
        Self {
            inner: Arc::new(CallbackManagerInner { handlers }),
            tracer: self.tracer,
            active_trace_spans: self.active_trace_spans,
        }
    }

    /// Attach a [`Tracer`] so every dispatched run is also published as a
    /// tracing span (Q6: minimal merge between callbacks and the tracing tree).
    ///
    /// With a tracer attached, `dispatch_*_start` starts a span named after the
    /// run and `dispatch_*_end`/`dispatch_*_error` ends it. Nested runs become
    /// parent/child spans via the tracer's task-local span stack.
    pub fn with_tracer(self, tracer: Arc<Tracer>) -> Self {
        Self {
            inner: self.inner,
            tracer: Some(tracer),
            active_trace_spans: self.active_trace_spans,
        }
    }

    /// Get all handlers
    pub fn handlers(&self) -> &[Arc<dyn CallbackHandler>] {
        &self.inner.handlers
    }

    /// Check if there are any handlers
    pub fn is_empty(&self) -> bool {
        self.inner.handlers.is_empty()
    }

    /// Merge `other`'s handlers into a new manager, keeping this manager's
    /// handlers first and appending `other`'s (Q8: config merges append
    /// callback handlers instead of replacing them wholesale).
    ///
    /// `self`'s tracer wins; if `self` has none, `other`'s tracer is used.
    /// The active trace-span table starts empty in the merged manager.
    pub fn merge_with(&self, other: &CallbackManager) -> CallbackManager {
        let mut handlers = self.inner.handlers.clone();
        handlers.extend(other.inner.handlers.iter().cloned());
        let tracer = self.tracer.clone().or_else(|| other.tracer.clone());
        CallbackManager {
            inner: Arc::new(CallbackManagerInner { handlers }),
            tracer,
            active_trace_spans: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    // ---- Tracing-span bridge (Q6) ----

    /// Publish `run` start as a tracing span when a tracer is attached.
    async fn begin_trace_span(&self, run: &RunTree) {
        if let Some(tracer) = &self.tracer {
            let kind = SpanKind::from(run.run_type);
            // start_child falls back to a root span when no span is active
            let guard = tracer.start_child(&run.name, kind);
            self.active_trace_spans
                .lock()
                .await
                .insert(run.id.to_string(), guard);
        }
    }

    /// End the tracing span for `run` (marking it errored if `error` is set).
    async fn end_trace_span(&self, run: &RunTree, error: Option<&str>) {
        if let Some(mut guard) = self
            .active_trace_spans
            .lock()
            .await
            .remove(&run.id.to_string())
        {
            if let Some(msg) = error {
                guard.set_error(msg);
            }
            guard.end();
        }
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CallbackManager {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            tracer: self.tracer.clone(),
            active_trace_spans: Arc::clone(&self.active_trace_spans),
        }
    }
}

// ============ Helper methods for dispatching callbacks ============

impl CallbackManager {
    /// Dispatch `on_run_start` to all handlers.
    pub async fn dispatch_run_start(&self, run: &RunTree) {
        self.begin_trace_span(run).await;
        for handler in &self.inner.handlers {
            handler.on_run_start(run).await;
        }
    }

    /// Dispatch `on_run_end` to all handlers.
    pub async fn dispatch_run_end(&self, run: &RunTree) {
        self.end_trace_span(run, None).await;
        for handler in &self.inner.handlers {
            handler.on_run_end(run).await;
        }
    }

    /// Dispatch `on_run_error` to all handlers.
    pub async fn dispatch_run_error(&self, run: &RunTree, error: &str) {
        self.end_trace_span(run, Some(error)).await;
        for handler in &self.inner.handlers {
            handler.on_run_error(run, error).await;
        }
    }

    /// Dispatch `on_chain_start` to all handlers.
    pub async fn dispatch_chain_start(&self, run: &RunTree, inputs: &serde_json::Value) {
        self.begin_trace_span(run).await;
        for handler in &self.inner.handlers {
            handler.on_chain_start(run, inputs).await;
        }
    }

    /// Dispatch `on_chain_end` to all handlers.
    pub async fn dispatch_chain_end(&self, run: &RunTree, outputs: &serde_json::Value) {
        self.end_trace_span(run, None).await;
        for handler in &self.inner.handlers {
            handler.on_chain_end(run, outputs).await;
        }
    }

    /// Dispatch `on_chain_error` to all handlers.
    pub async fn dispatch_chain_error(&self, run: &RunTree, error: &str) {
        self.end_trace_span(run, Some(error)).await;
        for handler in &self.inner.handlers {
            handler.on_chain_error(run, error).await;
        }
    }

    /// Dispatch `on_llm_start` to all handlers.
    pub async fn dispatch_llm_start(&self, run: &RunTree, messages: &[lc_schema::Message]) {
        self.begin_trace_span(run).await;
        for handler in &self.inner.handlers {
            handler.on_llm_start(run, messages).await;
        }
    }

    /// Dispatch `on_llm_end` to all handlers.
    pub async fn dispatch_llm_end(&self, run: &RunTree, response: &str) {
        self.end_trace_span(run, None).await;
        for handler in &self.inner.handlers {
            handler.on_llm_end(run, response).await;
        }
    }

    /// Dispatch `on_llm_error` to all handlers.
    pub async fn dispatch_llm_error(&self, run: &RunTree, error: &str) {
        self.end_trace_span(run, Some(error)).await;
        for handler in &self.inner.handlers {
            handler.on_llm_error(run, error).await;
        }
    }

    /// Dispatch `on_llm_new_token` to all handlers.
    pub async fn dispatch_llm_new_token(&self, run: &RunTree, token: &str) {
        for handler in &self.inner.handlers {
            handler.on_llm_new_token(run, token).await;
        }
    }

    /// Dispatch `on_llm_thinking` to all handlers.
    pub async fn dispatch_llm_thinking(&self, run: &RunTree, thinking: &str) {
        for handler in &self.inner.handlers {
            handler.on_llm_thinking(run, thinking).await;
        }
    }

    /// Dispatch `on_tool_start` to all handlers.
    pub async fn dispatch_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        self.begin_trace_span(run).await;
        for handler in &self.inner.handlers {
            handler.on_tool_start(run, tool_name, input).await;
        }
    }

    /// Dispatch `on_tool_end` to all handlers.
    pub async fn dispatch_tool_end(&self, run: &RunTree, output: &str) {
        self.end_trace_span(run, None).await;
        for handler in &self.inner.handlers {
            handler.on_tool_end(run, output).await;
        }
    }

    /// Dispatch `on_tool_error` to all handlers.
    pub async fn dispatch_tool_error(&self, run: &RunTree, error: &str) {
        self.end_trace_span(run, Some(error)).await;
        for handler in &self.inner.handlers {
            handler.on_tool_error(run, error).await;
        }
    }

    /// Dispatch `on_retriever_start` to all handlers.
    pub async fn dispatch_retriever_start(&self, run: &RunTree, query: &str) {
        self.begin_trace_span(run).await;
        for handler in &self.inner.handlers {
            handler.on_retriever_start(run, query).await;
        }
    }

    /// Dispatch `on_retriever_end` to all handlers.
    pub async fn dispatch_retriever_end(&self, run: &RunTree, documents: &[serde_json::Value]) {
        self.end_trace_span(run, None).await;
        for handler in &self.inner.handlers {
            handler.on_retriever_end(run, documents).await;
        }
    }

    /// Dispatch `on_retriever_error` to all handlers.
    pub async fn dispatch_retriever_error(&self, run: &RunTree, error: &str) {
        self.end_trace_span(run, Some(error)).await;
        for handler in &self.inner.handlers {
            handler.on_retriever_error(run, error).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::InMemoryTracingBackend;

    struct RecordingHandler {
        events: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl CallbackHandler for RecordingHandler {
        async fn on_run_start(&self, run: &RunTree) {
            self.events.lock().await.push(format!("start:{}", run.name));
        }
        async fn on_run_end(&self, run: &RunTree) {
            self.events.lock().await.push(format!("end:{}", run.name));
        }
        async fn on_run_error(&self, _run: &RunTree, error: &str) {
            self.events.lock().await.push(format!("error:{}", error));
        }
    }

    #[tokio::test]
    async fn test_dispatch_run_start_end() {
        let handler = Arc::new(RecordingHandler {
            events: Mutex::new(Vec::new()),
        });
        let cm = CallbackManager::new().add_handler(handler.clone());

        let run = RunTree::new("r1", crate::RunType::Chain, serde_json::json!({}));
        cm.dispatch_run_start(&run).await;
        cm.dispatch_run_end(&run).await;

        let events = handler.events.lock().await.clone();
        assert_eq!(events, vec!["start:r1".to_string(), "end:r1".to_string()]);
    }

    #[tokio::test]
    async fn test_dispatch_run_error_forwards_message() {
        let handler = Arc::new(RecordingHandler {
            events: Mutex::new(Vec::new()),
        });
        let cm = CallbackManager::new().add_handler(handler.clone());

        let run = RunTree::new("r1", crate::RunType::Tool, serde_json::json!({}));
        cm.dispatch_run_start(&run).await;
        cm.dispatch_run_error(&run, "boom").await;

        let events = handler.events.lock().await.clone();
        assert!(events.contains(&"error:boom".to_string()));
    }

    #[tokio::test]
    async fn test_dispatch_llm_thinking_forwards() {
        struct ThinkingHandler {
            received: Mutex<Vec<String>>,
        }
        #[async_trait]
        impl CallbackHandler for ThinkingHandler {
            async fn on_run_start(&self, _run: &RunTree) {}
            async fn on_run_end(&self, _run: &RunTree) {}
            async fn on_run_error(&self, _run: &RunTree, _e: &str) {}
            async fn on_llm_thinking(&self, _run: &RunTree, thinking: &str) {
                self.received.lock().await.push(thinking.to_string());
            }
        }
        let handler = Arc::new(ThinkingHandler {
            received: Mutex::new(Vec::new()),
        });
        let cm = CallbackManager::new().add_handler(handler.clone());

        let run = RunTree::new("llm1", crate::RunType::Llm, serde_json::json!({}));
        cm.dispatch_llm_start(&run, &[]).await;
        cm.dispatch_llm_thinking(&run, "let me think").await;
        cm.dispatch_llm_end(&run, "answer").await;

        let received = handler.received.lock().await.clone();
        assert_eq!(received, vec!["let me think".to_string()]);
    }

    #[tokio::test]
    async fn test_with_tracer_publishes_run_spans() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Arc::new(Tracer::new(backend.clone()));

        let cm = CallbackManager::new().with_tracer(tracer);
        let run = RunTree::new("span_run", crate::RunType::Chain, serde_json::json!({}));

        cm.dispatch_run_start(&run).await;
        cm.dispatch_run_end(&run).await;

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "span_run");
        assert_eq!(spans[0].kind, SpanKind::Chain);
    }

    #[tokio::test]
    async fn test_with_tracer_publishes_error_status() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Arc::new(Tracer::new(backend.clone()));

        let cm = CallbackManager::new().with_tracer(tracer);
        let run = RunTree::new("fail_run", crate::RunType::Llm, serde_json::json!({}));

        cm.dispatch_llm_start(&run, &[]).await;
        cm.dispatch_llm_error(&run, "model timeout").await;

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        assert!(matches!(
            &spans[0].status,
            crate::tracing::SpanStatus::Error(e) if e == "model timeout"
        ));
    }

    #[tokio::test]
    async fn test_nested_runs_become_parent_child_spans() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Arc::new(Tracer::new(backend.clone()));

        let cm = CallbackManager::new().with_tracer(tracer);

        let chain = RunTree::new("outer_chain", crate::RunType::Chain, serde_json::json!({}));
        cm.dispatch_chain_start(&chain, &serde_json::json!({}))
            .await;

        let llm = RunTree::new("inner_llm", crate::RunType::Llm, serde_json::json!({}));
        cm.dispatch_llm_start(&llm, &[]).await;
        cm.dispatch_llm_end(&llm, "answer").await;

        cm.dispatch_chain_end(&chain, &serde_json::json!({})).await;

        let spans = backend.spans();
        assert_eq!(spans.len(), 2);
        let outer = spans.iter().find(|s| s.name == "outer_chain").unwrap();
        let inner = spans.iter().find(|s| s.name == "inner_llm").unwrap();
        assert!(outer.parent_id.is_none());
        assert_eq!(inner.parent_id.as_deref(), Some(outer.id.as_str()));
    }
}
