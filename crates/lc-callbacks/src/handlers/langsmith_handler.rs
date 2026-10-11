// lc-callbacks/src/handlers/langsmith_handler.rs
//! LangSmith callback handler

use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::circuit_breaker::CircuitBreaker;
use crate::{CallbackHandler, LangSmithClient, LangSmithConfig, LangSmithError, RunTree};
use lc_schema::Message;

/// Cap on the degradation buffer so a long outage cannot grow memory without bound.
const LANGSMITH_BUFFER_CAP: usize = 512;

/// A trace operation held locally while the breaker is open. A run that was
/// buffered at `create` (start) is later completed by an `update` (end) op,
/// so both stages are replayed in order when the breaker closes.
enum BufferedOp {
    Create(RunTree),
    Update(RunTree),
}

struct HandlerInner {
    client: Arc<LangSmithClient>,
    async_mode: bool,
    /// Optional per-handler circuit breaker; when `None` (default) the handler
    /// behaves exactly as before (straight-through sends, no degradation).
    breaker: Option<Arc<CircuitBreaker>>,
    buffer: Mutex<VecDeque<BufferedOp>>,
}

impl HandlerInner {
    /// Push a trace op onto the degradation buffer, dropping the oldest entry
    /// once the buffer is at capacity.
    async fn buffer_push(&self, op: BufferedOp) {
        let mut b = self.buffer.lock().await;
        if b.len() >= LANGSMITH_BUFFER_CAP {
            b.pop_front();
        }
        b.push_back(op);
    }

    /// Send one `create_run` (trace started); returns whether it reached the server.
    async fn send_create(&self, run: &RunTree) -> bool {
        match self.client.create_run(run).await {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[LangSmith] create_run failed: {}", e);
                false
            }
        }
    }

    /// Send one `update_run` (trace ended); returns whether it reached the server.
    async fn send_update(&self, run: &RunTree) -> bool {
        match self.client.update_run(run).await {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[LangSmith] update_run failed: {}", e);
                false
            }
        }
    }

    /// Best-effort replay of buffered trace ops, run after a send succeeded (the
    /// breaker is closing or already closed). Stops at the first failure so a
    /// still-down endpoint is not hammered by the whole backlog.
    async fn flush_buffer(&self) {
        let drained: Vec<BufferedOp> = {
            let mut b = self.buffer.lock().await;
            b.drain(..).collect()
        };
        for op in drained {
            let ok = match op {
                BufferedOp::Create(r) => self.send_create(&r).await,
                BufferedOp::Update(r) => self.send_update(&r).await,
            };
            if !ok {
                if let Some(b) = &self.breaker {
                    b.record_failure();
                }
                return;
            }
        }
    }

    /// Dispatch a `create_run` through the breaker (degrading to the local
    /// buffer while open).
    async fn dispatch_create(&self, run: &RunTree) {
        match &self.breaker {
            None => {
                self.send_create(run).await;
            }
            Some(b) => {
                if !b.allow() {
                    self.buffer_push(BufferedOp::Create(run.clone())).await;
                    return;
                }
                if self.send_create(run).await {
                    b.record_success();
                    self.flush_buffer().await;
                } else {
                    b.record_failure();
                }
            }
        }
    }

    /// Dispatch an `update_run` through the breaker (degrading to the local
    /// buffer while open).
    async fn dispatch_update(&self, run: &RunTree) {
        match &self.breaker {
            None => {
                self.send_update(run).await;
            }
            Some(b) => {
                if !b.allow() {
                    self.buffer_push(BufferedOp::Update(run.clone())).await;
                    return;
                }
                if self.send_update(run).await {
                    b.record_success();
                    self.flush_buffer().await;
                } else {
                    b.record_failure();
                }
            }
        }
    }
}

/// LangSmith callback handler
///
/// Automatically sends trace data to LangSmith. When a [`CircuitBreaker`] is
/// attached via [`with_circuit_breaker`](LangSmithHandler::with_circuit_breaker),
/// repeated network failures cause it to degrade to bounded local buffering
/// instead of stalling or tail-latency-ing every subsequent trace.
pub struct LangSmithHandler {
    inner: Arc<HandlerInner>,
}

impl LangSmithHandler {
    /// Creates a new handler with the given LangSmith configuration.
    pub fn new(config: LangSmithConfig) -> Self {
        Self {
            inner: Arc::new(HandlerInner {
                client: Arc::new(LangSmithClient::new(config)),
                async_mode: false, // default to sync mode so the request completes
                breaker: None,
                buffer: Mutex::new(VecDeque::new()),
            }),
        }
    }

    /// Creates a handler from environment variables.
    pub fn from_env() -> Result<Self, LangSmithError> {
        let config = LangSmithConfig::from_env()?;
        Ok(Self::new(config))
    }

    /// Enables or disables asynchronous mode for sending traces.
    pub fn with_async_mode(mut self, async_mode: bool) -> Self {
        let current = &*self.inner;
        self.inner = Arc::new(HandlerInner {
            client: Arc::clone(&current.client),
            async_mode,
            breaker: current.breaker.clone(),
            buffer: Mutex::new(VecDeque::new()),
        });
        self
    }

    /// Attach a degradation [`CircuitBreaker`]: after `trip_threshold`
    /// consecutive send failures the handler stops hitting the network and
    /// buffers traces locally (capped at `LANGSMITH_BUFFER_CAP`), flushing
    /// them best-effort once a later half-open probe succeeds (T2 §5).
    pub fn with_circuit_breaker(mut self, trip_threshold: u32, cooldown: Duration) -> Self {
        let current = &*self.inner;
        self.inner = Arc::new(HandlerInner {
            client: Arc::clone(&current.client),
            async_mode: current.async_mode,
            breaker: Some(Arc::new(CircuitBreaker::new(trip_threshold, cooldown))),
            buffer: Mutex::new(VecDeque::new()),
        });
        self
    }
}

impl std::fmt::Debug for LangSmithHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangSmithHandler")
            .field("async_mode", &self.inner.async_mode)
            .field("circuit_breaker", &self.inner.breaker.is_some())
            .finish()
    }
}

#[async_trait]
impl CallbackHandler for LangSmithHandler {
    async fn on_run_start(&self, run: &RunTree) {
        if !self.inner.client.is_tracing_enabled() {
            return;
        }

        if self.inner.async_mode {
            let inner = Arc::clone(&self.inner);
            let run = run.clone();
            tokio::spawn(async move { inner.dispatch_create(&run).await });
        } else {
            self.inner.dispatch_create(run).await;
        }
    }

    async fn on_run_end(&self, run: &RunTree) {
        if !self.inner.client.is_tracing_enabled() {
            return;
        }

        if self.inner.async_mode {
            let inner = Arc::clone(&self.inner);
            let run = run.clone();
            tokio::spawn(async move { inner.dispatch_update(&run).await });
        } else {
            self.inner.dispatch_update(run).await;
        }
    }

    async fn on_run_error(&self, run: &RunTree, error: &str) {
        let mut run = run.clone();
        run.end_with_error(error);
        self.on_run_end(&run).await;
    }

    async fn on_llm_start(&self, run: &RunTree, _messages: &[Message]) {
        self.on_run_start(run).await;
    }

    async fn on_llm_end(&self, run: &RunTree, _response: &str) {
        self.on_run_end(run).await;
    }

    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }

    async fn on_chain_start(&self, run: &RunTree, _inputs: &serde_json::Value) {
        self.on_run_start(run).await;
    }

    async fn on_chain_end(&self, run: &RunTree, _outputs: &serde_json::Value) {
        self.on_run_end(run).await;
    }

    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }

    async fn on_tool_start(&self, run: &RunTree, _tool_name: &str, _input: &str) {
        self.on_run_start(run).await;
    }

    async fn on_tool_end(&self, run: &RunTree, _output: &str) {
        self.on_run_end(run).await;
    }

    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }

    async fn on_retriever_start(&self, run: &RunTree, _query: &str) {
        self.on_run_start(run).await;
    }

    async fn on_retriever_end(&self, run: &RunTree, _documents: &[serde_json::Value]) {
        self.on_run_end(run).await;
    }

    async fn on_retriever_error(&self, run: &RunTree, error: &str) {
        self.on_run_error(run, error).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(name: &str) -> RunTree {
        RunTree::new(name, crate::RunType::Llm, serde_json::json!({}))
    }

    fn broken_handler() -> (LangSmithHandler, Arc<HandlerInner>) {
        // An endpoint that can never be reached: 127.0.0.1 on a port that is not
        // listening. Wiring is the point, not the status code.
        let config = LangSmithConfig::new("ls_test").with_endpoint("http://127.0.0.1:9");
        let handler =
            LangSmithHandler::new(config).with_circuit_breaker(2, Duration::from_secs(3600));
        let inner = Arc::clone(&handler.inner);
        (handler, inner)
    }

    #[tokio::test]
    async fn breaker_buffers_after_repeated_failures() {
        // M18/§5: two consecutive create failures trip the breaker; the next
        // dispatch is degraded into the buffer and never touches the network.
        let (handler, inner) = broken_handler();
        assert!(inner.breaker.is_some());
        let b = inner.breaker.as_ref().expect("breaker configured");

        handler.on_run_start(&run("a")).await; // 1st failure
        handler.on_run_start(&run("b")).await; // 2nd failure → open
        assert!(b.is_open());

        handler.on_run_start(&run("c")).await; // degraded
        assert_eq!(b.status().refused, 1, "one more refused after open");
        assert_eq!(inner.buffer.lock().await.len(), 1, "one buffered trace op");
    }

    #[tokio::test]
    async fn breaker_stays_closed_on_success() {
        // A success on the first try keeps the breaker closed and the buffer empty.
        let (handler, inner) = broken_handler();
        // Simulate a success directly on the breaker to isolate the wiring:
        inner.breaker.as_ref().unwrap().record_success();
        assert!(!inner.breaker.as_ref().unwrap().is_open());
        assert_eq!(
            inner.buffer.lock().await.len(),
            0,
            "no degradation when healthy"
        );
        // After threshold failures it must open regardless.
        handler.on_run_start(&run("x")).await;
        handler.on_run_start(&run("y")).await;
        assert!(inner.breaker.as_ref().unwrap().is_open());
    }

    #[tokio::test]
    async fn buffer_caps_at_capacity() {
        let (handler, inner) = broken_handler();
        let b = inner.breaker.as_ref().unwrap();
        // Open the breaker outright (threshold 2 → two failures) so every later
        // dispatch degrades to buffering without touching the network.
        b.record_failure();
        b.record_failure();
        assert!(b.is_open());

        for i in 0..(LANGSMITH_BUFFER_CAP + 20) {
            handler.on_run_start(&run(&format!("r{i}"))).await;
        }

        let buf = inner.buffer.lock().await;
        assert_eq!(
            buf.len(),
            LANGSMITH_BUFFER_CAP,
            "buffer never grows past the cap"
        );
    }

    #[tokio::test]
    async fn buffer_preserves_create_then_update_order() {
        // A buffered run holds both a Create (start) and an Update (end) op, so a
        // closed breaker replays start-then-end in order.
        let (handler, inner) = broken_handler();
        handler.on_run_start(&run("a")).await;
        handler.on_run_start(&run("b")).await; // trips after threshold 2
        handler.on_run_start(&run("c")).await; // buffered Create(c)

        handler.on_run_end(&run("c")).await; // buffered Update(c) — distinct op

        let buf = inner.buffer.lock().await;
        assert_eq!(buf.len(), 2);
        assert!(matches!(buf[0], BufferedOp::Create(_)));
        assert!(matches!(buf[1], BufferedOp::Update(_)));
    }
}
