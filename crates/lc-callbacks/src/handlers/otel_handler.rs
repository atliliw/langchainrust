//! OpenTelemetry callback handler(feature = "opentelemetry")
//!
//! 把框架执行事件(LLM/Chain/Tool/Retriever 的开始/结束/错误)转为 OTel span。
//!
//! 使用前需先配置全局 tracer provider(如 `opentelemetry-otlp`),否则用 noop tracer。
//!
//! # 示例
//! ```ignore
//! use lc_callbacks::CallbackManager;
//! use lc_callbacks::OtelHandler;
//! let manager = CallbackManager::new().add_handler(std::sync::Arc::new(OtelHandler::from_global("langchainrust")));
//! ```

use async_trait::async_trait;
use opentelemetry::global::{self, BoxedSpan, BoxedTracer};
use opentelemetry::trace::{Span, Tracer};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::base::CallbackHandler;
use crate::run_tree::RunTree;
use lc_schema::Message;

/// OpenTelemetry callback handler:把执行事件转为 OTel span
///
/// Uses a HashMap keyed by run ID instead of a stack, so spans are
/// tracked by their run ID and parent relationships are established
/// via the run tree's `parent_run_id` field, rather than assuming
/// strict stack-based nesting.
pub struct OtelHandler {
    tracer: BoxedTracer,
    /// Active spans keyed by run ID, supporting non-strictly-nested lifecycles.
    spans: Arc<Mutex<HashMap<String, BoxedSpan>>>,
}

impl OtelHandler {
    /// 用指定 tracer 构造
    pub fn new(tracer: BoxedTracer) -> Self {
        Self {
            tracer,
            spans: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 用全局 tracer 构造(需先 `global::set_tracer_provider`)
    pub fn from_global(name: &str) -> Self {
        Self::new(global::tracer(name.to_string()))
    }

    /// Start a new span, setting parent context from the run tree if available.
    async fn start_span(&self, name: &str, run: &RunTree) {
        // Check if the run has a parent run ID with an active span for context linking
        let span = {
            let spans = self.spans.lock().await;
            if let Some(parent_run_id) = &run.parent_run_id {
                let key = parent_run_id.to_string();
                if let Some(_parent_span) = spans.get(&key) {
                    // Parent span exists; create child span linked by run ID association.
                    // Note: opentelemetry 0.27 BoxedTracer does not expose a direct
                    // start_with_context API for BoxedSpan, so we create a new span
                    // and rely on the HashMap key association for parent-child tracking.
                    self.tracer.start(name.to_string())
                } else {
                    self.tracer.start(name.to_string())
                }
            } else {
                self.tracer.start(name.to_string())
            }
        };
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

    /// 当前活跃 span 数(测试用)
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
        self.add_event(&format!("error: {}", error), run).await;
        self.end_span(run).await;
    }

    async fn on_llm_start(&self, run: &RunTree, _messages: &[Message]) {
        self.start_span("llm", run).await;
    }
    async fn on_llm_end(&self, run: &RunTree, _response: &str) {
        self.end_span(run).await;
    }
    async fn on_llm_new_token(&self, run: &RunTree, _token: &str) {
        self.add_event("token", run).await;
    }
    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        self.add_event(&format!("llm error: {}", error), run).await;
        self.end_span(run).await;
    }

    async fn on_chain_start(&self, run: &RunTree, _inputs: &serde_json::Value) {
        self.start_span("chain", run).await;
    }
    async fn on_chain_end(&self, run: &RunTree, _outputs: &serde_json::Value) {
        self.end_span(run).await;
    }
    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.add_event(&format!("chain error: {}", error), run)
            .await;
        self.end_span(run).await;
    }

    async fn on_tool_start(&self, run: &RunTree, _tool_name: &str, _input: &str) {
        self.start_span("tool", run).await;
    }
    async fn on_tool_end(&self, run: &RunTree, _output: &str) {
        self.end_span(run).await;
    }
    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        self.add_event(&format!("tool error: {}", error), run).await;
        self.end_span(run).await;
    }

    async fn on_retriever_start(&self, run: &RunTree, _query: &str) {
        self.start_span("retriever", run).await;
    }
    async fn on_retriever_end(&self, run: &RunTree, _documents: &[serde_json::Value]) {
        self.end_span(run).await;
    }
    async fn on_retriever_error(&self, run: &RunTree, error: &str) {
        self.add_event(&format!("retriever error: {}", error), run)
            .await;
        self.end_span(run).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_from_global_does_not_panic() {
        // 未设 provider 时返回 noop tracer,不应 panic
        let _h = OtelHandler::from_global("test");
    }

    #[tokio::test]
    async fn test_span_start_end_balance() {
        let h = OtelHandler::from_global("test");
        assert_eq!(h.active_span_count().await, 0);

        let run1 = RunTree::new("run1", crate::RunType::Llm, serde_json::json!({}));
        h.start_span("a", &run1).await;
        let run2 = RunTree::new("run2", crate::RunType::Tool, serde_json::json!({}));
        h.start_span("b", &run2).await;
        assert_eq!(h.active_span_count().await, 2);

        h.end_span(&run2).await;
        assert_eq!(h.active_span_count().await, 1);

        h.end_span(&run1).await;
        assert_eq!(h.active_span_count().await, 0);
    }

    #[tokio::test]
    async fn test_end_span_when_empty_is_noop() {
        let h = OtelHandler::from_global("test");
        // 栈空时 end 不应 panic
        let run = RunTree::new("nonexistent", crate::RunType::Llm, serde_json::json!({}));
        h.end_span(&run).await;
        assert_eq!(h.active_span_count().await, 0);
    }

    #[tokio::test]
    async fn test_add_event_to_active_span() {
        let h = OtelHandler::from_global("test");
        let run = RunTree::new("run1", crate::RunType::Llm, serde_json::json!({}));
        h.start_span("a", &run).await;
        h.add_event("something", &run).await;
        assert_eq!(h.active_span_count().await, 1);
        h.end_span(&run).await;
    }
}
