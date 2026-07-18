//! OpenTelemetry callback handler(feature = "opentelemetry")
//!
//! 把框架执行事件(LLM/Chain/Tool/Retriever 的开始/结束/错误)转为 OTel span。
//!
//! 使用前需先配置全局 tracer provider(如 `opentelemetry-otlp`),否则用 noop tracer。
//!
//! # 示例
//! ```ignore
//! use langchainrust::callbacks::CallbackManager;
//! use langchainrust::OtelHandler;
//! let manager = CallbackManager::new().add_handler(std::sync::Arc::new(OtelHandler::from_global("langchainrust")));
//! ```

use async_trait::async_trait;
use opentelemetry::global::{self, BoxedSpan, BoxedTracer};
use opentelemetry::trace::{Span, Tracer};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::callbacks::base::CallbackHandler;
use crate::callbacks::run_tree::RunTree;
use crate::schema::Message;

/// OpenTelemetry callback handler:把执行事件转为 OTel span
pub struct OtelHandler {
    tracer: BoxedTracer,
    /// 活跃 span 栈(按 start/end 配对,支持嵌套)
    spans: Arc<Mutex<Vec<BoxedSpan>>>,
}

impl OtelHandler {
    /// 用指定 tracer 构造
    pub fn new(tracer: BoxedTracer) -> Self {
        Self {
            tracer,
            spans: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// 用全局 tracer 构造(需先 `global::set_tracer_provider`)
    pub fn from_global(name: &str) -> Self {
        Self::new(global::tracer(name.to_string()))
    }

    async fn start_span(&self, name: &str) {
        let span = self.tracer.start(name.to_string());
        self.spans.lock().await.push(span);
    }

    async fn end_span(&self) {
        if let Some(mut span) = self.spans.lock().await.pop() {
            span.end();
        }
    }

    async fn add_event(&self, name: &str) {
        if let Some(span) = self.spans.lock().await.last_mut() {
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
    async fn on_run_start(&self, _run: &RunTree) {
        self.start_span("run").await;
    }
    async fn on_run_end(&self, _run: &RunTree) {
        self.end_span().await;
    }
    async fn on_run_error(&self, _run: &RunTree, error: &str) {
        self.add_event(&format!("error: {}", error)).await;
        self.end_span().await;
    }

    async fn on_llm_start(&self, _run: &RunTree, _messages: &[Message]) {
        self.start_span("llm").await;
    }
    async fn on_llm_end(&self, _run: &RunTree, _response: &str) {
        self.end_span().await;
    }
    async fn on_llm_new_token(&self, _run: &RunTree, _token: &str) {
        self.add_event("token").await;
    }
    async fn on_llm_error(&self, _run: &RunTree, error: &str) {
        self.add_event(&format!("llm error: {}", error)).await;
        self.end_span().await;
    }

    async fn on_chain_start(&self, _run: &RunTree, _inputs: &serde_json::Value) {
        self.start_span("chain").await;
    }
    async fn on_chain_end(&self, _run: &RunTree, _outputs: &serde_json::Value) {
        self.end_span().await;
    }
    async fn on_chain_error(&self, _run: &RunTree, error: &str) {
        self.add_event(&format!("chain error: {}", error)).await;
        self.end_span().await;
    }

    async fn on_tool_start(&self, _run: &RunTree, _tool_name: &str, _input: &str) {
        self.start_span("tool").await;
    }
    async fn on_tool_end(&self, _run: &RunTree, _output: &str) {
        self.end_span().await;
    }
    async fn on_tool_error(&self, _run: &RunTree, error: &str) {
        self.add_event(&format!("tool error: {}", error)).await;
        self.end_span().await;
    }

    async fn on_retriever_start(&self, _run: &RunTree, _query: &str) {
        self.start_span("retriever").await;
    }
    async fn on_retriever_end(&self, _run: &RunTree, _documents: &[serde_json::Value]) {
        self.end_span().await;
    }
    async fn on_retriever_error(&self, _run: &RunTree, error: &str) {
        self.add_event(&format!("retriever error: {}", error)).await;
        self.end_span().await;
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

        h.start_span("a").await;
        h.start_span("b").await;
        assert_eq!(h.active_span_count().await, 2);

        h.end_span().await;
        assert_eq!(h.active_span_count().await, 1);

        h.end_span().await;
        assert_eq!(h.active_span_count().await, 0);
    }

    #[tokio::test]
    async fn test_end_span_when_empty_is_noop() {
        let h = OtelHandler::from_global("test");
        // 栈空时 end 不应 panic
        h.end_span().await;
        assert_eq!(h.active_span_count().await, 0);
    }

    #[tokio::test]
    async fn test_add_event_to_active_span() {
        let h = OtelHandler::from_global("test");
        h.start_span("a").await;
        h.add_event("something").await;
        assert_eq!(h.active_span_count().await, 1);
        h.end_span().await;
    }
}
