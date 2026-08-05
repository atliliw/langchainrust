// lc-callbacks/src/base.rs
//! Base callback handler trait

use async_trait::async_trait;
use std::sync::Arc;

use super::run_tree::RunTree;
use lc_schema::Message;

/// Callback handler trait for tracing and monitoring
///
/// Implement this trait to receive callbacks during execution.
/// Can be used for logging, tracing, monitoring, etc.
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
pub struct CallbackManager {
    inner: Arc<CallbackManagerInner>,
}

struct CallbackManagerInner {
    handlers: Vec<Arc<dyn CallbackHandler>>,
}

impl std::fmt::Debug for CallbackManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackManager")
            .field("handlers_count", &self.inner.handlers.len())
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
        }
    }

    /// Add a callback handler
    pub fn add_handler(self, handler: Arc<dyn CallbackHandler>) -> Self {
        let mut handlers = self.inner.handlers.clone();
        handlers.push(handler);
        Self {
            inner: Arc::new(CallbackManagerInner { handlers }),
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
        }
    }
}

// ============ Helper methods for dispatching callbacks ============

impl CallbackManager {
    /// Dispatch `on_chain_start` to all handlers.
    pub async fn dispatch_chain_start(&self, run: &RunTree, inputs: &serde_json::Value) {
        for handler in &self.inner.handlers {
            handler.on_chain_start(run, inputs).await;
        }
    }

    /// Dispatch `on_chain_end` to all handlers.
    pub async fn dispatch_chain_end(&self, run: &RunTree, outputs: &serde_json::Value) {
        for handler in &self.inner.handlers {
            handler.on_chain_end(run, outputs).await;
        }
    }

    /// Dispatch `on_chain_error` to all handlers.
    pub async fn dispatch_chain_error(&self, run: &RunTree, error: &str) {
        for handler in &self.inner.handlers {
            handler.on_chain_error(run, error).await;
        }
    }

    /// Dispatch `on_llm_start` to all handlers.
    pub async fn dispatch_llm_start(&self, run: &RunTree, messages: &[lc_schema::Message]) {
        for handler in &self.inner.handlers {
            handler.on_llm_start(run, messages).await;
        }
    }

    /// Dispatch `on_llm_end` to all handlers.
    pub async fn dispatch_llm_end(&self, run: &RunTree, response: &str) {
        for handler in &self.inner.handlers {
            handler.on_llm_end(run, response).await;
        }
    }

    /// Dispatch `on_llm_error` to all handlers.
    pub async fn dispatch_llm_error(&self, run: &RunTree, error: &str) {
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

    /// Dispatch `on_tool_start` to all handlers.
    pub async fn dispatch_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        for handler in &self.inner.handlers {
            handler.on_tool_start(run, tool_name, input).await;
        }
    }

    /// Dispatch `on_tool_end` to all handlers.
    pub async fn dispatch_tool_end(&self, run: &RunTree, output: &str) {
        for handler in &self.inner.handlers {
            handler.on_tool_end(run, output).await;
        }
    }

    /// Dispatch `on_tool_error` to all handlers.
    pub async fn dispatch_tool_error(&self, run: &RunTree, error: &str) {
        for handler in &self.inner.handlers {
            handler.on_tool_error(run, error).await;
        }
    }

    /// Dispatch `on_retriever_start` to all handlers.
    pub async fn dispatch_retriever_start(&self, run: &RunTree, query: &str) {
        for handler in &self.inner.handlers {
            handler.on_retriever_start(run, query).await;
        }
    }

    /// Dispatch `on_retriever_end` to all handlers.
    pub async fn dispatch_retriever_end(&self, run: &RunTree, documents: &[serde_json::Value]) {
        for handler in &self.inner.handlers {
            handler.on_retriever_end(run, documents).await;
        }
    }

    /// Dispatch `on_retriever_error` to all handlers.
    pub async fn dispatch_retriever_error(&self, run: &RunTree, error: &str) {
        for handler in &self.inner.handlers {
            handler.on_retriever_error(run, error).await;
        }
    }
}
