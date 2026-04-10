// src/callbacks/base.rs
//! Base callback handler trait

use async_trait::async_trait;
use std::sync::Arc;

use super::run_tree::RunTree;
use crate::schema::Message;

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
    handlers: Vec<Arc<dyn CallbackHandler>>,
}

impl std::fmt::Debug for CallbackManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackManager")
            .field("handlers_count", &self.handlers.len())
            .finish()
    }
}

impl CallbackManager {
    /// Create a new callback manager
    pub fn new() -> Self {
        Self { handlers: Vec::new() }
    }
    
    /// Add a callback handler
    pub fn add_handler(mut self, handler: Arc<dyn CallbackHandler>) -> Self {
        self.handlers.push(handler);
        self
    }
    
    /// Get all handlers
    pub fn handlers(&self) -> &[Arc<dyn CallbackHandler>] {
        &self.handlers
    }
    
    /// Check if there are any handlers
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
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
            handlers: self.handlers.clone(),
        }
    }
}