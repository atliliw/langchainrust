// src/callbacks/handlers/langsmith_handler.rs
//! LangSmith callback handler

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::callbacks::{CallbackHandler, LangSmithClient, LangSmithConfig, RunTree};
use crate::schema::Message;

/// LangSmith callback handler
/// 
/// Automatically sends trace data to LangSmith.
pub struct LangSmithHandler {
    client: LangSmithClient,
    active_runs: Arc<RwLock<Vec<RunTree>>>,
    async_mode: bool,
}

impl LangSmithHandler {
    pub fn new(config: LangSmithConfig) -> Self {
        Self {
            client: LangSmithClient::new(config),
            active_runs: Arc::new(RwLock::new(Vec::new())),
            async_mode: false,  // 默认同步模式，确保请求完成
        }
    }
    
    pub fn from_env() -> Result<Self, String> {
        let config = LangSmithConfig::from_env()?;
        Ok(Self::new(config))
    }
    
    pub fn with_async_mode(mut self, async_mode: bool) -> Self {
        self.async_mode = async_mode;
        self
    }
    
    fn push_run(&self, run: RunTree) {
        let active_runs = self.active_runs.clone();
        tokio::spawn(async move {
            let mut runs = active_runs.write().await;
            runs.push(run);
        });
    }
    
    fn pop_run(&self) {
        let active_runs = self.active_runs.clone();
        tokio::spawn(async move {
            let mut runs = active_runs.write().await;
            runs.pop();
        });
    }
}

#[async_trait]
impl CallbackHandler for LangSmithHandler {
    async fn on_run_start(&self, run: &RunTree) {
        if !self.client.is_tracing_enabled() {
            return;
        }
        
        if self.async_mode {
            let run = run.clone();
            let config = self.client.config.clone();
            let client = LangSmithClient::new(config);
            tokio::spawn(async move {
                let _ = client.create_run(&run).await;
            });
        } else {
            let _ = self.client.create_run(run).await;
        }
        
        self.push_run(run.clone());
    }
    
    async fn on_run_end(&self, run: &RunTree) {
        if !self.client.is_tracing_enabled() {
            return;
        }
        
        if self.async_mode {
            let run = run.clone();
            let config = self.client.config.clone();
            let client = LangSmithClient::new(config);
            tokio::spawn(async move {
                let _ = client.update_run(&run).await;
            });
        } else {
            let _ = self.client.update_run(run).await;
        }
        
        self.pop_run();
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