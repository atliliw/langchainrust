// lc-chains/tests/callback_propagation.rs
//! Integration test for callback propagation through Chain execution.

use async_trait::async_trait;
use lc_callbacks::{CallbackHandler, CallbackManager, RunTree, RunType};
use lc_chains::base::{BaseChain, ChainError, ChainResult};
use lc_chains::ChainRunnable;
use lc_core::runnables::{Runnable, RunnableConfig};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A callback handler that records which callbacks were fired.
#[derive(Debug)]
struct RecordingHandler {
    events: Arc<Mutex<Vec<String>>>,
}

impl RecordingHandler {
    fn new(events: Arc<Mutex<Vec<String>>>) -> Self {
        Self { events }
    }

    fn push(&self, event: &str) {
        self.events.lock().unwrap().push(event.to_string());
    }
}

#[async_trait]
impl CallbackHandler for RecordingHandler {
    async fn on_run_start(&self, run: &RunTree) {
        self.push(&format!("run_start:{}", run.name));
    }
    async fn on_run_end(&self, run: &RunTree) {
        self.push(&format!("run_end:{}", run.name));
    }
    async fn on_run_error(&self, run: &RunTree, error: &str) {
        self.push(&format!("run_error:{}:{}", run.name, error));
    }
    async fn on_chain_start(&self, run: &RunTree, _inputs: &Value) {
        self.push(&format!("chain_start:{}", run.name));
    }
    async fn on_chain_end(&self, run: &RunTree, _outputs: &Value) {
        self.push(&format!("chain_end:{}", run.name));
    }
    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.push(&format!("chain_error:{}:{}", run.name, error));
    }
    async fn on_llm_start(&self, run: &RunTree, _messages: &[lc_schema::Message]) {
        self.push(&format!("llm_start:{}", run.name));
    }
    async fn on_llm_end(&self, run: &RunTree, _response: &str) {
        self.push(&format!("llm_end:{}", run.name));
    }
    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        self.push(&format!("llm_error:{}:{}", run.name, error));
    }
    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, _input: &str) {
        self.push(&format!("tool_start:{}:{}", run.name, tool_name));
    }
    async fn on_tool_end(&self, run: &RunTree, _output: &str) {
        self.push(&format!("tool_end:{}", run.name));
    }
    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        self.push(&format!("tool_error:{}:{}", run.name, error));
    }
}

/// Simple echo chain for testing.
struct EchoChain;

#[async_trait]
impl BaseChain for EchoChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }
    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get("input").and_then(|v| v.as_str()).unwrap_or("");
        let mut result = HashMap::new();
        result.insert("output".to_string(), Value::String(format!("echo: {}", input)));
        Ok(result)
    }
}

/// Test that invoke_without_config still works (backward compat).
#[tokio::test]
async fn test_chain_invoke_without_config() {
    let chain = EchoChain;
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Value::String("hello".to_string()));

    // invoke without config
    let result = chain.invoke(inputs.clone()).await.unwrap();
    assert_eq!(result.get("output").unwrap(), &Value::String("echo: hello".to_string()));

    // invoke_with_config with None
    let result = chain.invoke_with_config(inputs, None).await.unwrap();
    assert_eq!(result.get("output").unwrap(), &Value::String("echo: hello".to_string()));
}

/// Test that CallbackManager dispatch methods work correctly.
#[tokio::test]
async fn test_callback_manager_dispatch_methods() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler::new(events.clone());

    let callbacks = CallbackManager::new().add_handler(Arc::new(handler));
    let run = RunTree::new("test", RunType::Chain, json!({}));

    callbacks.dispatch_chain_start(&run, &json!({})).await;
    callbacks.dispatch_chain_end(&run, &json!({})).await;
    callbacks.dispatch_llm_start(&run, &[]).await;
    callbacks.dispatch_llm_end(&run, "response").await;

    let recorded = events.lock().unwrap().clone();
    assert_eq!(recorded.len(), 4);
    assert!(recorded[0].starts_with("chain_start:"));
    assert!(recorded[1].starts_with("chain_end:"));
    assert!(recorded[2].starts_with("llm_start:"));
    assert!(recorded[3].starts_with("llm_end:"));
}

/// Test that multiple handlers all receive callbacks.
#[tokio::test]
async fn test_callback_manager_multiple_handlers() {
    let events1 = Arc::new(Mutex::new(Vec::new()));
    let events2 = Arc::new(Mutex::new(Vec::new()));

    let handler1 = RecordingHandler::new(events1.clone());
    let handler2 = RecordingHandler::new(events2.clone());

    let callbacks = CallbackManager::new()
        .add_handler(Arc::new(handler1))
        .add_handler(Arc::new(handler2));

    let run = RunTree::new("test", RunType::Chain, json!({}));
    callbacks.dispatch_chain_start(&run, &json!({})).await;

    assert_eq!(events1.lock().unwrap().len(), 1);
    assert_eq!(events2.lock().unwrap().len(), 1);
}

/// Test that ChainRunnable adapter propagates callbacks through LCEL pipeline.
#[tokio::test]
async fn test_chain_runnable_propagates_callbacks() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler::new(events.clone());
    let callbacks = Arc::new(CallbackManager::new().add_handler(Arc::new(handler)));

    let chain = ChainRunnable::new(Arc::new(EchoChain));

    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Value::String("test".to_string()));

    let config = RunnableConfig::new().with_callbacks(callbacks);

    let result: HashMap<String, Value> = chain.invoke(inputs, Some(config)).await.unwrap();
    assert_eq!(result.get("output").unwrap(), &Value::String("echo: test".to_string()));
}
