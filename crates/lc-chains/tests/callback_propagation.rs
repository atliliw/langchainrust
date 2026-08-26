// lc-chains/tests/callback_propagation.rs
//! Integration test for callback propagation through Chain execution.

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_callbacks::{CallbackHandler, CallbackManager, RunTree, RunType};
use lc_chains::base::{BaseChain, ChainError, ChainResult};
use lc_chains::{ChainRunnable, LLMChain, RouterChain, SequentialChain};
use lc_core::language_models::{LLMResult, StreamChunk};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_core::{BaseChatModel, BaseLanguageModel};
use lc_providers::openai::OpenAIError;
use lc_providers::ProviderError;
use lc_schema::Message;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::pin::Pin;
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
        result.insert(
            "output".to_string(),
            Value::String(format!("echo: {}", input)),
        );
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
    assert_eq!(
        result.get("output").unwrap(),
        &Value::String("echo: hello".to_string())
    );

    // invoke_with_config with None
    let result = chain.invoke_with_config(inputs, None).await.unwrap();
    assert_eq!(
        result.get("output").unwrap(),
        &Value::String("echo: hello".to_string())
    );
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
    assert_eq!(
        result.get("output").unwrap(),
        &Value::String("echo: test".to_string())
    );
}

/// A chain with a custom name that doesn't override `invoke_with_config`
/// (exercises the base default which must no longer drop config).
struct NamedChain {
    name: String,
}

#[async_trait]
impl BaseChain for NamedChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }
    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }
    async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let mut result = HashMap::new();
        result.insert("output".to_string(), Value::String("ok".to_string()));
        Ok(result)
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Minimal chat model mock for LLMChain callback tests.
struct MockChatModel;

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockChatModel {
    type Error = ProviderError;
    async fn invoke(
        &self,
        _input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Ok(LLMResult {
            content: "rust rocks".to_string(),
            model: "mock-chat".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChatModel {
    fn model_name(&self) -> &str {
        "mock-chat"
    }
    fn get_num_tokens(&self, t: &str) -> usize {
        t.len()
    }
    fn with_temperature(self, _: f32) -> Self {
        self
    }
    fn with_max_tokens(self, _: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for MockChatModel {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Ok(LLMResult {
            content: "rust rocks".to_string(),
            model: "mock-chat".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }
    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        Err(ProviderError::OpenAI(OpenAIError::Api(
            "stream not supported".to_string(),
        )))
    }
}

/// P0-1: The default `invoke_with_config` must fire on_chain_start/end
/// (previously it did `let _ = config;` and dropped the callbacks).
#[tokio::test]
async fn test_invoke_with_config_fires_chain_callbacks() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler::new(events.clone());
    let callbacks = Arc::new(CallbackManager::new().add_handler(Arc::new(handler)));

    let chain = NamedChain {
        name: "leaf".to_string(),
    };
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Value::String("hi".to_string()));

    let config = RunnableConfig::new().with_callbacks(callbacks);
    let result = chain
        .invoke_with_config(inputs, Some(config))
        .await
        .unwrap();
    assert_eq!(
        result.get("output").unwrap(),
        &Value::String("ok".to_string())
    );

    let recorded = events.lock().unwrap().clone();
    assert!(recorded.contains(&"chain_start:leaf".to_string()));
    assert!(recorded.contains(&"chain_end:leaf".to_string()));
    assert!(!recorded.iter().any(|e| e.starts_with("chain_error")));
}

/// P0-1: SequentialChain must thread config into sub-chains — the inner chain
/// receives its own chain_start/chain_end (proving callbacks crossed the
/// composition boundary instead of being dropped).
#[tokio::test]
async fn test_sequential_chain_propagates_callbacks() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler::new(events.clone());
    let callbacks = Arc::new(CallbackManager::new().add_handler(Arc::new(handler)));

    let seq = SequentialChain::new().with_name("seq").add_chain(
        Arc::new(NamedChain {
            name: "inner".to_string(),
        }),
        vec!["input"],
        vec!["output"],
    );

    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Value::String("hi".to_string()));

    let config = RunnableConfig::new().with_callbacks(callbacks);
    let result = seq.invoke_with_config(inputs, Some(config)).await.unwrap();
    assert_eq!(
        result.get("output").unwrap(),
        &Value::String("ok".to_string())
    );

    let recorded = events.lock().unwrap().clone();
    // SequentialChain itself fires its own chain_start/end...
    assert!(recorded.contains(&"chain_start:seq".to_string()));
    assert!(recorded.contains(&"chain_end:seq".to_string()));
    // ...and the config reached the inner sub-chain.
    assert!(recorded.contains(&"chain_start:inner".to_string()));
    assert!(recorded.contains(&"chain_end:inner".to_string()));
}

/// P0-1: RouterChain must thread config into the routed destination chain.
#[tokio::test]
async fn test_router_chain_propagates_callbacks() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler::new(events.clone());
    let callbacks = Arc::new(CallbackManager::new().add_handler(Arc::new(handler)));

    let router = RouterChain::new()
        .with_name("router")
        .add_route_with_keywords(
            "math",
            "handles math",
            Arc::new(NamedChain {
                name: "math_chain".to_string(),
            }),
            vec!["calc"],
        )
        .add_route(
            "other",
            "anything else",
            Arc::new(NamedChain {
                name: "other_chain".to_string(),
            }),
        );

    let mut inputs = HashMap::new();
    inputs.insert(
        "input".to_string(),
        Value::String("please calc this".to_string()),
    );

    let config = RunnableConfig::new().with_callbacks(callbacks);
    let result = router
        .invoke_with_config(inputs, Some(config))
        .await
        .unwrap();
    assert_eq!(
        result.get("output").unwrap(),
        &Value::String("ok".to_string())
    );

    let recorded = events.lock().unwrap().clone();
    assert!(recorded.contains(&"chain_start:router".to_string()));
    assert!(recorded.contains(&"chain_end:router".to_string()));
    // The routed destination chain received config too.
    assert!(recorded.contains(&"chain_start:math_chain".to_string()));
    assert!(recorded.contains(&"chain_end:math_chain".to_string()));
    assert!(!recorded.iter().any(|e| e.contains("other_chain")));
}

/// P0-1: LLMChain fires chain_start → llm_start → llm_end → chain_end in order,
/// with exactly ONE LLM node per call (llm_start/llm_end reuse the same child
/// run — the double-object fix).
#[tokio::test]
async fn test_llm_chain_callback_order() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler::new(events.clone());
    let callbacks = Arc::new(CallbackManager::new().add_handler(Arc::new(handler)));

    let chain = LLMChain::new(MockChatModel, "Tell me about {topic}").with_input_key("topic");

    let mut inputs = HashMap::new();
    inputs.insert("topic".to_string(), Value::String("rust".to_string()));

    let config = RunnableConfig::new().with_callbacks(callbacks);
    let result = chain
        .invoke_with_config(inputs, Some(config))
        .await
        .unwrap();
    assert_eq!(
        result.get("text").unwrap(),
        &Value::String("rust rocks".to_string())
    );

    let recorded = events.lock().unwrap().clone();
    let llm_starts = recorded
        .iter()
        .filter(|e| e.starts_with("llm_start:"))
        .count();
    let llm_ends = recorded
        .iter()
        .filter(|e| e.starts_with("llm_end:"))
        .count();
    assert_eq!(llm_starts, 1, "exactly one llm_start");
    assert_eq!(llm_ends, 1, "exactly one llm_end");

    let pos = |prefix: &str| {
        recorded
            .iter()
            .position(|e| e.starts_with(prefix))
            .expect("event should fire")
    };
    assert!(pos("chain_start:") < pos("llm_start:"));
    assert!(pos("llm_start:") < pos("llm_end:"));
    assert!(pos("llm_end:") < pos("chain_end:"));
}

/// P0-1: The default `stream_with_config` fires on_chain_start up front and
/// on_chain_end once the token stream is exhausted.
#[tokio::test]
async fn test_stream_with_config_fires_chain_callbacks() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = RecordingHandler::new(events.clone());
    let callbacks = Arc::new(CallbackManager::new().add_handler(Arc::new(handler)));

    let chain = NamedChain {
        name: "leaf".to_string(),
    };
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), Value::String("hi".to_string()));

    let config = RunnableConfig::new().with_callbacks(callbacks);
    let mut stream = chain
        .stream_with_config(inputs, Some(config))
        .await
        .unwrap();
    while let Some(_token) = stream.next().await {}

    let recorded = events.lock().unwrap().clone();
    assert!(recorded.contains(&"chain_start:leaf".to_string()));
    assert!(recorded.contains(&"chain_end:leaf".to_string()));
    assert!(!recorded.iter().any(|e| e.starts_with("chain_error")));
}
