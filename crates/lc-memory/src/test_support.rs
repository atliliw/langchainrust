// lc-memory/src/test_support.rs
//! Shared test utilities: a mock LLM with injectable failures.
//!
//! Compiled only under `#[cfg(test)]`. Reused by test modules like `summary.rs` /
//! `summary_buffer.rs` to drive the "summary LLM call fails" path — no real API needed,
//! and no mock duplication.

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_schema::Message;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Mock error: simulates a summary LLM call failure.
#[derive(Debug)]
pub struct MockLlmError(pub String);

impl std::fmt::Display for MockLlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for MockLlmError {}

/// Mock LLM that consumes preset responses one at a time.
///
/// Each element is `Result<summary text, failure reason>`; `pop` consumes them, and an empty
/// queue falls back to returning "Summary". Returning `Err` simulates a failed summary LLM call.
#[derive(Clone)]
pub struct MockLlm {
    responses: Arc<Mutex<Vec<Result<String, String>>>>,
}

impl MockLlm {
    pub fn new(responses: Vec<Result<String, String>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLlm {
    fn model_name(&self) -> &str {
        "mock-llm"
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        text.len()
    }

    fn with_temperature(self, _temp: f32) -> Self
    where
        Self: Sized,
    {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        self
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockLlm {
    type Error = MockLlmError;

    async fn invoke(
        &self,
        _input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let mut responses = self.responses.lock().await;
        match responses.pop() {
            Some(Ok(content)) => Ok(LLMResult {
                content,
                model: "mock-llm".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            }),
            Some(Err(reason)) => Err(MockLlmError(reason)),
            None => Ok(LLMResult {
                content: "Summary".to_string(),
                model: "mock-llm".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            }),
        }
    }
}

#[async_trait]
impl BaseChatModel for MockLlm {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.invoke(messages, config).await
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        unimplemented!("stream_chat not needed for tests")
    }
}
