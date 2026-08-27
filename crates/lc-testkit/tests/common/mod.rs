//! Shared test double: an offline fake `BaseChatModel` (its Error sits at the real provider's layer).

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_providers::ProviderError;
use lc_schema::Message;
use std::pin::Pin;

/// A fake model returning a fixed reply, `Error = ProviderError` (same layer as a real provider,
/// satisfying `RecordingProvider<M>`'s `M::Error: Into<ProviderError>` bound).
#[derive(Clone)]
pub struct FakeModel {
    reply: String,
}

impl FakeModel {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
        }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for FakeModel {
    type Error = ProviderError;

    async fn invoke(
        &self,
        messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Ok(LLMResult {
            content: self.reply.clone(),
            model: "fake".to_string(),
            token_usage: Some(TokenUsage {
                prompt_tokens: messages.len(),
                completion_tokens: 1,
                total_tokens: messages.len() + 1,
            }),
            ..Default::default()
        })
    }
}

impl BaseLanguageModel<Vec<Message>, LLMResult> for FakeModel {
    fn model_name(&self) -> &str {
        "fake"
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        text.chars().count()
    }

    fn temperature(&self) -> Option<f32> {
        None
    }

    fn max_tokens(&self) -> Option<usize> {
        None
    }

    fn with_temperature(self, _temp: f32) -> Self {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for FakeModel {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.invoke(messages, config).await
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let full = self.invoke(messages, config).await?.content;
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(
            StreamChunk::new(full),
        )])))
    }
}
