// lc-providers/src/wrapper.rs
//! Wrapper that normalizes any `BaseChatModel` to use `ProviderError`.
//!
//! This enables `Arc<dyn BaseChatModel<Error = ProviderError>>` —
//! a single trait-object type that works with every provider.

use crate::error::ProviderError;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;
use std::pin::Pin;
use std::sync::Arc;

/// Wrapper that converts any `BaseChatModel`'s error into `ProviderError`.
///
/// This allows heterogeneous LLM providers to be stored behind a single
/// `Arc<dyn BaseChatModel<Error = ProviderError>>` trait object.
pub struct ChatModelWrapper<L> {
    inner: L,
}

impl<L> ChatModelWrapper<L> {
    /// Create a new wrapper around the given LLM.
    pub fn new(llm: L) -> Self {
        Self { inner: llm }
    }
}

#[async_trait]
impl<L> Runnable<Vec<Message>, LLMResult> for ChatModelWrapper<L>
where
    L: Runnable<Vec<Message>, LLMResult> + Send + Sync,
    L::Error: Into<ProviderError>,
{
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, ProviderError> {
        self.inner.invoke(input, config).await.map_err(Into::into)
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, ProviderError> {
        self.inner.batch(inputs, config).await.map_err(Into::into)
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, ProviderError>> + Send>>, ProviderError>
    {
        let stream = self.inner.stream(input, config).await.map_err(Into::into)?;
        Ok(Box::pin(stream.map(|r| r.map_err(Into::into))))
    }
}

#[async_trait]
impl<L> BaseLanguageModel<Vec<Message>, LLMResult> for ChatModelWrapper<L>
where
    L: BaseLanguageModel<Vec<Message>, LLMResult> + Send + Sync,
    L::Error: Into<ProviderError>,
{
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        self.inner.get_num_tokens(text)
    }

    fn temperature(&self) -> Option<f32> {
        self.inner.temperature()
    }

    fn max_tokens(&self) -> Option<usize> {
        self.inner.max_tokens()
    }

    fn with_temperature(self, temp: f32) -> Self
    where
        Self: Sized,
    {
        Self {
            inner: self.inner.with_temperature(temp),
        }
    }

    fn with_max_tokens(self, max: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            inner: self.inner.with_max_tokens(max),
        }
    }
}

#[async_trait]
impl<L> BaseChatModel for ChatModelWrapper<L>
where
    L: BaseChatModel + Send + Sync,
    L::Error: Into<ProviderError>,
{
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, ProviderError> {
        self.inner.chat(messages, config).await.map_err(Into::into)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>, ProviderError>
    {
        let stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(Into::into)?;
        Ok(Box::pin(stream.map(|r| r.map_err(Into::into))))
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = ProviderError> + Send + Sync>> {
        // We cannot easily wrap the bound model since bind_tools returns
        // a different type. Return None to indicate no tool binding support
        // through the wrapper. Callers should bind tools before wrapping.
        let _ = tools;
        None
    }
}

/// Wrap any `BaseChatModel` into an `Arc<dyn BaseChatModel<Error = ProviderError>>`.
///
/// This is the primary way to create a uniform trait object from any provider.
///
/// # Example
///
/// ```ignore
/// use lc_providers::wrap_chat_model;
///
/// let openai = OpenAIChat::new(config);
/// let llm: Arc<dyn BaseChatModel<Error = ProviderError>> = wrap_chat_model(openai);
/// ```
pub fn wrap_chat_model<L>(llm: L) -> Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>
where
    L: BaseChatModel + Send + Sync + 'static,
    L::Error: Into<ProviderError>,
{
    Arc::new(ChatModelWrapper::new(llm))
}
