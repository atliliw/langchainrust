// src/core/language_models/wrapper.rs
//! Wrapper that normalizes any `BaseChatModel` to use `crate::error::Error`.
//!
//! This enables `Arc<dyn BaseChatModel<Error = crate::error::Error>>` —
//! a single trait-object type that works with every provider.

use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use crate::core::runnables::Runnable;
use crate::core::tools::ToolDefinition;
use crate::error::Error;
use crate::schema::Message;
use crate::RunnableConfig;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

/// Wrapper that converts any `BaseChatModel`'s error into `crate::error::Error`.
///
/// This allows heterogeneous LLM providers to be stored behind a single
/// `Arc<dyn BaseChatModel<Error = Error>>` trait object.
///
/// # Example
///
/// ```ignore
/// use langchainrust::core::language_models::ChatModelWrapper;
///
/// let openai = OpenAIChat::new(config);
/// let wrapped: Arc<dyn BaseChatModel<Error = Error>> = Arc::new(ChatModelWrapper::new(openai));
/// ```
pub struct ChatModelWrapper<L> {
    inner: L,
}

impl<L> ChatModelWrapper<L> {
    /// Create a new wrapper around the given LLM.
    pub fn new(llm: L) -> Self {
        Self { inner: llm }
    }

    /// Get a reference to the inner LLM.
    pub fn inner(&self) -> &L {
        &self.inner
    }
}

impl<L> BaseLanguageModel<Vec<Message>, LLMResult> for ChatModelWrapper<L>
where
    L: BaseChatModel + Send + Sync,
    L::Error: Into<Error>,
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
impl<L> Runnable<Vec<Message>, LLMResult> for ChatModelWrapper<L>
where
    L: BaseChatModel + Send + Sync,
    L::Error: Into<Error>,
{
    type Error = Error;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Error> {
        self.inner
            .invoke(input, config)
            .await
            .map_err(Into::into)
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, Error> {
        self.inner
            .batch(inputs, config)
            .await
            .map_err(Into::into)
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>>, Error> {
        let inner_stream = self
            .inner
            .stream(input, config)
            .await
            .map_err(Into::into)?;

        let mapped: Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>> =
            Box::pin(inner_stream.map(|item| item.map_err(Into::into)));

        Ok(mapped)
    }
}

#[async_trait]
impl<L> BaseChatModel for ChatModelWrapper<L>
where
    L: BaseChatModel + Send + Sync,
    L::Error: Into<Error>,
{
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Error> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(Into::into)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>, Error> {
        let inner_stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(Into::into)?;

        let mapped: Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>> =
            Box::pin(inner_stream.map(|item| item.map_err(Into::into)));

        Ok(mapped)
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Error> + Send + Sync>> {
        self.inner
            .bind_tools(tools)
            .map(|boxed| -> Box<dyn BaseChatModel<Error = Error> + Send + Sync> {
                Box::new(BoxedModelWrapper { inner: boxed })
            })
    }
}

// ---------------------------------------------------------------------------
// BoxedModelWrapper — wraps a Box<dyn BaseChatModel<Error = E>>
// where E: Into<Error>, converting it to BaseChatModel<Error = Error>.
// ---------------------------------------------------------------------------

/// Wraps a `Box<dyn BaseChatModel<Error = E>>` and converts errors to `crate::error::Error`.
///
/// This is needed because `ChatModelWrapper<L>` requires `L: Sized`, but
/// `Box<dyn BaseChatModel<Error = E>>` is unsized. We solve this by
/// wrapping the boxed trait object in a new struct that implements
/// `BaseChatModel<Error = Error>` directly.
struct BoxedModelWrapper<E: std::error::Error + Send + Sync + Into<Error> + 'static> {
    inner: Box<dyn BaseChatModel<Error = E> + Send + Sync>,
}

impl<E> BaseLanguageModel<Vec<Message>, LLMResult> for BoxedModelWrapper<E>
where
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
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

    fn with_temperature(self, _temp: f32) -> Self
    where
        Self: Sized,
    {
        // Cannot modify a boxed trait object's temperature.
        // This is a no-op since the boxed model is already configured.
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        // Cannot modify a boxed trait object's max_tokens.
        self
    }
}

#[async_trait]
impl<E> Runnable<Vec<Message>, LLMResult> for BoxedModelWrapper<E>
where
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
{
    type Error = Error;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Error> {
        self.inner
            .invoke(input, config)
            .await
            .map_err(Into::into)
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, Error> {
        self.inner
            .batch(inputs, config)
            .await
            .map_err(Into::into)
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>>, Error> {
        let inner_stream = self
            .inner
            .stream(input, config)
            .await
            .map_err(Into::into)?;

        let mapped: Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>> =
            Box::pin(inner_stream.map(|item| item.map_err(Into::into)));

        Ok(mapped)
    }
}

#[async_trait]
impl<E> BaseChatModel for BoxedModelWrapper<E>
where
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
{
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Error> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(Into::into)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>, Error> {
        let inner_stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(Into::into)?;

        let mapped: Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>> =
            Box::pin(inner_stream.map(|item| item.map_err(Into::into)));

        Ok(mapped)
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Error> + Send + Sync>> {
        // The inner model already has tools bound; calling bind_tools again
        // may or may not work depending on the provider. We delegate.
        self.inner
            .bind_tools(tools)
            .map(|boxed| -> Box<dyn BaseChatModel<Error = Error> + Send + Sync> {
                Box::new(BoxedModelWrapper { inner: boxed })
            })
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Wrap any `BaseChatModel` into a trait object with `Error = crate::error::Error`.
///
/// This is the primary way to create a `Arc<dyn BaseChatModel<Error = Error>>`
/// from any concrete LLM provider.
///
/// # Example
///
/// ```ignore
/// let openai = OpenAIChat::new(config);
/// let llm: Arc<dyn BaseChatModel<Error = Error>> = wrap_chat_model(openai);
/// ```
pub fn wrap_chat_model<L>(llm: L) -> Arc<dyn BaseChatModel<Error = Error> + Send + Sync>
where
    L: BaseChatModel + Send + Sync + 'static,
    L::Error: Into<Error>,
{
    Arc::new(ChatModelWrapper::new(llm))
}
