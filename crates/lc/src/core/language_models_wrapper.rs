// src/core/language_models/wrapper.rs
//! Wrapper that normalizes any `BaseChatModel` to use `crate::error::Error`.
//!
//! This enables `Arc<dyn BaseChatModel<Error = crate::error::Error>>` —
//! a single trait-object type that works with every provider.

use crate::error::Error;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_core::language_models::{
    predict_tools, BaseChatModel, BaseLanguageModel, LLMResult, PredictToolsError, StreamChunk,
};
use lc_core::runnables::Runnable;
use lc_core::structured_output::{stream_structured_output, StructuredOutputError};
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::ops::Deref;
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
        self.inner.invoke(input, config).await.map_err(Into::into)
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, Error> {
        self.inner.batch(inputs, config).await.map_err(Into::into)
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>>, Error> {
        let inner_stream = self.inner.stream(input, config).await.map_err(Into::into)?;

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
        self.inner.chat(messages, config).await.map_err(Into::into)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>, Error> {
        let inner_stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(Into::into)?;

        let mapped: Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>> =
            Box::pin(inner_stream.map(|item| item.map_err(Into::into)));

        Ok(mapped)
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Error> + Send + Sync>> {
        self.inner.bind_tools(tools).map(
            |boxed| -> Box<dyn BaseChatModel<Error = Error> + Send + Sync> {
                Box::new(BoxedModelWrapper::new(boxed))
            },
        )
    }
}

// T2-4 (0.27.0): convenience passthroughs to `lc_core`'s *free* helper
// functions, so a caller that holds the concrete [`ChatModelWrapper`] (rather
// than an erased `dyn BaseChatModel`) does not have to thread the free
// function's `&M`/`M::Error` through the wrapper's erased error type. Because
// the wrapper itself implements `BaseChatModel<Error = Error>`, `M` is the
// wrapper and `M::Error = Error` — the streamed error stays `StructuredOutputError`
// and the tool-call error stays within `PredictToolsError<Error>`.
impl<L> ChatModelWrapper<L>
where
    L: BaseChatModel + Send + Sync,
    L::Error: Into<Error>,
{
    /// Stream a structured-typed output, delegating to
    /// [`lc_core::structured_output::stream_structured_output`] on the wrapped model.
    ///
    /// This is the streaming counterpart of the blanket `StructuredOutputExt`'s
    /// `with_structured_output`; `T` is the typed JSON value being parsed from
    /// the streamed response. Errors surface as [`StructuredOutputError`].
    pub async fn stream_structured_output<T>(
        &self,
        schema: Value,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<T, StructuredOutputError>> + Send>>,
        StructuredOutputError,
    >
    where
        T: DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync + 'static,
    {
        stream_structured_output(self, schema, prompt).await
    }

    /// Predict tool calls for a single turn, delegating to
    /// [`lc_core::language_models::predict_tools`] on the wrapped model.
    ///
    /// Does **not** run an agent loop — it returns one `LLMResult` whose
    /// `tool_calls` the caller is expected to execute. Because the wrapper
    /// erases the inner error to `crate::error::Error`, the returned
    /// [`PredictToolsError`] is `PredictToolsError<Error>`.
    pub async fn predict_tools(
        &self,
        prompt: impl Into<String>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LLMResult, PredictToolsError<Error>> {
        predict_tools(self, prompt, tools).await
    }
}

// ---------------------------------------------------------------------------
// PtrModelWrapper — wraps a boxed/arc'd `dyn BaseChatModel<Error = E>`
// where `E: Into<Error>`, converting it to `BaseChatModel<Error = Error>`.
// ---------------------------------------------------------------------------

/// One pointer-holding wrapper that normalizes any `Box<dyn ...>` or `Arc<dyn ...>`
/// chat model to `BaseChatModel<Error = crate::error::Error>`.
///
/// `BoxedModelWrapper` and `ArcModelWrapper` differ only in the deref pointer
/// type, so a single generic struct hosts both (`P` is the pointer; the aliases
/// below pin it to `Box` / `Arc`). Because `BaseLanguageModel::with_temperature`
/// is `where Self: Sized`, the value cannot be applied through a trait object,
/// so the setter captures it as an override surfaced by `temperature()` /
/// `max_tokens()` — the previous silent no-op is gone.
struct PtrModelWrapper<P, E>
where
    P: Deref<Target = dyn BaseChatModel<Error = E> + Send + Sync>,
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
{
    inner: P,
    temperature_override: Option<f32>,
    max_tokens_override: Option<usize>,
}

/// Boxed-pointer instantiation of [`PtrModelWrapper`].
type BoxedModelWrapper<E> = PtrModelWrapper<Box<dyn BaseChatModel<Error = E> + Send + Sync>, E>;

/// Arc-pointer instantiation of [`PtrModelWrapper`].
type ArcModelWrapper<E> = PtrModelWrapper<Arc<dyn BaseChatModel<Error = E> + Send + Sync>, E>;

impl<P, E> PtrModelWrapper<P, E>
where
    P: Deref<Target = dyn BaseChatModel<Error = E> + Send + Sync> + Send + Sync,
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
{
    fn new(inner: P) -> Self {
        Self {
            inner,
            temperature_override: None,
            max_tokens_override: None,
        }
    }
}

impl<P, E> BaseLanguageModel<Vec<Message>, LLMResult> for PtrModelWrapper<P, E>
where
    P: Deref<Target = dyn BaseChatModel<Error = E> + Send + Sync> + Send + Sync + 'static,
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
{
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        self.inner.get_num_tokens(text)
    }

    fn temperature(&self) -> Option<f32> {
        self.temperature_override
            .or_else(|| self.inner.temperature())
    }

    fn max_tokens(&self) -> Option<usize> {
        self.max_tokens_override.or_else(|| self.inner.max_tokens())
    }

    fn with_temperature(self, temp: f32) -> Self
    where
        Self: Sized,
    {
        Self {
            inner: self.inner,
            temperature_override: Some(temp),
            max_tokens_override: self.max_tokens_override,
        }
    }

    fn with_max_tokens(self, max: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            inner: self.inner,
            temperature_override: self.temperature_override,
            max_tokens_override: Some(max),
        }
    }
}

#[async_trait]
impl<P, E> Runnable<Vec<Message>, LLMResult> for PtrModelWrapper<P, E>
where
    P: Deref<Target = dyn BaseChatModel<Error = E> + Send + Sync> + Send + Sync + 'static,
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
{
    type Error = Error;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Error> {
        self.inner.invoke(input, config).await.map_err(Into::into)
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, Error> {
        self.inner.batch(inputs, config).await.map_err(Into::into)
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>>, Error> {
        let inner_stream = self.inner.stream(input, config).await.map_err(Into::into)?;

        let mapped: Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>> =
            Box::pin(inner_stream.map(|item| item.map_err(Into::into)));

        Ok(mapped)
    }
}

#[async_trait]
impl<P, E> BaseChatModel for PtrModelWrapper<P, E>
where
    P: Deref<Target = dyn BaseChatModel<Error = E> + Send + Sync> + Send + Sync + 'static,
    E: std::error::Error + Send + Sync + Into<Error> + 'static,
{
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Error> {
        self.inner.chat(messages, config).await.map_err(Into::into)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>, Error> {
        let inner_stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(Into::into)?;

        let mapped: Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>> =
            Box::pin(inner_stream.map(|item| item.map_err(Into::into)));

        Ok(mapped)
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Error> + Send + Sync>> {
        self.inner.bind_tools(tools).map(
            |boxed| -> Box<dyn BaseChatModel<Error = Error> + Send + Sync> {
                Box::new(BoxedModelWrapper::new(boxed))
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
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

/// Wrap a provider's `Arc<dyn BaseChatModel<Error = ProviderError>>` into
/// an `Arc<dyn BaseChatModel<Error = Error>>`.
///
/// This bridges the gap between the `lc-providers` crate's `ProviderError`
/// and the main crate's unified `Error` type.
pub fn wrap_provider_model(
    llm: Arc<dyn BaseChatModel<Error = lc_providers::ProviderError> + Send + Sync>,
) -> Arc<dyn BaseChatModel<Error = Error> + Send + Sync> {
    Arc::new(ArcModelWrapper::new(llm))
}
