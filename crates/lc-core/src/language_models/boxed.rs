// src/core/language_models/boxed.rs
//! Forwarding `BaseChatModel` implementations for boxed trait objects.
//!
//! `BaseChatModel::bind_tools` returns `Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>`.
//! For a wrapper (e.g. `TokenTrackingLLM`) to re-wrap that box and keep
//! counting, the box itself must satisfy the `L: BaseChatModel` bound — this
//! module is the thin glue forwarding every method to the boxed model.
//!
//! The orphan rule is satisfied here because `Runnable`, `BaseLanguageModel`
//! and `BaseChatModel` are all **local** to `lc-core` (unlike
//! `lc-providers::wrapper::BoundModel`, which needed a private adapter because
//! those traits are foreign there).

use crate::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use crate::runnables::Runnable;
use crate::tools::ToolDefinition;
use crate::RunnableConfig;
use async_trait::async_trait;
use futures_util::Stream;
use lc_schema::Message;
use std::pin::Pin;

#[async_trait]
impl<E> Runnable<Vec<Message>, LLMResult> for Box<dyn BaseChatModel<Error = E> + Send + Sync>
where
    E: std::error::Error + Send + Sync + 'static,
{
    type Error = E;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        (**self).invoke(input, config).await
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        (**self).stream(input, config).await
    }
}

#[async_trait]
impl<E> BaseLanguageModel<Vec<Message>, LLMResult>
    for Box<dyn BaseChatModel<Error = E> + Send + Sync>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        (**self).get_num_tokens(text)
    }

    fn temperature(&self) -> Option<f32> {
        (**self).temperature()
    }

    fn max_tokens(&self) -> Option<usize> {
        (**self).max_tokens()
    }

    // `with_temperature`/`with_max_tokens` are `Self: Sized` and, on a boxed
    // trait object, would replace the box — returning `self` is the honest
    // no-op for this glue; sampling overrides flow through `RunnableConfig`
    // (same note as `lc-providers::wrapper::BoundModel`).
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
impl<E> BaseChatModel for Box<dyn BaseChatModel<Error = E> + Send + Sync>
where
    E: std::error::Error + Send + Sync + 'static,
{
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        (**self).chat(messages, config).await
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        (**self).stream_chat(messages, config).await
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // `Self::Error = E`, so the inner `bind_tools` already returns exactly
        // `Option<Box<dyn BaseChatModel<Error = E> + Send + Sync>>`.
        (**self).bind_tools(tools)
    }
}
