// src/core/language_models/chat.rs
//! Chat model base trait.

use super::BaseLanguageModel;
use crate::core::tools::ToolCall;
use crate::core::tools::ToolDefinition;
use crate::schema::Message;
use crate::RunnableConfig;
use async_trait::async_trait;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// LLM result containing response content and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LLMResult {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub token_usage: Option<TokenUsage>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_content: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input token count.
    pub prompt_tokens: usize,

    /// Output token count.
    pub completion_tokens: usize,

    /// Total token count.
    pub total_tokens: usize,
}

/// Base trait for chat models.
///
/// Extends BaseLanguageModel for chat scenarios.
/// Accepts message list as input, returns AI message.
#[async_trait]
pub trait BaseChatModel: BaseLanguageModel<Vec<Message>, LLMResult> {
    /// Chat with the model.
    ///
    /// # Arguments
    /// * `messages` - Message list.
    /// * `config` - Optional configuration.
    ///
    /// # Returns
    /// LLM result.
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error>;

    /// Stream chat with the model.
    ///
    /// # Arguments
    /// * `messages` - Message list.
    /// * `config` - Optional configuration.
    ///
    /// # Returns
    /// Stream of output chunks.
    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>;

    /// Chat with system prompt.
    ///
    /// # Arguments
    /// * `system` - System prompt.
    /// * `messages` - Message list.
    ///
    /// # Returns
    /// LLM result.
    async fn chat_with_system(
        &self,
        system: String,
        messages: Vec<Message>,
    ) -> Result<LLMResult, Self::Error> {
        let full_messages = vec![Message::system(system)]
            .into_iter()
            .chain(messages)
            .collect();

        self.chat(full_messages, None).await
    }

    /// Bind tool definitions for function calling.
    ///
    /// Returns `None` by default if the provider does not support tool binding.
    /// Providers that support function calling (OpenAI, Ollama) override this
    /// to return a boxed chat model with the tools attached.
    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        None
    }
}
