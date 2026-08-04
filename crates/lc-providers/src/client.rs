// lc-providers/src/client.rs
//! LLMClient — zero-config unified entry point for switching providers
//!
//! Provides three creation modes:
//! 1. `from_env()` — auto-detect environment variables
//! 2. `openai(config)` / `anthropic(config)` etc. — explicit Config
//! 3. `openai(OpenAIConfig::from_env_result()?)` — read config from env, then override
//!
//! # Example
//!
//! ```ignore
//! // Mode 1: auto-detect
//! let llm = LLMClient::from_env()?;
//!
//! // Mode 2: explicit Config
//! let llm = LLMClient::openai(OpenAIConfig::new("sk-...").with_model("gpt-4o"));
//!
//! // Mode 3: from env + override
//! let llm = LLMClient::openai(OpenAIConfig::from_env_result()?.with_model("gpt-4o"));
//! ```

use crate::error::ProviderError;
use crate::ollama::OllamaChat;
use crate::ollama::OllamaConfig;
use crate::openai::OpenAIChat;
use crate::openai::OpenAIConfig;
use crate::providers::anthropic::AnthropicChat;
use crate::providers::anthropic::AnthropicConfig;
use crate::providers::deepseek::DeepSeekChat;
use crate::providers::deepseek::DeepSeekConfig;
use crate::providers::gemini::GeminiChat;
use crate::providers::gemini::GeminiConfig;
use crate::providers::moonshot::MoonshotChat;
use crate::providers::moonshot::MoonshotConfig;
use crate::providers::qwen::QwenChat;
use crate::providers::qwen::QwenConfig;
use crate::providers::zhipu::ZhipuChat;
use crate::providers::zhipu::ZhipuConfig;
use crate::wrap_chat_model;
use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;
use std::sync::Arc;

/// LLM Client unified entry point
///
/// Wraps any `BaseChatModel` as `Arc<dyn BaseChatModel<Error = ProviderError>>`,
/// providing zero-config auto-detection and explicit construction.
///
/// Implements `Deref<Target = dyn BaseChatModel>`, so you can call `.chat()` etc. directly.
pub struct LLMClient {
    inner: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
}

impl std::fmt::Debug for LLMClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMClient")
            .field("model_name", &self.inner.model_name())
            .finish()
    }
}

impl LLMClient {
    // -----------------------------------------------------------------------
    // Auto-detect
    // -----------------------------------------------------------------------

    /// Create LLM Client from environment variables (auto-detect)
    ///
    /// Detection priority:
    /// 1. `OPENAI_API_KEY` -> OpenAIChat
    /// 2. `ANTHROPIC_API_KEY` -> AnthropicChat
    /// 3. `OLLAMA_BASE_URL` -> OllamaChat
    ///
    /// # Errors
    ///
    /// Returns error if no known environment variable is set.
    pub fn from_env() -> Result<Self, String> {
        // Priority 1: OpenAI
        if std::env::var("OPENAI_API_KEY").is_ok() {
            let config = OpenAIConfig::from_env_result()?;
            return Ok(Self::openai(config));
        }

        // Priority 2: Anthropic
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            let config = AnthropicConfig::from_env_result()?;
            return Ok(Self::anthropic(config));
        }

        // Priority 3: Ollama
        if std::env::var("OLLAMA_BASE_URL").is_ok() {
            let config = OllamaConfig::from_env_result()?;
            return Ok(Self::ollama(config));
        }

        Err(
            "No LLM provider detected. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL"
                .to_string(),
        )
    }

    // -----------------------------------------------------------------------
    // Explicit construction
    // -----------------------------------------------------------------------

    /// Create OpenAI Client
    pub fn openai(config: OpenAIConfig) -> Self {
        let llm = OpenAIChat::new(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create Anthropic Client
    pub fn anthropic(config: AnthropicConfig) -> Self {
        let llm = AnthropicChat::new(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create Ollama Client
    pub fn ollama(config: OllamaConfig) -> Self {
        let llm = OllamaChat::with_config(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create Gemini Client
    pub fn gemini(config: GeminiConfig) -> Self {
        let llm = GeminiChat::new(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create DeepSeek Client
    pub fn deepseek(config: DeepSeekConfig) -> Self {
        let llm = DeepSeekChat::new(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create Qwen Client
    pub fn qwen(config: QwenConfig) -> Self {
        let llm = QwenChat::new(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create Moonshot Client
    pub fn moonshot(config: MoonshotConfig) -> Self {
        let llm = MoonshotChat::new(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create Zhipu Client
    pub fn zhipu(config: ZhipuConfig) -> Self {
        let llm = ZhipuChat::new(config);
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    // -----------------------------------------------------------------------
    // Generic construction
    // -----------------------------------------------------------------------

    /// Create Client from any `BaseChatModel`
    pub fn from_llm<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            inner: wrap_chat_model(llm),
        }
    }

    /// Create Client from `Arc<dyn BaseChatModel>`
    pub fn from_arc(llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>) -> Self {
        Self { inner: llm }
    }

    // -----------------------------------------------------------------------
    // Access inner
    // -----------------------------------------------------------------------

    /// Get the inner `Arc<dyn BaseChatModel>`, can be passed directly to Agent
    pub fn into_inner(self) -> Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync> {
        self.inner
    }

    /// Get inner reference
    pub fn inner(&self) -> &Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync> {
        &self.inner
    }
}

// LLMClient implements the full trait hierarchy: Runnable -> BaseLanguageModel -> BaseChatModel

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for LLMClient {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, ProviderError> {
        self.inner.invoke(input, config).await
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, ProviderError> {
        self.inner.batch(inputs, config).await
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<LLMResult, ProviderError>> + Send>>,
        ProviderError,
    > {
        self.inner.stream(input, config).await
    }
}

impl BaseLanguageModel<Vec<Message>, LLMResult> for LLMClient {
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
        // Cannot modify a wrapped LLM's temperature.
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        // Cannot modify a wrapped LLM's max_tokens.
        self
    }
}

#[async_trait]
impl BaseChatModel for LLMClient {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, ProviderError> {
        self.inner.chat(messages, config).await
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>,
        ProviderError,
    > {
        self.inner.stream_chat(messages, config).await
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = ProviderError> + Send + Sync>> {
        self.inner.bind_tools(tools)
    }
}

impl std::ops::Deref for LLMClient {
    type Target = dyn BaseChatModel<Error = ProviderError> + Send + Sync;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{OpenAIChat, OpenAIConfig};

    #[test]
    fn test_from_llm_openai() {
        let config = OpenAIConfig::new("test_key");
        let _client = LLMClient::from_llm(OpenAIChat::new(config));
    }

    #[test]
    fn test_openai_constructor() {
        let config = OpenAIConfig::new("test_key");
        let _client = LLMClient::openai(config);
    }

    #[test]
    fn test_from_arc() {
        let config = OpenAIConfig::new("test_key");
        let arc = wrap_chat_model(OpenAIChat::new(config));
        let _client = LLMClient::from_arc(arc);
    }

    #[test]
    fn test_into_inner() {
        let config = OpenAIConfig::new("test_key");
        let client = LLMClient::openai(config);
        let _arc = client.into_inner();
    }

    #[test]
    fn test_from_env_no_keys() {
        // Clear env vars, ensure from_env errors
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("OLLAMA_BASE_URL");

        let result = LLMClient::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No LLM provider detected"));
    }

    #[test]
    fn test_deref_works() {
        let config = OpenAIConfig::new("test_key");
        let client = LLMClient::openai(config);
        // Can directly call BaseChatModel methods
        let _name = client.model_name();
    }
}
