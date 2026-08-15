// lc-providers/src/providers/mistral.rs
//! Mistral AI API implementation (OpenAI-compatible).
//!
//! Mistral's chat API is compatible with the OpenAI `/v1/chat/completions` format,
//! so this implementation wraps `OpenAIChat` and delegates all calls.
//!
//! # Supported Models
//!
//! - `mistral-large-latest` — flagship model
//! - `mistral-medium-latest` — balanced performance
//! - `mistral-small-latest` — fast and cost-effective
//! - `open-mistral-nemo` — open-weight model
//! - `codestral-latest` — code generation
//! - `mistral-embed` — embedding model (use `MistralEmbeddings` in lc-embeddings)
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_providers::providers::{MistralChat, MistralConfig};
//!
//! let llm = MistralChat::new(MistralConfig::new("your-api-key"));
//! let result = llm.chat(messages, None).await?;
//! ```

use crate::error::ProviderError;
use crate::openai::{OpenAIChat, OpenAIConfig, OpenAIError, StructuredOutputMethod};
use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::env;
use std::pin::Pin;

/// Mistral AI API endpoint.
pub const MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";

/// Mistral model list.
pub const MISTRAL_MODELS: [&str; 6] = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "open-mistral-nemo",
    "codestral-latest",
    "mistral-embed",
];

/// Mistral AI configuration.
#[derive(Debug, Clone)]
pub struct MistralConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: MISTRAL_BASE_URL.to_string(),
            model: "mistral-large-latest".to_string(),
            temperature: None,
            max_tokens: None,
        }
    }
}

impl MistralConfig {
    /// Creates a new MistralConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a MistralConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `MISTRAL_API_KEY`: API key (required)
    /// - `MISTRAL_BASE_URL`: API endpoint (optional)
    /// - `MISTRAL_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = env::var("MISTRAL_API_KEY")
            .map_err(|_| "MISTRAL_API_KEY environment variable not set".to_string())?;

        let base_url =
            env::var("MISTRAL_BASE_URL").unwrap_or_else(|_| MISTRAL_BASE_URL.to_string());

        let model =
            env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-large-latest".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Creates a MistralConfig from environment variables.
    #[deprecated(
        since = "0.9.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    #[allow(deprecated)]
    pub fn from_env() -> Result<Self, String> {
        Self::from_env_result()
    }

    /// Sets the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the temperature parameter.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the max tokens limit.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Converts to OpenAI config (reuses OpenAI implementation).
    pub fn into_openai_config(self) -> OpenAIConfig {
        OpenAIConfig {
            api_key: self.api_key,
            base_url: self.base_url,
            model: self.model,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            streaming: false,
            organization: None,
            tools: None,
            tool_choice: None,
        }
    }
}

/// Mistral AI chat client.
///
/// Wraps `OpenAIChat` internally since Mistral's API is OpenAI-compatible.
#[derive(Clone)]
pub struct MistralChat {
    inner: OpenAIChat,
}

impl std::fmt::Debug for MistralChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralChat").finish_non_exhaustive()
    }
}

impl MistralChat {
    /// Creates a MistralChat with the given configuration.
    pub fn new(config: MistralConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    /// Creates a MistralChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, String> {
        Ok(Self::new(MistralConfig::from_env_result()?))
    }

    /// Creates a MistralChat from environment variables.
    #[deprecated(
        since = "0.9.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    #[allow(deprecated)]
    pub fn from_env() -> Result<Self, String> {
        Self::from_env_result()
    }

    /// Delegate chat to inner OpenAIChat.
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, OpenAIError> {
        self.inner.chat(messages, config).await
    }

    /// Delegate stream_chat to inner OpenAIChat.
    pub async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, OpenAIError>> + Send>>, OpenAIError> {
        self.inner.stream_chat(messages, config).await
    }

    /// Delegate bind_tools to inner OpenAIChat.
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        Self {
            inner: self.inner.bind_tools(tools),
        }
    }

    /// Delegate with_tool_choice to inner OpenAIChat.
    pub fn with_tool_choice(self, choice: impl Into<String>) -> Self {
        Self {
            inner: self.inner.with_tool_choice(choice),
        }
    }

    /// Delegate with_structured_output to inner OpenAIChat.
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> StructuredOutputMethod<T> {
        self.inner.with_structured_output()
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for MistralChat {
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

    fn with_temperature(self, temp: f32) -> Self {
        Self {
            inner: self.inner.with_temperature(temp),
        }
    }

    fn with_max_tokens(self, max: usize) -> Self {
        Self {
            inner: self.inner.with_max_tokens(max),
        }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MistralChat {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .invoke(input, config)
            .await
            .map_err(ProviderError::Mistral)
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;
        let stream = self
            .inner
            .stream(input, config)
            .await
            .map_err(ProviderError::Mistral)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::Mistral))))
    }
}

#[async_trait]
impl BaseChatModel for MistralChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(ProviderError::Mistral)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        use futures_util::StreamExt;
        let stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(ProviderError::Mistral)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::Mistral))))
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // Expose the inherent tool-binding capability at the trait level so it
        // survives being wrapped by `ChatModelWrapper` / `LLMClient` (Q1).
        Some(Box::new(self.bind_tools(tools)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ENV_TEST_LOCK;

    fn save_and_set(key: &str, value: &str) -> Option<String> {
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        old
    }

    fn restore(key: &str, old: Option<String>) {
        match old {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn test_config_new() {
        let config = MistralConfig::new("test-key");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, MISTRAL_BASE_URL);
        assert_eq!(config.model, "mistral-large-latest");
    }

    #[test]
    fn test_config_builder() {
        let config = MistralConfig::new("key")
            .with_model("mistral-small-latest")
            .with_base_url("https://custom.mistral.ai/v1")
            .with_temperature(0.7)
            .with_max_tokens(1024);
        assert_eq!(config.model, "mistral-small-latest");
        assert_eq!(config.base_url, "https://custom.mistral.ai/v1");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(1024));
    }

    #[test]
    fn test_config_from_env_result_ok() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("MISTRAL_API_KEY", "env-key");
        let result = MistralConfig::from_env_result();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().api_key, "env-key");
        restore("MISTRAL_API_KEY", old);
    }

    #[test]
    fn test_config_from_env_result_err_when_missing() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old = std::env::var("MISTRAL_API_KEY").ok();
        std::env::remove_var("MISTRAL_API_KEY");
        let result = MistralConfig::from_env_result();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("MISTRAL_API_KEY"));
        restore("MISTRAL_API_KEY", old);
    }

    #[test]
    fn test_into_openai_config() {
        let config = MistralConfig::new("key").with_model("codestral-latest");
        let openai_config = config.into_openai_config();
        assert_eq!(openai_config.api_key, "key");
        assert_eq!(openai_config.base_url, MISTRAL_BASE_URL);
        assert_eq!(openai_config.model, "codestral-latest");
    }

    #[test]
    fn test_chat_new() {
        let config = MistralConfig::new("test-key");
        let _chat = MistralChat::new(config);
    }

    #[test]
    fn test_model_name() {
        let config = MistralConfig::new("key").with_model("mistral-small-latest");
        let chat = MistralChat::new(config);
        assert_eq!(chat.model_name(), "mistral-small-latest");
    }
}
