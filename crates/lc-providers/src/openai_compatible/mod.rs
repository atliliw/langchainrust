// lc-providers/src/openai_compatible/mod.rs
//! Generic OpenAI-compatible chat client (B5, v0.22.4).
//!
//! One transport for every endpoint speaking the OpenAI Chat Completions
//! protocol:
//!
//! - **Hosted routers** via presets: [Groq](OpenAICompatibleConfig::groq),
//!   [OpenRouter](OpenAICompatibleConfig::openrouter),
//!   [xAI / Grok](OpenAICompatibleConfig::xai) — or their `*_from_env`
//!   constructors.
//! - **Self-hosted/private endpoints** (vLLM, LM Studio, SGLang, Ollama's
//!   `/v1` shim, internal gateways) via
//!   [`OpenAICompatibleConfig::new`] — keyless by default, with
//!   [`with_api_key`](OpenAICompatibleConfig::with_api_key) and arbitrary
//!   [extra headers](OpenAICompatibleConfig::with_extra_header) when the
//!   deployment needs them.
//!
//! Rather than duplicating ~300 lines of delegation per vendor (the old
//! DeepSeek/Qwen/Moonshot pattern), presets differ only in config: base URL,
//!   auth, default model, and the error label. Streaming, function calling,
//! engine-side structured output, retries and callbacks all come from the
//! shared [`OpenAIChat`](crate::openai::OpenAIChat) implementation.
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use lc_providers::openai_compatible::OpenAICompatibleConfig;
//! use lc_providers::openai_compatible::GROQ_MODELS;
//! use lc_schema::Message;
//!
//! // Groq preset.
//! let groq = lc_providers::openai_compatible::OpenAICompatibleChat::groq_from_env()?;
//! let reply = groq.chat(vec![Message::human("hi")], None).await?;
//!
//! // Keyless local vLLM/LM Studio endpoint.
//! let local = lc_providers::openai_compatible::OpenAICompatibleConfig::new(
//!     "http://127.0.0.1:8000/v1",
//!     "local-model",
//! );
//! let _ = local;
//! let _ = GROQ_MODELS;
//! # Ok(())
//! # }
//! ```

mod config;

pub use config::{
    OpenAICompatibleConfig, DEFAULT_GROQ_MODEL, DEFAULT_XAI_MODEL, GROQ_BASE_URL, GROQ_MODELS,
    OPENROUTER_BASE_URL, XAI_BASE_URL, XAI_MODELS,
};

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::pin::Pin;

use crate::error::ProviderError;
use crate::openai::{OpenAIChat, OpenAIError, StructuredOutputMethod};

/// Chat client for any OpenAI-compatible endpoint.
///
/// Construct via [`OpenAICompatibleConfig`] — [`new`](Self::new) for a custom
/// endpoint or one of the `*_from_env` preset constructors.
#[derive(Clone)]
pub struct OpenAICompatibleChat {
    pub(crate) inner: OpenAIChat,
    /// Endpoint label carried into [`ProviderError`] on failure.
    pub(crate) provider: &'static str,
}

/// Groq chat client — a named preset sharing the generic transport.
pub type GroqChat = OpenAICompatibleChat;
/// OpenRouter chat client — a named preset sharing the generic transport.
pub type OpenRouterChat = OpenAICompatibleChat;
/// xAI (Grok) chat client — a named preset sharing the generic transport.
pub type XaiChat = OpenAICompatibleChat;

impl std::fmt::Debug for OpenAICompatibleChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatibleChat")
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl OpenAICompatibleChat {
    /// Creates a client for the given endpoint configuration.
    pub fn new(config: OpenAICompatibleConfig) -> Self {
        let provider = config.provider;
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
            provider,
        }
    }

    /// Generic endpoint from `OPENAI_COMPATIBLE_*` environment variables.
    pub fn from_env_result() -> Result<Self, ProviderError> {
        Ok(Self::new(OpenAICompatibleConfig::from_env_result()?))
    }

    /// Groq preset from `GROQ_API_KEY` (+ optional `GROQ_MODEL`, `GROQ_BASE_URL`).
    pub fn groq_from_env() -> Result<Self, ProviderError> {
        Ok(Self::new(OpenAICompatibleConfig::groq_from_env()?))
    }

    /// OpenRouter preset from `OPENROUTER_API_KEY` + `OPENROUTER_MODEL`.
    pub fn openrouter_from_env() -> Result<Self, ProviderError> {
        Ok(Self::new(OpenAICompatibleConfig::openrouter_from_env()?))
    }

    /// xAI preset from `XAI_API_KEY` (+ optional `XAI_MODEL`, `XAI_BASE_URL`).
    pub fn xai_from_env() -> Result<Self, ProviderError> {
        Ok(Self::new(OpenAICompatibleConfig::xai_from_env()?))
    }

    /// Endpoint label used in error messages (`"groq"`, `"openrouter"`, …).
    pub fn provider_label(&self) -> &str {
        self.provider
    }

    fn map_err(&self, e: OpenAIError) -> ProviderError {
        ProviderError::OpenAICompatible {
            provider: self.provider.to_string(),
            source: e,
        }
    }

    /// Single-shot chat against the compatible endpoint.
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, ProviderError> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(|e| self.map_err(e))
    }

    /// Per-token stream against the compatible endpoint.
    pub async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, ProviderError>> + Send>>, ProviderError>
    {
        use futures_util::StreamExt;
        let provider = self.provider;
        let stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(|e| self.map_err(e))?;
        Ok(Box::pin(stream.map(move |r| {
            r.map_err(|e| ProviderError::OpenAICompatible {
                provider: provider.to_string(),
                source: e,
            })
        })))
    }

    /// Binds tool definitions for function calling (preset is retained).
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        Self {
            inner: self.inner.bind_tools(tools),
            provider: self.provider,
        }
    }

    /// Sets the tool choice strategy.
    pub fn with_tool_choice(self, choice: impl Into<String>) -> Self {
        Self {
            inner: self.inner.with_tool_choice(choice),
            provider: self.provider,
        }
    }

    /// Tool-based structured output (works on any OpenAI-compatible backend).
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> StructuredOutputMethod<T> {
        self.inner.with_structured_output()
    }

    /// Engine-side `response_format: json_schema` structured output.
    ///
    /// Requires provider-side support (Groq/OpenRouter/xAI generally forward
    /// it when the underlying model supports it); unsupported models answer
    /// with a 4xx, surfaced through [`ProviderError`].
    pub fn with_json_schema_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> StructuredOutputMethod<T> {
        self.inner.with_json_schema_output()
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for OpenAICompatibleChat {
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

    fn with_temperature(mut self, temp: f32) -> Self {
        self.inner = self.inner.with_temperature(temp);
        self
    }

    fn with_max_tokens(mut self, max: usize) -> Self {
        self.inner = self.inner.with_max_tokens(max);
        self
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for OpenAICompatibleChat {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .invoke(input, config)
            .await
            .map_err(|e| self.map_err(e))
    }

    // True per-token streaming (same delegation rationale as DeepSeek).
    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;
        let provider = self.provider;
        let stream = self
            .inner
            .stream(input, config)
            .await
            .map_err(|e| self.map_err(e))?;
        Ok(Box::pin(stream.map(move |r| {
            r.map_err(|e| ProviderError::OpenAICompatible {
                provider: provider.to_string(),
                source: e,
            })
        })))
    }
}

#[async_trait]
impl BaseChatModel for OpenAICompatibleChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(|e| self.map_err(e))
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        OpenAICompatibleChat::stream_chat(self, messages, config).await
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        Some(Box::new(self.bind_tools(tools)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)] // fields exist only to exercise schema generation
    struct City {
        name: String,
        population: u64,
    }

    fn presets() -> Vec<(&'static str, OpenAICompatibleConfig, &'static str)> {
        vec![
            (
                "groq",
                OpenAICompatibleConfig::groq("k", "llama-3.3-70b-versatile"),
                GROQ_BASE_URL,
            ),
            (
                "openrouter",
                OpenAICompatibleConfig::openrouter("k", "anthropic/claude-sonnet-4-5"),
                OPENROUTER_BASE_URL,
            ),
            (
                "xai",
                OpenAICompatibleConfig::xai("k", "grok-4"),
                XAI_BASE_URL,
            ),
        ]
    }

    #[test]
    fn client_carries_endpoint_model_and_label() {
        let config = OpenAICompatibleConfig::new("http://127.0.0.1:8000/v1/", "vllm-model");
        let chat = OpenAICompatibleChat::new(config);
        assert_eq!(chat.model_name(), "vllm-model");
        assert_eq!(chat.provider_label(), "openai-compatible");
        assert_eq!(chat.inner.config.base_url, "http://127.0.0.1:8000/v1");
        assert!(!chat.inner.config.send_auth);
    }

    #[test]
    fn sampling_builders_return_same_preset() {
        let chat = OpenAICompatibleChat::new(OpenAICompatibleConfig::groq("k", "m"))
            .with_temperature(0.2)
            .with_max_tokens(128);
        assert_eq!(chat.temperature(), Some(0.2));
        assert_eq!(chat.max_tokens(), Some(128));
        assert_eq!(chat.provider_label(), "groq");
    }

    #[test]
    fn bind_tools_binds_and_keeps_preset() {
        let chat =
            OpenAICompatibleChat::new(OpenAICompatibleConfig::openrouter("k", "openai/gpt-5"));
        let tool = ToolDefinition::new("search", "Search the web");
        let bound = chat.bind_tools(vec![tool]);
        assert_eq!(bound.provider_label(), "openrouter");
        assert_eq!(bound.inner.config.tools.as_ref().unwrap().len(), 1);
        assert_eq!(bound.model_name(), "openai/gpt-5");
    }

    #[test]
    fn structured_output_snapshot_per_preset() {
        for (label, config, base_url) in presets() {
            let chat = OpenAICompatibleChat::new(config);
            let method = chat.with_structured_output::<City>();
            assert_eq!(method.config.model, chat.model_name(), "{label}");
            assert_eq!(method.config.base_url, base_url, "{label}");
            let tools = method.config.tools.as_ref().unwrap();
            assert_eq!(tools.len(), 1, "{label}");
            assert_eq!(tools[0].function.name, "structured_output");
        }
    }

    #[test]
    fn json_schema_output_sets_response_format() {
        let chat = OpenAICompatibleChat::new(OpenAICompatibleConfig::xai("k", "grok-4"));
        let method = chat.with_json_schema_output::<City>();
        assert!(method.config.response_format.is_some());
        assert!(method.config.tools.is_none());
    }

    #[test]
    fn works_as_boxed_base_chat_model() {
        let chat: Box<dyn BaseChatModel<Error = ProviderError> + Send + Sync> = Box::new(
            OpenAICompatibleChat::new(OpenAICompatibleConfig::groq("k", DEFAULT_GROQ_MODEL)),
        );
        assert_eq!(chat.model_name(), DEFAULT_GROQ_MODEL);
        let bound = chat
            .bind_tools(vec![ToolDefinition::new("echo", "echo")])
            .unwrap();
        assert_eq!(bound.model_name(), DEFAULT_GROQ_MODEL);
    }

    #[test]
    fn error_label_mentions_preset() {
        let chat = OpenAICompatibleChat::new(OpenAICompatibleConfig::groq("k", "m"));
        let err = chat.map_err(OpenAIError::Api("HTTP 404: nope".into()));
        let text = err.to_string();
        assert!(text.contains("groq"), "{text}");
        assert!(text.contains("HTTP 404"), "{text}");
    }

    #[tokio::test]
    #[ignore = "hits a live OpenAI-compatible endpoint; set LANGCHAINRUST_TEST_OPENAI_COMPAT_BASE_URL and LANGCHAINRUST_TEST_OPENAI_COMPAT_MODEL (optional LANGCHAINRUST_TEST_OPENAI_COMPAT_API_KEY)"]
    async fn live_endpoint_connectivity_smoke() {
        let base_url = std::env::var("LANGCHAINRUST_TEST_OPENAI_COMPAT_BASE_URL")
            .expect("LANGCHAINRUST_TEST_OPENAI_COMPAT_BASE_URL");
        let model = std::env::var("LANGCHAINRUST_TEST_OPENAI_COMPAT_MODEL")
            .expect("LANGCHAINRUST_TEST_OPENAI_COMPAT_MODEL");
        let mut config = OpenAICompatibleConfig::new(base_url, model);
        if let Ok(key) = std::env::var("LANGCHAINRUST_TEST_OPENAI_COMPAT_API_KEY") {
            config = config.with_api_key(key);
        }
        let chat = OpenAICompatibleChat::new(config);
        let result = chat
            .chat(vec![Message::human("Reply with exactly: pong")], None)
            .await
            .expect("live endpoint responds");
        assert!(!result.content.trim().is_empty());
    }
}
