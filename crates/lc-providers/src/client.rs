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
use crate::providers::azure::AzureOpenAIChat;
use crate::providers::azure::AzureOpenAIConfig;
use crate::providers::cohere::CohereChat;
use crate::providers::cohere::CohereConfig;
use crate::providers::deepseek::DeepSeekChat;
use crate::providers::deepseek::DeepSeekConfig;
use crate::providers::gemini::GeminiChat;
use crate::providers::gemini::GeminiConfig;
use crate::providers::mistral::MistralChat;
use crate::providers::mistral::MistralConfig;
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
use std::sync::{Arc, Mutex};

/// Per-client sampling overrides (providers Q2).
///
/// Stored on `LLMClient` so the consuming `with_temperature` /
/// `with_max_tokens` builders can affect later `chat` / `stream_chat` calls
/// even though the wrapped model sits behind a trait object. The overrides
/// are merged into the `RunnableConfig` per call.
#[derive(Debug, Default)]
struct ClientOverrides {
    temperature: Option<f32>,
    max_tokens: Option<usize>,
}

/// LLM Client unified entry point
///
/// Wraps any `BaseChatModel` as `Arc<dyn BaseChatModel<Error = ProviderError>>`,
/// providing zero-config auto-detection and explicit construction.
///
/// Implements `Deref<Target = dyn BaseChatModel>`, so you can call `.chat()` etc. directly.
pub struct LLMClient {
    inner: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    overrides: Mutex<ClientOverrides>,
}

impl LLMClient {
    /// Wrap an already-normalized `Arc<dyn BaseChatModel>` with fresh overrides.
    fn from_inner(inner: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>) -> Self {
        Self {
            inner,
            overrides: Mutex::new(ClientOverrides::default()),
        }
    }

    /// Merge per-client sampling overrides into the invocation config.
    fn apply_overrides(&self, config: Option<RunnableConfig>) -> Option<RunnableConfig> {
        let overrides = self.overrides.lock().unwrap_or_else(|e| e.into_inner());
        if overrides.temperature.is_none() && overrides.max_tokens.is_none() {
            return config;
        }
        let mut cfg = config.unwrap_or_default();
        cfg.temperature = overrides.temperature.or(cfg.temperature);
        cfg.max_tokens = overrides.max_tokens.or(cfg.max_tokens);
        Some(cfg)
    }
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
    /// 3. `AZURE_OPENAI_API_KEY` -> AzureOpenAIChat
    /// 4. `DEEPSEEK_API_KEY` -> DeepSeekChat
    /// 5. `QWEN_API_KEY` -> QwenChat
    /// 6. `MOONSHOT_API_KEY` -> MoonshotChat
    /// 7. `ZHIPU_API_KEY` -> ZhipuChat
    /// 8. `MISTRAL_API_KEY` -> MistralChat
    /// 9. `COHERE_API_KEY` -> CohereChat
    /// 10. `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) -> GeminiChat
    /// 11. `OLLAMA_BASE_URL` -> OllamaChat
    ///
    /// # Errors
    ///
    /// Returns error if no known environment variable is set.
    pub fn from_env() -> Result<Self, ProviderError> {
        // Priority 1: OpenAI
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return Ok(Self::openai(OpenAIConfig::from_env_result()?));
        }

        // Priority 2: Anthropic
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            return Ok(Self::anthropic(AnthropicConfig::from_env_result()?));
        }

        // Priority 3: Azure OpenAI (endpoint + deployment based)
        if std::env::var("AZURE_OPENAI_API_KEY").is_ok() {
            return Ok(Self::azure(AzureOpenAIConfig::from_env_result()?));
        }

        // Priority 4: DeepSeek
        if std::env::var("DEEPSEEK_API_KEY").is_ok() {
            return Ok(Self::deepseek(DeepSeekConfig::from_env_result()?));
        }

        // Priority 5: Qwen
        if std::env::var("QWEN_API_KEY").is_ok() {
            return Ok(Self::qwen(QwenConfig::from_env_result()?));
        }

        // Priority 6: Moonshot
        if std::env::var("MOONSHOT_API_KEY").is_ok() {
            return Ok(Self::moonshot(MoonshotConfig::from_env_result()?));
        }

        // Priority 7: Zhipu
        if std::env::var("ZHIPU_API_KEY").is_ok() {
            return Ok(Self::zhipu(ZhipuConfig::from_env_result()?));
        }

        // Priority 8: Mistral
        if std::env::var("MISTRAL_API_KEY").is_ok() {
            return Ok(Self::mistral(MistralConfig::from_env_result()?));
        }

        // Priority 9: Cohere
        if std::env::var("COHERE_API_KEY").is_ok() {
            return Ok(Self::cohere(CohereConfig::from_env_result()?));
        }

        // Priority 10: Gemini
        if std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_API_KEY").is_ok() {
            return Ok(Self::gemini(GeminiConfig::from_env_result()?));
        }

        // Priority 11: Ollama (local, no API key required)
        if std::env::var("OLLAMA_BASE_URL").is_ok() {
            return Ok(Self::ollama(OllamaConfig::from_env_result()?));
        }

        Err(ProviderError::Config(
            "No LLM provider detected. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, \
             AZURE_OPENAI_API_KEY, DEEPSEEK_API_KEY, QWEN_API_KEY, MOONSHOT_API_KEY, \
             ZHIPU_API_KEY, MISTRAL_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, OLLAMA_BASE_URL"
                .to_string(),
        ))
    }

    // -----------------------------------------------------------------------
    // Explicit construction
    // -----------------------------------------------------------------------

    /// Create OpenAI Client
    pub fn openai(config: OpenAIConfig) -> Self {
        let llm = OpenAIChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Anthropic Client
    pub fn anthropic(config: AnthropicConfig) -> Self {
        let llm = AnthropicChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Ollama Client
    pub fn ollama(config: OllamaConfig) -> Self {
        let llm = OllamaChat::with_config(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Gemini Client
    pub fn gemini(config: GeminiConfig) -> Self {
        let llm = GeminiChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create DeepSeek Client
    pub fn deepseek(config: DeepSeekConfig) -> Self {
        let llm = DeepSeekChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Qwen Client
    pub fn qwen(config: QwenConfig) -> Self {
        let llm = QwenChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Moonshot Client
    pub fn moonshot(config: MoonshotConfig) -> Self {
        let llm = MoonshotChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Zhipu Client
    pub fn zhipu(config: ZhipuConfig) -> Self {
        let llm = ZhipuChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Mistral Client
    pub fn mistral(config: MistralConfig) -> Self {
        let llm = MistralChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Azure OpenAI Client
    pub fn azure(config: AzureOpenAIConfig) -> Self {
        let llm = AzureOpenAIChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Cohere Client
    pub fn cohere(config: CohereConfig) -> Self {
        let llm = CohereChat::new(config);
        Self::from_inner(wrap_chat_model(llm))
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
        Self::from_inner(wrap_chat_model(llm))
    }

    /// Create Client from `Arc<dyn BaseChatModel>`
    pub fn from_arc(llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>) -> Self {
        Self::from_inner(llm)
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
        self.inner.invoke(input, self.apply_overrides(config)).await
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, ProviderError> {
        self.inner.batch(inputs, self.apply_overrides(config)).await
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<LLMResult, ProviderError>> + Send>>,
        ProviderError,
    > {
        self.inner.stream(input, self.apply_overrides(config)).await
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
        // Per-client override takes precedence over the wrapped model's own value.
        self.overrides
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .temperature
            .or_else(|| self.inner.temperature())
    }

    fn max_tokens(&self) -> Option<usize> {
        self.overrides
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .max_tokens
            .or_else(|| self.inner.max_tokens())
    }

    fn with_temperature(self, temp: f32) -> Self
    where
        Self: Sized,
    {
        // Store the override; `chat`/`stream_chat` merge it into the
        // `RunnableConfig` for the wrapped model (providers Q2).
        self.overrides
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .temperature = Some(temp);
        self
    }

    fn with_max_tokens(self, max: usize) -> Self
    where
        Self: Sized,
    {
        self.overrides
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .max_tokens = Some(max);
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
        self.inner
            .chat(messages, self.apply_overrides(config))
            .await
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>,
        ProviderError,
    > {
        self.inner
            .stream_chat(messages, self.apply_overrides(config))
            .await
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
    use crate::ENV_TEST_LOCK;

    /// Env vars that `LLMClient::from_env` checks, in detection order.
    const DETECTION_ENV_VARS: [&str; 11] = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "QWEN_API_KEY",
        "MOONSHOT_API_KEY",
        "ZHIPU_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "GEMINI_API_KEY",
        "OLLAMA_BASE_URL",
    ];

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
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Clear all detection vars, ensure from_env errors
        let saved: Vec<(&str, Option<String>)> = DETECTION_ENV_VARS
            .iter()
            .map(|k| {
                let old = std::env::var(k).ok();
                std::env::remove_var(k);
                (*k, old)
            })
            .collect();

        let result = LLMClient::from_env();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No LLM provider detected"));

        for (k, old) in saved {
            restore(k, old);
        }
    }

    #[test]
    fn test_from_env_detects_each_provider() {
        for key in DETECTION_ENV_VARS {
            let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            // Clear every detection var, then set exactly one.
            let saved: Vec<(&str, Option<String>)> = DETECTION_ENV_VARS
                .iter()
                .map(|k| {
                    let old = std::env::var(k).ok();
                    std::env::remove_var(k);
                    (*k, old)
                })
                .collect();

            let old = save_and_set(key, "test-value");
            // Azure also needs endpoint + deployment to build its config.
            let azure_extra: Vec<(&str, Option<String>)> = if key == "AZURE_OPENAI_API_KEY" {
                vec![
                    (
                        "AZURE_OPENAI_ENDPOINT",
                        save_and_set("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com"),
                    ),
                    (
                        "AZURE_OPENAI_DEPLOYMENT_NAME",
                        save_and_set("AZURE_OPENAI_DEPLOYMENT_NAME", "test-deployment"),
                    ),
                    (
                        "AZURE_OPENAI_API_VERSION",
                        save_and_set("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                    ),
                ]
            } else {
                vec![]
            };

            let result = LLMClient::from_env();
            assert!(result.is_ok(), "expected detection via {key}");

            restore(key, old);
            for (k, v) in azure_extra {
                restore(k, v);
            }
            for (k, old) in saved {
                restore(k, old);
            }
        }
    }

    #[test]
    fn test_with_temperature_override_applies_to_config() {
        let config = OpenAIConfig::new("test_key");
        let client = LLMClient::openai(config)
            .with_temperature(0.7)
            .with_max_tokens(128);

        // Getter reflects the per-client override (providers Q2).
        assert_eq!(client.temperature(), Some(0.7));
        assert_eq!(client.max_tokens(), Some(128));

        // The merged config carries the overrides to the wrapped model.
        let merged = client.apply_overrides(None).unwrap();
        assert_eq!(merged.temperature, Some(0.7));
        assert_eq!(merged.max_tokens, Some(128));

        // Per-client override takes precedence over a per-call config value.
        let cfg = RunnableConfig::default().with_temperature(0.2);
        let merged = client.apply_overrides(Some(cfg)).unwrap();
        assert_eq!(merged.temperature, Some(0.7));
        assert_eq!(merged.max_tokens, Some(128));
    }

    #[test]
    fn test_no_overrides_passes_config_through() {
        let config = OpenAIConfig::new("test_key");
        let client = LLMClient::openai(config);

        assert_eq!(client.temperature(), None);
        assert_eq!(client.max_tokens(), None);

        // No overrides: config is returned as-is (not cloned).
        let cfg = RunnableConfig::default().with_temperature(0.5);
        let merged = client.apply_overrides(Some(cfg.clone())).unwrap();
        assert_eq!(merged.temperature, Some(0.5));
        assert!(client.apply_overrides(None).is_none());
    }

    #[test]
    fn test_deref_works() {
        let config = OpenAIConfig::new("test_key");
        let client = LLMClient::openai(config);
        // Can directly call BaseChatModel methods
        let _name = client.model_name();
    }
}
