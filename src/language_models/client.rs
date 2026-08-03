// src/language_models/client.rs
//! LLMClient — 零配置切换 Provider 的统一入口
//!
//! 提供三种创建方式：
//! 1. `from_env()` — 自动检测环境变量
//! 2. `openai(config)` / `anthropic(config)` 等 — 显式传 Config
//! 3. `openai(OpenAIConfig::from_env_result()?)` — 从环境读配置，再覆盖参数
//!
//! # Example
//!
//! ```ignore
//! // 方式 1：自动检测
//! let llm = LLMClient::from_env()?;
//!
//! // 方式 2：显式传 Config
//! let llm = LLMClient::openai(OpenAIConfig::new("sk-...").with_model("gpt-4o"));
//!
//! // 方式 3：从环境读 + 覆盖
//! let llm = LLMClient::openai(OpenAIConfig::from_env_result()?.with_model("gpt-4o"));
//! ```

use crate::core::language_models::{BaseChatModel, LLMResult};
use crate::error::Error;
use crate::schema::Message;
use crate::RunnableConfig;
use async_trait::async_trait;
use std::sync::Arc;

/// LLM Client 统一入口
///
/// 包装任意 `BaseChatModel` 为 `Arc<dyn BaseChatModel<Error = Error>>`，
/// 提供零配置自动检测和显式构造两种方式。
///
/// 实现了 `Deref<Target = dyn BaseChatModel>`，可以直接调用 `.chat()` 等方法。
pub struct LLMClient {
    inner: Arc<dyn BaseChatModel<Error = Error> + Send + Sync>,
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
    // 自动检测
    // -----------------------------------------------------------------------

    /// 从环境变量自动检测并创建 LLM Client
    ///
    /// 检测优先级：
    /// 1. `OPENAI_API_KEY` → OpenAIChat
    /// 2. `ANTHROPIC_API_KEY` → AnthropicChat
    /// 3. `OLLAMA_BASE_URL` → OllamaChat
    ///
    /// # Errors
    ///
    /// 如果没有任何已知的环境变量，返回错误。
    pub fn from_env() -> Result<Self, String> {
        // 检测优先级 1: OpenAI
        if std::env::var("OPENAI_API_KEY").is_ok() {
            let config = crate::language_models::openai::OpenAIConfig::from_env_result()?;
            return Ok(Self::openai(config));
        }

        // 检测优先级 2: Anthropic
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            let config =
                crate::language_models::providers::anthropic::config::AnthropicConfig::from_env_result()?;
            return Ok(Self::anthropic(config));
        }

        // 检测优先级 3: Ollama
        if std::env::var("OLLAMA_BASE_URL").is_ok() {
            let config = crate::language_models::ollama::config::OllamaConfig::from_env_result()?;
            return Ok(Self::ollama(config));
        }

        Err(
            "No LLM provider detected. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL"
                .to_string(),
        )
    }

    // -----------------------------------------------------------------------
    // 显式构造
    // -----------------------------------------------------------------------

    /// 创建 OpenAI Client
    pub fn openai(config: crate::language_models::openai::OpenAIConfig) -> Self {
        let llm = crate::language_models::OpenAIChat::new(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 创建 Anthropic Client
    pub fn anthropic(
        config: crate::language_models::providers::anthropic::config::AnthropicConfig,
    ) -> Self {
        let llm = crate::language_models::providers::anthropic::AnthropicChat::new(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 创建 Ollama Client
    pub fn ollama(config: crate::language_models::ollama::config::OllamaConfig) -> Self {
        let llm = crate::language_models::ollama::chat::OllamaChat::with_config(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 创建 Gemini Client
    pub fn gemini(config: crate::language_models::providers::gemini::GeminiConfig) -> Self {
        let llm = crate::language_models::providers::gemini::GeminiChat::new(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 创建 DeepSeek Client
    pub fn deepseek(config: crate::language_models::providers::deepseek::DeepSeekConfig) -> Self {
        let llm = crate::language_models::providers::deepseek::DeepSeekChat::new(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 创建 Qwen Client
    pub fn qwen(config: crate::language_models::providers::qwen::QwenConfig) -> Self {
        let llm = crate::language_models::providers::qwen::QwenChat::new(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 创建 Moonshot Client
    pub fn moonshot(config: crate::language_models::providers::moonshot::MoonshotConfig) -> Self {
        let llm = crate::language_models::providers::moonshot::MoonshotChat::new(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 创建 Zhipu Client
    pub fn zhipu(config: crate::language_models::providers::zhipu::ZhipuConfig) -> Self {
        let llm = crate::language_models::providers::zhipu::ZhipuChat::new(config);
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    // -----------------------------------------------------------------------
    // 通用构造
    // -----------------------------------------------------------------------

    /// 从任意 `BaseChatModel` 创建 Client
    pub fn from_llm<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<Error>,
    {
        Self {
            inner: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 从 `Arc<dyn BaseChatModel>` 创建 Client
    pub fn from_arc(llm: Arc<dyn BaseChatModel<Error = Error> + Send + Sync>) -> Self {
        Self { inner: llm }
    }

    // -----------------------------------------------------------------------
    // 访问底层
    // -----------------------------------------------------------------------

    /// 获取内部 `Arc<dyn BaseChatModel>`，可直接传给 Agent
    pub fn into_inner(self) -> Arc<dyn BaseChatModel<Error = Error> + Send + Sync> {
        self.inner
    }

    /// 获取内部引用
    pub fn inner(&self) -> &Arc<dyn BaseChatModel<Error = Error> + Send + Sync> {
        &self.inner
    }
}

// LLMClient 实现完整的 trait 层次: Runnable → BaseLanguageModel → BaseChatModel

use crate::core::language_models::BaseLanguageModel;
use crate::core::runnables::Runnable;
use futures_util::Stream;

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for LLMClient {
    type Error = Error;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Error> {
        self.inner.invoke(input, config).await
    }

    async fn batch(
        &self,
        inputs: Vec<Vec<Message>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<LLMResult>, Error> {
        self.inner.batch(inputs, config).await
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<LLMResult, Error>> + Send>>, Error> {
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
    ) -> Result<LLMResult, Error> {
        self.inner.chat(messages, config).await
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>, Error>
    {
        self.inner.stream_chat(messages, config).await
    }

    fn bind_tools(
        &self,
        tools: Vec<crate::core::tools::ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Error> + Send + Sync>> {
        self.inner.bind_tools(tools)
    }
}

impl std::ops::Deref for LLMClient {
    type Target = dyn BaseChatModel<Error = Error> + Send + Sync;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::{OpenAIChat, OpenAIConfig};

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
        let arc = crate::core::language_models::wrap_chat_model(OpenAIChat::new(config));
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
        // 清除环境变量，确保 from_env 报错
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
        // 可以直接调用 BaseChatModel 的方法
        let _name = client.model_name();
    }
}
