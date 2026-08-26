// src/language_models/providers/deepseek.rs
//! DeepSeek API 实现 (OpenAI 兼容)

use crate::error::ProviderError;
use crate::openai::{OpenAIChat, OpenAIConfig, OpenAIError, StructuredOutputMethod};
use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::env;
use std::pin::Pin;

/// DeepSeek API 端点
pub const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";

/// DeepSeek 模型列表
pub const DEEPSEEK_MODELS: [&str; 4] = [
    "deepseek-chat",     // 通用对话模型
    "deepseek-coder",    // 代码专用模型
    "deepseek-reasoner", // 推理模型 (R1)
    "deepseek-v3",       // V3 版本
];

/// DeepSeek 配置
#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    /// DeepSeek API key.
    pub api_key: String,
    /// Base URL of the DeepSeek API endpoint.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: DEEPSEEK_BASE_URL.to_string(),
            model: "deepseek-chat".to_string(),
            temperature: None,
            max_tokens: None,
        }
    }
}

impl DeepSeekConfig {
    /// Creates a new DeepSeekConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a DeepSeekConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `DEEPSEEK_API_KEY`: API key (required)
    /// - `DEEPSEEK_BASE_URL`: API endpoint (optional)
    /// - `DEEPSEEK_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let api_key = env::var("DEEPSEEK_API_KEY").map_err(|_| {
            ProviderError::Config("DEEPSEEK_API_KEY environment variable not set".to_string())
        })?;

        let base_url =
            env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| DEEPSEEK_BASE_URL.to_string());

        let model = env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
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

    /// 转换为 OpenAI 配置 (复用 OpenAI 实现)
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

/// DeepSeek 聊天客户端
#[derive(Clone)]
pub struct DeepSeekChat {
    inner: OpenAIChat,
}

impl std::fmt::Debug for DeepSeekChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekChat").finish_non_exhaustive()
    }
}

impl DeepSeekChat {
    /// Creates a DeepSeekChat with the given configuration.
    pub fn new(config: DeepSeekConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    /// Creates a DeepSeekChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, ProviderError> {
        Ok(Self::new(DeepSeekConfig::from_env_result()?))
    }
}

impl DeepSeekChat {
    /// Delegate chat to inner OpenAIChat
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, OpenAIError> {
        self.inner.chat(messages, config).await
    }

    /// Delegate stream_chat to inner OpenAIChat
    pub async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>>, OpenAIError>
    {
        self.inner.stream_chat(messages, config).await
    }

    /// Delegate bind_tools to inner OpenAIChat
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        Self {
            inner: self.inner.bind_tools(tools),
        }
    }

    /// L3 fix: delegate with_tool_choice to inner OpenAIChat
    pub fn with_tool_choice(self, choice: impl Into<String>) -> Self {
        Self {
            inner: self.inner.with_tool_choice(choice),
        }
    }

    /// Delegate with_structured_output to inner OpenAIChat
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> StructuredOutputMethod<T> {
        self.inner.with_structured_output()
    }
}

// H8: Implement BaseChatModel for DeepSeekChat
#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for DeepSeekChat {
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
impl Runnable<Vec<Message>, LLMResult> for DeepSeekChat {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .invoke(input, config)
            .await
            .map_err(ProviderError::DeepSeek)
    }

    // H6 fix: override stream() to delegate to inner OpenAIChat,
    // enabling true per-token streaming instead of default single-shot.
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
            .map_err(ProviderError::DeepSeek)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::DeepSeek))))
    }
}

#[async_trait]
impl BaseChatModel for DeepSeekChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(ProviderError::DeepSeek)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;
        let stream = self
            .inner
            .stream_chat(messages, config)
            .await
            .map_err(ProviderError::DeepSeek)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::DeepSeek))))
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
