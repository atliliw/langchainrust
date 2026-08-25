// src/language_models/providers/zhipu.rs
//! Zhipu GLM API 实现 (OpenAI 兼容)

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

/// Zhipu API 端点
pub const ZHIPU_BASE_URL: &str = "https://open.bigmodel.cn/api/paas/v4";

/// Zhipu GLM 模型列表
pub const ZHIPU_MODELS: [&str; 4] = [
    "glm-4",       // GLM-4 基础模型
    "glm-4-flash", // GLM-4 快速版
    "glm-4-plus",  // GLM-4 Plus
    "glm-4-long",  // GLM-4 长文本
];

/// Zhipu 配置
#[derive(Debug, Clone)]
pub struct ZhipuConfig {
    /// Zhipu API key.
    pub api_key: String,
    /// Base URL of the Zhipu (BigModel) API endpoint.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
}

impl Default for ZhipuConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: ZHIPU_BASE_URL.to_string(),
            model: "glm-4-flash".to_string(),
            temperature: None,
            max_tokens: None,
        }
    }
}

impl ZhipuConfig {
    /// Creates a new ZhipuConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a ZhipuConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `ZHIPU_API_KEY`: API key (required)
    /// - `ZHIPU_BASE_URL`: API endpoint (optional)
    /// - `ZHIPU_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let api_key = env::var("ZHIPU_API_KEY").map_err(|_| {
            ProviderError::Config("ZHIPU_API_KEY environment variable not set".to_string())
        })?;

        let base_url = env::var("ZHIPU_BASE_URL").unwrap_or_else(|_| ZHIPU_BASE_URL.to_string());

        let model = env::var("ZHIPU_MODEL").unwrap_or_else(|_| "glm-4-flash".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Sets the model name (e.g., glm-4, glm-4-flash).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of tokens to generate.
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

/// Zhipu 聊天客户端
#[derive(Clone)]
pub struct ZhipuChat {
    inner: OpenAIChat,
}

impl std::fmt::Debug for ZhipuChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZhipuChat").finish_non_exhaustive()
    }
}

impl ZhipuChat {
    /// Creates a ZhipuChat with the given configuration.
    pub fn new(config: ZhipuConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    /// Creates a ZhipuChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, ProviderError> {
        Ok(Self::new(ZhipuConfig::from_env_result()?))
    }

    /// Creates a ZhipuChat with a specific model.
    pub fn with_model(model: impl Into<String>) -> Result<Self, ProviderError> {
        let config = ZhipuConfig::from_env_result()?.with_model(model);
        Ok(Self::new(config))
    }
}

impl ZhipuChat {
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, OpenAIError>> + Send>>, OpenAIError> {
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

// H8: Implement BaseChatModel for ZhipuChat
#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for ZhipuChat {
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
impl Runnable<Vec<Message>, LLMResult> for ZhipuChat {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .invoke(input, config)
            .await
            .map_err(ProviderError::Zhipu)
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
            .map_err(ProviderError::Zhipu)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::Zhipu))))
    }
}

#[async_trait]
impl BaseChatModel for ZhipuChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(ProviderError::Zhipu)
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
            .map_err(ProviderError::Zhipu)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::Zhipu))))
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
