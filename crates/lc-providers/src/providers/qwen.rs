// src/language_models/providers/qwen.rs
//! Alibaba Qwen (通义千问) API 实现 (OpenAI 兼容)

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

/// Qwen API 端点 (DashScope)
pub const QWEN_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/// Qwen 模型列表
pub const QWEN_MODELS: [&str; 6] = [
    "qwen-turbo",           // 快速版
    "qwen-plus",            // Plus 版本
    "qwen-max",             // Max 版本
    "qwen-max-longcontext", // 长文本
    "qwen2.5-72b-instruct", // Qwen2.5 开源版
    "qwen-coder-plus",      // 代码专用
];

/// Qwen 配置
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: QWEN_BASE_URL.to_string(),
            model: "qwen-plus".to_string(),
            temperature: None,
            max_tokens: None,
        }
    }
}

impl QwenConfig {
    /// Creates a new QwenConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a QwenConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `QWEN_API_KEY`: API key (required)
    /// - `QWEN_BASE_URL`: API endpoint (optional)
    /// - `QWEN_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = env::var("QWEN_API_KEY")
            .map_err(|_| "QWEN_API_KEY environment variable not set".to_string())?;

        let base_url = env::var("QWEN_BASE_URL").unwrap_or_else(|_| QWEN_BASE_URL.to_string());

        let model = env::var("QWEN_MODEL").unwrap_or_else(|_| "qwen-plus".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Sets the model name (e.g., qwen-plus, qwen-max).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

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

/// Qwen 聊天客户端
#[derive(Clone)]
pub struct QwenChat {
    inner: OpenAIChat,
}

impl std::fmt::Debug for QwenChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenChat").finish_non_exhaustive()
    }
}

impl QwenChat {
    /// Creates a QwenChat with the given configuration.
    pub fn new(config: QwenConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    /// Creates a QwenChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, String> {
        Ok(Self::new(QwenConfig::from_env_result()?))
    }
}

impl QwenChat {
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

// H8: Implement BaseChatModel for QwenChat
#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for QwenChat {
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
impl Runnable<Vec<Message>, LLMResult> for QwenChat {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .invoke(input, config)
            .await
            .map_err(ProviderError::Qwen)
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
            .map_err(ProviderError::Qwen)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::Qwen))))
    }
}

#[async_trait]
impl BaseChatModel for QwenChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.inner
            .chat(messages, config)
            .await
            .map_err(ProviderError::Qwen)
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
            .map_err(ProviderError::Qwen)?;
        Ok(Box::pin(stream.map(|r| r.map_err(ProviderError::Qwen))))
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
