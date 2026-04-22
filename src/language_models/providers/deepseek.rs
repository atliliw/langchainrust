// src/language_models/providers/deepseek.rs
//! DeepSeek API 实现 (OpenAI 兼容)

use crate::language_models::openai::{OpenAIChat, OpenAIConfig};
use std::env;

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
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
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

    /// Creates a DeepSeekConfig from environment variables.
    /// Reads DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL.
    pub fn from_env() -> Self {
        let api_key =
            env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY environment variable not set");

        let base_url =
            env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| DEEPSEEK_BASE_URL.to_string());

        let model = env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string());

        Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        }
    }

    /// Sets the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
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
pub struct DeepSeekChat {
    inner: OpenAIChat,
}

impl DeepSeekChat {
    /// Creates a DeepSeekChat with the given configuration.
    pub fn new(config: DeepSeekConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    /// Creates a DeepSeekChat from environment variables.
    pub fn from_env() -> Self {
        Self::new(DeepSeekConfig::from_env())
    }

    /// Creates a DeepSeekChat with a specific model.
    pub fn with_model(model: impl Into<String>) -> Self {
        let config = DeepSeekConfig::from_env().with_model(model);
        Self::new(config)
    }
}

impl std::ops::Deref for DeepSeekChat {
    type Target = OpenAIChat;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for DeepSeekChat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
