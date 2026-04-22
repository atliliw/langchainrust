// src/language_models/providers/qwen.rs
//! Alibaba Qwen (通义千问) API 实现 (OpenAI 兼容)

use crate::language_models::openai::{OpenAIChat, OpenAIConfig};
use std::env;

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

    /// Creates a QwenConfig from environment variables.
    pub fn from_env() -> Self {
        let api_key = env::var("QWEN_API_KEY").expect("QWEN_API_KEY environment variable not set");

        let base_url = env::var("QWEN_BASE_URL").unwrap_or_else(|_| QWEN_BASE_URL.to_string());

        let model = env::var("QWEN_MODEL").unwrap_or_else(|_| "qwen-plus".to_string());

        Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        }
    }

    /// Sets the model name (e.g., qwen-plus, qwen-max).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
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
pub struct QwenChat {
    inner: OpenAIChat,
}

impl QwenChat {
    /// Creates a QwenChat with the given configuration.
    pub fn new(config: QwenConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    /// Creates a QwenChat from environment variables.
    pub fn from_env() -> Self {
        Self::new(QwenConfig::from_env())
    }

    /// Creates a QwenChat with a specific model.
    pub fn with_model(model: impl Into<String>) -> Self {
        let config = QwenConfig::from_env().with_model(model);
        Self::new(config)
    }
}

impl std::ops::Deref for QwenChat {
    type Target = OpenAIChat;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for QwenChat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
