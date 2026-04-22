// src/language_models/providers/moonshot.rs
//! Moonshot (Kimi) API 实现 (OpenAI 兼容)

use crate::language_models::openai::{OpenAIChat, OpenAIConfig};
use std::env;

/// Moonshot API 端点
pub const MOONSHOT_BASE_URL: &str = "https://api.moonshot.cn/v1";

/// Moonshot 模型列表
pub const MOONSHOT_MODELS: [&str; 3] = [
    "moonshot-v1-8k",   // 8K 上下文
    "moonshot-v1-32k",  // 32K 上下文
    "moonshot-v1-128k", // 128K 长文本
];

/// Moonshot 配置
#[derive(Debug, Clone)]
pub struct MoonshotConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl Default for MoonshotConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: MOONSHOT_BASE_URL.to_string(),
            model: "moonshot-v1-8k".to_string(),
            temperature: None,
            max_tokens: None,
        }
    }
}

impl MoonshotConfig {
    /// Creates a new MoonshotConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a MoonshotConfig from environment variables.
    pub fn from_env() -> Self {
        let api_key =
            env::var("MOONSHOT_API_KEY").expect("MOONSHOT_API_KEY environment variable not set");

        let base_url =
            env::var("MOONSHOT_BASE_URL").unwrap_or_else(|_| MOONSHOT_BASE_URL.to_string());

        let model = env::var("MOONSHOT_MODEL").unwrap_or_else(|_| "moonshot-v1-8k".to_string());

        Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        }
    }

    /// Sets the model name (e.g., moonshot-v1-128k for long context).
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

/// Moonshot 聊天客户端
pub struct MoonshotChat {
    inner: OpenAIChat,
}

impl MoonshotChat {
    /// Creates a MoonshotChat with the given configuration.
    pub fn new(config: MoonshotConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    pub fn from_env() -> Self {
        Self::new(MoonshotConfig::from_env())
    }

    /// Creates a MoonshotChat with a specific model.
    pub fn with_model(model: impl Into<String>) -> Self {
        let config = MoonshotConfig::from_env().with_model(model);
        Self::new(config)
    }
}

impl std::ops::Deref for MoonshotChat {
    type Target = OpenAIChat;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for MoonshotChat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
