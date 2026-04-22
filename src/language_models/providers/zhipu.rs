// src/language_models/providers/zhipu.rs
//! Zhipu GLM API 实现 (OpenAI 兼容)

use crate::language_models::openai::{OpenAIChat, OpenAIConfig};
use std::env;

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
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
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

    /// Creates a ZhipuConfig from environment variables.
    pub fn from_env() -> Self {
        let api_key =
            env::var("ZHIPU_API_KEY").expect("ZHIPU_API_KEY environment variable not set");

        let base_url = env::var("ZHIPU_BASE_URL").unwrap_or_else(|_| ZHIPU_BASE_URL.to_string());

        let model = env::var("ZHIPU_MODEL").unwrap_or_else(|_| "glm-4-flash".to_string());

        Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        }
    }

    /// Sets the model name (e.g., glm-4, glm-4-flash).
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

/// Zhipu 聊天客户端
pub struct ZhipuChat {
    inner: OpenAIChat,
}

impl ZhipuChat {
    /// Creates a ZhipuChat with the given configuration.
    pub fn new(config: ZhipuConfig) -> Self {
        Self {
            inner: OpenAIChat::new(config.into_openai_config()),
        }
    }

    /// Creates a ZhipuChat from environment variables.
    pub fn from_env() -> Self {
        Self::new(ZhipuConfig::from_env())
    }

    /// Creates a ZhipuChat with a specific model.
    pub fn with_model(model: impl Into<String>) -> Self {
        let config = ZhipuConfig::from_env().with_model(model);
        Self::new(config)
    }
}

impl std::ops::Deref for ZhipuChat {
    type Target = OpenAIChat;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for ZhipuChat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
