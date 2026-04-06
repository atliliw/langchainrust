// src/language_models/openai/config.rs
//! OpenAI 配置结构
//!
//! 参考 Python 版本: langchain/libs/partners/openai/langchain_openai/chat_models/base.py

use std::env;

/// OpenAI 配置
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API 密钥 (从环境变量或显式设置)
    pub api_key: String,

    /// 基础 URL (默认: https://api.openai.com/v1)
    pub base_url: String,

    /// 模型名称 (gpt-4, gpt-3.5-turbo 等)
    pub model: String,

    /// 温度 (0.0 - 2.0)
    pub temperature: Option<f32>,

    /// 最大 token 数
    pub max_tokens: Option<usize>,

    /// Top P 采样
    pub top_p: Option<f32>,

    /// 频率惩罚
    pub frequency_penalty: Option<f32>,

    /// 存在惩罚
    pub presence_penalty: Option<f32>,

    /// 是否启用流式
    pub streaming: bool,

    /// 组织 ID
    pub organization: Option<String>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            streaming: false,
            organization: None,
        }
    }
}

impl OpenAIConfig {
    /// 创建新配置
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// 从环境变量创建
    pub fn from_env() -> Self {
        Self::default()
    }

    /// 设置模型
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// 设置基础 URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// 设置温度
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// 设置最大 token 数
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// 启用流式
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// 设置组织 ID
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
}
