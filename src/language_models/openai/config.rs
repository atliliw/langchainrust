// src/language_models/openai/config.rs
//! OpenAI 配置结构

use crate::core::tools::ToolDefinition;
use std::env;

/// OpenAI 配置
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub streaming: bool,
    pub organization: Option<String>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<String>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            temperature: None,
            max_tokens: None,
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

impl OpenAIConfig {
    /// 创建新配置
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// 从环境变量创建配置
    ///
    /// 环境变量:
    /// - `OPENAI_API_KEY`: API 密钥 (必需)
    /// - `OPENAI_BASE_URL`: API 端点 (可选，默认: https://api.openai.com/v1)
    /// - `OPENAI_MODEL`: 模型名称 (可选，默认: gpt-3.5-turbo)
    #[deprecated(
        since = "0.5.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    pub fn from_env() -> Self {
        Self::from_env_result().unwrap_or_else(|_| Self::default())
    }

    /// 从环境变量创建配置，返回 Result
    ///
    /// 环境变量:
    /// - `OPENAI_API_KEY`: API 密钥 (必需)
    /// - `OPENAI_BASE_URL`: API 端点 (可选，默认: https://api.openai.com/v1)
    /// - `OPENAI_MODEL`: 模型名称 (可选，默认: gpt-3.5-turbo)
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY environment variable not set".to_string())?;

        let base_url =
            env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Self::default()
        })
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

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.tool_choice = Some(choice.into());
        self
    }
}
