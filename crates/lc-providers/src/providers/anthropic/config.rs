// src/language_models/providers/anthropic/config.rs
//! Anthropic configuration types and constants.

use lc_core::tools::ToolDefinition;
use serde::{Deserialize, Serialize};
use std::env;

use crate::ProviderError;

/// Anthropic API endpoint.
pub const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Claude model list.
pub const CLAUDE_MODELS: [&str; 5] = [
    "claude-3-5-sonnet-20241022", // Claude 3.5 Sonnet
    "claude-3-5-haiku-20241022",  // Claude 3.5 Haiku
    "claude-3-opus-20240229",     // Claude 3 Opus
    "claude-3-sonnet-20240229",   // Claude 3 Sonnet
    "claude-3-haiku-20240307",    // Claude 3 Haiku
];

/// Type of extended thinking mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ThinkingType {
    /// Extended thinking enabled with a token budget.
    Enabled,
    /// Extended thinking disabled (default).
    #[default]
    Disabled,
}

/// Configuration for Anthropic extended thinking.
///
/// When enabled, the model emits a "thinking" content block before the
/// final text answer, allowing callers to observe the reasoning process.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    /// Maximum number of tokens the model may spend on thinking.
    pub budget_tokens: usize,
    /// Whether thinking is enabled or disabled.
    pub r#type: ThinkingType,
}

impl ThinkingConfig {
    /// Create a new enabled thinking config with the given budget.
    pub fn enabled(budget_tokens: usize) -> Self {
        Self {
            budget_tokens,
            r#type: ThinkingType::Enabled,
        }
    }

    /// Create a disabled thinking config.
    pub fn disabled() -> Self {
        Self {
            budget_tokens: 0,
            r#type: ThinkingType::Disabled,
        }
    }

    /// Returns true if thinking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.r#type == ThinkingType::Enabled
    }
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Anthropic Claude configuration.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Anthropic API key.
    pub api_key: String,
    /// Base URL of the Anthropic API endpoint.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Optional system prompt.
    pub system_prompt: Option<String>,
    /// Extended thinking configuration.
    pub thinking: ThinkingConfig,
    /// Tool definitions for function calling.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice strategy (e.g. "auto", "any", or a specific tool name).
    pub tool_choice: Option<String>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: ANTHROPIC_BASE_URL.to_string(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: 4096,
            temperature: None,
            system_prompt: None,
            thinking: ThinkingConfig::default(),
            tools: None,
            tool_choice: None,
        }
    }
}

impl AnthropicConfig {
    /// Creates a new AnthropicConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates an AnthropicConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `ANTHROPIC_API_KEY`: API key (required)
    /// - `ANTHROPIC_BASE_URL`: API endpoint (optional)
    /// - `ANTHROPIC_MODEL`: Model name (optional)
    /// - `ANTHROPIC_MAX_TOKENS`: Max tokens (optional)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| {
            ProviderError::Config("ANTHROPIC_API_KEY environment variable not set".to_string())
        })?;

        let base_url =
            env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| ANTHROPIC_BASE_URL.to_string());

        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());

        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        Ok(Self {
            api_key,
            base_url,
            model,
            max_tokens,
            ..Default::default()
        })
    }

    /// Sets the Claude model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the max tokens limit.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Sets the temperature parameter.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enables extended thinking with the given token budget.
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = thinking;
        self
    }
}
