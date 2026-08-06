// lc-providers/src/openai/config.rs
//! OpenAI configuration types

use lc_core::tools::ToolDefinition;
use std::env;

/// OpenAI configuration
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
    /// Create new configuration
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Create configuration from environment variables
    ///
    /// Environment variables:
    /// - `OPENAI_API_KEY`: API key (required)
    /// - `OPENAI_BASE_URL`: API endpoint (optional, default: <https://api.openai.com/v1>)
    /// - `OPENAI_MODEL`: Model name (optional, default: gpt-3.5-turbo)
    #[deprecated(
        since = "0.5.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    pub fn from_env() -> Self {
        Self::from_env_result().unwrap_or_else(|_| Self::default())
    }

    /// Create configuration from environment variables, returning Result
    ///
    /// Environment variables:
    /// - `OPENAI_API_KEY`: API key (required)
    /// - `OPENAI_BASE_URL`: API endpoint (optional, default: <https://api.openai.com/v1>)
    /// - `OPENAI_MODEL`: Model name (optional, default: gpt-3.5-turbo)
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

    /// Set model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Set organization ID
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

#[cfg(test)]
mod tests {
    use super::*;
    
    use std::env;

    fn save_and_set(key: &str, value: &str) -> Option<String> {
        let old = env::var(key).ok();
        env::set_var(key, value);
        old
    }

    fn restore(key: &str, old: Option<String>) {
        match old {
            Some(v) => env::set_var(key, v),
            None => env::remove_var(key),
        }
    }

    #[test]
    fn test_from_env_result_ok_when_key_set() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("OPENAI_API_KEY", "test-key-123");
        let result = OpenAIConfig::from_env_result();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().api_key, "test-key-123");
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");
        let result = OpenAIConfig::from_env_result();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("OPENAI_API_KEY"));
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_uses_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("OPENAI_API_KEY", "key");
        let old_url = save_and_set("OPENAI_BASE_URL", "https://custom.api.com/v1");
        let old_model = save_and_set("OPENAI_MODEL", "gpt-4");
        let config = OpenAIConfig::from_env_result().unwrap();
        assert_eq!(config.base_url, "https://custom.api.com/v1");
        assert_eq!(config.model, "gpt-4");
        restore("OPENAI_API_KEY", old_key);
        restore("OPENAI_BASE_URL", old_url);
        restore("OPENAI_MODEL", old_model);
    }

    #[test]
    fn test_from_env_result_uses_defaults_for_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("OPENAI_API_KEY", "key");
        let old_url = env::var("OPENAI_BASE_URL").ok();
        env::remove_var("OPENAI_BASE_URL");
        let old_model = env::var("OPENAI_MODEL").ok();
        env::remove_var("OPENAI_MODEL");
        let config = OpenAIConfig::from_env_result().unwrap();
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.model, "gpt-3.5-turbo");
        restore("OPENAI_API_KEY", old_key);
        restore("OPENAI_BASE_URL", old_url);
        restore("OPENAI_MODEL", old_model);
    }
}
