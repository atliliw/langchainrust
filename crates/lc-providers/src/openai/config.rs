// lc-providers/src/openai/config.rs
//! OpenAI configuration types

use lc_core::tools::ToolDefinition;
use std::env;

use crate::openai::response_format::ResponseFormat;
use crate::ProviderError;

/// A **non-exhaustive** hint list of current OpenAI model ids (B5, v0.22.4).
///
/// The `model` field accepts any string, so newer ids work without a crate
/// upgrade; this list only documents the 2025–2026 flagship line.
pub const OPENAI_MODELS: [&str; 6] = [
    "gpt-5",       // GPT-5 flagship
    "gpt-5-mini",  // GPT-5 balanced tier
    "gpt-4o",      // GPT-4o
    "gpt-4o-mini", // GPT-4o mini
    "o3",          // reasoning
    "o4-mini",     // small reasoning
];

/// OpenAI configuration
#[derive(Clone)]
pub struct OpenAIConfig {
    /// OpenAI API key.
    pub api_key: String,
    /// Base URL of the OpenAI-compatible API endpoint.
    pub base_url: String,
    /// Model name to use for chat completions.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
    /// Nucleus sampling probability mass.
    pub top_p: Option<f32>,
    /// Frequency penalty applied to repeated tokens.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty applied to already-seen tokens.
    pub presence_penalty: Option<f32>,
    /// Whether to stream responses.
    pub streaming: bool,
    /// Organization ID for the API request.
    pub organization: Option<String>,
    /// Tool definitions for function calling.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice strategy (e.g. "auto" or a function name).
    pub tool_choice: Option<String>,
    /// Response format constraint (0.21.0 S3.1): engine-side structured output
    /// (`json_object` / `json_schema`). `None` keeps the default text mode.
    pub response_format: Option<ResponseFormat>,
    /// B5: additional HTTP headers attached to every request (e.g. OpenRouter
    /// attribution headers, tenant headers on private gateways).
    pub extra_headers: Vec<(String, String)>,
    /// B5: whether to send the `Authorization: Bearer` header. Keyless local
    /// endpoints (LM Studio, vLLM, Ollama's OpenAI shim) set this to `false`.
    pub send_auth: bool,
}

impl std::fmt::Debug for OpenAIConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A13: redact the API key so `{:?}` on a config never leaks the secret.
        f.debug_struct("OpenAIConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("top_p", &self.top_p)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("presence_penalty", &self.presence_penalty)
            .field("streaming", &self.streaming)
            .field("organization", &self.organization)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .field("response_format", &self.response_format)
            .field("extra_headers", &self.extra_headers)
            .field("send_auth", &self.send_auth)
            .finish()
    }
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
            response_format: None,
            extra_headers: Vec::new(),
            send_auth: true,
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

    /// Create configuration from environment variables, returning Result
    ///
    /// Environment variables:
    /// - `OPENAI_API_KEY`: API key (required)
    /// - `OPENAI_BASE_URL`: API endpoint (optional, default: <https://api.openai.com/v1>)
    /// - `OPENAI_MODEL`: Model name (optional, default: gpt-3.5-turbo)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
            ProviderError::Config("OPENAI_API_KEY environment variable not set".to_string())
        })?;

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

    /// Set tool definitions for function calling.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set tool choice strategy.
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.tool_choice = Some(choice.into());
        self
    }

    /// Set the response format constraint (0.21.0 S3.1).
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// B5: append an extra HTTP header sent on every request.
    pub fn with_extra_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// B5: append multiple extra HTTP headers sent on every request.
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers.extend(headers);
        self
    }

    /// B5: control whether the `Authorization: Bearer` header is sent.
    ///
    /// Set to `false` for keyless local OpenAI-compatible endpoints
    /// (LM Studio, vLLM, Ollama's `/v1` shim).
    pub fn with_send_auth(mut self, send_auth: bool) -> Self {
        self.send_auth = send_auth;
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
        assert!(result.unwrap_err().to_string().contains("OPENAI_API_KEY"));
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
    fn test_debug_redacts_api_key() {
        let config = OpenAIConfig::new("sk-my-secret-key-123");
        let debug = format!("{:?}", config);
        assert!(
            !debug.contains("sk-my-secret-key-123"),
            "Debug output must not contain the raw API key"
        );
        assert!(debug.contains("***"));
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
