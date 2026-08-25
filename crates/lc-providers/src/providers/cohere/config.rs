// lc-providers/src/providers/cohere/config.rs
//! Cohere configuration.

use std::env;

use crate::ProviderError;

/// Cohere API endpoint.
pub const COHERE_BASE_URL: &str = "https://api.cohere.com/v2";

/// Cohere model list.
pub const COHERE_MODELS: [&str; 4] = ["command-r-plus", "command-r", "command", "command-light"];

/// Cohere configuration.
#[derive(Debug, Clone)]
pub struct CohereConfig {
    /// Cohere API key.
    pub api_key: String,
    /// Base URL of the Cohere API endpoint.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
    /// Optional system preamble for the model.
    pub preamble: Option<String>,
}

impl Default for CohereConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: COHERE_BASE_URL.to_string(),
            model: "command-r-plus".to_string(),
            temperature: None,
            max_tokens: None,
            preamble: None,
        }
    }
}

impl CohereConfig {
    /// Creates a new CohereConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a CohereConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `COHERE_API_KEY`: API key (required)
    /// - `COHERE_BASE_URL`: API endpoint (optional)
    /// - `COHERE_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let api_key = env::var("COHERE_API_KEY").map_err(|_| {
            ProviderError::Config("COHERE_API_KEY environment variable not set".to_string())
        })?;

        let base_url = env::var("COHERE_BASE_URL").unwrap_or_else(|_| COHERE_BASE_URL.to_string());

        let model = env::var("COHERE_MODEL").unwrap_or_else(|_| "command-r-plus".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Sets the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the max tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Sets the preamble (system prompt).
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }
}
