// lc-providers/src/providers/cohere/config.rs
//! Cohere configuration.

use std::env;

use crate::ProviderError;
use lc_core::tools::ToolDefinition;

/// Cohere API endpoint.
pub const COHERE_BASE_URL: &str = "https://api.cohere.com/v2";

/// Cohere model list.
pub const COHERE_MODELS: [&str; 4] = ["command-r-plus", "command-r", "command", "command-light"];

/// Cohere configuration.
#[derive(Clone)]
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
    /// Bound tool definitions for function calling (0.25.0: the Cohere
    /// implementation previously never sent `tools`).
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice strategy. Cohere v2 accepts `NONE`/`AUTO`/`ANY` (an
    /// OpenAI-style `{"type":"function",...}` object also works on the wire,
    /// but this configuration passes the string through verbatim).
    pub tool_choice: Option<String>,
}

impl std::fmt::Debug for CohereConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A13: redact the API key so `{:?}` never leaks the secret.
        f.debug_struct("CohereConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("preamble", &self.preamble)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .finish()
    }
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
            tools: None,
            tool_choice: None,
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

    /// Sets the tool choice strategy (`NONE`/`AUTO`/`ANY` on Cohere v2).
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.tool_choice = Some(choice.into());
        self
    }
}
