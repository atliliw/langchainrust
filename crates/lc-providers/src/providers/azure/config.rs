// lc-providers/src/providers/azure/config.rs
//! Azure OpenAI configuration.

use std::env;

use crate::ProviderError;

/// Azure OpenAI API version.
pub const AZURE_DEFAULT_API_VERSION: &str = "2024-02-15-preview";

/// Azure OpenAI configuration.
#[derive(Debug, Clone)]
pub struct AzureOpenAIConfig {
    /// Azure OpenAI resource endpoint (e.g., https://myresource.openai.azure.com).
    pub endpoint: String,
    /// Deployment name (e.g., "my-gpt4-deployment").
    pub deployment_name: String,
    /// API key for authentication.
    pub api_key: String,
    /// API version string (default: 2024-02-15-preview).
    pub api_version: String,
    /// Model name for LLMResult metadata (optional, defaults to deployment_name).
    pub model: Option<String>,
    /// Temperature for generation.
    pub temperature: Option<f32>,
    /// Maximum tokens for generation.
    pub max_tokens: Option<usize>,
    /// Top-p for nucleus sampling.
    pub top_p: Option<f32>,
}

impl AzureOpenAIConfig {
    /// Creates a new AzureOpenAIConfig.
    pub fn new(
        endpoint: impl Into<String>,
        deployment_name: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            deployment_name: deployment_name.into(),
            api_key: api_key.into(),
            api_version: AZURE_DEFAULT_API_VERSION.to_string(),
            model: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
        }
    }

    /// Creates an AzureOpenAIConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `AZURE_OPENAI_ENDPOINT`: Resource endpoint (required)
    /// - `AZURE_OPENAI_DEPLOYMENT_NAME`: Deployment name (required)
    /// - `AZURE_OPENAI_API_KEY`: API key (required)
    /// - `AZURE_OPENAI_API_VERSION`: API version (optional)
    /// - `AZURE_OPENAI_MODEL`: Model name override (optional)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let endpoint = env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| {
            ProviderError::Config("AZURE_OPENAI_ENDPOINT environment variable not set".to_string())
        })?;

        let deployment_name = env::var("AZURE_OPENAI_DEPLOYMENT_NAME").map_err(|_| {
            ProviderError::Config(
                "AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set".to_string(),
            )
        })?;

        let api_key = env::var("AZURE_OPENAI_API_KEY").map_err(|_| {
            ProviderError::Config("AZURE_OPENAI_API_KEY environment variable not set".to_string())
        })?;

        let api_version = env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| AZURE_DEFAULT_API_VERSION.to_string());

        let model = env::var("AZURE_OPENAI_MODEL").ok();

        Ok(Self {
            endpoint,
            deployment_name,
            api_key,
            api_version,
            model,
            temperature: None,
            max_tokens: None,
            top_p: None,
        })
    }

    /// Sets the API version.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Sets the model name for metadata.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
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

    /// Sets the top-p.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Builds the chat completions URL.
    pub(crate) fn chat_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint.trim_end_matches('/'),
            self.deployment_name,
            self.api_version,
        )
    }

    /// Returns the effective model name.
    pub(crate) fn effective_model(&self) -> &str {
        self.model.as_deref().unwrap_or(&self.deployment_name)
    }
}
