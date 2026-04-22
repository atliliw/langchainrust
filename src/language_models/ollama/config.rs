// src/language_models/ollama/config.rs
//! Configuration for Ollama local LLM integration.
//!
//! Ollama provides local deployment of open-source LLMs like Llama, Mistral, etc.
//! This module provides configuration options for connecting to an Ollama server.

use crate::core::tools::ToolDefinition;
use std::env;

/// Configuration for Ollama chat model.
///
/// This struct holds all configuration options for connecting to an Ollama server
/// and controlling the behavior of the chat model.
///
/// # Example
/// ```rust
/// use langchainrust::OllamaConfig;
///
/// let config = OllamaConfig::new("llama3.2")
///     .with_base_url("http://localhost:11434/v1")
///     .with_temperature(0.7)
///     .with_streaming(true);
/// ```
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// The base URL of the Ollama server (default: "http://localhost:11434/v1").
    pub base_url: String,
    /// The model name to use (e.g., "llama3.2", "mistral", "codellama").
    pub model: String,
    /// Sampling temperature (0.0-2.0). Higher values produce more random outputs.
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
    /// Top-p sampling parameter for nucleus sampling.
    pub top_p: Option<f32>,
    /// Whether to enable streaming output.
    pub streaming: bool,
    /// Tool definitions for function calling support.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice strategy ("auto", "none", or specific tool name).
    pub tool_choice: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_string(),
            model: String::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            streaming: false,
            tools: None,
            tool_choice: None,
        }
    }
}

impl OllamaConfig {
    /// Creates a new OllamaConfig with the specified model name.
    ///
    /// # Arguments
    /// * `model` - The name of the Ollama model to use.
    ///
    /// # Returns
    /// A new OllamaConfig instance with default settings.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Creates an OllamaConfig from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `OLLAMA_BASE_URL`: The Ollama server URL (default: "http://localhost:11434/v1")
    /// - `OLLAMA_MODEL`: The model name (default: empty)
    ///
    /// # Returns
    /// A new OllamaConfig instance configured from environment.
    pub fn from_env() -> Self {
        let base_url =
            env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

        let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| String::new());

        Self {
            base_url,
            model,
            ..Default::default()
        }
    }

    /// Sets a custom base URL for the Ollama server.
    ///
    /// # Arguments
    /// * `url` - The base URL (e.g., "http://localhost:11434/v1").
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the model name to use.
    ///
    /// # Arguments
    /// * `model` - The model name (e.g., "llama3.2", "mistral").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the sampling temperature.
    ///
    /// # Arguments
    /// * `temp` - Temperature value (0.0-2.0). Higher = more random.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of tokens to generate.
    ///
    /// # Arguments
    /// * `max` - Maximum token count.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Enables or disables streaming output.
    ///
    /// # Arguments
    /// * `streaming` - Whether to enable streaming.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Sets the tool definitions for function calling.
    ///
    /// # Arguments
    /// * `tools` - List of tool definitions to bind to the model.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the tool choice strategy.
    ///
    /// # Arguments
    /// * `choice` - Tool choice ("auto", "none", or tool name).
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.tool_choice = Some(choice.into());
        self
    }
}
