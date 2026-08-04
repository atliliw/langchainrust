// src/language_models/providers/anthropic/error.rs
//! Anthropic error types.

/// Errors that can occur when interacting with the Anthropic API.
#[derive(Debug)]
pub enum AnthropicError {
    Http(String),
    Api(String),
    Parse(String),
}

impl std::fmt::Display for AnthropicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnthropicError::Http(msg) => write!(f, "HTTP error: {}", msg),
            AnthropicError::Api(msg) => write!(f, "API error: {}", msg),
            AnthropicError::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for AnthropicError {}

// L2 fix: add From<String> for AnthropicError, matching OpenAIError pattern
impl From<String> for AnthropicError {
    fn from(s: String) -> Self {
        AnthropicError::Api(s)
    }
}
