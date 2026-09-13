// lc-providers/src/openai/chat/error.rs
//! Error types for the OpenAI chat provider.

use crate::ProviderError;

/// OpenAI error type
#[derive(Debug)]
#[non_exhaustive]
pub enum OpenAIError {
    /// HTTP request error
    Http(String),
    /// API-returned error
    Api(String),
    /// Response parse error
    Parse(String),
    /// Configuration error (missing/malformed environment variables, etc.).
    Config(String),
    /// The SSE stream ended before a terminal marker (`[DONE]`, or a chunk
    /// carrying `finish_reason`) was observed — i.e. the connection dropped
    /// mid-generation. A12: the partial text streamed so far is truncated and
    /// must not be presented as a complete answer.
    StreamInterrupted(String),
}

impl std::fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIError::Http(msg) => write!(f, "HTTP error: {}", msg),
            OpenAIError::Api(msg) => write!(f, "API error: {}", msg),
            OpenAIError::Parse(msg) => write!(f, "Parse error: {}", msg),
            OpenAIError::Config(msg) => write!(f, "Configuration error: {}", msg),
            OpenAIError::StreamInterrupted(msg) => write!(
                f,
                "stream interrupted before terminal event (output truncated): {}",
                msg
            ),
        }
    }
}

impl std::error::Error for OpenAIError {}

impl From<String> for OpenAIError {
    fn from(s: String) -> Self {
        OpenAIError::Api(s)
    }
}

impl From<ProviderError> for OpenAIError {
    fn from(e: ProviderError) -> Self {
        OpenAIError::Config(e.to_string())
    }
}
