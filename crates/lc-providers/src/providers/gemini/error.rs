// lc-providers/src/providers/gemini/error.rs
//! Error types for the Gemini provider.

/// Gemini error type
#[derive(Debug)]
#[non_exhaustive]
pub enum GeminiError {
    /// Gemini API-returned error
    ApiError(String),
    /// HTTP request error
    HttpError(String),
    /// Response parse error
    ParseError(String),
    /// No response
    NoResponse,
    /// Blocked by safety filter
    SafetyBlock(String),
}

impl std::fmt::Display for GeminiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeminiError::ApiError(msg) => write!(f, "Gemini API error: {}", msg),
            GeminiError::HttpError(msg) => write!(f, "Gemini HTTP error: {}", msg),
            GeminiError::ParseError(msg) => write!(f, "Gemini parse error: {}", msg),
            GeminiError::NoResponse => write!(f, "Gemini returned no response"),
            GeminiError::SafetyBlock(msg) => write!(f, "Gemini blocked by safety filter: {}", msg),
        }
    }
}

impl std::error::Error for GeminiError {}

// L2 fix: add From<String> for GeminiError, matching OpenAIError pattern
impl From<String> for GeminiError {
    fn from(s: String) -> Self {
        GeminiError::ApiError(s)
    }
}
