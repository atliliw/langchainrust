// lc-providers/src/providers/cohere/error.rs
//! Error types for the Cohere provider.

/// Cohere error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum CohereError {
    /// HTTP request error.
    Http(String),
    /// API error (non-2xx response).
    Api(String),
    /// Response parsing error.
    Parse(String),
}

impl std::fmt::Display for CohereError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CohereError::Http(msg) => write!(f, "Cohere HTTP error: {}", msg),
            CohereError::Api(msg) => write!(f, "Cohere API error: {}", msg),
            CohereError::Parse(msg) => write!(f, "Cohere parse error: {}", msg),
        }
    }
}

impl std::error::Error for CohereError {}

impl From<String> for CohereError {
    fn from(s: String) -> Self {
        CohereError::Api(s)
    }
}
