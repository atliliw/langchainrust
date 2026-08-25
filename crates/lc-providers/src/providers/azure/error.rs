// lc-providers/src/providers/azure/error.rs
//! Error types for the Azure OpenAI provider.

/// Azure OpenAI error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum AzureOpenAIError {
    /// HTTP request error.
    Http(String),
    /// API error (non-2xx response).
    Api(String),
    /// Response parsing error.
    Parse(String),
}

impl std::fmt::Display for AzureOpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AzureOpenAIError::Http(msg) => write!(f, "Azure OpenAI HTTP error: {}", msg),
            AzureOpenAIError::Api(msg) => write!(f, "Azure OpenAI API error: {}", msg),
            AzureOpenAIError::Parse(msg) => write!(f, "Azure OpenAI parse error: {}", msg),
        }
    }
}

impl std::error::Error for AzureOpenAIError {}

impl From<String> for AzureOpenAIError {
    fn from(s: String) -> Self {
        AzureOpenAIError::Api(s)
    }
}
