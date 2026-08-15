// lc-providers/src/error.rs
//! Unified error type for the lc-providers crate.
//!
//! Aggregates all provider-specific error types so the `?` operator works
//! seamlessly across provider boundaries.

pub use crate::ollama::OllamaError;
pub use crate::openai::responses::types::ResponsesError;
pub use crate::openai::AssistantError;
pub use crate::openai::OpenAIError;
pub use crate::providers::anthropic::error::AnthropicError;
pub use crate::providers::azure::AzureOpenAIError;
pub use crate::providers::cohere::CohereError;
pub use crate::providers::gemini::GeminiError;

/// Unified error type that aggregates all LLM provider errors.
///
/// This allows using `?` across provider boundaries without manually
/// mapping error types. Each variant wraps the original provider
/// error, preserving full context.
#[derive(Debug)]
pub enum ProviderError {
    /// OpenAI API error.
    OpenAI(OpenAIError),
    /// Anthropic API error.
    Anthropic(AnthropicError),
    /// Gemini API error.
    Gemini(GeminiError),
    /// Azure OpenAI API error.
    Azure(AzureOpenAIError),
    /// Cohere API error.
    Cohere(CohereError),
    /// Ollama API error.
    Ollama(OllamaError),
    /// OpenAI Assistants API error.
    Assistant(AssistantError),
    /// OpenAI Responses API error.
    Responses(ResponsesError),
    /// DeepSeek API error (OpenAI-compatible endpoint).
    DeepSeek(OpenAIError),
    /// Qwen (Alibaba) API error (OpenAI-compatible endpoint).
    Qwen(OpenAIError),
    /// Moonshot (Kimi) API error (OpenAI-compatible endpoint).
    Moonshot(OpenAIError),
    /// Zhipu (ChatGLM) API error (OpenAI-compatible endpoint).
    Zhipu(OpenAIError),
    /// Mistral API error (OpenAI-compatible endpoint).
    Mistral(OpenAIError),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::OpenAI(e) => write!(f, "OpenAI error: {e}"),
            ProviderError::Anthropic(e) => write!(f, "Anthropic error: {e}"),
            ProviderError::Gemini(e) => write!(f, "Gemini error: {e}"),
            ProviderError::Azure(e) => write!(f, "Azure OpenAI error: {e}"),
            ProviderError::Cohere(e) => write!(f, "Cohere error: {e}"),
            ProviderError::Ollama(e) => write!(f, "Ollama error: {e}"),
            ProviderError::Assistant(e) => write!(f, "Assistant error: {e}"),
            ProviderError::Responses(e) => write!(f, "Responses error: {e}"),
            ProviderError::DeepSeek(e) => write!(f, "DeepSeek error: {e}"),
            ProviderError::Qwen(e) => write!(f, "Qwen error: {e}"),
            ProviderError::Moonshot(e) => write!(f, "Moonshot error: {e}"),
            ProviderError::Zhipu(e) => write!(f, "Zhipu error: {e}"),
            ProviderError::Mistral(e) => write!(f, "Mistral error: {e}"),
        }
    }
}

impl std::error::Error for ProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProviderError::OpenAI(e) => Some(e),
            ProviderError::Anthropic(e) => Some(e),
            ProviderError::Gemini(e) => Some(e),
            ProviderError::Azure(e) => Some(e),
            ProviderError::Cohere(e) => Some(e),
            ProviderError::Ollama(e) => Some(e),
            ProviderError::Assistant(e) => Some(e),
            ProviderError::Responses(e) => Some(e),
            ProviderError::DeepSeek(e) => Some(e),
            ProviderError::Qwen(e) => Some(e),
            ProviderError::Moonshot(e) => Some(e),
            ProviderError::Zhipu(e) => Some(e),
            ProviderError::Mistral(e) => Some(e),
        }
    }
}

// ---- From impls for all provider error types ----

impl From<OpenAIError> for ProviderError {
    fn from(e: OpenAIError) -> Self {
        ProviderError::OpenAI(e)
    }
}
impl From<AnthropicError> for ProviderError {
    fn from(e: AnthropicError) -> Self {
        ProviderError::Anthropic(e)
    }
}
impl From<GeminiError> for ProviderError {
    fn from(e: GeminiError) -> Self {
        ProviderError::Gemini(e)
    }
}
impl From<OllamaError> for ProviderError {
    fn from(e: OllamaError) -> Self {
        ProviderError::Ollama(e)
    }
}
impl From<AssistantError> for ProviderError {
    fn from(e: AssistantError) -> Self {
        ProviderError::Assistant(e)
    }
}
impl From<ResponsesError> for ProviderError {
    fn from(e: ResponsesError) -> Self {
        ProviderError::Responses(e)
    }
}
impl From<AzureOpenAIError> for ProviderError {
    fn from(e: AzureOpenAIError) -> Self {
        ProviderError::Azure(e)
    }
}
impl From<CohereError> for ProviderError {
    fn from(e: CohereError) -> Self {
        ProviderError::Cohere(e)
    }
}

// ---- LCEL Error conversion ----

/// Allow `ProviderError` to convert into `LcelError` for LCEL pipeline compatibility.
/// This enables LLM providers to participate in `pipe()` chains.
impl From<ProviderError> for lc_core::LcelError {
    fn from(err: ProviderError) -> Self {
        lc_core::LcelError::Provider(err.to_string())
    }
}
