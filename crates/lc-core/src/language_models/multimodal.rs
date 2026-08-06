// lc-core/src/language_models/multimodal.rs
//! Multimodal model trait — extends BaseChatModel with audio, speech, and image generation.
//!
//! Not all providers support all multimodal capabilities. Implement only the
//! methods that the provider supports; unsupported methods return
//! `MultimodalError::Unsupported`.

use async_trait::async_trait;
use lc_schema::{AudioContent, ImageContent};

use super::BaseChatModel;

/// Errors from multimodal operations.
#[derive(Debug, thiserror::Error)]
pub enum MultimodalError {
    /// The provider does not support this operation.
    #[error("Unsupported multimodal operation: {0}")]
    Unsupported(String),

    /// HTTP/network error.
    #[error("HTTP error: {0}")]
    HttpError(String),

    /// API returned an error.
    #[error("API error: {0}")]
    ApiError(String),

    /// Response parsing error.
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Multimodal model trait — extends `BaseChatModel` with audio, speech, and image generation.
///
/// Providers implement only the methods they support. Callers should check
/// for `MultimodalError::Unsupported` when using optional capabilities.
///
/// # Example
///
/// ```rust,ignore
/// use lc_core::language_models::MultimodalModel;
///
/// // Transcribe audio
/// let transcript = llm.transcribe(AudioContent::from_url("https://...")).await?;
///
/// // Generate speech
/// let audio_bytes = llm.generate_speech("Hello, world!").await?;
///
/// // Generate image
/// let image = llm.generate_image("A cat wearing a hat").await?;
/// ```
#[async_trait]
pub trait MultimodalModel: BaseChatModel + Send + Sync {
    /// Transcribes audio to text (Speech-to-Text).
    ///
    /// Returns the transcribed text.
    async fn transcribe(&self, _audio: AudioContent) -> Result<String, MultimodalError> {
        Err(MultimodalError::Unsupported("transcribe".to_string()))
    }

    /// Generates audio from text (Text-to-Speech).
    ///
    /// Returns the raw audio bytes (format depends on provider, typically MP3 or PCM).
    async fn generate_speech(&self, _text: &str) -> Result<Vec<u8>, MultimodalError> {
        Err(MultimodalError::Unsupported("generate_speech".to_string()))
    }

    /// Generates an image from a text prompt.
    ///
    /// Returns the generated image content.
    async fn generate_image(&self, _prompt: &str) -> Result<ImageContent, MultimodalError> {
        Err(MultimodalError::Unsupported("generate_image".to_string()))
    }
}
