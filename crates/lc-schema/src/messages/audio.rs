// lc-schema/src/messages/audio.rs
//! Audio content type (multimodal audio support).
//!
//! Supports both URL-based and base64-encoded audio content,
//! following the same pattern as `ImageContent`.

use serde::{Deserialize, Serialize};

/// Audio content (URL or base64 data URI).
///
/// Used for audio input/output in multimodal interactions,
/// such as speech-to-text (Whisper) and text-to-speech (TTS).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioContent {
    /// Audio URL or base64 data URI.
    pub url: String,
}

impl AudioContent {
    /// Creates from a URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// Creates from base64 data (auto-wraps as data URI).
    pub fn from_base64(data: impl Into<String>) -> Self {
        Self {
            url: format!("data:audio/wav;base64,{}", data.into()),
        }
    }

    /// Creates from base64 data with a specific MIME type.
    pub fn from_base64_with_mime(data: impl Into<String>, mime: &str) -> Self {
        Self {
            url: format!("data:{};base64,{}", mime, data.into()),
        }
    }

    /// Returns whether this is a base64 data URI.
    pub fn is_base64(&self) -> bool {
        self.url.starts_with("data:")
    }

    /// Extracts the base64 raw data (if this is a data URI).
    pub fn base64_data(&self) -> Option<&str> {
        self.url
            .split_once(',')
            .filter(|(prefix, _)| prefix.contains("base64"))
            .map(|(_, data)| data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_url() {
        let audio = AudioContent::from_url("https://example.com/audio.mp3");
        assert_eq!(audio.url, "https://example.com/audio.mp3");
        assert!(!audio.is_base64());
    }

    #[test]
    fn test_from_base64() {
        let audio = AudioContent::from_base64("abc123");
        assert!(audio.is_base64());
        assert_eq!(audio.base64_data(), Some("abc123"));
    }

    #[test]
    fn test_from_base64_with_mime() {
        let audio = AudioContent::from_base64_with_mime("xyz", "audio/mp3");
        assert!(audio.url.starts_with("data:audio/mp3;base64,"));
        assert_eq!(audio.base64_data(), Some("xyz"));
    }

    #[test]
    fn test_url_not_base64() {
        let audio = AudioContent::from_url("https://example.com/speech.wav");
        assert_eq!(audio.base64_data(), None);
    }
}
