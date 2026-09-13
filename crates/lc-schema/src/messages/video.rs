// lc-schema/src/messages/video.rs
//! Video content type (multimodal video support).
//!
//! Supports both URL-based and base64-encoded video content,
//! following the same pattern as [`super::AudioContent`].

use serde::{Deserialize, Serialize};

/// Video content (URL or base64 data URI).
///
/// Used for video input in multimodal interactions. Provider support varies:
/// OpenAI accepts `input_video` (https URL / `file_id` / data URI depending on
/// the model), Gemini accepts `inline_data` / `fileData` video parts, while
/// Anthropic does not accept video input — providers that cannot map video
/// surface an explicit error at request-build time instead of dropping it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoContent {
    /// Video URL or base64 data URI.
    pub url: String,
}

impl VideoContent {
    /// Creates from a URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// Creates from base64 data (auto-wraps as data URI, defaults to `video/mp4`).
    pub fn from_base64(data: impl Into<String>) -> Self {
        Self {
            url: format!("data:video/mp4;base64,{}", data.into()),
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
        let video = VideoContent::from_url("https://example.com/clip.mp4");
        assert_eq!(video.url, "https://example.com/clip.mp4");
        assert!(!video.is_base64());
    }

    #[test]
    fn test_from_base64() {
        let video = VideoContent::from_base64("abc123");
        assert!(video.is_base64());
        assert!(video.url.starts_with("data:video/mp4;base64,"));
        assert_eq!(video.base64_data(), Some("abc123"));
    }

    #[test]
    fn test_from_base64_with_mime() {
        let video = VideoContent::from_base64_with_mime("xyz", "video/webm");
        assert!(video.url.starts_with("data:video/webm;base64,"));
        assert_eq!(video.base64_data(), Some("xyz"));
    }

    #[test]
    fn test_url_not_base64() {
        let video = VideoContent::from_url("https://example.com/movie.mov");
        assert_eq!(video.base64_data(), None);
    }
}
