//! Image content type (multimodal vision support)

use serde::{Deserialize, Serialize};

/// Image content (URL or base64 data URI)
///
/// OpenAI Vision uses `image_url.url` (an https URL or a `data:image/...;base64,...` data URI);
/// Ollama uses raw base64 bytes. This type uniformly stores the value in the `url` field,
/// letting each provider convert it at serialization time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageContent {
    /// Image URL or base64 data URI
    pub url: String,
}

impl ImageContent {
    /// Create from a URL
    pub fn from_url(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// Create from base64 data (auto-wrapped as a data URI)
    pub fn from_base64(data: impl Into<String>) -> Self {
        Self {
            url: format!("data:image/png;base64,{}", data.into()),
        }
    }

    /// Create from base64 data with the given MIME type
    pub fn from_base64_with_mime(data: impl Into<String>, mime: &str) -> Self {
        Self {
            url: format!("data:{};base64,{}", mime, data.into()),
        }
    }

    /// Whether this is a base64 data URI
    pub fn is_base64(&self) -> bool {
        self.url.starts_with("data:")
    }

    /// Extract the raw base64 data (when this is a data URI)
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
        let img = ImageContent::from_url("https://example.com/image.jpg");
        assert_eq!(img.url, "https://example.com/image.jpg");
        assert!(!img.is_base64());
    }

    #[test]
    fn test_from_base64() {
        let img = ImageContent::from_base64("abc123");
        assert!(img.is_base64());
        assert_eq!(img.base64_data(), Some("abc123"));
    }

    #[test]
    fn test_from_base64_with_mime() {
        let img = ImageContent::from_base64_with_mime("xyz", "image/jpeg");
        assert!(img.url.starts_with("data:image/jpeg;base64,"));
        assert_eq!(img.base64_data(), Some("xyz"));
    }

    #[test]
    fn test_url_not_base64() {
        let img = ImageContent::from_url("https://example.com/img.png");
        assert_eq!(img.base64_data(), None);
    }
}
