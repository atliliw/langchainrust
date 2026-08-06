// lc-schema/src/messages/file.rs
//! File content type (multimodal document/file support).
//!
//! Supports both URL-based and base64-encoded file content,
//! with explicit MIME type support for diverse document formats.

use serde::{Deserialize, Serialize};

/// File content (URL or base64 data URI).
///
/// Used for document/file input in multimodal interactions,
/// such as PDF processing or document analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileContent {
    /// URL or data URI for the file.
    pub url: String,
    /// MIME type of the file (e.g., "application/pdf", "text/csv").
    pub mime_type: Option<String>,
    /// Optional filename for reference.
    pub name: Option<String>,
}

impl FileContent {
    /// Creates from a URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            mime_type: None,
            name: None,
        }
    }

    /// Creates from base64 data with an explicit MIME type.
    pub fn from_base64(data: impl Into<String>, mime: &str) -> Self {
        Self {
            url: format!("data:{};base64,{}", mime, data.into()),
            mime_type: Some(mime.to_string()),
            name: None,
        }
    }

    /// Creates from a URL with a known MIME type.
    pub fn from_url_with_mime(url: impl Into<String>, mime: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            mime_type: Some(mime.into()),
            name: None,
        }
    }

    /// Sets the filename.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
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
        let file = FileContent::from_url("https://example.com/doc.pdf");
        assert_eq!(file.url, "https://example.com/doc.pdf");
        assert!(!file.is_base64());
        assert!(file.mime_type.is_none());
    }

    #[test]
    fn test_from_url_with_mime() {
        let file = FileContent::from_url_with_mime("https://example.com/data.csv", "text/csv");
        assert_eq!(file.mime_type, Some("text/csv".to_string()));
    }

    #[test]
    fn test_from_base64() {
        let file = FileContent::from_base64("abc123", "application/pdf");
        assert!(file.is_base64());
        assert_eq!(file.mime_type, Some("application/pdf".to_string()));
        assert_eq!(file.base64_data(), Some("abc123"));
    }

    #[test]
    fn test_with_name() {
        let file = FileContent::from_url("https://example.com/doc.pdf").with_name("report.pdf");
        assert_eq!(file.name, Some("report.pdf".to_string()));
    }
}
