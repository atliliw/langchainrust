//! HTML document loader
//!
//! Loads documents from an HTML string or URL: strips script/style, removes tags,
//! decodes entities, and extracts plain text.

use std::collections::HashMap;
use std::sync::LazyLock;

use async_trait::async_trait;
use regex::Regex;

use super::{DocumentLoader, LoaderError};
use lc_vector_stores::Document;

static SCRIPT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<script.*?</script>").unwrap());
static STYLE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<style.*?</style>").unwrap());
static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static WHITESPACE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// HTML loader: strips script/style, removes tags, decodes entities, extracts plain text
pub struct HTMLLoader {
    html: Option<String>,
    url: Option<String>,
}

impl HTMLLoader {
    /// Creates a loader from an HTML string
    pub fn new(html: impl Into<String>) -> Self {
        Self {
            html: Some(html.into()),
            url: None,
        }
    }

    /// Creates a loader from a URL (fetches the HTML asynchronously, then parses it)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            html: None,
            url: Some(url.into()),
        }
    }

    /// Extracts plain text from HTML (a pure function, convenient for testing)
    pub fn extract_text(html: &str) -> String {
        let mut text = html.to_string();
        text = SCRIPT_RE.replace_all(&text, "").to_string();
        text = STYLE_RE.replace_all(&text, "").to_string();
        text = TAG_RE.replace_all(&text, " ").to_string();
        // Decode common entities
        text = text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", "\"")
            .replace("&#39;", "'");
        // Compress whitespace
        WHITESPACE_RE.replace_all(&text, " ").trim().to_string()
    }

    /// Fetches HTML content from a URL
    async fn fetch_html(url: &str) -> Result<String, LoaderError> {
        let response = reqwest::get(url)
            .await
            .map_err(|e| LoaderError::Other(format!("HTTP request failed: {}", e)))?;
        let status = response.status();
        if !status.is_success() {
            return Err(LoaderError::Other(format!("HTTP error: {}", status)));
        }
        response
            .text()
            .await
            .map_err(|e| LoaderError::Other(format!("failed to read response: {}", e)))
    }
}

#[async_trait]
impl DocumentLoader for HTMLLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let html = if let Some(ref html) = self.html {
            html.clone()
        } else if let Some(ref url) = self.url {
            Self::fetch_html(url).await?
        } else {
            return Err(LoaderError::Other(
                "HTMLLoader has neither html nor url set".to_string(),
            ));
        };

        let text = Self::extract_text(&html);
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "html".to_string().into());
        if let Some(ref url) = self.url {
            metadata.insert("source".to_string(), url.clone().into());
        }
        Ok(vec![Document {
            content: text,
            metadata,
            id: None,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_removes_scripts_and_styles() {
        let html = r#"<html><head><script>alert(1)</script><style>body{}</style></head><body><p>Hello</p></body></html>"#;
        let text = HTMLLoader::extract_text(html);
        assert!(text.contains("Hello"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("body{}"));
        assert!(!text.contains('<'));
    }

    #[test]
    fn test_extract_text_decodes_entities() {
        let html = "<p>a &amp; b &lt; c</p>";
        let text = HTMLLoader::extract_text(html);
        assert_eq!(text, "a & b < c");
    }

    #[test]
    fn test_extract_text_decodes_more_entities() {
        let html = "<p>&quot;hello&quot; &#39;world&#39;</p>";
        let text = HTMLLoader::extract_text(html);
        assert_eq!(text, "\"hello\" 'world'");
    }

    #[test]
    fn test_extract_text_compresses_whitespace() {
        let html = "<p>hello</p>\n\n<p>world</p>";
        let text = HTMLLoader::extract_text(html);
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn test_load_returns_document() {
        let loader = HTMLLoader::new("<p>test</p>");
        let docs = loader.load().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "test");
        assert_eq!(
            docs[0].metadata.get("format"),
            Some(&serde_json::Value::String("html".to_string()))
        );
    }

    #[tokio::test]
    async fn test_load_from_url_has_source_metadata() {
        let loader = HTMLLoader::from_url("https://example.com");
        // No actual request is made; only verify construction
        assert!(loader.url.is_some());
        assert!(loader.html.is_none());
        assert_eq!(loader.url.as_deref(), Some("https://example.com"));
    }

    #[tokio::test]
    async fn test_load_from_url_invalid_url() {
        let loader = HTMLLoader::from_url("http://nonexistent.invalid.example");
        let result = loader.load().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_from_html_with_source_in_metadata() {
        let loader = HTMLLoader::new("<p>hello</p>");
        let docs = loader.load().await.unwrap();
        // No source when loading from an HTML string
        assert!(!docs[0].metadata.contains_key("source"));
    }

    #[test]
    fn test_extract_text_empty() {
        assert_eq!(HTMLLoader::extract_text(""), "");
    }

    #[test]
    fn test_extract_text_nested_tags() {
        let html = "<div><p><b>bold</b> text</p></div>";
        let text = HTMLLoader::extract_text(html);
        assert_eq!(text, "bold text");
    }

    #[test]
    fn test_extract_text_realistic_page() {
        let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Page</title>
    <script src="app.js"></script>
    <style>body { margin: 0; }</style>
</head>
<body>
    <h1>Welcome</h1>
    <p>This is a <strong>test</strong> page.</p>
    <footer>&copy; 2026</footer>
</body>
</html>"#;
        let text = HTMLLoader::extract_text(html);
        assert!(text.contains("Welcome"));
        assert!(text.contains("test page"));
        assert!(!text.contains("app.js"));
        assert!(!text.contains("margin"));
        assert!(!text.contains("DOCTYPE"));
    }
}
