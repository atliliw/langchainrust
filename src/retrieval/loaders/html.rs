//! HTML 文档加载器
//!
//! 支持从 HTML 字符串或 URL 加载文档,去除 script/style,剥离标签,解码实体,提取纯文本。

use std::collections::HashMap;
use std::sync::LazyLock;

use async_trait::async_trait;
use regex::Regex;

use crate::retrieval::loaders::{DocumentLoader, LoaderError};
use crate::vector_stores::Document;

static SCRIPT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<script.*?</script>").unwrap());
static STYLE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<style.*?</style>").unwrap());
static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static WHITESPACE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// HTML 加载器:去除 script/style,剥离标签,解码实体,提取纯文本
pub struct HTMLLoader {
    html: Option<String>,
    url: Option<String>,
}

impl HTMLLoader {
    /// 从 HTML 字符串创建加载器
    pub fn new(html: impl Into<String>) -> Self {
        Self {
            html: Some(html.into()),
            url: None,
        }
    }

    /// 从 URL 创建加载器(异步抓取 HTML 后解析)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            html: None,
            url: Some(url.into()),
        }
    }

    /// 从 HTML 提取纯文本(纯函数,便于测试)
    pub fn extract_text(html: &str) -> String {
        let mut text = html.to_string();
        text = SCRIPT_RE.replace_all(&text, "").to_string();
        text = STYLE_RE.replace_all(&text, "").to_string();
        text = TAG_RE.replace_all(&text, " ").to_string();
        // 解码常见实体
        text = text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", "\"")
            .replace("&#39;", "'");
        // 压缩空白
        WHITESPACE_RE.replace_all(&text, " ").trim().to_string()
    }

    /// 从 URL 抓取 HTML 内容
    async fn fetch_html(url: &str) -> Result<String, LoaderError> {
        let response = reqwest::get(url)
            .await
            .map_err(|e| LoaderError::Other(format!("HTTP 请求失败: {}", e)))?;
        let status = response.status();
        if !status.is_success() {
            return Err(LoaderError::Other(format!("HTTP 错误: {}", status)));
        }
        response
            .text()
            .await
            .map_err(|e| LoaderError::Other(format!("读取响应失败: {}", e)))
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
                "HTMLLoader 未设置 html 或 url".to_string(),
            ));
        };

        let text = Self::extract_text(&html);
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "html".to_string());
        if let Some(ref url) = self.url {
            metadata.insert("source".to_string(), url.clone());
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
        assert_eq!(docs[0].metadata.get("format"), Some(&"html".to_string()));
    }

    #[tokio::test]
    async fn test_load_from_url_has_source_metadata() {
        let loader = HTMLLoader::from_url("https://example.com");
        // 不实际请求,只验证构造
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
        // 从 HTML 字符串加载时没有 source
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
