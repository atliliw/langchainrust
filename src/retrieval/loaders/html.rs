//! HTML 文档加载器

use std::collections::HashMap;

use async_trait::async_trait;
use regex::Regex;

use crate::retrieval::loaders::{DocumentLoader, LoaderError};
use crate::vector_stores::Document;

/// HTML 加载器:去除 script/style,剥离标签,解码实体,提取纯文本
pub struct HTMLLoader {
    html: String,
}

impl HTMLLoader {
    pub fn new(html: impl Into<String>) -> Self {
        Self { html: html.into() }
    }

    /// 从 HTML 提取纯文本(纯函数,便于测试)
    pub fn extract_text(html: &str) -> String {
        let script_re = Regex::new(r"(?s)<script.*?</script>").unwrap();
        let style_re = Regex::new(r"(?s)<style.*?</style>").unwrap();
        let tag_re = Regex::new(r"<[^>]+>").unwrap();
        let whitespace_re = Regex::new(r"\s+").unwrap();

        let mut text = html.to_string();
        text = script_re.replace_all(&text, "").to_string();
        text = style_re.replace_all(&text, "").to_string();
        text = tag_re.replace_all(&text, " ").to_string();
        // 解码常见实体
        text = text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", "\"");
        // 压缩空白
        whitespace_re.replace_all(&text, " ").trim().to_string()
    }
}

#[async_trait]
impl DocumentLoader for HTMLLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let text = Self::extract_text(&self.html);
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "html".to_string());
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
}
