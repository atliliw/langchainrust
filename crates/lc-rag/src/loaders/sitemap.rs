//! Sitemap 加载器
//!
//! 解析 sitemap.xml,批量爬取页面内容。

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;
use std::sync::LazyLock;

use super::{DocumentLoader, LoaderError};
use lc_vector_stores::Document;

/// M59: compile regex once using LazyLock instead of on every call
static LOC_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<loc>\s*(.*?)\s*</loc>").unwrap());

/// H8: 默认单次 HTTP 请求超时——目标站挂起时不会永久阻塞。
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Sitemap 加载器
///
/// 从 sitemap.xml URL 或内容加载页面,提取正文文本。
pub struct SitemapLoader {
    /// sitemap URL 或 XML 内容
    source: SitemapSource,
    /// 最大爬取页面数
    max_pages: usize,
    /// H8: 单次 HTTP 请求超时,防目标站挂起导致爬虫永久阻塞
    timeout: Duration,
}

/// sitemap 来源
enum SitemapSource {
    /// 从 URL 获取
    Url(String),
    /// 直接 XML 内容
    Xml(String),
}

impl SitemapLoader {
    /// 从 sitemap URL 创建
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: SitemapSource::Url(url.into()),
            max_pages: 100,
            timeout: DEFAULT_HTTP_TIMEOUT,
        }
    }

    /// 从 sitemap XML 内容创建
    pub fn from_xml(xml: impl Into<String>) -> Self {
        Self {
            source: SitemapSource::Xml(xml.into()),
            max_pages: 100,
            timeout: DEFAULT_HTTP_TIMEOUT,
        }
    }

    /// 设置最大爬取页面数
    pub fn with_max_pages(mut self, max: usize) -> Self {
        self.max_pages = max;
        self
    }

    /// 设置单次 HTTP 请求超时(H8,默认 30s)
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// 解析 sitemap XML,提取 URL 列表
    fn parse_urls(xml: &str) -> Vec<String> {
        LOC_REGEX
            .captures_iter(xml)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    /// 爬取单个页面
    async fn fetch_page(url: &str, timeout: Duration) -> Result<String, LoaderError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| LoaderError::Other(format!("failed to build HTTP client: {}", e)))?;
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| LoaderError::Other(format!("HTTP request failed {}: {}", url, e)))?;
        let status = response.status();
        if !status.is_success() {
            return Err(LoaderError::Other(format!(
                "HTTP error {}: {}",
                url, status
            )));
        }
        response
            .text()
            .await
            .map_err(|e| LoaderError::Other(format!("failed to read response {}: {}", url, e)))
    }
}

#[async_trait]
impl DocumentLoader for SitemapLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        // 获取 sitemap XML
        let xml = match &self.source {
            SitemapSource::Url(url) => Self::fetch_page(url, self.timeout).await?,
            SitemapSource::Xml(content) => content.clone(),
        };

        // 解析 URL 列表
        let urls = Self::parse_urls(&xml);
        let mut documents = Vec::new();

        for url in urls.iter().take(self.max_pages) {
            match Self::fetch_page(url, self.timeout).await {
                Ok(html) => {
                    let text = super::HTMLLoader::extract_text(&html);
                    let mut metadata = HashMap::new();
                    metadata.insert("format".to_string(), "html".to_string().into());
                    metadata.insert("source".to_string(), url.clone().into());

                    documents.push(Document {
                        content: text,
                        metadata,
                        id: None,
                    });
                }
                Err(e) => {
                    log::warn!("Failed to crawl {} (skipped from results): {}", url, e);
                    continue;
                }
            }
        }

        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_urls_simple() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/</loc></url>
            <url><loc>https://example.com/about</loc></url>
            <url><loc>https://example.com/contact</loc></url>
        </urlset>"#;
        let urls = SitemapLoader::parse_urls(xml);
        assert_eq!(urls.len(), 3);
        assert_eq!(urls[0], "https://example.com/");
        assert_eq!(urls[1], "https://example.com/about");
    }

    #[test]
    fn test_parse_urls_with_whitespace() {
        let xml = r#"<urlset>
            <url><loc>  https://example.com/page1  </loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>"#;
        let urls = SitemapLoader::parse_urls(xml);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://example.com/page1");
    }

    #[test]
    fn test_parse_urls_empty() {
        let xml = r#"<?xml version="1.0"?><urlset></urlset>"#;
        let urls = SitemapLoader::parse_urls(xml);
        assert!(urls.is_empty());
    }

    #[test]
    fn test_from_url() {
        let loader = SitemapLoader::from_url("https://example.com/sitemap.xml");
        assert_eq!(loader.max_pages, 100);
    }

    #[test]
    fn test_with_max_pages() {
        let loader = SitemapLoader::from_url("https://example.com/sitemap.xml").with_max_pages(5);
        assert_eq!(loader.max_pages, 5);
    }

    #[tokio::test]
    async fn test_load_from_xml() {
        let xml = r#"<?xml version="1.0"?>
        <urlset>
            <url><loc>https://example.com/</loc></url>
        </urlset>"#;
        // 这个测试会尝试真实爬取,但 max_pages=0 可以跳过
        let loader = SitemapLoader::from_xml(xml).with_max_pages(0);
        let docs = loader.load().await.unwrap();
        assert!(docs.is_empty()); // max_pages=0,不爬任何页面
    }
}
