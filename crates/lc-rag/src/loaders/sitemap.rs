//! Sitemap loader
//!
//! Parses a sitemap.xml and crawls the page contents in bulk.

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

/// H8: default per-HTTP-request timeout — a hung target site will not block forever.
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Sitemap loader
///
/// Loads pages from a sitemap.xml URL or content, extracting the body text.
pub struct SitemapLoader {
    /// Sitemap URL or XML content
    source: SitemapSource,
    /// Maximum number of pages to crawl
    max_pages: usize,
    /// H8: per-HTTP-request timeout, preventing the crawler from blocking forever on a hung target site
    timeout: Duration,
}

/// Sitemap source
enum SitemapSource {
    /// Fetched from a URL
    Url(String),
    /// Direct XML content
    Xml(String),
}

impl SitemapLoader {
    /// Creates from a sitemap URL
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: SitemapSource::Url(url.into()),
            max_pages: 100,
            timeout: DEFAULT_HTTP_TIMEOUT,
        }
    }

    /// Creates from sitemap XML content
    pub fn from_xml(xml: impl Into<String>) -> Self {
        Self {
            source: SitemapSource::Xml(xml.into()),
            max_pages: 100,
            timeout: DEFAULT_HTTP_TIMEOUT,
        }
    }

    /// Sets the maximum number of pages to crawl
    pub fn with_max_pages(mut self, max: usize) -> Self {
        self.max_pages = max;
        self
    }

    /// Sets the per-HTTP-request timeout (H8, default 30s)
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Parses sitemap XML, extracting the URL list
    fn parse_urls(xml: &str) -> Vec<String> {
        LOC_REGEX
            .captures_iter(xml)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    /// Crawls a single page
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
        // Fetch the sitemap XML
        let xml = match &self.source {
            SitemapSource::Url(url) => Self::fetch_page(url, self.timeout).await?,
            SitemapSource::Xml(content) => content.clone(),
        };

        // Parse the URL list
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
        // This test would actually crawl, but max_pages=0 skips the crawl
        let loader = SitemapLoader::from_xml(xml).with_max_pages(0);
        let docs = loader.load().await.unwrap();
        assert!(docs.is_empty()); // max_pages=0, no pages are crawled
    }
}
