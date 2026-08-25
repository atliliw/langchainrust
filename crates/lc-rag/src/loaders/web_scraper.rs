//! 网页爬取加载器
//!
//! 从 URL 爬取网页内容,提取正文文本,支持递归链接跟踪。
//! 基于 HTMLLoader 的文本提取逻辑,增加链接发现与批量爬取能力。

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;

use super::{DocumentLoader, LoaderError};
use lc_vector_stores::Document;

/// H8: 默认单次 HTTP 请求超时——目标站挂起时爬虫不会永久阻塞。
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

// M9: Pre-compile regexes once instead of on every call.
static HREF_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"href\s*=\s*["']([^"']+)["']"#).unwrap());
static DOMAIN_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"https?://([^/]+)").unwrap());
static DOMAIN_PREFIX_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"https?://[^/]+").unwrap());

/// 网页爬取加载器
///
/// 从 URL 爬取网页,提取正文文本。可选递归跟踪同域链接。
pub struct WebScraperLoader {
    /// 起始 URL
    url: String,
    /// 最大递归深度(0 = 仅爬起始页)
    max_depth: usize,
    /// 最大爬取页面数
    max_pages: usize,
    /// 是否在爬取失败时返回错误(默认 false,跳过失败页面)
    fail_on_error: bool,
    /// H8: 单次 HTTP 请求超时,防目标站挂起导致爬虫永久阻塞
    timeout: Duration,
}

impl WebScraperLoader {
    /// 从 URL 创建加载器(仅爬取指定页面)
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_depth: 0,
            max_pages: 1,
            fail_on_error: false,
            timeout: DEFAULT_HTTP_TIMEOUT,
        }
    }

    /// 设置最大递归深度
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// 设置最大爬取页面数
    pub fn with_max_pages(mut self, pages: usize) -> Self {
        self.max_pages = pages;
        self
    }

    /// 设置爬取失败时是否返回错误(默认跳过失败页面)
    pub fn with_fail_on_error(mut self, fail: bool) -> Self {
        self.fail_on_error = fail;
        self
    }

    /// 设置单次 HTTP 请求超时(H8,默认 30s)
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// 从 HTML 提取纯文本(复用 HTMLLoader 的逻辑)
    fn extract_text(html: &str) -> String {
        super::HTMLLoader::extract_text(html)
    }

    /// 从 HTML 提取链接
    fn extract_links(html: &str, base_url: &str) -> Vec<String> {
        let base_domain = Self::extract_domain(base_url);
        HREF_RE
            .captures_iter(html)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .filter(|link| !link.starts_with('#') && !link.starts_with("javascript:"))
            .filter_map(|link| Self::resolve_url(base_url, &link))
            .filter(|url| Self::extract_domain(url) == base_domain)
            .collect()
    }

    /// 提取域名
    fn extract_domain(url: &str) -> String {
        DOMAIN_RE
            .captures(url)
            .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
            .unwrap_or_default()
    }

    /// 解析相对 URL 为绝对 URL
    fn resolve_url(base: &str, href: &str) -> Option<String> {
        if href.starts_with("http://") || href.starts_with("https://") {
            Some(href.to_string())
        } else if href.starts_with('/') {
            // 找到 scheme://domain 部分
            let domain = DOMAIN_PREFIX_RE.find(base)?.as_str();
            Some(format!("{}{}", domain, href))
        } else {
            // 相对路径
            let base_dir = base.rfind('/').map(|i| &base[..=i]).unwrap_or(base);
            Some(format!("{}{}", base_dir, href))
        }
    }

    /// 爬取单个页面
    async fn fetch_page(url: &str, timeout: Duration) -> Result<(String, String), LoaderError> {
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
        let html = response
            .text()
            .await
            .map_err(|e| LoaderError::Other(format!("failed to read response {}: {}", url, e)))?;
        Ok((url.to_string(), html))
    }
}

#[async_trait]
impl DocumentLoader for WebScraperLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let mut documents = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![(self.url.clone(), 0usize)];
        let mut failed_count: usize = 0;

        while let Some((url, depth)) = queue.pop() {
            if visited.contains(&url) || documents.len() >= self.max_pages {
                continue;
            }
            visited.insert(url.clone());

            let (fetched_url, html) = match Self::fetch_page(&url, self.timeout).await {
                Ok(r) => r,
                Err(e) => {
                    failed_count += 1;
                    if self.fail_on_error {
                        return Err(e);
                    }
                    // 跳过失败页面,继续爬取其他(经日志门面暴露,便于宿主捕获)
                    log::warn!("Failed to crawl {} (failure #{}): {}", url, failed_count, e);
                    continue;
                }
            };

            let text = Self::extract_text(&html);

            let mut metadata = HashMap::new();
            metadata.insert("format".to_string(), "html".to_string().into());
            metadata.insert("source".to_string(), fetched_url.clone().into());

            documents.push(Document {
                content: text,
                metadata,
                id: None,
            });

            // 递归跟踪链接
            if depth < self.max_depth {
                let links = Self::extract_links(&html, &fetched_url);
                for link in links {
                    if !visited.contains(&link) {
                        queue.push((link, depth + 1));
                    }
                }
            }
        }

        if failed_count > 0 {
            log::warn!(
                "Crawling finished: {} pages failed, {} pages succeeded",
                failed_count,
                documents.len()
            );
        }

        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_links() {
        let html = "<html><body><a href=\"/about\">About</a><a href=\"https://example.com/contact\">Contact</a><a href=\"#top\">Top</a></body></html>";
        let links = WebScraperLoader::extract_links(html, "https://example.com/");
        assert!(links.contains(&"https://example.com/about".to_string()));
        assert!(links.contains(&"https://example.com/contact".to_string()));
        // # 链接应被过滤
        assert!(!links.iter().any(|l| l.contains('#')));
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(
            WebScraperLoader::extract_domain("https://example.com/path"),
            "example.com"
        );
        assert_eq!(
            WebScraperLoader::extract_domain("http://sub.example.com:8080/path"),
            "sub.example.com:8080"
        );
    }

    #[test]
    fn test_resolve_url_absolute() {
        let result =
            WebScraperLoader::resolve_url("https://example.com/", "https://other.com/page");
        assert_eq!(result, Some("https://other.com/page".to_string()));
    }

    #[test]
    fn test_resolve_url_relative() {
        let result = WebScraperLoader::resolve_url("https://example.com/dir/page", "other");
        assert_eq!(result, Some("https://example.com/dir/other".to_string()));
    }

    #[test]
    fn test_resolve_url_root_relative() {
        let result = WebScraperLoader::resolve_url("https://example.com/dir/page", "/root");
        assert_eq!(result, Some("https://example.com/root".to_string()));
    }

    #[test]
    fn test_extract_text() {
        let html = "<html><body><p>Hello World</p></body></html>";
        let text = WebScraperLoader::extract_text(html);
        assert!(text.contains("Hello World"));
    }

    #[test]
    fn test_new_creates_single_page_scraper() {
        let loader = WebScraperLoader::new("https://example.com");
        assert_eq!(loader.max_depth, 0);
        assert_eq!(loader.max_pages, 1);
    }

    #[test]
    fn test_with_options() {
        let loader = WebScraperLoader::new("https://example.com")
            .with_max_depth(2)
            .with_max_pages(10);
        assert_eq!(loader.max_depth, 2);
        assert_eq!(loader.max_pages, 10);
    }
}
