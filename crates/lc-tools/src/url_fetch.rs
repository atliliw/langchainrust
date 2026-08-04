// lc-tools/src/url_fetch.rs
//! 网页抓取工具
//!
//! 提供网页内容抓取和解析功能。

use async_trait::async_trait;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

use lc_core::tools::{BaseTool, Tool, ToolError};

static SCRIPT_REGEX: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"<script[^>]*>.*?</script>").unwrap());

static STYLE_REGEX: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"<style[^>]*>.*?</style>").unwrap());

static TAG_REGEX: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());

static WHITESPACE_REGEX: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"\s+").unwrap());

static LINK_REGEX: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r#"<a[^>]+href\s*=\s*['"]([^'"]+)['"][^>]*>"#).unwrap());

static IMG_REGEX: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r#"<img[^>]+src\s*=\s*['"]([^'"]+)['"][^>]*>"#).unwrap()
});

static TITLE_REGEX: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"<title[^>]*>(.*?)</title>").unwrap());

static DESC_REGEX: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r#"<meta[^>]+name\s*=\s*['"]description['"][^>]+content\s*=\s*['"]([^'"]+)['"]"#)
        .unwrap()
});

static KW_REGEX: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r#"<meta[^>]+name\s*=\s*['"]keywords['"][^>]+content\s*=\s*['"]([^'"]+)['"]"#)
        .unwrap()
});

/// Check if an IP address is private/internal (SSRF protection).
fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            octets[0] == 127
                || octets[0] == 10
                || (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31)
                || (octets[0] == 192 && octets[1] == 168)
                || (octets[0] == 169 && octets[1] == 254)
                || *v4 == std::net::Ipv4Addr::UNSPECIFIED
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || matches!(v6.segments(), [0xfe80, ..])
                || *v6 == std::net::Ipv6Addr::UNSPECIFIED
        }
    }
}

/// Resolve a URL hostname and check if it points to a private IP (async).
async fn url_points_to_private_ip(url: &str) -> Result<bool, ToolError> {
    let parsed =
        url::Url::parse(url).map_err(|e| ToolError::InvalidInput(format!("Invalid URL: {}", e)))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| ToolError::InvalidInput("URL has no host".to_string()))?;

    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(is_private_ip(&ip));
    }

    let port = parsed.port_or_known_default().unwrap_or(80);
    let addr_str = format!("{}:{}", host, port);
    let addrs: Vec<IpAddr> = tokio::net::lookup_host(&addr_str)
        .await
        .map_err(|e| {
            ToolError::ExecutionFailed(format!("DNS resolution failed for {}: {}", host, e))
        })?
        .map(|sa| sa.ip())
        .collect();

    if addrs.is_empty() {
        return Err(ToolError::ExecutionFailed(format!(
            "DNS resolution returned no addresses for {}",
            host
        )));
    }

    Ok(addrs.iter().any(is_private_ip))
}

/// URLFetch 工具输入
#[derive(Debug, Deserialize, JsonSchema)]
pub struct URLFetchInput {
    /// 操作类型: "fetch", "extract_text", "extract_links", "extract_images", "metadata"
    pub operation: String,

    /// URL 地址
    pub url: String,

    /// 是否包含头部信息（用于 fetch 操作）
    pub include_headers: Option<bool>,

    /// 最大内容长度（字节）
    pub max_length: Option<usize>,
}

/// URLFetch 工具输出
#[derive(Debug, Serialize)]
pub struct URLFetchOutput {
    /// 操作结果
    pub result: String,

    /// 操作类型
    pub operation: String,

    /// URL
    pub url: String,

    /// 内容长度
    pub content_length: usize,

    /// 额外信息
    pub details: Option<String>,
}

/// 网页抓取工具
pub struct URLFetchTool {
    /// HTTP 客户端
    client: reqwest::Client,
    /// 是否允许访问内网 IP（默认 false）
    allow_private_ips: bool,
}

impl URLFetchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("LangChainRust/0.1 (URL Fetch Tool)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            allow_private_ips: false,
        }
    }

    /// Allow requests to private/internal IP addresses (SSRF opt-in).
    pub fn with_allow_private_ips(mut self, allow: bool) -> Self {
        self.allow_private_ips = allow;
        self
    }

    /// Check SSRF protection before making a request.
    async fn check_ssrf(&self, url: &str) -> Result<(), ToolError> {
        if self.allow_private_ips {
            return Ok(());
        }
        if url_points_to_private_ip(url).await? {
            return Err(ToolError::ExecutionFailed(
                "Request to private/internal IP address is blocked by SSRF protection. \
                 Call .with_allow_private_ips(true) to allow."
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// 抓取网页内容
    async fn fetch_url(
        &self,
        url: &str,
        max_length: Option<usize>,
    ) -> Result<URLFetchOutput, ToolError> {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(ToolError::InvalidInput(
                "URL 必须以 http:// 或 https:// 开头".to_string(),
            ));
        }

        self.check_ssrf(url).await?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP 请求失败: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "HTTP 错误: {} - {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("未知")
            )));
        }

        let content = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("读取响应失败: {}", e)))?;

        let max_len = max_length.unwrap_or(50000);
        let content_len = content.len();
        let truncated = content_len > max_len;
        let result = if truncated {
            content.chars().take(max_len).collect::<String>() + "\n... [内容已截断]"
        } else {
            content
        };

        Ok(URLFetchOutput {
            result,
            operation: "fetch".to_string(),
            url: url.to_string(),
            content_length: content_len,
            details: Some(format!(
                "状态码: {}, 内容长度: {} 字节{}",
                status.as_u16(),
                content_len,
                if truncated { " (已截断)" } else { "" }
            )),
        })
    }

    /// 提取纯文本内容
    async fn extract_text(&self, url: &str) -> Result<URLFetchOutput, ToolError> {
        let fetch_result = self.fetch_url(url, Some(100000)).await?;
        let html = &fetch_result.result;

        let html = SCRIPT_REGEX.replace_all(html, "");
        let html = STYLE_REGEX.replace_all(&html, "");

        let text = TAG_REGEX.replace_all(&html, "");

        let clean_text = WHITESPACE_REGEX.replace_all(&text, " ").trim().to_string();

        let max_len = 5000;
        let clean_len = clean_text.len();
        let result = if clean_len > max_len {
            clean_text.chars().take(max_len).collect::<String>() + "..."
        } else {
            clean_text
        };

        Ok(URLFetchOutput {
            result,
            operation: "extract_text".to_string(),
            url: url.to_string(),
            content_length: clean_len,
            details: Some(format!("提取了 {} 字符的纯文本", clean_len)),
        })
    }

    /// 提取链接
    async fn extract_links(&self, url: &str) -> Result<URLFetchOutput, ToolError> {
        let fetch_result = self.fetch_url(url, Some(100000)).await?;
        let html = &fetch_result.result;

        let links: Vec<String> = LINK_REGEX
            .captures_iter(html)
            .map(|cap| cap[1].to_string())
            .collect();

        let unique_links: Vec<String> = links.into_iter().collect();
        let result = unique_links.join("\n");

        Ok(URLFetchOutput {
            result,
            operation: "extract_links".to_string(),
            url: url.to_string(),
            content_length: unique_links.len(),
            details: Some(format!("找到 {} 个链接", unique_links.len())),
        })
    }

    /// 提取图片链接
    async fn extract_images(&self, url: &str) -> Result<URLFetchOutput, ToolError> {
        let fetch_result = self.fetch_url(url, Some(100000)).await?;
        let html = &fetch_result.result;

        let images: Vec<String> = IMG_REGEX
            .captures_iter(html)
            .map(|cap| cap[1].to_string())
            .collect();

        let result = images.join("\n");

        Ok(URLFetchOutput {
            result,
            operation: "extract_images".to_string(),
            url: url.to_string(),
            content_length: images.len(),
            details: Some(format!("找到 {} 张图片", images.len())),
        })
    }

    /// 提取元数据
    async fn extract_metadata(&self, url: &str) -> Result<URLFetchOutput, ToolError> {
        let fetch_result = self.fetch_url(url, Some(50000)).await?;
        let html = &fetch_result.result;

        let title = TITLE_REGEX
            .captures(html)
            .map(|cap| cap[1].trim().to_string())
            .unwrap_or_default();

        let description = DESC_REGEX
            .captures(html)
            .map(|cap| cap[1].to_string())
            .unwrap_or_default();

        let keywords = KW_REGEX
            .captures(html)
            .map(|cap| cap[1].to_string())
            .unwrap_or_default();

        let result = format!(
            "标题: {}\n描述: {}\n关键词: {}",
            title, description, keywords
        );

        Ok(URLFetchOutput {
            result,
            operation: "metadata".to_string(),
            url: url.to_string(),
            content_length: title.len() + description.len() + keywords.len(),
            details: Some("提取了网页元数据".to_string()),
        })
    }
}

impl Default for URLFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for URLFetchTool {
    type Input = URLFetchInput;
    type Output = URLFetchOutput;

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        match input.operation.as_str() {
            "fetch" => self.fetch_url(&input.url, input.max_length).await,
            "extract_text" => self.extract_text(&input.url).await,
            "extract_links" => self.extract_links(&input.url).await,
            "extract_images" => self.extract_images(&input.url).await,
            "metadata" => self.extract_metadata(&input.url).await,
            _ => Err(ToolError::InvalidInput(
                format!("不支持的操作: {}，请使用: fetch, extract_text, extract_links, extract_images, metadata", input.operation)
            )),
        }
    }
}

#[async_trait]
impl BaseTool for URLFetchTool {
    fn name(&self) -> &str {
        "url_fetch"
    }

    fn description(&self) -> &str {
        "网页抓取工具。支持多种操作：

操作类型:
- fetch: 抓取完整网页内容
- extract_text: 提取纯文本内容（去除HTML标签）
- extract_links: 提取所有链接
- extract_images: 提取所有图片链接
- metadata: 提取网页元数据（标题、描述、关键词）

参数:
- url: 网页地址（必须以 http:// 或 https:// 开头）
- max_length: 最大内容长度（可选，默认50KB）
- include_headers: 是否包含头部信息（可选）

示例:
- 抓取网页: {\"operation\": \"fetch\", \"url\": \"https://example.com\"}
- 提取文本: {\"operation\": \"extract_text\", \"url\": \"https://example.com\"}
- 提取链接: {\"operation\": \"extract_links\", \"url\": \"https://example.com\"}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: URLFetchInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON 解析失败: {}", e)))?;

        let output = self.invoke(parsed).await?;

        Ok(format!(
            "URL: {}\n操作: {}\n内容长度: {} 字节\n\n{}\n详细信息: {}",
            output.url,
            output.operation,
            output.content_length,
            output.result,
            output.details.unwrap_or_default()
        ))
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(URLFetchInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_validation() {
        let valid_url = "https://example.com";
        assert!(valid_url.starts_with("http://") || valid_url.starts_with("https://"));

        let valid_url2 = "http://example.org";
        assert!(valid_url2.starts_with("http://") || valid_url2.starts_with("https://"));
    }

    #[tokio::test]
    async fn test_url_fetch_invalid_url() {
        let tool = URLFetchTool::new();

        let input = URLFetchInput {
            operation: "fetch".to_string(),
            url: "invalid-url".to_string(),
            include_headers: None,
            max_length: None,
        };

        let result = tool.invoke(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("http://"));
    }

    #[tokio::test]
    #[ignore = "需要网络连接"]
    async fn test_url_fetch_real() {
        let tool = URLFetchTool::new();

        let input = URLFetchInput {
            operation: "fetch".to_string(),
            url: "https://example.com".to_string(),
            include_headers: None,
            max_length: Some(5000),
        };

        let result = tool.invoke(input).await.unwrap();
        assert!(result.result.contains("example"));
        assert!(result.content_length > 0);
    }

    #[tokio::test]
    #[ignore = "需要网络连接"]
    async fn test_url_extract_text_real() {
        let tool = URLFetchTool::new();

        let input = URLFetchInput {
            operation: "extract_text".to_string(),
            url: "https://example.com".to_string(),
            include_headers: None,
            max_length: None,
        };

        let result = tool.invoke(input).await.unwrap();
        assert!(!result.result.contains("<"));
    }

    #[tokio::test]
    #[ignore = "需要网络连接"]
    async fn test_url_extract_links_real() {
        let tool = URLFetchTool::new();

        let input = URLFetchInput {
            operation: "extract_links".to_string(),
            url: "https://example.com".to_string(),
            include_headers: None,
            max_length: None,
        };

        let result = tool.invoke(input).await.unwrap();
        assert!(result.details.unwrap().contains("链接"));
    }

    #[tokio::test]
    #[ignore = "需要网络连接"]
    async fn test_url_extract_metadata_real() {
        let tool = URLFetchTool::new();

        let input = URLFetchInput {
            operation: "metadata".to_string(),
            url: "https://example.com".to_string(),
            include_headers: None,
            max_length: None,
        };

        let result = tool.invoke(input).await.unwrap();
        assert!(result.result.contains("标题"));
    }

    #[test]
    fn test_tool_properties() {
        let tool = URLFetchTool::new();

        assert_eq!(tool.name(), "url_fetch");
        assert!(tool.description().contains("fetch"));
        assert!(BaseTool::args_schema(&tool).is_some());
    }
}
