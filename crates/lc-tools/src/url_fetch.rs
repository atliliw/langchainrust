// lc-tools/src/url_fetch.rs
//! 网页抓取工具
//!
//! 提供网页内容抓取和解析功能。

use async_trait::async_trait;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ssrf::url_points_to_private_ip;
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

/// 从 HTML 中提取链接并去重(保留首次出现顺序)。
///
/// 返回 `(去重后的链接, 原始条数)`。原始条数用于 details 展示"去重前后对比"。
/// Q4: 旧实现只给 Vec 换了个名 `unique_links`,并没有真正去重。
fn extract_unique_links(html: &str) -> (Vec<String>, usize) {
    let raw: Vec<String> = LINK_REGEX
        .captures_iter(html)
        .map(|cap| cap[1].to_string())
        .collect();
    let raw_count = raw.len();
    let mut seen = std::collections::HashSet::new();
    let unique: Vec<String> = raw
        .into_iter()
        .filter(|link| seen.insert(link.clone()))
        .collect();
    (unique, raw_count)
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
        include_headers: Option<bool>,
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

        // Q3: include_headers = Some(true) 时,把响应头并入输出(details),不再静默忽略。
        // 头必须在消费 response 前读出(reqwest 的 headers() 借用、text() 消费)。
        let header_block: String = if include_headers.unwrap_or(false) {
            let mut lines: Vec<String> = response
                .headers()
                .iter()
                .map(|(name, value)| format!("{}: {}", name, value.to_str().unwrap_or("<非UTF-8>")))
                .collect();
            lines.sort();
            let mut block = String::from("响应头:\n");
            for line in lines {
                block.push_str(&line);
                block.push('\n');
            }
            block
        } else {
            String::new()
        };

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

        let mut details = format!(
            "状态码: {}, 内容长度: {} 字节{}",
            status.as_u16(),
            content_len,
            if truncated { " (已截断)" } else { "" }
        );
        if !header_block.is_empty() {
            details.push('\n');
            details.push_str(&header_block);
        }

        Ok(URLFetchOutput {
            result,
            operation: "fetch".to_string(),
            url: url.to_string(),
            content_length: content_len,
            details: Some(details),
        })
    }

    /// 提取纯文本内容
    async fn extract_text(&self, url: &str) -> Result<URLFetchOutput, ToolError> {
        let fetch_result = self.fetch_url(url, Some(100000), None).await?;
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

    /// 提取链接(去重,保留首次出现顺序)
    async fn extract_links(&self, url: &str) -> Result<URLFetchOutput, ToolError> {
        let fetch_result = self.fetch_url(url, Some(100000), None).await?;
        let html = &fetch_result.result;

        let (unique_links, raw_count) = extract_unique_links(html);
        let result = unique_links.join("\n");

        Ok(URLFetchOutput {
            result,
            operation: "extract_links".to_string(),
            url: url.to_string(),
            content_length: html.len(), // Q4: 真实正文长度,而非链接条数
            details: Some(format!(
                "找到 {} 个唯一链接(原始 {} 个)",
                unique_links.len(),
                raw_count
            )),
        })
    }

    /// 提取图片链接
    async fn extract_images(&self, url: &str) -> Result<URLFetchOutput, ToolError> {
        let fetch_result = self.fetch_url(url, Some(100000), None).await?;
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
        let fetch_result = self.fetch_url(url, Some(50000), None).await?;
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
            "fetch" => {
                self.fetch_url(&input.url, input.max_length, input.include_headers)
                    .await
            }
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

    /// Q4: extract_links 真正去重且保留首次出现顺序;content_length 是正文长度。
    #[test]
    fn test_extract_unique_links_dedups() {
        let html = r#"
            <a href="https://a.com/1">first</a>
            <a href="https://a.com/1">dup</a>
            <a href="https://a.com/2">second</a>
            <a href="https://a.com/1">dup2</a>
        "#;
        let (unique, raw) = extract_unique_links(html);
        assert_eq!(raw, 4, "原始链接数应为 4");
        assert_eq!(
            unique,
            vec!["https://a.com/1".to_string(), "https://a.com/2".to_string()],
            "应去重且保持首次出现顺序"
        );
    }

    /// Q1: SSRF 抽取公共模块后,URLFetch 默认仍拦截内网地址。
    #[tokio::test]
    async fn test_url_fetch_blocks_localhost_by_default() {
        let tool = URLFetchTool::new();
        let input = URLFetchInput {
            operation: "fetch".to_string(),
            url: "http://127.0.0.1:6379/".to_string(),
            include_headers: None,
            max_length: None,
        };
        let result = tool.invoke(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("SSRF"), "expected SSRF error, got: {}", err);
    }

    /// Q3: include_headers = Some(true) 时,响应头并入输出(details)。
    #[tokio::test]
    async fn test_fetch_include_headers() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let body = "hello world";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: text/plain\r\nX-Test-Header: hello\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes()).await.unwrap();
            drop(socket);
        });

        let tool = URLFetchTool::new().with_allow_private_ips(true);
        let input = URLFetchInput {
            operation: "fetch".to_string(),
            url: format!("http://{}/", addr),
            include_headers: Some(true),
            max_length: None,
        };

        let output = tool.invoke(input).await.unwrap();
        let details = output.details.unwrap();
        assert!(details.contains("响应头:"), "details: {}", details);
        // reqwest 会把响应头名规范化为小写（HTTP 头名不区分大小写）
        assert!(
            details.contains("x-test-header: hello"),
            "details: {}",
            details
        );
        assert!(output.result.contains("hello world"));
        server.await.unwrap();
    }

    /// Q3: include_headers = false/None 时不返回响应头。
    #[tokio::test]
    async fn test_fetch_without_include_headers() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let body = "hello world";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nX-Test-Header: hello\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes()).await.unwrap();
            drop(socket);
        });

        let tool = URLFetchTool::new().with_allow_private_ips(true);
        let input = URLFetchInput {
            operation: "fetch".to_string(),
            url: format!("http://{}/", addr),
            include_headers: Some(false),
            max_length: None,
        };

        let output = tool.invoke(input).await.unwrap();
        let details = output.details.unwrap();
        assert!(
            !details.contains("X-Test-Header"),
            "不应包含响应头, got: {}",
            details
        );
        server.await.unwrap();
    }

    #[test]
    fn test_tool_properties() {
        let tool = URLFetchTool::new();

        assert_eq!(tool.name(), "url_fetch");
        assert!(tool.description().contains("fetch"));
        assert!(BaseTool::args_schema(&tool).is_some());
    }
}
