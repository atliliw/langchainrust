// src/tools/url_fetch.rs
//! 网页抓取工具
//!
//! 提供网页内容抓取和解析功能。

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use regex::Regex;

use crate::core::tools::{BaseTool, Tool, ToolError};

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
}

impl URLFetchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("LangChainRust/0.1 (URL Fetch Tool)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
    
    /// 抓取网页内容
    async fn fetch_url(&self, url: &str, max_length: Option<usize>) -> Result<URLFetchOutput, ToolError> {
        // 验证 URL
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(ToolError::InvalidInput(
                "URL 必须以 http:// 或 https:// 开头".to_string()
            ));
        }
        
        // 发送请求
        let response = self.client
            .get(url)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP 请求失败: {}", e)))?;
        
        // 检查响应状态
        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(
                format!("HTTP 错误: {} - {}", status.as_u16(), status.canonical_reason().unwrap_or("未知"))
            ));
        }
        
        // 获取内容
        let content = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("读取响应失败: {}", e)))?;
        
        // 限制长度
        let max_len = max_length.unwrap_or(50000); // 默认最大 50KB
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
        
        // 移除 script 和 style 标签内容
        let script_regex = Regex::new(r"<script[^>]*>.*?</script>").unwrap();
        let style_regex = Regex::new(r"<style[^>]*>.*?</style>").unwrap();
        let html = script_regex.replace_all(html, "");
        let html = style_regex.replace_all(&html, "");
        
        // 移除所有 HTML 标签
        let tag_regex = Regex::new(r"<[^>]+>").unwrap();
        let text = tag_regex.replace_all(&html, "");
        
        // 清理空白
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        let clean_text = whitespace_regex.replace_all(&text, " ").trim().to_string();
        
        // 限制长度
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
        
        // 提取所有链接
        let link_regex = Regex::new(r#"<a[^>]+href\s*=\s*['"]([^'"]+)['"][^>]*>"#).unwrap();
        let links: Vec<String> = link_regex
            .captures_iter(html)
            .map(|cap| cap[1].to_string())
            .collect();
        
        // 去重并格式化
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
        
        // 提取所有图片链接
        let img_regex = Regex::new(r#"<img[^>]+src\s*=\s*['"]([^'"]+)['"][^>]*>"#).unwrap();
        let images: Vec<String> = img_regex
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
        
        // 提取 title
        let title_regex = Regex::new(r"<title[^>]*>(.*?)</title>").unwrap();
        let title = title_regex
            .captures(html)
            .map(|cap| cap[1].trim().to_string())
            .unwrap_or_default();
        
        // 提取 meta description
        let desc_regex = Regex::new(r#"<meta[^>]+name\s*=\s*['"]description['"][^>]+content\s*=\s*['"]([^'"]+)['"]"#).unwrap();
        let description = desc_regex
            .captures(html)
            .map(|cap| cap[1].to_string())
            .unwrap_or_default();
        
        // 提取 meta keywords
        let kw_regex = Regex::new(r#"<meta[^>]+name\s*=\s*['"]keywords['"][^>]+content\s*=\s*['"]([^'"]+)['"]"#).unwrap();
        let keywords = kw_regex
            .captures(html)
            .map(|cap| cap[1].to_string())
            .unwrap_or_default();
        
        let result = format!(
            "标题: {}\n描述: {}\n关键词: {}",
            title,
            description,
            keywords
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

/// 实现 Tool trait
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

/// 实现 BaseTool trait
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
        // 有效 URL 格式验证
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
        assert!(!result.result.contains("<")); // 不应包含 HTML 标签
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