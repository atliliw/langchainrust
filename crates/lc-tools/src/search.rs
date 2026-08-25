// lc-tools/src/search.rs
//! DuckDuckGo 搜索工具
//!
//! 通过 DuckDuckGo Instant Answer API 进行网页搜索。
//! 无需 API Key，免费使用。

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use lc_core::tools::{BaseTool, Tool, ToolError};

/// 搜索工具输入
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchInput {
    /// 搜索查询
    pub query: String,
    /// 返回结果数量（默认 5）
    pub top_k: Option<usize>,
}

/// 搜索工具输出
#[derive(Debug, Serialize)]
pub struct SearchOutput {
    /// 查询
    pub query: String,
    /// 结果列表
    pub results: Vec<SearchResult>,
    /// 结果数量
    pub total: usize,
    /// 摘要（如有）
    pub abstract_text: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    /// 标题
    pub title: String,
    /// 摘要
    pub snippet: String,
    /// URL
    pub url: String,
}

/// DuckDuckGo 网页搜索工具
///
/// 使用 DuckDuckGo 的 Instant Answer API 进行搜索，无需 API Key。
pub struct DuckDuckGoSearchTool {
    client: reqwest::Client,
}

impl DuckDuckGoSearchTool {
    /// 创建 DuckDuckGo 搜索工具。
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .user_agent("LangChainRust/0.1 (Search Tool)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for DuckDuckGoSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for DuckDuckGoSearchTool {
    type Input = SearchInput;
    type Output = SearchOutput;

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        if input.query.trim().is_empty() {
            return Err(ToolError::InvalidInput(
                "search query must not be empty".to_string(),
            ));
        }

        let top_k = input.top_k.unwrap_or(5);
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
            urlencoding(&input.query)
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("search request failed: {}", e)))?;

        let body: serde_json::Value = response.json().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to parse search results: {}", e))
        })?;

        let abstract_text = body["AbstractText"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let mut results = Vec::new();
        if let Some(topics) = body["RelatedTopics"].as_array() {
            for topic in topics.iter() {
                if results.len() >= top_k {
                    break;
                }
                if let Some(text) = topic["Text"].as_str() {
                    let title = topic["FirstURL"]
                        .as_str()
                        .map(|u| u.rsplit('/').next().unwrap_or(u).replace('_', " "))
                        .unwrap_or_default();
                    let url = topic["FirstURL"].as_str().unwrap_or("");
                    results.push(SearchResult {
                        title,
                        snippet: text.to_string(),
                        url: url.to_string(),
                    });
                }
                if let Some(nested) = topic["Topics"].as_array() {
                    for nt in nested.iter() {
                        if results.len() >= top_k {
                            break;
                        }
                        if let Some(text) = nt["Text"].as_str() {
                            let title = nt["FirstURL"]
                                .as_str()
                                .map(|u| u.rsplit('/').next().unwrap_or(u).replace('_', " "))
                                .unwrap_or_default();
                            let url = nt["FirstURL"].as_str().unwrap_or("");
                            results.push(SearchResult {
                                title,
                                snippet: text.to_string(),
                                url: url.to_string(),
                            });
                        }
                    }
                }
            }
        }

        if results.is_empty() {
            if let Some(ref abstract_text) = abstract_text {
                let source_url = body["AbstractURL"].as_str().unwrap_or("");
                results.push(SearchResult {
                    title: input.query.clone(),
                    snippet: abstract_text.clone(),
                    url: source_url.to_string(),
                });
            }
        }

        Ok(SearchOutput {
            query: input.query,
            total: results.len(),
            results,
            abstract_text,
        })
    }
}

/// URL 编码 (using urlencoding crate for complete encoding)
fn urlencoding(s: &str) -> String {
    urlencoding::encode(s).to_string()
}

#[async_trait]
impl BaseTool for DuckDuckGoSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "网页搜索工具。使用 DuckDuckGo 搜索引擎搜索网络信息，无需 API Key。

参数:
- query: 搜索关键词
- top_k: 返回结果数量（默认 5）

示例:
- {\"query\": \"Rust programming language\", \"top_k\": 3}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: SearchInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse failed: {}", e)))?;

        let output = self.invoke(parsed).await?;

        let mut text = format!("搜索结果 (查询: {})\n\n", output.query);

        if let Some(ref abstract_text) = output.abstract_text {
            text.push_str(&format!("摘要: {}\n\n", abstract_text));
        }

        for (i, result) in output.results.iter().enumerate() {
            text.push_str(&format!("{}. {}\n", i + 1, result.title));
            text.push_str(&format!("   {}\n", result.snippet));
            text.push_str(&format!("   URL: {}\n\n", result.url));
        }

        if output.results.is_empty() {
            text.push_str("未找到相关结果");
        } else {
            text.push_str(&format!("共 {} 条结果", output.total));
        }

        Ok(text)
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(SearchInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_tool_properties() {
        let tool = DuckDuckGoSearchTool::new();
        assert_eq!(tool.name(), "web_search");
        assert!(tool.description().contains("DuckDuckGo"));
        assert!(BaseTool::args_schema(&tool).is_some());
    }

    #[tokio::test]
    async fn test_search_empty_query() {
        let tool = DuckDuckGoSearchTool::new();
        let result = tool.run(r#"{"query": ""}"#.to_string()).await;
        assert!(result.is_err());
    }
}
