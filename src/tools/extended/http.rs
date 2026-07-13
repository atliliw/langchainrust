//! HTTP 工具

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::core::tools::ToolError;
use crate::BaseTool;

/// HTTP 请求工具(GET / POST)
pub struct HTTPTool {
    client: reqwest::Client,
}

impl HTTPTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    pub async fn get(&self, url: &str) -> Result<String, ToolError> {
        self.client
            .get(url)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }

    pub async fn post(&self, url: &str, body: Value) -> Result<String, ToolError> {
        self.client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }
}

impl Default for HTTPTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseTool for HTTPTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "发起 HTTP 请求。输入 JSON: {\"url\": \"...\", \"method\": \"get|post\", \"body\": {...}}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let v: Value =
            serde_json::from_str(&input).map_err(|e| ToolError::InvalidInput(e.to_string()))?;
        let url = v
            .get("url")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ToolError::InvalidInput("缺 url".to_string()))?;
        let method = v.get("method").and_then(|x| x.as_str()).unwrap_or("get");
        match method {
            "get" => self.get(url).await,
            "post" => {
                self.post(url, v.get("body").cloned().unwrap_or(Value::Null))
                    .await
            }
            other => Err(ToolError::InvalidInput(format!("未知 method: {}", other))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_description() {
        let t = HTTPTool::new();
        assert_eq!(t.name(), "http_request");
        assert!(t.description().contains("HTTP"));
    }

    #[tokio::test]
    async fn test_run_invalid_json() {
        let t = HTTPTool::new();
        assert!(t.run("not json".to_string()).await.is_err());
    }

    #[tokio::test]
    async fn test_run_missing_url() {
        let t = HTTPTool::new();
        assert!(t.run(r#"{"method":"get"}"#.to_string()).await.is_err());
    }

    #[tokio::test]
    async fn test_run_unknown_method() {
        let t = HTTPTool::new();
        let r = t.run(r#"{"url":"http://x","method":"put"}"#.to_string()).await;
        assert!(r.is_err());
    }
}
