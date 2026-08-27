//! HTTP tool with SSRF protection

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::ssrf::{guarded_get, url_points_to_private_ip};
use lc_core::tools::ToolError;
use lc_core::BaseTool;

/// HTTP request tool (GET/POST) with SSRF protection.
pub struct HTTPTool {
    client: reqwest::Client,
    allow_private_ips: bool,
}

impl HTTPTool {
    /// Creates an HTTP tool with a 30s timeout and SSRF protection enabled.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                // SSRF: disable auto-redirects, guarded_get re-checks each hop
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            allow_private_ips: false,
        }
    }

    /// Creates an HTTP tool with a custom timeout (SSRF protection enabled).
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(timeout)
                .redirect(reqwest::redirect::Policy::none())
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

    /// Sends a GET request, following redirects with SSRF checks per hop.
    pub async fn get(&self, url: &str) -> Result<String, ToolError> {
        // SSRF: guarded_get checks each hop and follows redirects manually
        guarded_get(&self.client, url, !self.allow_private_ips)
            .await?
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }

    /// Sends a POST request with a JSON body (single-hop SSRF check).
    pub async fn post(&self, url: &str, body: Value) -> Result<String, ToolError> {
        // POST has auto-redirect disabled (3xx returned as-is), a single-hop SSRF check suffices
        self.check_ssrf(url).await?;
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
        "Make HTTP requests. Input JSON: {\"url\": \"...\", \"method\": \"get|post\", \"body\": {...}}. \
         SSRF protection enabled by default (blocks private IPs)."
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let v: Value =
            serde_json::from_str(&input).map_err(|e| ToolError::InvalidInput(e.to_string()))?;
        let url = v
            .get("url")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'url' field".to_string()))?;
        let method = v.get("method").and_then(|x| x.as_str()).unwrap_or("get");
        match method {
            "get" => self.get(url).await,
            "post" => {
                self.post(url, v.get("body").cloned().unwrap_or(Value::Null))
                    .await
            }
            other => Err(ToolError::InvalidInput(format!(
                "Unknown method: {}. Supported: get, post",
                other
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssrf::is_private_ip;
    use std::net::IpAddr;

    #[test]
    fn test_name_description() {
        let t = HTTPTool::new();
        assert_eq!(t.name(), "http_request");
        assert!(t.description().contains("HTTP"));
    }

    #[test]
    fn test_private_ip_detection() {
        assert!(is_private_ip(&IpAddr::from([127, 0, 0, 1])));
        assert!(is_private_ip(&IpAddr::from([10, 0, 0, 1])));
        assert!(is_private_ip(&IpAddr::from([172, 16, 0, 1])));
        assert!(is_private_ip(&IpAddr::from([172, 31, 255, 255])));
        assert!(is_private_ip(&IpAddr::from([192, 168, 1, 1])));
        assert!(is_private_ip(&IpAddr::from([169, 254, 169, 254])));
        assert!(is_private_ip(&IpAddr::from([0, 0, 0, 0])));

        assert!(!is_private_ip(&IpAddr::from([8, 8, 8, 8])));
        assert!(!is_private_ip(&IpAddr::from([1, 1, 1, 1])));
        assert!(!is_private_ip(&IpAddr::from([172, 15, 0, 1])));
        assert!(!is_private_ip(&IpAddr::from([172, 32, 0, 1])));
    }

    #[tokio::test]
    async fn test_ssrf_blocks_localhost() {
        let tool = HTTPTool::new();
        let result = tool.check_ssrf("http://127.0.0.1:6379/").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SSRF"));
    }

    #[tokio::test]
    async fn test_ssrf_blocks_cloud_metadata() {
        let tool = HTTPTool::new();
        let result = tool
            .check_ssrf("http://169.254.169.254/latest/meta-data/")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ssrf_allows_when_opt_in() {
        let tool = HTTPTool::new().with_allow_private_ips(true);
        let result = tool.check_ssrf("http://127.0.0.1:6379/").await;
        assert!(result.is_ok());
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
        let r = t
            .run(r#"{"url":"http://x","method":"put"}"#.to_string())
            .await;
        assert!(r.is_err());
    }
}
