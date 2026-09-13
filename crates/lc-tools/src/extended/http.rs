//! HTTP tool with SSRF protection

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::ssrf::{guarded_get, guarded_post_json};
use lc_core::tools::ToolError;
use lc_core::BaseTool;

/// HTTP request tool (GET/POST) with SSRF protection.
pub struct HTTPTool {
    /// Per-request timeout handed to the per-request pinned client.
    timeout: Duration,
    allow_private_ips: bool,
}

impl HTTPTool {
    /// Creates an HTTP tool with a 30s timeout and SSRF protection enabled.
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            allow_private_ips: false,
        }
    }

    /// Creates an HTTP tool with a custom timeout (SSRF protection enabled).
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout,
            allow_private_ips: false,
        }
    }

    /// Allow requests to private/internal IP addresses (SSRF opt-in).
    pub fn with_allow_private_ips(mut self, allow: bool) -> Self {
        self.allow_private_ips = allow;
        self
    }

    /// Sends a GET request, following redirects with SSRF checks per hop.
    pub async fn get(&self, url: &str) -> Result<String, ToolError> {
        // SSRF: guarded_get resolves once, validates every answer, pins the validated
        // IPs, then follows redirects manually with the same treatment per hop.
        guarded_get(url, !self.allow_private_ips, Some(self.timeout))
            .await?
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }

    /// Sends a POST request with a JSON body (single hop, IP-pinned SSRF guard).
    pub async fn post(&self, url: &str, body: Value) -> Result<String, ToolError> {
        // POST never follows redirects (the 3xx response is returned as-is); resolve-once
        // plus IP pinning still closes the DNS-rebinding window on this single hop.
        guarded_post_json(url, &body, !self.allow_private_ips, Some(self.timeout))
            .await?
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
        // Rejection happens after resolve but before connect, so no listener is
        // contacted — works offline even though Windows answers every loopback port.
        let tool = HTTPTool::new();
        let result = tool.get("http://127.0.0.1:6379/").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SSRF"));
    }

    #[tokio::test]
    async fn test_ssrf_blocks_cloud_metadata() {
        let tool = HTTPTool::new();
        let result = tool.get("http://169.254.169.254/latest/meta-data/").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SSRF"));
    }

    #[test]
    fn test_with_timeout_is_recorded() {
        // The opt-in flag and custom timeout are plumbed into guarded_get as
        // (!allow, Some(timeout)); assert the configuration side directly.
        let tool = HTTPTool::with_timeout(Duration::from_secs(7)).with_allow_private_ips(true);
        assert_eq!(tool.timeout, Duration::from_secs(7));
        assert!(tool.allow_private_ips);
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
