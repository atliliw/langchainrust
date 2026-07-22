//! HTTP tool with SSRF protection

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::core::tools::ToolError;
use crate::BaseTool;

/// Check if an IP address is private/internal (SSRF protection).
///
/// Blocks:
/// - 127.0.0.0/8 (loopback)
/// - 10.0.0.0/8 (private class A)
/// - 172.16.0.0/12 (private class B)
/// - 192.168.0.0/16 (private class C)
/// - 169.254.0.0/16 (link-local / cloud metadata)
/// - ::1 (IPv6 loopback)
/// - fc00::/7 (IPv6 unique local)
/// - fe80::/10 (IPv6 link-local)
fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            // 127.0.0.0/8
            octets[0] == 127
            // 10.0.0.0/8
            || octets[0] == 10
            // 172.16.0.0/12
            || (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31)
            // 192.168.0.0/16
            || (octets[0] == 192 && octets[1] == 168)
            // 169.254.0.0/16 (link-local / cloud metadata)
            || (octets[0] == 169 && octets[1] == 254)
            // 0.0.0.0
            || *v4 == Ipv4Addr::UNSPECIFIED
        }
        IpAddr::V6(v6) => {
            // ::1 loopback
            v6.is_loopback()
            // fc00::/7 unique local (manual check for MSRV compat; is_unique_local needs 1.84)
            || (v6.segments()[0] & 0xfe00) == 0xfc00
            // fe80::/10 link-local
            || matches!(v6.segments(), [0xfe80, ..])
            // :: (unspecified)
            || *v6 == Ipv6Addr::UNSPECIFIED
        }
    }
}

/// Resolve a URL hostname and check if it points to a private IP.
fn url_points_to_private_ip(url: &str) -> Result<bool, ToolError> {
    let parsed = url::Url::parse(url)
        .map_err(|e| ToolError::InvalidInput(format!("Invalid URL: {}", e)))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| ToolError::InvalidInput("URL has no host".to_string()))?;

    // Try parsing as IP directly
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(is_private_ip(&ip));
    }

    // DNS resolution for hostnames
    // Use std::net for synchronous resolution (acceptable in this security check context)
    let addrs: Vec<IpAddr> = std::net::ToSocketAddrs::to_socket_addrs(&format!("{}:{}", host, parsed.port_or_known_default().unwrap_or(80)))
        .map_err(|e| ToolError::ExecutionFailed(format!("DNS resolution failed for {}: {}", host, e)))?
        .map(|a| a.ip())
        .collect();

    Ok(addrs.iter().any(is_private_ip))
}

/// HTTP request tool (GET/POST) with SSRF protection.
///
/// By default, requests to private/internal IP addresses are blocked.
/// Call `with_allow_private_ips(true)` to explicitly opt-in.
pub struct HTTPTool {
    client: reqwest::Client,
    allow_private_ips: bool,
}

impl HTTPTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            allow_private_ips: false,
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            allow_private_ips: false,
        }
    }

    /// Allow requests to private/internal IP addresses (SSRF opt-in).
    ///
    /// # Security Warning
    /// Enabling this allows the tool to access internal services (e.g., cloud metadata,
    /// local databases, internal APIs). Only enable in controlled environments.
    pub fn with_allow_private_ips(mut self, allow: bool) -> Self {
        self.allow_private_ips = allow;
        self
    }

    /// Check SSRF protection before making a request.
    fn check_ssrf(&self, url: &str) -> Result<(), ToolError> {
        if self.allow_private_ips {
            return Ok(());
        }
        if url_points_to_private_ip(url)? {
            return Err(ToolError::ExecutionFailed(
                "Request to private/internal IP address is blocked by SSRF protection. \
                 Call .with_allow_private_ips(true) to allow."
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub async fn get(&self, url: &str) -> Result<String, ToolError> {
        self.check_ssrf(url)?;
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
        self.check_ssrf(url)?;
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

        // Public IPs should NOT be blocked
        assert!(!is_private_ip(&IpAddr::from([8, 8, 8, 8])));
        assert!(!is_private_ip(&IpAddr::from([1, 1, 1, 1])));
        assert!(!is_private_ip(&IpAddr::from([172, 15, 0, 1])));
        assert!(!is_private_ip(&IpAddr::from([172, 32, 0, 1])));
    }

    #[test]
    fn test_ssrf_blocks_localhost() {
        let tool = HTTPTool::new();
        let result = tool.check_ssrf("http://127.0.0.1:6379/");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SSRF"));
    }

    #[test]
    fn test_ssrf_blocks_cloud_metadata() {
        let tool = HTTPTool::new();
        let result = tool.check_ssrf("http://169.254.169.254/latest/meta-data/");
        assert!(result.is_err());
    }

    #[test]
    fn test_ssrf_allows_when_opt_in() {
        let tool = HTTPTool::new().with_allow_private_ips(true);
        let result = tool.check_ssrf("http://127.0.0.1:6379/");
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
        let r = t.run(r#"{"url":"http://x","method":"put"}"#.to_string()).await;
        assert!(r.is_err());
    }
}
