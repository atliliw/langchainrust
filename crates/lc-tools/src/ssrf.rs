// lc-tools/src/ssrf.rs
//! SSRF 防护——单一实现,禁止复制。
//!
//! `is_private_ip` / `url_points_to_private_ip` 是安全关键逻辑,必须在全 crate 只有
//! 一份实现,供 [`crate::url_fetch`] 与 [`crate::extended::http`] 复用(评审 Q1)。
//! 任何规则演化(补 CGNAT 100.64.0.0/10、新 IPv6 特殊段等)都只能在此处修改,
//! 否则两个入口会规则分叉:"URLFetch 拦住内网、HTTP 工具放行内网"。

use std::net::IpAddr;

use lc_core::tools::ToolError;

/// Check if an IP address is private/internal (SSRF protection).
pub(crate) fn is_private_ip(ip: &IpAddr) -> bool {
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
pub(crate) async fn url_points_to_private_ip(url: &str) -> Result<bool, ToolError> {
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
