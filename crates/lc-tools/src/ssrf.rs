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
            // IPv4-mapped IPv6 (::ffff:a.b.c.d) 直连的是 IPv4 端点,必须转回 V4 判定,
            // 否则 ::ffff:127.0.0.1 / ::ffff:169.254.169.254 这类地址会绕过防护
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_private_ip(&IpAddr::V4(v4));
            }
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

/// 手动跟随重定向的最大跳数(与 reqwest 默认一致)。
pub(crate) const MAX_REDIRECTS: usize = 10;

/// 带 SSRF 逐跳检查的 GET 请求,手动跟随重定向。
///
/// reqwest 默认跟随 30x 但不会重查重定向目标,是"检查首跳放行、重定向进内网"
/// 这条 SSRF 绕过的根源。这里每一跳都先做 `url_points_to_private_ip` 再发送,
/// 重定向目标用 Location 解析(支持相对 URL),且拒绝非 http(s) 协议。
///
/// `check_ssrf = false` 时跳过 SSRF 检查(对应 `with_allow_private_ips(true)`),
/// 但保留手动重定向跟随行为。
pub(crate) async fn guarded_get(
    client: &reqwest::Client,
    url: &str,
    check_ssrf: bool,
) -> Result<reqwest::Response, ToolError> {
    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        if check_ssrf && url_points_to_private_ip(&current).await? {
            return Err(ToolError::ExecutionFailed(
                "Request to private/internal IP address is blocked by SSRF protection. \
                 Call .with_allow_private_ips(true) to allow."
                    .to_string(),
            ));
        }

        let resp = client
            .get(&current)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP 请求失败: {}", e)))?;

        if !resp.status().is_redirection() {
            return Ok(resp);
        }

        // 有 Location 才继续跟随;没有则把 3xx 响应原样交给调用方
        let Some(location) = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
        else {
            return Ok(resp);
        };
        current = resolve_redirect(&current, location)?;
    }
    Err(ToolError::ExecutionFailed(format!(
        "请求重定向次数超过上限 {} 次",
        MAX_REDIRECTS
    )))
}

/// 把 Location 头(可能相对)解析为绝对 URL,拒绝非 http(s) 协议。
fn resolve_redirect(base: &str, location: &str) -> Result<String, ToolError> {
    let joined = url::Url::parse(base)
        .and_then(|base_url| base_url.join(location))
        .map_err(|e| ToolError::InvalidInput(format!("非法重定向目标: {}", e)))?;
    if joined.scheme() != "http" && joined.scheme() != "https" {
        return Err(ToolError::InvalidInput(format!(
            "重定向目标协议不受支持: {}",
            joined.scheme()
        )));
    }
    Ok(joined.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipv4_mapped_ipv6_private_is_blocked() {
        assert!(is_private_ip(&"::ffff:127.0.0.1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"::ffff:10.0.0.1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"::ffff:169.254.169.254".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"::ffff:192.168.1.1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"::ffff:172.16.0.1".parse::<IpAddr>().unwrap()));
    }

    #[test]
    fn ipv4_mapped_ipv6_public_allowed() {
        assert!(!is_private_ip(&"::ffff:8.8.8.8".parse::<IpAddr>().unwrap()));
        assert!(!is_private_ip(&"::ffff:1.1.1.1".parse::<IpAddr>().unwrap()));
    }

    #[test]
    fn regular_ipv6_unchanged() {
        assert!(is_private_ip(&"::1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"fc00::1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"fe80::1".parse::<IpAddr>().unwrap()));
        assert!(!is_private_ip(&"2001:db8::1".parse::<IpAddr>().unwrap()));
    }

    #[test]
    fn resolve_redirect_relative_and_absolute() {
        assert_eq!(
            resolve_redirect("https://a.com/x", "/internal").unwrap(),
            "https://a.com/internal"
        );
        assert_eq!(
            resolve_redirect("https://a.com/x", "https://b.com/y").unwrap(),
            "https://b.com/y"
        );
    }

    #[test]
    fn resolve_redirect_rejects_non_http() {
        assert!(resolve_redirect("https://a.com/x", "file:///etc/passwd").is_err());
        assert!(resolve_redirect("https://a.com/x", "ftp://b.com").is_err());
    }
}
