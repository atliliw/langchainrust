// lc-core/src/ssrf.rs
//! SSRF protection — a single shared implementation, no copies allowed.
//!
//! `is_private_ip` / `url_points_to_private_ip` / `guarded_get` are security-critical;
//! the whole workspace must have exactly one implementation. Originally authored for
//! `lc-tools` (review Q1) and lifted into `lc-core` (0.20.0 S4 P1) so provider crates
//! that cannot depend on `lc-tools` (e.g. `lc-providers`) share the same rules. Any
//! rule evolution (adding CGNAT 100.64.0.0/10, new IPv6 special ranges, etc.) must
//! only change here, otherwise the entry points would diverge: "URLFetch blocks
//! intranet, Whisper allows intranet".

use std::net::IpAddr;

use crate::tools::ToolError;

/// Check if an IP address is private/internal (SSRF protection).
pub fn is_private_ip(ip: &IpAddr) -> bool {
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
            // IPv4-mapped IPv6 (::ffff:a.b.c.d) targets an IPv4 endpoint directly, so it must
            // be converted back to a V4 check; otherwise addresses like ::ffff:127.0.0.1 /
            // ::ffff:169.254.169.254 would bypass the protection
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
pub async fn url_points_to_private_ip(url: &str) -> Result<bool, ToolError> {
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

/// Maximum number of hops for manual redirect following (matches the reqwest default).
const MAX_REDIRECTS: usize = 10;

/// GET request with per-hop SSRF checks, following redirects manually.
///
/// reqwest follows 30x by default but does not re-check the redirect target, which is the
/// root of the "first hop checked, redirect into the intranet" SSRF bypass. Here every hop
/// runs `url_points_to_private_ip` before sending, the redirect target is resolved via the
/// Location header (relative URLs supported), and non-http(s) protocols are rejected.
///
/// When `check_ssrf = false`, the SSRF check is skipped (corresponding to
/// `with_allow_private_ips(true)`), but manual redirect following is preserved.
pub async fn guarded_get(
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
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        if !resp.status().is_redirection() {
            return Ok(resp);
        }

        // Follow only when a Location header is present; otherwise hand the 3xx response back as-is
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
        "request redirect count exceeded the limit of {} times",
        MAX_REDIRECTS
    )))
}

/// Resolves the Location header (possibly relative) into an absolute URL, rejecting non-http(s) protocols.
fn resolve_redirect(base: &str, location: &str) -> Result<String, ToolError> {
    let joined = url::Url::parse(base)
        .and_then(|base_url| base_url.join(location))
        .map_err(|e| ToolError::InvalidInput(format!("invalid redirect target: {}", e)))?;
    if joined.scheme() != "http" && joined.scheme() != "https" {
        return Err(ToolError::InvalidInput(format!(
            "redirect target protocol not supported: {}",
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
        assert!(is_private_ip(
            &"::ffff:127.0.0.1".parse::<IpAddr>().unwrap()
        ));
        assert!(is_private_ip(&"::ffff:10.0.0.1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(
            &"::ffff:169.254.169.254".parse::<IpAddr>().unwrap()
        ));
        assert!(is_private_ip(
            &"::ffff:192.168.1.1".parse::<IpAddr>().unwrap()
        ));
        assert!(is_private_ip(
            &"::ffff:172.16.0.1".parse::<IpAddr>().unwrap()
        ));
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
