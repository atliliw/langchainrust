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

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use crate::tools::ToolError;

/// Check if an IP address is private/internal or otherwise non-routable for SSRF
/// purposes.
///
/// A3: the set now covers the full RFC 6890 / 5735 / 7913 special-purpose ranges that
/// a server would never legitimately need to fetch (CGNAT `100.64/10`, benchmarking
/// `198.18/15`, TEST-NET documentation blocks, multicast, reserved) in addition to the
/// classic private/link-local/loopback ranges. Blocking these closes SSRF paths that
/// probe cloud-metadata, loopback services, or internal benchmarking hosts.
pub fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            // 0.0.0.0/8  "this network"
            if v4.octets()[0] == 0 {
                return true;
            }
            is_private_ipv4(*v4)
        }
        IpAddr::V6(v6) => {
            // IPv4-mapped IPv6 (::ffff:a.b.c.d) targets an IPv4 endpoint directly, so it must
            // be converted back to a V4 check; otherwise addresses like ::ffff:127.0.0.1 /
            // ::ffff:169.254.169.254 would bypass the protection
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_private_ip(&IpAddr::V4(v4));
            }
            is_private_ipv6(*v6)
        }
    }
}

/// IPv4 private / special-purpose ranges (RFC 6890 / 5735).
fn is_private_ipv4(v4: std::net::Ipv4Addr) -> bool {
    let [a, b, _, _] = v4.octets();
    match a {
        // 0.0.0.0/8       "this network"
        0 => true,
        // 10.0.0.0/8      private
        10 => true,
        // 100.64.0.0/10   shared address space (CGNAT)
        100 => b & 0b1100_0000 == 0b0100_0000,
        // 127.0.0.0/8     loopback
        127 => true,
        // 169.254.0.0/16  link-local
        169 => b == 254,
        // 172.16.0.0/12   private
        172 => (16..=31).contains(&b),
        // 192.0.0.0/24 IETF protocol assignments + 192.0.2.0/24 TEST-NET-1 (doc)
        192 if b == 0 => true,
        // 192.88.99.0/24  6to4 relay anycast (deprecated)
        192 if b == 88 => true,
        // 192.168.0.0/16  private
        192 if b == 168 => true,
        // 198.18.0.0/15 benchmarking + 198.51.100.0/24 TEST-NET-2 (doc)
        198 => (b & 0xfe) == 0x12 || b == 51,
        // 203.0.113.0/24   TEST-NET-3 (doc)
        203 if b == 0 => v4.octets()[2] == 113,
        // 224.0.0.0/4 multicast + 240.0.0.0/4 reserved
        224..=255 => true,
        _ => false,
    }
}

/// IPv6 private / special-purpose ranges.
fn is_private_ipv6(v6: std::net::Ipv6Addr) -> bool {
    let seg = v6.segments();
    // ::1 loopback
    if v6.is_loopback() {
        return true;
    }
    // fc00::/7 unique-local
    if (seg[0] & 0xfe00) == 0xfc00 {
        return true;
    }
    // fe80::/10 link-local
    if matches!(seg, [0xfe80, ..]) {
        return true;
    }
    // :: unspecified
    if v6 == std::net::Ipv6Addr::UNSPECIFIED {
        return true;
    }
    // fec0::/10          site-local (deprecated, RFC 3879)
    if (seg[0] & 0xffc0) == 0xfec0 {
        return true;
    }
    // ff00::/8            multicast
    if (seg[0] & 0xff00) == 0xff00 {
        return true;
    }
    // 2001:db8::/32        documentation
    if seg[0] == 0x2001 && seg[1] == 0x0db8 {
        return true;
    }
    // 2002::/16            6to4 (RFC 3056)
    if seg[0] == 0x2002 {
        return true;
    }
    // 64:ff9b::/96         NAT64 well-known prefix (RFC 6052): the low 32 bits
    //                      embed an IPv4 host (::7f00:1 == 127.0.0.1). On a
    //                      NAT64 network http://[64:ff9b::7f00:1]/ reaches the
    //                      loopback — without this the private-IP table claimed
    //                      to be complete while 127.x/10.x/169.254.x slipped in.
    if seg[0] == 0x0064 && seg[1] == 0xff9b {
        return is_private_ipv4(embedded_v4(seg));
    }
    // ::/96                IPv4-compatible IPv6 (RFC 4291): same low-32 embed;
    //                      e.g. ::7f00:1 == 127.0.0.1. The first six groups are
    //                      zero and the v4 lives in the last two.
    if seg[0..6].iter().all(|&s| s == 0) && (seg[6] != 0 || seg[7] != 0) {
        return is_private_ipv4(embedded_v4(seg));
    }
    false
}

/// Low 32 bits of the last two 16-bit segments, interpreted as an IPv4 address
/// (both the NAT64 and the IPv4-compatible embed schemes above put the v4 in
/// segments 6..8; reading the high half instead silently mapped private v4 to the
/// 0.0.0.0 route and mis-classified every public embed as private).
fn embedded_v4(seg: [u16; 8]) -> std::net::Ipv4Addr {
    let hi = seg[6] as u32;
    let lo = seg[7] as u32;
    std::net::Ipv4Addr::from((hi << 16) | lo)
}

/// Default timeout for guarded requests when the caller does not supply one.
pub const DEFAULT_GUARDED_TIMEOUT: Duration = Duration::from_secs(30);

/// Resolve a URL hostname and check if it points to a private IP (async).
///
/// If DNS returns several addresses (A + AAAA, round-robin) the result is
/// `true` when **any** of them is private: a checker that validated only the
/// first answer could be bypassed by an answer list whose first entry is
/// public and whose later entries are internal.
pub async fn url_points_to_private_ip(url: &str) -> Result<bool, ToolError> {
    let parsed = parse_http_url(url)?;
    let addrs = resolve_url_addrs(&parsed).await?;
    Ok(addrs.iter().any(|sa| is_private_ip(&sa.ip())))
}

/// Parse an http(s) URL, rejecting missing hosts and non-http(s) schemes.
fn parse_http_url(url: &str) -> Result<url::Url, ToolError> {
    let parsed =
        url::Url::parse(url).map_err(|e| ToolError::InvalidInput(format!("Invalid URL: {}", e)))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(ToolError::InvalidInput(format!(
            "URL scheme not supported: {}",
            parsed.scheme()
        )));
    }
    if parsed.host_str().is_none() {
        return Err(ToolError::InvalidInput("URL has no host".to_string()));
    }
    Ok(parsed)
}

/// Resolve a parsed URL's host to the concrete socket addresses the connection
/// will be pinned to. IP-literal hosts short-circuit without DNS.
///
/// A3: this is the *only* resolution in the guarded path. The addresses
/// returned here are validated and then handed to reqwest via
/// `resolve_to_addrs`, so the checker and the actual TCP connection can never
/// disagree (see [`pinned_client`]) — the previous check-then-re-resolve
/// implementation left a DNS-rebinding (TOCTOU) window.
async fn resolve_url_addrs(parsed: &url::Url) -> Result<Vec<SocketAddr>, ToolError> {
    let port = parsed.port_or_known_default().unwrap_or(80);

    // IP literal: no DNS lookup at all, just attach the URL port.
    //
    // Must match on `Url::host()`'s typed enum rather than parsing
    // `host_str()`: for IPv6 literals `host_str()` keeps the square brackets
    // ("[::1]"), which fails `IpAddr` parsing and falls through to DNS. Linux
    // getaddrinfo rejects the bracketed name with EAI_NONAME while Windows
    // tolerates it — so the IPv6 branch silently worked only on Windows.
    let host = match parsed.host() {
        Some(url::Host::Ipv4(ip)) => return Ok(vec![SocketAddr::new(IpAddr::V4(ip), port)]),
        Some(url::Host::Ipv6(ip)) => return Ok(vec![SocketAddr::new(IpAddr::V6(ip), port)]),
        Some(url::Host::Domain(h)) => h,
        None => {
            return Err(ToolError::InvalidInput("URL has no host".to_string()));
        }
    };

    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| {
            ToolError::ExecutionFailed(format!("DNS resolution failed for {}: {}", host, e))
        })?
        .collect();

    if addrs.is_empty() {
        return Err(ToolError::ExecutionFailed(format!(
            "DNS resolution returned no addresses for {}",
            host
        )));
    }
    Ok(addrs)
}

/// Reject the address set when **any** resolved address is private.
fn ensure_all_public(addrs: &[SocketAddr]) -> Result<(), ToolError> {
    if let Some(bad) = addrs.iter().map(SocketAddr::ip).find(is_private_ip) {
        return Err(ToolError::ExecutionFailed(format!(
            "Request to private/internal IP address ({bad}) is blocked by SSRF protection. \
             Call .with_allow_private_ips(true) to allow."
        )));
    }
    Ok(())
}

/// Build a one-shot client whose DNS for `host` is pinned to the already
/// validated `addrs`. reqwest resolves the override key by hostname (the URL's
/// port in the SocketAddr list is ignored for non-literal hosts), while the
/// `Host` header and TLS SNI keep using the URL hostname, so virtual hosting
/// and certificate validation are unaffected. IP-literal hosts need no pin:
/// reqwest connects to the literal directly, which is the validated address.
fn pinned_client(
    parsed: &url::Url,
    addrs: &[SocketAddr],
    timeout: Option<Duration>,
) -> Result<reqwest::Client, ToolError> {
    let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
    builder = builder.timeout(timeout.unwrap_or(DEFAULT_GUARDED_TIMEOUT));
    // Typed host enum: a bracketed IPv6 literal from host_str() would fail
    // IpAddr parsing and be mis-pinned as a DNS name (see resolve_url_addrs).
    let is_ip_literal = matches!(
        parsed.host(),
        Some(url::Host::Ipv4(_)) | Some(url::Host::Ipv6(_))
    );
    if !is_ip_literal {
        let host = parsed.host_str().expect("host checked above");
        builder = builder.resolve_to_addrs(host, addrs);
    }
    builder
        .build()
        .map_err(|e| ToolError::ExecutionFailed(format!("failed to build HTTP client: {}", e)))
}

/// Maximum number of hops for manual redirect following (matches the reqwest default).
const MAX_REDIRECTS: usize = 10;

/// GET request with per-hop SSRF checks and IP pinning, following redirects manually.
///
/// reqwest follows 30x by default but does not re-check the redirect target, which is the
/// root of the "first hop checked, redirect into the intranet" SSRF bypass. Here every hop
/// is resolved once, **all** resolved addresses are validated, and the same address set is
/// pinned onto a one-shot client (`ClientBuilder::resolve_to_addrs`) before sending — the
/// `Host` header and TLS SNI still carry the hostname. A hostile DNS answer therefore
/// cannot pass the check with a public IP and re-resolve to an internal one at connect
/// time (DNS-rebinding / TOCTOU closed, A3). The redirect target is taken from the
/// Location header (relative URLs supported) and non-http(s) protocols are rejected.
///
/// A fresh short-lived client is built per hop because reqwest only accepts DNS overrides
/// on the client builder. `timeout` bounds the whole request; `None` falls back to
/// [`DEFAULT_GUARDED_TIMEOUT`] (30s).
///
/// When `check_ssrf = false`, the SSRF check is skipped (corresponding to
/// `with_allow_private_ips(true)`), but address pinning and manual redirect following are
/// preserved.
pub async fn guarded_get(
    url: &str,
    check_ssrf: bool,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, ToolError> {
    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        let parsed = parse_http_url(&current)?;
        let addrs = resolve_url_addrs(&parsed).await?;
        if check_ssrf {
            ensure_all_public(&addrs)?;
        }
        let client = pinned_client(&parsed, &addrs, timeout)?;

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

/// POST request with a JSON body, behind the same resolve-once / validate-all / pin-IP
/// protection as [`guarded_get`]. POST intentionally does not follow redirects (the 3xx
/// response is handed back as-is), so only the first hop is checked. (A3)
pub async fn guarded_post_json(
    url: &str,
    body: &serde_json::Value,
    check_ssrf: bool,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, ToolError> {
    let parsed = parse_http_url(url)?;
    let addrs = resolve_url_addrs(&parsed).await?;
    if check_ssrf {
        ensure_all_public(&addrs)?;
    }
    let client = pinned_client(&parsed, &addrs, timeout)?;

    client
        .post(url)
        .json(body)
        .send()
        .await
        .map_err(|e| ToolError::ExecutionFailed(format!("HTTP request failed: {}", e)))
}

/// Hard ceiling on a single fetch body (M2): a caller or hostile server can never
/// force unbounded buffering. Consumers apply their own (usually much smaller) limit
/// on top of this.
pub const MAX_FETCH_BYTES: usize = 50 * 1024 * 1024;

/// Read a response body with a hard byte cap, **streaming** so an external URL cannot
/// force unbounded buffering (M2). The earlier pattern (`response.text()`) buffered the
/// whole body into memory before truncating, so a server could stream arbitrary data
/// within the 30s wall-clock timeout and OOM the process.
///
/// Returns the body text (its tail trimmed to a UTF-8 char boundary, so a cut never
/// mangles a multi-byte character) and whether it was truncated by hitting `max_bytes`.
pub async fn read_body_bounded(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<(String, bool), ToolError> {
    use futures_util::StreamExt;

    let mut body: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    let mut truncated = false;
    while let Some(next) = stream.next().await {
        let chunk = next.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to read response body: {e}"))
        })?;
        let remaining = max_bytes.saturating_sub(body.len());
        if chunk.len() > remaining {
            body.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        body.extend_from_slice(&chunk);
    }

    let text = if truncated {
        // Trim the capped body to a UTF-8 char boundary so we never emit a mangled char.
        let bytes = &body;
        let valid = match std::str::from_utf8(bytes) {
            Ok(_) => bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        std::str::from_utf8(&bytes[..valid])
            .unwrap_or("")
            .to_string()
    } else {
        String::from_utf8_lossy(&body).into_owned()
    };
    Ok((text, truncated))
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
    fn nat64_and_ipv4_compatible_embed_private_v4() {
        // M1: 64:ff9b::/96 (NAT64 well-known) and ::/96 (IPv4-compatible) embed the
        // IPv4 in the low 32 bits — a private v4 inside must be caught.
        assert!(is_private_ip(&"64:ff9b::7f00:1".parse::<IpAddr>().unwrap())); // 127.0.0.1
        assert!(is_private_ip(&"64:ff9b::a00:1".parse::<IpAddr>().unwrap())); // 10.0.0.1
        assert!(is_private_ip(
            &"64:ff9b::a9fe:a9fe".parse::<IpAddr>().unwrap()
        )); // 169.254.169.254
        assert!(is_private_ip(
            &"64:ff9b::c000:201".parse::<IpAddr>().unwrap()
        )); // 192.0.2.1 doc
            // A public IPv4 embedded in the NAT64 prefix stays allowed.
        assert!(!is_private_ip(
            &"64:ff9b::808:808".parse::<IpAddr>().unwrap()
        )); // 8.8.8.8
            // IPv4-compatible IPv6 (::/96) embeds v4 in the low 32 bits too.
        assert!(is_private_ip(&"::7f00:1".parse::<IpAddr>().unwrap())); // 127.0.0.1
        assert!(is_private_ip(&"::a00:1".parse::<IpAddr>().unwrap())); // 10.0.0.1
    }

    #[test]
    fn regular_ipv6_unchanged() {
        assert!(is_private_ip(&"::1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"fc00::1".parse::<IpAddr>().unwrap()));
        assert!(is_private_ip(&"fe80::1".parse::<IpAddr>().unwrap()));
        // 2001:db8::/32 is the documentation range (RFC 6890) — now flagged special.
        assert!(is_private_ip(&"2001:db8::1".parse::<IpAddr>().unwrap()));
        // A genuine public IPv6 must still be allowed.
        assert!(!is_private_ip(
            &"2606:4700:4700::1111".parse::<IpAddr>().unwrap()
        ));
    }

    #[test]
    fn ipv4_special_ranges_are_blocked() {
        // A3 additions per RFC 6890/5735.
        let blocked: &[&str] = &[
            "100.64.0.1",    // CGNAT
            "100.127.255.1", // CGNAT upper bound
            "198.18.0.1",    // benchmarking
            "198.19.255.1",  // benchmarking upper bound
            "192.0.0.1",     // IETF protocol assignments
            "192.0.2.1",     // TEST-NET-1
            "198.51.100.1",  // TEST-NET-2
            "203.0.113.1",   // TEST-NET-3
            "224.0.0.1",     // multicast
            "240.0.0.1",     // reserved
            "0.1.2.3",       // "this network"
        ];
        for s in blocked {
            assert!(
                is_private_ip(&s.parse::<IpAddr>().unwrap()),
                "expected {s} to be flagged"
            );
        }

        // Public + CGNAT-adjacent-but-public addresses must still pass.
        for s in &["100.128.0.1", "198.20.0.1", "8.8.8.8", "1.1.1.1"] {
            assert!(
                !is_private_ip(&s.parse::<IpAddr>().unwrap()),
                "expected {s} to be allowed"
            );
        }
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

    // ---- A3: resolve-once / validate-all / IP pinning -----------------------

    #[test]
    fn parse_http_url_rejects_scheme_and_missing_host() {
        assert!(parse_http_url("file:///etc/passwd").is_err());
        assert!(parse_http_url("ftp://b.com/x").is_err());
        assert!(parse_http_url("not a url").is_err());
        // url::Url accepts http:/etc but records no host — still rejected.
        assert!(parse_http_url("http://").is_err());
        assert!(parse_http_url("https://example.com/x").is_ok());
    }

    #[test]
    fn ensure_all_public_blocks_when_any_answer_is_private() {
        // DNS round-robin with one internal answer must reject the whole set.
        let mixed: Vec<SocketAddr> = vec![
            "8.8.8.8:443".parse().unwrap(),
            "10.0.0.5:443".parse().unwrap(),
            "1.1.1.1:443".parse().unwrap(),
        ];
        let err = ensure_all_public(&mixed).unwrap_err();
        assert!(err.to_string().contains("SSRF"), "got: {err}");

        let public: Vec<SocketAddr> = vec![
            "8.8.8.8:443".parse().unwrap(),
            "[2606:4700:4700::1111]:443".parse().unwrap(),
        ];
        ensure_all_public(&public).expect("all-public set passes");

        // IPv4-mapped IPv6 in the answer set is unmasked by is_private_ip.
        let mapped: Vec<SocketAddr> = vec!["[::ffff:169.254.169.254]:80".parse().unwrap()];
        assert!(ensure_all_public(&mapped).is_err());
    }

    #[tokio::test]
    async fn resolve_url_addrs_ip_literals_bypass_dns() {
        // Numeric hosts must resolve locally (no DNS query, works offline) and
        // carry the URL's effective port.
        let url = parse_http_url("http://127.0.0.1:8080/").unwrap();
        let addrs = resolve_url_addrs(&url).await.unwrap();
        assert_eq!(addrs, vec!["127.0.0.1:8080".parse::<SocketAddr>().unwrap()]);

        let url = parse_http_url("https://8.8.8.8/").unwrap();
        let addrs = resolve_url_addrs(&url).await.unwrap();
        assert_eq!(addrs, vec!["8.8.8.8:443".parse::<SocketAddr>().unwrap()]);

        // Bracketed IPv6 literal (incl. IPv4-mapped) must NOT fall through to
        // DNS: host_str() keeps the brackets, which IpAddr::parse rejects and
        // Linux getaddrinfo rejects with EAI_NONAME (Windows tolerated it).
        let url = parse_http_url("http://[::ffff:169.254.169.254]/latest").unwrap();
        let addrs = resolve_url_addrs(&url).await.unwrap();
        assert_eq!(
            addrs,
            vec!["[::ffff:169.254.169.254]:80".parse::<SocketAddr>().unwrap()]
        );

        let url = parse_http_url("http://[::1]:9000/").unwrap();
        let addrs = resolve_url_addrs(&url).await.unwrap();
        assert_eq!(addrs, vec!["[::1]:9000".parse::<SocketAddr>().unwrap()]);
    }

    #[tokio::test]
    async fn guarded_get_blocks_loopback_before_connecting() {
        // The rejection happens after resolution but before the pinned client is
        // built/sent, so no network I/O occurs — no listener needed (and on
        // Windows every loopback port would otherwise appear open).
        let err = guarded_get("http://127.0.0.1:1/", true, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("SSRF"), "got: {err}");
    }

    #[tokio::test]
    async fn guarded_get_blocks_link_local_before_connecting() {
        let err = guarded_get("http://[::ffff:169.254.169.254]/latest", true, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("SSRF"), "got: {err}");
    }

    #[tokio::test]
    async fn guarded_get_rejects_non_http_scheme_without_sending() {
        let err = guarded_get("file:///etc/passwd", true, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("scheme"), "got: {err}");
    }

    #[tokio::test]
    async fn guarded_post_json_blocks_private_before_connecting() {
        let err = guarded_post_json(
            "http://169.254.169.254/latest/meta-data/",
            &serde_json::json!({}),
            true,
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("SSRF"), "got: {err}");
    }
}
