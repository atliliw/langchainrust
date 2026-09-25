// lc-mcp/src/transport/stateless.rs
//! Stateless HTTP POST transport (2026-07-28 track, 0.22.0 S2.1).
//!
//! Every request is self-contained: one HTTP POST carrying the JSON-RPC
//! envelope plus `_meta` (protocol version / client identity / optional
//! `requestState`), tagged with the `Mcp-Method` / `Mcp-Name` headers so
//! gateways can route and throttle without parsing the body. No handshake,
//! no session id, no sticky routing.

use reqwest::Client;
use std::time::Duration;

use crate::auth::AuthScheme;
use crate::protocol::{
    MCPError, MCPRequest, MCPResponse, RequestMeta, MCP_METHOD_HEADER, MCP_NAME_HEADER,
    MCP_VERSION_STATELESS,
};

/// Default request timeout for stateless calls.
const STATELESS_TIMEOUT: Duration = Duration::from_secs(60);

/// Stateless HTTP POST transport.
///
/// Each `post_jsonrpc` posts one JSON-RPC envelope and awaits the response
/// body. There is no connection state: construction is infallible, failures
/// surface per-request. Server-initiated interaction goes through MRTR
/// `input_required` rather than a push channel.
pub struct StatelessTransport {
    url: String,
    http: Client,
    auth: Option<AuthScheme>,
}

impl StatelessTransport {
    /// Creates a transport for a stateless MCP endpoint (e.g.
    /// `https://host/mcp`).
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            http: Client::builder()
                .timeout(STATELESS_TIMEOUT)
                // F2: never follow redirects. A hostile server 302ing into an
                // intranet host would otherwise be fetched silently; disabling
                // automatic redirects guarantees the request (and any bearer
                // token from `with_auth`) is only ever sent to the configured
                // host. A 3xx is surfaced to the caller to inspect instead.
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default(),
            auth: None,
        }
    }

    /// Creates a transport with bearer auth attached to every request.
    pub fn with_auth(url: impl Into<String>, auth: AuthScheme) -> Self {
        Self {
            url: url.into(),
            http: Client::builder()
                .timeout(STATELESS_TIMEOUT)
                // F2: see [`Self::new`] — redirects are not followed, so the
                // bearer token is never forwarded to a redirect target.
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default(),
            auth: Some(auth),
        }
    }

    /// The endpoint URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Sends one JSON-RPC request over POST with the routing headers.
    ///
    /// Public so gateway adapters can reuse the exact wire behavior.
    pub async fn post_jsonrpc(&self, req: &MCPRequest) -> Result<MCPResponse, MCPError> {
        // F2: validate the target before building the request, so neither the
        // JSON-RPC payload nor a `with_auth` bearer token is ever handed to a
        // non-https / hostless / intranet target (SSRF + credential leak).
        validate_target_url(&self.url)?;
        let mut builder = self
            .http
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header(MCP_METHOD_HEADER, req.method.as_str())
            .header(MCP_NAME_HEADER, "mcp");
        if let Some(auth) = &self.auth {
            if let Some(value) = auth.header_value() {
                builder = builder.header("Authorization", value);
            }
        }
        let resp = builder
            .json(req)
            .send()
            .await
            .map_err(|e| MCPError::new(-32000, format!("stateless POST failed: {e}")))?;

        let status = resp.status();
        if status.as_u16() == 401 {
            return Err(MCPError::new(
                crate::protocol::MCP_ERROR_UNAUTHORIZED,
                "unauthorized: token rejected by server".to_string(),
            ));
        }
        if !status.is_success() {
            return Err(MCPError::new(
                -32000,
                format!("stateless POST failed: HTTP {}", status.as_u16()),
            ));
        }
        let resp: MCPResponse = resp
            .json()
            .await
            .map_err(|e| MCPError::new(-32700, format!("invalid JSON-RPC response: {e}")))?;
        // M-4: the server's response id must echo this request's id. A missing or
        // mismatched id (replay / interleaved response / promiscuous server) must
        // not be accepted as the answer to this request — the stdio/SSE transports
        // all match; only the stateless track skipped it.
        if resp.id.as_ref() != Some(&req.id) {
            return Err(MCPError::new(
                -32700,
                format!(
                    "JSON-RPC response id mismatch: sent {:?}, got {:?}",
                    req.id, resp.id
                ),
            ));
        }
        Ok(resp)
    }
}

/// Convenience: the default [`RequestMeta`] for the stateless track.
pub fn default_meta() -> RequestMeta {
    RequestMeta::default_for(MCP_VERSION_STATELESS)
}

/// F2 target-URL validation: the stateless endpoint must be `https`, except
/// cleartext `http` on a loopback host (local dev servers only). This rejects
/// sending request payloads or bearer tokens to cleartext-on-the-internet and
/// to URLs without a resolvable host string, before any network I/O.
fn validate_target_url(url: &str) -> Result<(), MCPError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| MCPError::new(-32000, format!("invalid target URL: {e}")))?;
    let is_loopback = match parsed.host_str() {
        Some(h) => {
            let lower = h.trim_matches(['[', ']']).to_ascii_lowercase();
            lower == "localhost"
                || lower == "localhost."
                || lower == "127.0.0.1"
                || lower == "::1"
        }
        None => false,
    };
    if parsed.scheme() != "https" && !(parsed.scheme() == "http" && is_loopback) {
        return Err(MCPError::new(
            -32000,
            format!("stateless target must use HTTPS (got scheme '{}')", parsed.scheme()),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{MCP_METHOD_HEADER, MCP_NAME_HEADER, MCP_VERSION_STATELESS};

    /// F2: cleartext `http` on a non-loopback host is rejected (the target
    /// would be reachable by a path that leaks payload/token); https and
    /// loopback http are accepted.
    #[test]
    fn f2_validate_target_url_scheme_and_host() {
        assert!(validate_target_url("https://example.com/mcp").is_ok());
        assert!(validate_target_url("https://10.0.0.5/mcp").is_ok());
        assert!(validate_target_url("http://localhost:8080/mcp").is_ok());
        assert!(validate_target_url("http://127.0.0.1:8080/mcp").is_ok());
        assert!(validate_target_url("http://[::1]:8080/mcp").is_ok());

        assert!(validate_target_url("http://example.com/mcp").is_err());
        assert!(validate_target_url("http://192.168.1.10/mcp").is_err());
        assert!(validate_target_url("ftp://example.com/mcp").is_err());
        assert!(validate_target_url("not a url").is_err());
    }

    /// F2: a redirect is NOT followed, so a bearer token is never forwarded to
    /// a redirect target. A server answering 302 must yield the 3xx as-is, not
    /// march on to the Location.
    #[tokio::test]
    async fn f2_redirect_is_not_followed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // A server that 302s to a second listener (a different host string).
        let sink = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let sink_addr = sink.local_addr().unwrap();
        let sink_task = tokio::spawn(async move {
            if let Ok((mut sock, _)) = sink.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                    .await;
            }
        });

        let origin = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin_port = origin.local_addr().unwrap().port();
        let target = format!("http://127.0.0.1:{origin_port}/mcp");
        let origin_task = tokio::spawn(async move {
            if let Ok((mut sock, _)) = origin.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let head = format!(
                    "HTTP/1.1 302 Found\r\nContent-Length: 0\r\nLocation: http://{}/evil\r\n\r\n",
                    sink_addr
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.flush().await;
            }
        });

        // If redirects were followed, the client would reach the sink (200) and
        // parse `{}`; hitting the 302 proves the policy is `none`.
        let transport = StatelessTransport::with_auth(target, AuthScheme::Bearer("secret-token".into()));
        let err = transport
            .post_jsonrpc(&MCPRequest::new(1, "ping", None))
            .await
            .expect_err("a 302 must not be followed");
        assert!(
            err.message.contains("302"),
            "error: {:?}",
            err.message
        );

        origin_task.abort();
        sink_task.abort();
    }

    /// Meta defaults carry the stateless version and client identity.
    #[test]
    fn default_meta_carries_stateless_version() {
        let meta = default_meta();
        assert_eq!(meta.protocol_version, MCP_VERSION_STATELESS);
        assert_eq!(meta.client_info.name, "langchainrust-mcp-client");
        assert!(meta.request_state.is_none());
    }

    /// Meta with request_state roundtrips through serde (M2 snapshot check).
    #[test]
    fn meta_request_state_roundtrip() {
        let meta = default_meta().with_request_state("tok-123");
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("2026-07-28"), "{json}");
        assert!(json.contains("requestState"), "{json}");
        assert!(json.contains("requestState\":\"tok-123"), "{json}");
        let back: RequestMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back, meta);
    }

    /// Header constant values (contract for gateways).
    #[test]
    fn header_constants() {
        assert_eq!(MCP_METHOD_HEADER, "Mcp-Method");
        assert_eq!(MCP_NAME_HEADER, "Mcp-Name");
    }
}
