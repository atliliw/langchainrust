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
        resp.json::<MCPResponse>()
            .await
            .map_err(|e| MCPError::new(-32700, format!("invalid JSON-RPC response: {e}")))
    }
}

/// Convenience: the default [`RequestMeta`] for the stateless track.
pub fn default_meta() -> RequestMeta {
    RequestMeta::default_for(MCP_VERSION_STATELESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{MCP_METHOD_HEADER, MCP_NAME_HEADER, MCP_VERSION_STATELESS};

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
