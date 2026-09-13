//! B1 Streamable HTTP interop matrix: the framework's official-track
//! Streamable HTTP client (`StreamableMcpClient` / `StreamableHttpTransport`)
//! talking to a spec-faithful MCP Streamable HTTP server fixture
//! (`tests/common/mod.rs`) — the same role an official TS/Python SDK remote
//! server plays in CI.
//!
//! Covered: JSON and SSE response delivery, interleaved server notifications,
//! the initialize handshake + `Mcp-Session-Id` echo/enforcement, stateless
//! servers (no session), 404 session-lost + re-handshake, OAuth 2.1
//! resource-server challenge, provider invalidate/retry, RFC 9728/8414
//! metadata discovery + token-endpoint exchange, version negotiation
//! (Degrade/Reject), per-request timeout, SSE truncation, and the A16
//! fail-closed gate through `MCPToolAdapter`.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lc_core::tools::ToolError;
use lc_core::BaseTool;
use lc_mcp::{
    discover_authorization_server, discover_protected_resource, negotiate_protocol_version,
    BearerTokenProvider, MCPError, MCPToolAdapter, OAuthChallenge, OAuthTokenClient,
    StreamableHttpTransport, StreamableMcpClient, VersionPolicy, MCP_ERROR_REQUEST_TIMEOUT,
    MCP_ERROR_SESSION_LOST, MCP_ERROR_UNAUTHORIZED, MCP_ERROR_VERSION_UNSUPPORTED, MCP_VERSION,
};
use serde_json::{json, Value};

use common::{start, FixtureConfig, ResponseMode};

// ---------------------------------------------------------------------------
// Handshake + sessions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn initialize_handshake_json_mode() {
    let server = start(FixtureConfig::json()).await;
    let client = StreamableMcpClient::connect(&server.url)
        .await
        .expect("handshake");

    let info = client.protocol_info();
    assert_eq!(info.requested, MCP_VERSION);
    assert_eq!(info.negotiated, MCP_VERSION);
    assert!(info.supported);

    assert_eq!(client.session_id().as_deref(), Some("sess-1"));
    assert_eq!(
        client.initialize_result()["serverInfo"]["name"],
        "echo-streamable"
    );
    assert_eq!(
        server.initialized_notifications.load(Ordering::SeqCst),
        1,
        "initialized notification must follow initialize"
    );

    // The initialize POST carried no session id.
    let seen = server.session_header_seen.lock().unwrap();
    assert_eq!(seen[0], None, "initialize must not send a session id");
    assert!(seen.len() >= 2); // initialize + notification (202, no id)
}

#[tokio::test]
async fn session_id_is_echoed_on_later_posts() {
    let server = start(FixtureConfig::json()).await;
    let client = StreamableMcpClient::connect(&server.url).await.unwrap();

    client.list_tools().await.unwrap();
    let seen = server.session_header_seen.lock().unwrap();
    // Some row after the initialize must carry the issued id.
    assert!(
        seen.iter().any(|h| h.as_deref() == Some("sess-1")),
        "session id must be echoed: {seen:?}"
    );
}

#[tokio::test]
async fn stateless_server_without_session_id_works() {
    let server = start(FixtureConfig::stateless()).await;
    let client = StreamableMcpClient::connect(&server.url)
        .await
        .expect("stateless handshake");
    assert!(client.session_id().is_none());

    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    let result = client
        .call_tool("echo", json!({"msg": "nostate"}))
        .await
        .unwrap();
    assert!(result.text().contains("nostate"));
}

#[tokio::test]
async fn post_without_handshake_is_session_lost() {
    let server = start(FixtureConfig::json()).await;
    // Raw transport: no initialize → no session id → enforced 404.
    let transport = StreamableHttpTransport::new(&server.url);
    let req = lc_mcp::MCPRequest::new(7, "tools/list", None);
    let err = transport.request(&req).await.expect_err("404 expected");
    assert_eq!(err.code, MCP_ERROR_SESSION_LOST, "{err}");
    assert!(err.is_session_lost());
}

#[tokio::test]
async fn reconnect_reinitializes_after_server_drops_session() {
    let server = start(FixtureConfig::json()).await;
    let client = StreamableMcpClient::connect(&server.url).await.unwrap();
    assert_eq!(client.session_id().as_deref(), Some("sess-1"));

    // Server forgets the session mid-life.
    server.kill_session("sess-1");
    let err = client.list_tools().await.expect_err("dead session");
    assert!(err.is_session_lost(), "{err}");

    // Re-handshake issues a fresh session and service resumes.
    client.reconnect().await.expect("reconnect");
    assert_eq!(client.session_id().as_deref(), Some("sess-2"));
    let tools = client
        .list_tools()
        .await
        .expect("tools/list after reconnect");
    assert_eq!(tools.len(), 1);
    assert_eq!(
        server.sessions_issued.lock().unwrap().as_slice(),
        ["sess-1", "sess-2"]
    );
}

// ---------------------------------------------------------------------------
// Tool round trips: JSON and SSE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tools_roundtrip_json_mode() {
    let server = start(FixtureConfig::json()).await;
    let client = StreamableMcpClient::connect(&server.url).await.unwrap();

    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools[0].name, "echo");
    let result = client
        .call_tool("echo", json!({"msg": "hello"}))
        .await
        .unwrap();
    assert!(!result.is_error);
    assert!(result.text().contains("hello"));

    client.ping().await.unwrap();
}

#[tokio::test]
async fn tools_roundtrip_sse_mode_ignores_interleaved_notification() {
    let server = start(FixtureConfig::sse()).await;
    let client = StreamableMcpClient::connect(&server.url).await.unwrap();

    // The fixture emits a notifications/progress frame before every response;
    // none of these may confuse response routing.
    assert_eq!(client.list_tools().await.unwrap().len(), 1);
    let result = client
        .call_tool("echo", json!({"msg": "via sse"}))
        .await
        .unwrap();
    assert!(result.text().contains("via sse"), "{}", result.text());
    client.ping().await.unwrap();

    // SSE path also assigned and enforced the session.
    assert_eq!(client.session_id().as_deref(), Some("sess-1"));
}

#[tokio::test]
async fn unknown_tool_jsonrpc_error_json_mode_session_survives() {
    let server = start(FixtureConfig::json()).await;
    let client = StreamableMcpClient::connect(&server.url).await.unwrap();
    let err = client
        .call_tool("nope", json!({}))
        .await
        .expect_err("unknown tool");
    assert_eq!(err.code, -32601, "{err}");
    assert!(err.to_string().contains("unknown tool"), "{err}");

    assert!(client
        .call_tool("echo", json!({"msg": "after"}))
        .await
        .is_ok());
}

#[tokio::test]
async fn unknown_tool_jsonrpc_error_sse_mode() {
    let server = start(FixtureConfig::sse()).await;
    let client = StreamableMcpClient::connect(&server.url).await.unwrap();
    let err = client
        .call_tool("nope", json!({}))
        .await
        .expect_err("unknown tool over SSE");
    assert_eq!(err.code, -32601, "{err}");
}

#[tokio::test]
async fn empty_sse_stream_surfaces_clear_transport_error() {
    let config = FixtureConfig {
        mode: ResponseMode::EmptySse,
        ..FixtureConfig::json()
    };
    let server = start(config).await;
    let err = StreamableMcpClient::connect(&server.url)
        .await
        .expect_err("initialize cannot complete");
    assert_eq!(err.code, -32000, "{err}");
    assert!(err.to_string().contains("without a response"), "{err}");
}

// ---------------------------------------------------------------------------
// OAuth 2.1 resource server
// ---------------------------------------------------------------------------

/// First token is stale; after invalidate() the provider rotates to the good
/// token — exercises the transport's single 401 retry.
struct RotatingProvider {
    calls: AtomicUsize,
    invalidated: Mutex<Vec<String>>,
}

#[async_trait]
impl BearerTokenProvider for RotatingProvider {
    async fn token(&self) -> Result<String, MCPError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(if n == 0 {
            "stale".into()
        } else {
            "good".into()
        })
    }

    async fn invalidate(&self, token: &str) {
        self.invalidated.lock().unwrap().push(token.to_string());
    }
}

#[tokio::test]
async fn bearer_provider_retries_once_after_401() {
    let mut config = FixtureConfig::json();
    config.require_bearer = Some("good".into());
    let server = start(config).await;

    let provider = Arc::new(RotatingProvider {
        calls: AtomicUsize::new(0),
        invalidated: Mutex::new(Vec::new()),
    });
    let client = StreamableMcpClient::connect_with_token_provider(&server.url, provider.clone())
        .await
        .expect("retry must complete the handshake");

    assert_eq!(client.session_id().as_deref(), Some("sess-1"));
    // token() is consulted per POST: twice for the retried initialize, plus
    // the initialized notification. What matters is one rotation cycle.
    assert!(provider.calls.load(Ordering::SeqCst) >= 2);
    assert_eq!(
        provider.invalidated.lock().unwrap().as_slice(),
        ["stale".to_string()]
    );
    // The server saw the stale token first, then the rotated bearer.
    let auth_seen = server.auth_headers_seen.lock().unwrap();
    assert_eq!(
        auth_seen[0].as_deref(),
        Some("Bearer stale"),
        "first attempt carried the stale token"
    );
    assert!(
        auth_seen
            .iter()
            .any(|h| h.as_deref() == Some("Bearer good")),
        "retry must carry the rotated token: {auth_seen:?}"
    );
}

#[tokio::test]
async fn unauthorized_challenge_and_metadata_discovery() {
    let mut config = FixtureConfig::json();
    config.require_bearer = Some("good".into());
    let server = start(config).await;

    // No provider: the 401 surfaces with the parsed challenge.
    let err = StreamableMcpClient::connect(&server.url)
        .await
        .expect_err("401");
    assert_eq!(err.code, MCP_ERROR_UNAUTHORIZED, "{err}");
    let challenge = OAuthChallenge::from_error(&err).expect("oauth challenge in error data");
    assert_eq!(challenge.scheme, "Bearer");
    assert_eq!(
        challenge.resource_metadata.as_deref(),
        Some(server.metadata_url.as_str())
    );

    // RFC 9728 discovery.
    let resource = discover_protected_resource(&server.metadata_url)
        .await
        .expect("protected-resource metadata");
    assert_eq!(resource.resource, server.url);
    assert!(resource.authorization_servers.contains(&server.as_issuer));
    assert!(resource.scopes_supported.contains(&"mcp.read".to_string()));

    // RFC 8414 discovery.
    let auth_server = discover_authorization_server(&server.as_issuer)
        .await
        .expect("authorization-server metadata");
    assert_eq!(auth_server.issuer, server.as_issuer);
    let token_endpoint = auth_server.token_endpoint.expect("token_endpoint");
    assert!(auth_server
        .code_challenge_methods_supported
        .contains(&"S256".to_string()));

    // Token exchange against the fixture endpoint (refresh grant).
    let token_client = OAuthTokenClient::new(token_endpoint, "test-client");
    let tokens = token_client
        .refresh("rt-1", Some("mcp.read"))
        .await
        .expect("refresh");
    assert_eq!(tokens.access_token, "rotated-access");
    assert_eq!(tokens.token_type, "Bearer");
    assert_eq!(tokens.refresh_token.as_deref(), Some("rotated-refresh"));

    // The fixture verifies client_id is sent in the form body.
    assert!(server
        .token_requests
        .lock()
        .unwrap()
        .iter()
        .any(|body| body.contains("grant_type=refresh_token")
            && body.contains("client_id=test-client")));
}

// ---------------------------------------------------------------------------
// Version negotiation + timeouts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn version_reject_fails_handshake() {
    let config = FixtureConfig {
        forced_version: Some("1999-01-01".into()),
        ..FixtureConfig::json()
    };
    let server = start(config).await;
    let err = StreamableMcpClient::connect_with(
        &server.url,
        VersionPolicy::Reject,
        Duration::from_secs(10),
    )
    .await
    .expect_err("unsupported version under Reject");
    assert_eq!(err.code, MCP_ERROR_VERSION_UNSUPPORTED, "{err}");
}

#[tokio::test]
async fn version_degrade_pins_library_version() {
    let config = FixtureConfig {
        forced_version: Some("1999-01-01".into()),
        ..FixtureConfig::json()
    };
    let server = start(config).await;
    let client = StreamableMcpClient::connect_with(
        &server.url,
        VersionPolicy::Degrade,
        Duration::from_secs(10),
    )
    .await
    .expect("Degrade continues");
    let info = client.protocol_info();
    assert_eq!(info.server_version, "1999-01-01");
    assert!(!info.supported);
    assert_eq!(info.negotiated, MCP_VERSION);
    assert_eq!(client.list_tools().await.unwrap().len(), 1);
}

#[tokio::test]
async fn request_timeout_is_enforced_on_handshake() {
    let config = FixtureConfig {
        delay: Duration::from_secs(2),
        ..FixtureConfig::json()
    };
    let server = start(config).await;
    let err = StreamableMcpClient::connect_with(
        &server.url,
        VersionPolicy::Degrade,
        Duration::from_millis(50),
    )
    .await
    .expect_err("50ms cannot beat a 2s server");
    assert_eq!(err.code, MCP_ERROR_REQUEST_TIMEOUT, "{err}");
}

// ---------------------------------------------------------------------------
// Framework integration (McpToolClient → MCPToolAdapter, A16 gate)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streamable_tool_runs_through_base_tool_adapter() {
    let server = start(FixtureConfig::json()).await;
    let client = Arc::new(StreamableMcpClient::connect(&server.url).await.unwrap());
    let definition = client.list_tools().await.unwrap().pop().unwrap();

    // Fail-closed default (A16): refused before dispatch.
    let gated = MCPToolAdapter::from_client(client.clone(), definition.clone());
    let err = gated
        .run(json!({"msg": "denied"}).to_string())
        .await
        .expect_err("default gate denies");
    assert!(matches!(err, ToolError::PermissionDenied(_)), "{err:?}");

    // Explicit opt-in: real call over Streamable HTTP.
    let adapter =
        MCPToolAdapter::from_client(client.clone(), definition).allow_unattended_execution();
    assert_eq!(adapter.name(), "echo");
    let out = adapter
        .run(json!({"msg": "via adapter"}).to_string())
        .await
        .unwrap();
    assert!(out.contains("via adapter"), "{out}");
}

// ---------------------------------------------------------------------------
// Small protocol-helper guard used by both handshake clients
// ---------------------------------------------------------------------------

#[test]
fn shared_negotiation_helper_matches_track_policy() {
    let (v, ok) = negotiate_protocol_version("1999-01-01", VersionPolicy::Degrade).unwrap();
    assert!(!ok && v == MCP_VERSION);
    assert!(negotiate_protocol_version("1999-01-01", VersionPolicy::Reject).is_err());
    let _: Value = json!({"sanity": true});
}
