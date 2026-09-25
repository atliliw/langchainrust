//! P1-7: out-of-the-box axum HTTP serving for [`A2AServer`].
//!
//! Feature-gated behind `feature = "axum"`. Adds [`A2AServer::serve`] so an
//! agent can be exposed over HTTP without hand-wiring a framework. The module
//! is intentionally thin — all logic lives in the handlers the server already
//! exposes, so it stays a convenience wrapper rather than a second code path.
//!
//! Routes:
//!
//! - `GET /.well-known/agent-card.json` → the [`AgentCard`]
//! - `POST /` → `handle_a2a_request_authenticated` (bearer token enforced when
//!   the server was configured with `with_auth_token`)
//! - `GET /events` → SSE stream of [`crate::protocol::TaskPushNotification`]s, only when
//!   streaming was enabled with `with_streaming`
//!
//! A CORS layer restricted to localhost origins is applied by default so
//! browser-based A2A clients can connect during development; tighten
//! `router()` with an explicit origin allowlist before exposing the agent to
//! untrusted cross-origin callers (0.22.0 audit fix: the layer used to be
//! all-open).

use std::convert::Infallible;
use std::net::IpAddr;
use std::str::FromStr;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::net::TcpListener;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::protocol::{A2ARequest, A2AResponse, AgentCard};
use crate::server::A2AServer;
use crate::A2AError;

/// Agent card route (A2A standard location).
const CARD_PATH: &str = "/.well-known/agent-card.json";
/// Alias route for the agent card; some A2A tooling walks `/.well-known/agent.json`
/// as a discovery fallback, so serve the same card there (B7 wire-compat fix).
const AGENT_JSON_PATH: &str = "/.well-known/agent.json";
/// SSE route for streaming task notifications (P2-1).
const SSE_PATH: &str = "/events";
/// Default bind address for [`A2AServer::serve`]: the loopback interface, so an
/// unauthenticated server is not silently exposed to the network (B7 security
/// default). Reach further hosts explicitly via [`A2AServer::serve_with`].
const DEFAULT_BIND: &str = "127.0.0.1";
/// Default port for [`A2AServer::serve`].
const DEFAULT_PORT: u16 = 3000;

/// HTTP serving configuration for [`A2AServer::serve_with`].
///
/// Security defaults (B7): the server binds loopback, and a non-loopback bind
/// requires a bearer token (`.auth_token`, or `with_auth_token` on the server)
/// or an explicit `insecure_public = true`. Cross-origin requests are denied
/// unless the origin is on the exact `cors_origins` allow-list (or is a
/// loopback dev origin).
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Host/IP to bind. Default `127.0.0.1`.
    pub bind: String,
    /// TCP port to bind. Default `3000`.
    pub port: u16,
    /// Bearer token to require of callers, applied when the server does not
    /// already have one configured.
    pub auth_token: Option<String>,
    /// Exact cross-origin allow-list. Empty (default) means no cross-origin
    /// requests are allowed; loopback dev origins are always permitted.
    pub cors_origins: Vec<String>,
    /// Opt out of the B7 "public bind requires auth" guard. Reckless — set
    /// only when you know there is an authenticating proxy in front.
    pub insecure_public: bool,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND.to_string(),
            port: DEFAULT_PORT,
            auth_token: None,
            cors_origins: Vec::new(),
            insecure_public: false,
        }
    }
}

impl ServeConfig {
    /// A config with only the port set (everything else default).
    pub fn for_port(port: u16) -> Self {
        Self {
            port,
            ..Self::default()
        }
    }
}

impl A2AServer {
    /// Serve the agent over HTTP on `127.0.0.1:{port}` using axum.
    ///
    /// B7 security default: `serve` binds the *loopback* interface, so an
    /// unauthenticated dev server is never silently reachable over the
    /// network. Bind a real host (and provide a bearer token) explicitly with
    /// [`A2AServer::serve_with`]. Returns `Ok(())` when the server shuts down,
    /// or the bind/serve error.
    pub async fn serve(self, port: u16) -> Result<(), A2AError> {
        self.serve_with(ServeConfig::for_port(port)).await
    }

    /// Serve over an already-bound listener (custom address, TLS, unix socket,
    /// or an ephemeral `:0` port for tests).
    ///
    /// Loopback dev origins are allowed cross-origin; every other origin is
    /// denied (0.22.0 audit fix). This is the escape hatch used by the tests
    /// to bind a private ephemeral port; production deployments should use
    /// [`A2AServer::serve_with`] with an explicit origin allow-list.
    pub async fn serve_on(self, listener: TcpListener) -> Result<(), A2AError> {
        // B7 fail-closed: a caller-supplied bound listener is a second, unvetted
        // path into serving. Enforce the same public-bind guard as
        // [`Self::serve_with`] — a non-loopback bind without any configured auth
        // is refused up front rather than silently exposing an open agent to the
        // network. Loopback binds stay allowed so the test / ephemeral-port
        // escape hatch keeps working; an intentionally open proxy-fronted public
        // bind must go through `serve_with` (with `ServeConfig.insecure_public`).
        if let Ok(addr) = listener.local_addr() {
            if !addr.ip().is_loopback() && !self.has_auth() {
                return Err(A2AError::Http(format!(
                    "refusing to serve A2A on non-loopback bound address '{addr}' without a \
                     bearer token: use A2AServer::serve_with with a token (or \
                     ServeConfig.insecure_public) for a public bind"
                )));
            }
        }
        let server = Arc::new(self);
        axum::serve(listener, router(server, &[]))
            .await
            .map_err(|e| A2AError::Http(format!("Server error: {e}")))?;
        Ok(())
    }

    /// Serve over HTTP with full [`ServeConfig`], enforcing the B7 secure-bind
    /// guard.
    ///
    /// # Security guard
    ///
    /// Binding a **non-loopback** address without a bearer token and without
    /// `insecure_public` is refused up front with a clear error — an open agent
    /// must never be silently exposed to the network. Loopback binds need no
    /// token (they cannot authenticate the cross-machine case by default, so
    /// they stay a local-only dev surface).
    ///
    /// `config.auth_token` is applied to the server only when it does not
    /// already have one configured (e.g. via [`A2AServer::with_auth_token`]).
    pub async fn serve_with(self, config: ServeConfig) -> Result<(), A2AError> {
        let auth_configured = config.auth_token.is_some() || self.has_auth();
        if !is_loopback_host(&config.bind) && !auth_configured && !config.insecure_public {
            return Err(A2AError::Http(format!(
                "refusing to serve A2A on non-loopback bind '{}' without a bearer token: \
                 set a token (with_auth_token / ServeConfig.auth_token) or explicitly enable \
                 insecure_public for a proxy-fronted deployment",
                config.bind
            )));
        }

        let ip = IpAddr::from_str(&config.bind).map_err(|_| {
            A2AError::Http(format!(
                "invalid bind '{}': expected an IP address",
                config.bind
            ))
        })?;
        let addr = std::net::SocketAddr::new(ip, config.port);
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| A2AError::Http(format!("Failed to bind {addr}: {e}")))?;

        // Apply the config's token when the server has none.
        let server = if config.auth_token.is_some() && !self.has_auth() {
            let token = config.auth_token.clone().expect("checked Some above");
            self.with_auth_token(token)
        } else {
            self
        };

        let server = Arc::new(server);
        axum::serve(listener, router(server, &config.cors_origins))
            .await
            .map_err(|e| A2AError::Http(format!("Server error: {e}")))?;
        Ok(())
    }
}

/// Whether a bind host string is the loopback interface (any spelling).
fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "0:0:0:0:0:0:0:1")
        || host.starts_with("127.")
}

/// 0.22.0 audit fix: the CORS layer used to allow every origin (`Any`),
/// letting any website call the agent with the visitor's credentials.
///
/// B7 hardened: cross-origin callers are admitted only when their origin is
/// *exactly* in the configured allow-list, plus the loopback dev origins
/// (localhost / 127.0.0.1) so local browser clients keep working out of the
/// box. An empty allow-list therefore means "no cross-origin access" — the
/// agent is only reachable same-origin.
fn cors_layer(cors_origins: &[String]) -> CorsLayer {
    let allowed: Vec<String> = cors_origins.to_vec();
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _| {
            let Ok(s) = origin.to_str() else {
                return false;
            };
            is_loopback_origin(s) || allowed.iter().any(|o| o == s)
        }))
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any)
}

/// Whether an Origin header value is a loopback dev origin.
fn is_loopback_origin(origin: &str) -> bool {
    origin.starts_with("http://localhost") || origin.starts_with("http://127.0.0.1")
}

/// Build the axum [`Router`] that exposes the server over HTTP, honoring the
/// exact cross-origin allow-list.
fn router(server: Arc<A2AServer>, cors_origins: &[String]) -> Router {
    let cors_origins = cors_origins.to_vec();
    Router::new()
        .route(CARD_PATH, get(get_agent_card))
        .route(AGENT_JSON_PATH, get(get_agent_card))
        .route("/", post(post_request))
        .route(SSE_PATH, get(sse_stream))
        .layer(cors_layer(&cors_origins))
        .with_state(server)
}

/// `GET /.well-known/agent-card.json` (and `/.well-known/agent.json` alias).
async fn get_agent_card(State(server): State<Arc<A2AServer>>) -> Json<AgentCard> {
    Json(server.get_agent_card().clone())
}

/// `POST /` — dispatch an A2A request, enforcing bearer auth when configured.
async fn post_request(
    State(server): State<Arc<A2AServer>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // 0.25.0 sandbox wiring: reject oversized raw wire bodies before JSON
    // parsing (the framework-neutral handler re-checks the parsed request).
    if let Some(sandbox) = server.sandbox() {
        let size = body.len();
        if !sandbox.accepts_payload(size) {
            let resp = A2AResponse::error(
                0,
                413,
                format!("request payload of {size} bytes exceeds sandbox limit"),
            );
            return (StatusCode::PAYLOAD_TOO_LARGE, Json(resp)).into_response();
        }
    }
    let req: A2ARequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(e) => {
            let resp = A2AResponse::error(0, -32700, format!("Invalid request: {e}"));
            return (StatusCode::BAD_REQUEST, Json(resp)).into_response();
        }
    };
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let resp = server.handle_a2a_request_authenticated(req, bearer).await;
    // 0.22.0 audit fix: an auth failure must surface as HTTP 401, not HTTP 200
    // with an error body.
    let status = if resp.error.as_ref().is_some_and(|e| e.code == 401) {
        StatusCode::UNAUTHORIZED
    } else {
        StatusCode::OK
    };
    (status, Json(resp)).into_response()
}

/// Query params accepted on the SSE endpoint.
///
/// `?taskId=...` narrows the stream to a single task. On an identity-bound
/// server, requesting another tenant's task id is refused up front (403) so a
/// foreign task's bytes are never streamed to this connection (B7 cross-tenant
/// fix).
#[derive(Debug, Deserialize)]
struct SseQuery {
    /// Restrict notifications to one task id.
    #[serde(rename = "taskId")]
    task_id: Option<String>,
}

/// `GET /events` — SSE stream of task notifications (P2-1).
///
/// Enforces the same bearer auth as [`post_request`] when the server was
/// configured with `with_auth_token`, so the streaming endpoint is not a
/// bypass for the JSON-RPC one (0.20.0 S4 G1).
async fn sse_stream(
    State(server): State<Arc<A2AServer>>,
    Query(query): Query<SseQuery>,
    headers: HeaderMap,
) -> Response {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let principal = match server.check_auth(bearer) {
        Ok(principal) => principal,
        Err(resp) => return (StatusCode::UNAUTHORIZED, Json(resp)).into_response(),
    };
    let Some(rx) = server.subscribe() else {
        return (
            StatusCode::NOT_FOUND,
            "SSE not enabled (call A2AServer::with_streaming)",
        )
            .into_response();
    };

    // B7 cross-tenant guard: an identity-bound principal asking to subscribe to
    // a task it does not own is refused before any event is emitted, so no
    // foreign-task payload is even serialized onto the wire for this caller.
    if let (Some(principal), Some(task_id)) = (&principal, &query.task_id) {
        if !server.notification_visible(Some(principal), task_id).await {
            return (
                StatusCode::FORBIDDEN,
                Json(A2AResponse::error(
                    0,
                    -32003,
                    "caller does not own the requested task",
                )),
            )
                .into_response();
        }
    }

    // 0.25.0 authz fix: the bus is process-global, so filter every event down
    // to tasks this connection's principal owns. Previously any authenticated
    // connection received every tenant's task notifications. B7 additionally
    // honors an explicit `?taskId=` restriction.
    let task_id = query.task_id.clone();
    let stream = BroadcastStream::new(rx)
        .then(move |item| {
            let server = server.clone();
            let principal = principal.clone();
            let task_id = task_id.clone();
            async move {
                match item {
                    Ok(notification) => {
                        // Hard filter on the owning principal and/or an exact
                        // requested task id.
                        let owned = server
                            .notification_visible(principal.as_deref(), notification.id())
                            .await;
                        let wanted = task_id
                            .as_ref()
                            .is_none_or(|want| want == notification.id());
                        if !owned || !wanted {
                            None
                        } else {
                            serde_json::to_string(&notification)
                                .ok()
                                .map(|data| Event::default().event("task").data(data))
                        }
                    }
                    // A slow subscriber dropped events; signal a reset rather
                    // than stall (a dropped sender ends the stream).
                    Err(_lagged) => Some(Event::default().event("reset").data("lagged")),
                }
            }
        })
        .filter_map(|event| event.map(Ok::<_, Infallible>));
    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    use lc_chains::base::{BaseChain, ChainError, ChainResult};
    use serde_json::Value;

    use crate::protocol::{A2AMessage, TaskStatus};
    use crate::A2AClient;

    /// A trivial chain that echoes its input.
    struct EchoChain;

    #[async_trait::async_trait]
    impl BaseChain for EchoChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }

        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }

        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let input = inputs
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut out = HashMap::new();
            out.insert("output".to_string(), Value::String(input));
            Ok(out)
        }

        fn name(&self) -> &str {
            "echo-chain"
        }
    }

    /// Spawn `server` on an ephemeral port; returns its base URL and a handle.
    async fn spawn(server: A2AServer) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            let _ = server.serve_on(listener).await;
        });
        (format!("http://127.0.0.1:{port}"), handle)
    }

    #[tokio::test]
    async fn serve_exposes_agent_card() {
        let server = A2AServer::new(Arc::new(EchoChain));
        let (base, _handle) = spawn(server).await;
        let client = A2AClient::new(base).unwrap();
        let card = client.get_agent_card().await.unwrap();
        assert_eq!(card.name, "echo-chain");
    }

    #[tokio::test]
    async fn serve_dispatches_tasks_end_to_end() {
        let server = A2AServer::new(Arc::new(EchoChain));
        let (base, _handle) = spawn(server).await;
        let client = A2AClient::new(base).unwrap();
        let result = client
            .send_task_and_wait(A2AMessage::user("hello"), Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(result.output, "hello");
    }

    #[tokio::test]
    async fn serve_enforces_bearer_token() {
        let server = A2AServer::new(Arc::new(EchoChain)).with_auth_token("secret-token");
        let (base, _handle) = spawn(server).await;
        let client = A2AClient::new(base.clone()).unwrap();

        // No token -> rejected with a 401 API error.
        let err = client.send_task(A2AMessage::user("hi")).await.unwrap_err();
        assert!(err.to_string().contains("Authentication required"));

        // Correct token -> accepted.
        let client = A2AClient::builder(base)
            .bearer_token("secret-token")
            .build()
            .unwrap();
        let result = client
            .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(result.output, "hi");
    }

    #[tokio::test]
    async fn serve_returns_http_401_on_auth_failure() {
        // 0.22.0 audit fix: auth failure must surface as HTTP 401, not 200.
        let server = A2AServer::new(Arc::new(EchoChain)).with_auth_token("secret-token");
        let (base, _handle) = spawn(server).await;
        let req = crate::protocol::A2ARequest::send_task(1, &A2AMessage::user("hi"));
        let resp = reqwest::Client::new()
            .post(format!("{base}/"))
            .header("content-type", "application/json")
            .body(serde_json::to_string(&req).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn cors_allows_localhost_but_blocks_other_origins() {
        // 0.22.0 audit fix: the CORS layer must not be all-open; localhost
        // dev origins are allowed, everything else gets no allow-origin.
        let server = A2AServer::new(Arc::new(EchoChain));
        let (base, _handle) = spawn(server).await;
        let client = reqwest::Client::new();
        for origin in ["http://localhost:3000", "http://127.0.0.1:5173"] {
            let resp = client
                .get(format!("{base}/.well-known/agent-card.json"))
                .header("Origin", origin)
                .send()
                .await
                .unwrap();
            assert_eq!(
                resp.headers()
                    .get("access-control-allow-origin")
                    .and_then(|v| v.to_str().ok()),
                Some(origin),
                "localhost origin {origin} should be allowed"
            );
        }
        let resp = client
            .get(format!("{base}/.well-known/agent-card.json"))
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "non-localhost origins must not be allowed"
        );
    }

    #[tokio::test]
    async fn serve_streams_task_notifications() {
        let server = A2AServer::new(Arc::new(EchoChain)).with_streaming(64);
        let (base, _handle) = spawn(server).await;
        let client = A2AClient::new(base.clone()).unwrap();

        let mut stream = client
            .send_task_streaming(&format!("{base}/events"), A2AMessage::user("hi"))
            .await
            .unwrap();

        let mut saw_working = false;
        let mut saw_completed = false;
        while let Some(event) = stream.next().await {
            let event = event.unwrap();
            match event.status_value() {
                Some(TaskStatus::Working) => saw_working = true,
                Some(TaskStatus::Completed) => {
                    saw_completed = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(saw_working, "expected a working status-update");
        assert!(saw_completed, "expected a completed status-update");
    }

    #[tokio::test]
    async fn serve_sse_requires_bearer_token_when_configured() {
        // 0.20.0 S4 G1: the SSE endpoint must not be an auth bypass for a
        // token-configured server.
        let server = A2AServer::new(Arc::new(EchoChain))
            .with_auth_token("secret-token")
            .with_streaming(64);
        let (base, _handle) = spawn(server).await;

        // No token -> 401 before any SSE bytes.
        let resp = reqwest::get(format!("{base}/events")).await.unwrap();
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::UNAUTHORIZED,
            "unauthenticated SSE must be rejected"
        );

        // Correct token -> SSE stream opens.
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/events"))
            .bearer_auth("secret-token")
            .send()
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "authenticated SSE should open, got: {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn serve_rejects_oversized_body_under_sandbox() {
        // 0.25.0 C3: the sandbox payload cap is enforced on the raw wire body
        // before JSON-RPC parsing, surfaced as HTTP 413 with a JSON-RPC body.
        let server = A2AServer::new(Arc::new(EchoChain)).with_sandbox(Arc::new(
            crate::security::SandboxConfig::new().with_max_payload(256),
        ));
        let (base, _handle) = spawn(server).await;

        let big = serde_json::to_string(&crate::protocol::A2ARequest::send_task(
            1,
            &A2AMessage::user("x".repeat(4096)),
        ))
        .unwrap();
        let resp = reqwest::Client::new()
            .post(format!("{base}/"))
            .header("content-type", "application/json")
            .body(big)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);
        let body: A2AResponse = resp.json().await.unwrap();
        assert_eq!(body.error.unwrap().code, 413);
    }

    /// Spawn `server` over an ephemeral port with an explicit CORS allow-list
    /// (mirrors what `serve_with` builds).
    async fn spawn_with_cors(
        server: A2AServer,
        origins: &[String],
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let origins = origins.to_vec();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router(Arc::new(server), &origins))
                .await
                .unwrap();
        });
        (format!("http://127.0.0.1:{port}"), handle)
    }

    #[tokio::test]
    async fn serve_with_refuses_public_bind_without_token() {
        // B7 points 1 & 4: binding a non-loopback address without a bearer
        // token must be refused up front — an open agent is never silently
        // exposed to the network.
        let server = A2AServer::new(Arc::new(EchoChain));
        let cfg = ServeConfig {
            bind: "0.0.0.0".to_string(),
            port: 0, // would bind beyond loopback if the guard failed
            ..ServeConfig::default()
        };
        let err = server.serve_with(cfg).await.unwrap_err();
        assert!(
            err.to_string().contains("refusing to serve"),
            "public bind without a token must be refused, got: {err}"
        );
    }

    #[tokio::test]
    async fn serve_with_permits_public_bind_with_token_or_fallback() {
        // Positive control for the B7 guard: an explicit token (or
        // insecure_public opt-out) clears the "refusing to serve" guard. To keep
        // the test from running a real server forever, the target port is
        // already held by a blocker listener, so serve_with proceeds past the
        // guard and then fails fast at bind — the assertion is that the returned
        // error is a bind conflict, *not* the guard's refusal.
        let blocker = TcpListener::bind("0.0.0.0:0").await.unwrap();
        let port = blocker.local_addr().unwrap().port();

        let server = A2AServer::new(Arc::new(EchoChain));
        let cfg_with_token = ServeConfig {
            bind: "0.0.0.0".to_string(),
            port,
            auth_token: Some("secret-token".to_string()),
            ..ServeConfig::default()
        };
        let err = server.serve_with(cfg_with_token).await.unwrap_err();
        assert!(
            err.to_string().contains("Failed to bind")
                && !err.to_string().contains("refusing to serve"),
            "a token must clear the public-bind guard and proceed to bind, got: {err}"
        );

        let server = A2AServer::new(Arc::new(EchoChain));
        let cfg_insecure = ServeConfig {
            bind: "0.0.0.0".to_string(),
            port,
            insecure_public: true,
            ..ServeConfig::default()
        };
        let err = server.serve_with(cfg_insecure).await.unwrap_err();
        assert!(
            err.to_string().contains("Failed to bind")
                && !err.to_string().contains("refusing to serve"),
            "insecure_public must clear the public-bind guard, got: {err}"
        );
    }

    #[tokio::test]
    async fn cors_exact_allow_list_admits_listed_blocks_unlisted() {
        // B7 point 2: only origins *exactly* on the allow-list (or loopback)
        // receive an allow-origin header; a non-listed cross-origin caller is
        // denied, and the loopback carve-out still works for local dev.
        let server = A2AServer::new(Arc::new(EchoChain));
        let allowed = vec!["https://app.example.com".to_string()];
        let (base, _handle) = spawn_with_cors(server, &allowed).await;
        let client = reqwest::Client::new();
        let card = format!("{base}/.well-known/agent-card.json");

        // Listed origin -> reflected allow-origin.
        let resp = client
            .get(&card)
            .header("Origin", "https://app.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("https://app.example.com")
        );

        // A suffix-lookalike of a listed origin must NOT be allowed (exact match).
        let resp = client
            .get(&card)
            .header("Origin", "https://app.example.com.evil.io")
            .send()
            .await
            .unwrap();
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "origin that only prefixes a listed origin must be denied"
        );

        // Unrelated cross-origin -> denied.
        let resp = client
            .get(&card)
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "unlisted cross-origin must be denied"
        );

        // Loopback carve-out still applies.
        let resp = client
            .get(&card)
            .header("Origin", "http://localhost:4000")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://localhost:4000")
        );
    }

    #[tokio::test]
    async fn serve_sse_denies_cross_tenant_task_filter() {
        // B7 point 5: an identity-bound principal asking to stream another
        // tenant's task via `?taskId=` gets an up-front 403, so the foreign
        // task's bytes are never serialized onto the connection. The owning
        // principal can still open the stream.
        struct TenantAuth;
        impl crate::Authenticator for TenantAuth {
            fn authenticate(
                &self,
                bearer: Option<&str>,
            ) -> Result<Option<crate::Principal>, crate::AuthError> {
                match bearer {
                    Some("alice") => Ok(Some(crate::Principal("alice".to_string()))),
                    Some("bob") => Ok(Some(crate::Principal("bob".to_string()))),
                    _ => Err(crate::AuthError::Required),
                }
            }
        }

        let server = A2AServer::new(Arc::new(EchoChain))
            .with_authenticator(Arc::new(TenantAuth))
            .with_streaming(64);
        let (base, _handle) = spawn(server).await;
        let client = reqwest::Client::new();

        // alice sends a task, which becomes owned by alice.
        let req = crate::protocol::A2ARequest::send_task(1, &A2AMessage::user("hi"));
        let resp = client
            .post(format!("{base}/"))
            .header("content-type", "application/json")
            .bearer_auth("alice")
            .body(serde_json::to_string(&req).unwrap())
            .send()
            .await
            .unwrap();
        let body: Value = resp.json().await.unwrap();
        let msg = format!("no task id in send response envelope: {body}");
        let task_id = body["result"]["task"]["id"]
            .as_str()
            .expect(&msg)
            .to_string();

        // bob asks to stream alice's task -> 403, no task bytes emitted.
        let resp = client
            .get(format!("{base}/events?taskId={task_id}"))
            .bearer_auth("bob")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::FORBIDDEN,
            "cross-tenant task filter must be refused with 403"
        );

        // alice (the owner) may open the stream.
        let resp = client
            .get(format!("{base}/events?taskId={task_id}"))
            .bearer_auth("alice")
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
    }
}
