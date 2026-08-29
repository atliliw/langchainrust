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
//! - `GET /events` → SSE stream of [`TaskPushNotification`]s, only when
//!   streaming was enabled with `with_streaming`
//!
//! A permissive CORS layer is applied by default so browser-based A2A clients
//! can connect; lock it down before exposing the agent to untrusted callers.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::net::TcpListener;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};

use crate::protocol::{A2ARequest, A2AResponse, AgentCard};
use crate::server::A2AServer;
use crate::A2AError;

/// Agent card route (A2A standard location).
const CARD_PATH: &str = "/.well-known/agent-card.json";
/// SSE route for streaming task notifications (P2-1).
const SSE_PATH: &str = "/events";

impl A2AServer {
    /// Serve the agent over HTTP on `0.0.0.0:{port}` using axum.
    ///
    /// Returns `Ok(())` when the server shuts down, or the bind/serve error.
    pub async fn serve(self, port: u16) -> Result<(), A2AError> {
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| A2AError::Http(format!("Failed to bind {addr}: {e}")))?;
        self.serve_on(listener).await
    }

    /// Serve over an already-bound listener (custom address, TLS, unix socket,
    /// or an ephemeral `:0` port for tests).
    pub async fn serve_on(self, listener: TcpListener) -> Result<(), A2AError> {
        let server = Arc::new(self);
        axum::serve(listener, router(server))
            .await
            .map_err(|e| A2AError::Http(format!("Server error: {e}")))?;
        Ok(())
    }
}

/// Build the axum [`Router`] that exposes the server over HTTP.
fn router(server: Arc<A2AServer>) -> Router {
    Router::new()
        .route(CARD_PATH, get(get_agent_card))
        .route("/", post(post_request))
        .route(SSE_PATH, get(sse_stream))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers(Any),
        )
        .with_state(server)
}

/// `GET /.well-known/agent-card.json`.
async fn get_agent_card(State(server): State<Arc<A2AServer>>) -> Json<AgentCard> {
    Json(server.get_agent_card().clone())
}

/// `POST /` — dispatch an A2A request, enforcing bearer auth when configured.
async fn post_request(
    State(server): State<Arc<A2AServer>>,
    headers: HeaderMap,
    body: String,
) -> Response {
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
    (StatusCode::OK, Json(resp)).into_response()
}

/// `GET /events` — SSE stream of task notifications (P2-1).
///
/// Enforces the same bearer auth as [`post_request`] when the server was
/// configured with `with_auth_token`, so the streaming endpoint is not a
/// bypass for the JSON-RPC one (0.20.0 S4 G1).
async fn sse_stream(State(server): State<Arc<A2AServer>>, headers: HeaderMap) -> Response {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if let Err(resp) = server.check_auth(bearer) {
        return (StatusCode::UNAUTHORIZED, Json(resp)).into_response();
    }
    let Some(rx) = server.subscribe() else {
        return (
            StatusCode::NOT_FOUND,
            "SSE not enabled (call A2AServer::with_streaming)",
        )
            .into_response();
    };
    let stream = BroadcastStream::new(rx).filter_map(|item| match item {
        Ok(notification) => {
            let data = serde_json::to_string(&notification).ok()?;
            Some(Ok::<_, Infallible>(
                Event::default().event("task").data(data),
            ))
        }
        // A slow subscriber dropped events; skip ahead rather than stall.
        // (When the broadcast sender is dropped the stream simply ends.)
        Err(BroadcastStreamRecvError::Lagged(_)) => {
            Some(Ok(Event::default().event("reset").data("lagged")))
        }
    });
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
}
