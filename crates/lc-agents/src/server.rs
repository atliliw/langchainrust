//! OpenAI-compatible HTTP surface.
//!
//! Exposes the framework behind the two endpoints existing OpenAI SDKs and UIs
//! already speak:
//!
//! - `GET  /v1/models`
//! - `POST /v1/chat/completions` (JSON and SSE streaming)
//!
//! The backend is abstracted by [`ChatBackend`], so an agent, a chain or a bare
//! chat model can be served without changing the wire contract. Sampling
//! parameters are intentionally owned by the backend/model configuration: the
//! endpoint accepts them for compatibility but does not re-plumb them per
//! request in this version.
//!
//! Enabled by the `openai-server` feature.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{extract::State, http::StatusCode, Json, Router};
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};

use lc_core::language_models::{BaseChatModel, LLMResult, StreamChunk};
use lc_schema::Message;

/// Backend that answers chat completions.
///
/// Implement this to serve an agent/chain; [`ModelBackend`] adapts any
/// [`BaseChatModel`].
#[async_trait]
pub trait ChatBackend: Send + Sync {
    /// Non-streaming completion.
    async fn complete(&self, messages: Vec<Message>) -> Result<LLMResult, String>;

    /// Streaming completion.
    async fn stream_chat(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>, String>;
}

/// Adapts a [`BaseChatModel`] into a [`ChatBackend`].
pub struct ModelBackend<M> {
    model: Arc<M>,
}

impl<M> ModelBackend<M> {
    /// Wraps a chat model.
    pub fn new(model: Arc<M>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl<M> ChatBackend for ModelBackend<M>
where
    M: BaseChatModel + Send + Sync + 'static,
    M::Error: std::fmt::Display,
{
    async fn complete(&self, messages: Vec<Message>) -> Result<LLMResult, String> {
        self.model
            .chat(messages, None)
            .await
            .map_err(|e| e.to_string())
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>, String> {
        let stream = self
            .model
            .stream_chat(messages, None)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Box::pin(stream.map(|item| item.map_err(|e| e.to_string()))))
    }
}

struct AppState<B> {
    backend: Arc<B>,
    model: String,
}

// Manual impl: `#[derive(Clone)]` would add a `B: Clone` bound, but only the
// `Arc<B>` needs cloning.
impl<B> Clone for AppState<B> {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            model: self.model.clone(),
        }
    }
}

/// Builds the router. Mount it directly or nest it under a prefix.
pub fn openai_compat_router<B>(backend: Arc<B>, model: impl Into<String>) -> Router
where
    B: ChatBackend + 'static,
{
    let state = AppState {
        backend,
        model: model.into(),
    };
    Router::new()
        .route("/v1/models", get(list_models::<B>))
        .route("/v1/chat/completions", post(chat_completions::<B>))
        .with_state(state)
}

async fn list_models<B: ChatBackend>(
    State(state): State<AppState<B>>,
) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model,
            "object": "model",
            "created": 0,
            "owned_by": "langchainrust"
        }]
    }))
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<WireMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    temperature: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
}

fn to_messages(wire: Vec<WireMessage>) -> Result<Vec<Message>, String> {
    wire.into_iter()
        .map(|m| {
            let content = m.content.unwrap_or_default();
            match m.role.as_str() {
                "system" => Ok(Message::system(content)),
                "user" => Ok(Message::human(content)),
                "assistant" => Ok(Message::ai(content)),
                other => Err(format!(
                    "unsupported message role '{other}' (supported: system, user, assistant)"
                )),
            }
        })
        .collect()
}

async fn chat_completions<B: ChatBackend + 'static>(
    State(state): State<AppState<B>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let messages = match to_messages(req.messages) {
        Ok(m) if !m.is_empty() => m,
        Ok(_) => return error_response(StatusCode::BAD_REQUEST, "messages must not be empty"),
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };

    if req.stream.unwrap_or(false) {
        stream_completion(state, messages).await
    } else {
        match state.backend.complete(messages).await {
            Ok(result) => Json(json!({
                "id": completion_id(),
                "object": "chat.completion",
                "created": now(),
                "model": state.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.content
                    },
                    "finish_reason": "stop"
                }],
                "usage": usage_json(&result)
            }))
            .into_response(),
            Err(e) => error_response(StatusCode::BAD_GATEWAY, e),
        }
    }
}

async fn stream_completion<B: ChatBackend + 'static>(
    state: AppState<B>,
    messages: Vec<Message>,
) -> Response {
    let stream = match state.backend.stream_chat(messages).await {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, e),
    };

    let id = completion_id();
    let created = now();
    let model = state.model;

    let events = stream
        .map(move |item| -> Result<Event, Infallible> {
            let payload = match item {
                Ok(chunk) => json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": chunk.text },
                        "finish_reason": null
                    }]
                }),
                Err(e) => json!({
                    "error": { "message": e, "type": "server_error" }
                }),
            };
            Ok(Event::default().data(payload.to_string()))
        })
        .chain(futures_util::stream::once(async {
            Ok(Event::default().data("[DONE]"))
        }));

    Sse::new(events).into_response()
}

fn completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn usage_json(result: &LLMResult) -> Value {
    match &result.token_usage {
        Some(u) => json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        }),
        None => Value::Null,
    }
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    struct MockBackend;

    #[async_trait]
    impl ChatBackend for MockBackend {
        async fn complete(&self, _messages: Vec<Message>) -> Result<LLMResult, String> {
            Ok(LLMResult {
                content: "hello".to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>, String>
        {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::new("he")),
                Ok(StreamChunk::new("llo")),
            ])))
        }
    }

    fn app() -> Router {
        openai_compat_router(Arc::new(MockBackend), "mock-model")
    }

    fn post(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn non_streaming_completion_returns_openai_shape() {
        let resp = app()
            .oneshot(post(
                r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn streaming_completion_emits_sse_and_done() {
        let resp = app()
            .oneshot(post(
                r#"{"model":"mock-model","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(body.contains("chat.completion.chunk"), "body: {body}");
        assert!(body.contains("\"he\""), "body: {body}");
        assert!(body.contains("\"llo\""), "body: {body}");
        assert!(body.contains("[DONE]"), "body: {body}");
    }

    #[tokio::test]
    async fn models_endpoint_lists_configured_model() {
        let resp = app()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["data"][0]["id"], "mock-model");
    }

    #[tokio::test]
    async fn empty_messages_are_rejected() {
        let resp = app()
            .oneshot(post(r#"{"model":"mock-model","messages":[]}"#))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unsupported_role_is_rejected() {
        let resp = app()
            .oneshot(post(
                r#"{"messages":[{"role":"tool","content":"x"}]}"#,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
