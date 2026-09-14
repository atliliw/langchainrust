// lc-agents/src/streaming/sse.rs
//! Agent events → Server-Sent Events framing (B8, v0.22.4).
//!
//! Turns a stream of [`AgentStreamEvent`] into
//! [`text/event-stream`](https://html.spec.whatwg.org/multipage/server-sent-events.html)
//! frames any SSE client can consume. The module is web-framework agnostic:
//! [`agent_sse_frames`] emits [`SseFrame`]s and [`encode_sse_frame`] renders
//! the exact wire bytes, so actix/rocket/a hand-rolled hyper service can
//! serve them. The `sse-server` feature adds an axum 0.7 endpoint
//! (`agent_sse_router`, `agent_sse_handler`).
//!
//! # Wire contract
//!
//! | SSE `event:` | Backed by | Payload (`data:` JSON, one compact line) |
//!|---|---|---|
//! | `text`          | [`AgentStreamEvent::Text`]        | `{"content": "..."}` |
//! | `tool_call`     | [`AgentStreamEvent::ToolCall`]    | `{"state":"started"\|"arguments_streaming"\|"arguments_complete"\|"executing"\|"completed"\|"failed", "tool_name", "call_id", ...}` |
//! | `tool_start`    | [`AgentStreamEvent::ToolStart`]   | `{"name","input"}` |
//! | `tool_end`      | [`AgentStreamEvent::ToolEnd`]     | `{"name","output"}` |
//! | `pipeline_step` | [`AgentStreamEvent::PipelineStep`]| `{"step","detail": string\|null}` |
//! | `final_answer`  | [`AgentStreamEvent::FinalAnswer`] | `{"content":"..."}` |
//! | `error`         | [`AgentStreamEvent::Error`]       | `{"message":"..."}` |
//!
//! Every content frame carries a monotonic `id:` starting at 1. The run ends
//! with an **unnumbered** `done` frame (`data:{"status":"done"}`); an error
//! during execution arrives as `event:error` and is still followed by `done`.
//! With [`SseOptions::with_heartbeat`], idle periods emit SSE comment frames
//! (`: keep-alive`, no id) so proxies cannot idle-timeout the connection.
//! With [`SseOptions::with_retry`], the first content frame carries a
//! `retry:` reconnection hint (milliseconds).
//!
//! # Routes (`sse-server` feature)
//!
//! - `POST /agent/stream` with a JSON [`AgentSseRequest`] — for programmatic
//!   / `fetch()` streaming clients (arbitrarily long prompts).
//! - `GET /agent/stream?input=...` — native browser `EventSource` is
//!   GET-only, so the same run is reachable from `new EventSource(url)`.
//!
//! # Disconnect / resume semantics
//!
//! The frame stream is driven by a spawned task that forwards from the agent
//! stream. When the HTTP client goes away, axum drops the response future,
//! this stream drops, and the agent producer observes a closed channel on its
//! next send — the run is cancelled promptly (see the module's disconnect
//! test). Cross-connection run resumption (replaying an already-finished run)
//! is deliberately out of scope for a stateless endpoint; clients reconnect
//! and send `Last-Event-ID` (header, or [`AgentSseRequest::last_event_id`]),
//! which [`SseOptions::resume_from`] honors **within the new run's** frame
//! sequence (frames at or below the id are suppressed) — enough for
//! dedupe-aware clients, without pretending a dropped run can be rewound
//! server-side.

use std::pin::Pin;
use std::time::Duration;

#[cfg(feature = "sse-server")]
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::state::{AgentStreamEvent, ToolCallState};

/// One rendered-but-not-yet-encoded SSE message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseFrame {
    /// A named event with an optional last-event-id, an optional `retry:`
    /// reconnection hint, and a JSON payload.
    Event {
        /// Monotonic event id (absent on the terminal `done` frame).
        id: Option<u64>,
        /// SSE `retry:` hint in milliseconds (sent on the first frame only).
        retry: Option<Duration>,
        /// SSE `event:` field, e.g. `text`.
        event: &'static str,
        /// Compact, single-line JSON payload for the `data:` field.
        data: String,
    },
    /// An SSE comment line (`: …`), used for keep-alive heartbeats.
    Comment(String),
}

impl SseFrame {
    /// Creates a numbered content event frame.
    pub fn event(id: u64, event: &'static str, data: String) -> Self {
        SseFrame::Event {
            id: Some(id),
            retry: None,
            event,
            data,
        }
    }
}

/// Options for [`agent_sse_frames`].
#[derive(Debug, Clone, Default)]
pub struct SseOptions {
    /// Suppress frames whose id is at or below this value (SSE `Last-Event-ID`).
    pub resume_from: Option<u64>,
    /// When set, emit a `: keep-alive` comment whenever no event arrives for
    /// this long.
    pub heartbeat: Option<Duration>,
    /// When set, the first content frame carries an SSE `retry:` hint.
    pub retry: Option<Duration>,
}

impl SseOptions {
    /// Suppresses frames at or below `last_event_id` (SSE `Last-Event-ID`).
    pub fn with_resume_from(mut self, last_event_id: u64) -> Self {
        self.resume_from = Some(last_event_id);
        self
    }

    /// Emits keep-alive comments on idle connections at the given interval.
    pub fn with_heartbeat(mut self, interval: Duration) -> Self {
        self.heartbeat = Some(interval);
        self
    }

    /// Sends the SSE `retry:` reconnection hint on the first content frame.
    pub fn with_retry(mut self, retry: Duration) -> Self {
        self.retry = Some(retry);
        self
    }
}

/// Maps an agent event to its stable SSE event name.
pub fn sse_event_name(event: &AgentStreamEvent) -> &'static str {
    match event {
        AgentStreamEvent::Text { .. } => "text",
        AgentStreamEvent::ToolCall { .. } => "tool_call",
        AgentStreamEvent::ToolStart { .. } => "tool_start",
        AgentStreamEvent::ToolEnd { .. } => "tool_end",
        AgentStreamEvent::PipelineStep { .. } => "pipeline_step",
        AgentStreamEvent::FinalAnswer { .. } => "final_answer",
        AgentStreamEvent::Error { .. } => "error",
    }
}

/// Maps an agent event to its compact single-line JSON `data:` payload.
pub fn sse_event_payload(event: &AgentStreamEvent) -> serde_json::Value {
    match event {
        AgentStreamEvent::Text { content } => serde_json::json!({ "content": content }),
        AgentStreamEvent::ToolStart { name, input } => {
            serde_json::json!({ "name": name, "input": input })
        }
        AgentStreamEvent::ToolEnd { name, output } => {
            serde_json::json!({ "name": name, "output": output })
        }
        AgentStreamEvent::PipelineStep { step, detail } => {
            serde_json::json!({ "step": step, "detail": detail })
        }
        AgentStreamEvent::FinalAnswer { content } => {
            serde_json::json!({ "content": content })
        }
        AgentStreamEvent::Error { message } => {
            serde_json::json!({ "message": message })
        }
        AgentStreamEvent::ToolCall { state } => tool_call_payload(state),
    }
}

fn tool_call_payload(state: &ToolCallState) -> serde_json::Value {
    // State-specific key first; tool_name/call_id are uniform across states.
    let state_key: Option<(&'static str, serde_json::Value)> = match state {
        ToolCallState::Started { .. } | ToolCallState::Executing { .. } => None,
        ToolCallState::ArgumentsStreaming { partial_args, .. } => {
            Some(("partial_args", partial_args.clone().into()))
        }
        ToolCallState::ArgumentsComplete { args, .. } => Some(("args", args.clone())),
        ToolCallState::Completed { result, .. } => Some(("result", result.clone().into())),
        ToolCallState::Failed { error, .. } => Some(("error", error.clone().into())),
    };
    let state_name = match state {
        ToolCallState::Started { .. } => "started",
        ToolCallState::ArgumentsStreaming { .. } => "arguments_streaming",
        ToolCallState::ArgumentsComplete { .. } => "arguments_complete",
        ToolCallState::Executing { .. } => "executing",
        ToolCallState::Completed { .. } => "completed",
        ToolCallState::Failed { .. } => "failed",
    };
    let mut payload = serde_json::json!({
        "state": state_name,
        "tool_name": state.tool_name(),
        "call_id": state.call_id(),
    });
    if let (Some((key, value)), Some(obj)) = (state_key, payload.as_object_mut()) {
        obj.insert(key.to_string(), value);
    }
    payload
}

/// Renders an [`SseFrame`] into its exact `text/event-stream` wire form.
///
/// The data payload is emitted verbatim — callers pass compact JSON from
/// [`sse_event_payload`] (always one physical line; embedded newlines are
/// JSON-escaped, never raw). A blank line terminates the frame.
pub fn encode_sse_frame(frame: &SseFrame) -> String {
    match frame {
        SseFrame::Comment(text) => format!(": {text}\n\n"),
        SseFrame::Event {
            id,
            retry,
            event,
            data,
        } => {
            let mut rendered = String::new();
            if let Some(retry) = retry {
                rendered.push_str(&format!("retry: {}\n", retry.as_millis()));
            }
            if let Some(id) = id {
                rendered.push_str(&format!("id: {id}\n"));
            }
            rendered.push_str(&format!("event: {event}\n"));
            // SSE spec: every physical line of a multi-line datum gets its own
            // `data:` prefix. Compact JSON is single-line, but honor the rule
            // for hand-built payloads too.
            for line in data.split('\n') {
                rendered.push_str(&format!("data: {line}\n"));
            }
            rendered.push('\n');
            rendered
        }
    }
}

/// Agent event stream with a concrete item type, as produced by
/// [`StreamingFunctionCallingAgent::invoke_stream`](crate::streaming::StreamingFunctionCallingAgent::invoke_stream).
pub type AgentEventStream = Pin<Box<dyn Stream<Item = AgentStreamEvent> + Send>>;

/// Wraps an agent event stream with SSE framing: monotonic ids, optional
/// `Last-Event-ID` suppression, idle heartbeats, a `retry:` hint, and a
/// terminal `done` frame.
///
/// Framing runs in a spawned task; dropping the returned stream stops polling
/// the agent so its producer task unwinds instead of running to completion.
pub fn agent_sse_frames<S>(events: S, options: SseOptions) -> ReceiverStream<SseFrame>
where
    S: Stream<Item = AgentStreamEvent> + Send + 'static,
{
    let (tx, rx) = mpsc::channel(16);
    tokio::spawn(async move {
        let mut events = Box::pin(events);
        let resume_from = options.resume_from.unwrap_or(0);
        let mut next_id: u64 = 1;
        let mut retry_hint = options.retry;

        // The interval's first tick is immediate — burn it so the heartbeat
        // measures idle time rather than firing at stream start.
        let mut ticker = match options.heartbeat {
            Some(interval) => {
                let mut interval = tokio::time::interval(interval);
                interval.tick().await;
                Some(interval)
            }
            None => None,
        };

        loop {
            let next = if let Some(ticker) = ticker.as_mut() {
                tokio::select! {
                    event = events.next() => event,
                    _ = ticker.tick() => {
                        if tx
                            .send(SseFrame::Comment("keep-alive".to_string()))
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                }
            } else {
                events.next().await
            };

            let Some(event) = next else { break };

            let id = next_id;
            next_id += 1;
            if id <= resume_from {
                continue;
            }
            let payload = serde_json::to_string(&sse_event_payload(&event)).unwrap_or_else(|_| {
                "{\"message\":\"event payload serialization failed\"}".to_string()
            });
            let mut frame = SseFrame::event(id, sse_event_name(&event), payload);
            if let SseFrame::Event { retry, .. } = &mut frame {
                *retry = retry_hint.take();
            }
            if tx.send(frame).await.is_err() {
                // Client went away: stop polling the agent so its producer
                // task unwinds instead of running to completion.
                return;
            }
        }

        // Terminal marker carries no id: reconnecting after `done` must not
        // resume past the end of a (new) run.
        let _ = tx
            .send(SseFrame::Event {
                id: None,
                retry: None,
                event: "done",
                data: "{\"status\":\"done\"}".to_string(),
            })
            .await;
    });

    ReceiverStream::new(rx)
}

/// Request body for the `POST /agent/stream` endpoint (`sse-server` feature).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSseRequest {
    /// User input forwarded verbatim to the agent.
    pub input: String,
    /// Optional client-side `Last-Event-ID`; an actual `Last-Event-ID`
    /// request header takes precedence over this field.
    #[serde(default)]
    pub last_event_id: Option<u64>,
}

/// Query parameters for the `GET /agent/stream?input=...` route, which exists
/// so browser-native `EventSource` (GET-only) can consume the stream.
#[cfg(feature = "sse-server")]
#[derive(Debug, Clone, Deserialize)]
pub struct AgentSseQuery {
    /// User input forwarded verbatim to the agent.
    pub input: String,
    /// Optional `Last-Event-ID` equivalent; the real header wins.
    #[serde(default)]
    pub last_event_id: Option<u64>,
}

/// Default keep-alive interval for the built-in router: proxies and browsers
/// commonly idle-cut SSE connections at 30–60s, so beat well under that.
#[cfg(feature = "sse-server")]
pub const DEFAULT_SSE_HEARTBEAT: Duration = Duration::from_secs(20);

/// Server-wide settings for the built-in axum router.
#[cfg(feature = "sse-server")]
#[derive(Debug, Clone, Default)]
pub struct AgentSseServerConfig {
    /// Idle heartbeat interval. `None` (the default) means
    /// [`DEFAULT_SSE_HEARTBEAT`]; an explicit [`Duration::ZERO`] (set via
    /// [`AgentSseServerConfig::without_heartbeat`]) disables heartbeats.
    pub heartbeat: Option<Duration>,
    /// Optional SSE `retry:` hint sent on the first frame of every run.
    pub retry: Option<Duration>,
}

#[cfg(feature = "sse-server")]
impl AgentSseServerConfig {
    /// Sets the idle heartbeat interval.
    pub fn with_heartbeat(mut self, interval: Duration) -> Self {
        self.heartbeat = Some(interval);
        self
    }

    /// Disables idle heartbeats.
    pub fn without_heartbeat(mut self) -> Self {
        self.heartbeat = Some(Duration::ZERO);
        self
    }

    /// Sets the SSE `retry:` reconnection hint.
    pub fn with_retry(mut self, retry: Duration) -> Self {
        self.retry = Some(retry);
        self
    }
}

/// Boxed future returned by an [`AgentStreamFactory`].
#[cfg(feature = "sse-server")]
pub type AgentStreamFuture = Pin<Box<dyn std::future::Future<Output = AgentEventStream> + Send>>;

/// Builds a fresh agent event stream per HTTP request.
///
/// A blanket impl covers `Fn(String) -> Fut`, so an `Arc`-shared agent is
/// wired with a move closure that clones the Arc and awaits its stream:
///
/// ```ignore
/// let agent = Arc::new(StreamingFunctionCallingAgent::new(chat));
/// let factory: Arc<dyn AgentStreamFactory> = Arc::new(move |input: String| {
///     let agent = agent.clone();
///     async move { agent.invoke_stream(input).await }
/// });
/// ```
#[cfg(feature = "sse-server")]
pub trait AgentStreamFactory: Send + Sync {
    /// Starts one run and returns its event stream.
    fn start(&self, input: String) -> AgentStreamFuture;
}

#[cfg(feature = "sse-server")]
impl<F, Fut> AgentStreamFactory for F
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = AgentEventStream> + Send + 'static,
{
    fn start(&self, input: String) -> AgentStreamFuture {
        Box::pin(self(input))
    }
}

/// Router state: the per-request stream factory plus server-wide options.
#[cfg(feature = "sse-server")]
#[derive(Clone)]
pub struct AgentSseState {
    factory: Arc<dyn AgentStreamFactory>,
    config: AgentSseServerConfig,
}

/// The boxed SSE response body used by the axum handlers.
#[cfg(feature = "sse-server")]
type AgentSseBody = axum::response::sse::Sse<
    Pin<
        Box<dyn Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> + Send>,
    >,
>;

#[cfg(feature = "sse-server")]
fn build_options(
    headers: &axum::http::HeaderMap,
    body_last_event_id: Option<u64>,
    config: &AgentSseServerConfig,
) -> SseOptions {
    let last_event_id = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .or(body_last_event_id);

    let mut options = SseOptions::default();
    if let Some(id) = last_event_id {
        options = options.with_resume_from(id);
    }
    // ZERO is the explicit "heartbeats disabled" marker.
    match config.heartbeat {
        Some(interval) if interval > Duration::ZERO => options.heartbeat = Some(interval),
        Some(_) => {}
        None => options.heartbeat = Some(DEFAULT_SSE_HEARTBEAT),
    }
    options.retry = config.retry;
    options
}

#[cfg(feature = "sse-server")]
async fn sse_response(
    state: AgentSseState,
    headers: axum::http::HeaderMap,
    input: String,
    body_last_event_id: Option<u64>,
) -> Result<AgentSseBody, (axum::http::StatusCode, String)> {
    use std::convert::Infallible;

    if input.trim().is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "input must not be empty".to_string(),
        ));
    }

    let options = build_options(&headers, body_last_event_id, &state.config);
    let events = state.factory.start(input).await;
    let frames = agent_sse_frames(events, options);
    let body: Pin<Box<dyn Stream<Item = Result<_, Infallible>> + Send>> =
        Box::pin(frames.map(|frame| {
            // Every byte source here is newline-free: compact JSON, static
            // event names, numeric ids, and the fixed "keep-alive" comment.
            let event = match frame {
                // In axum 0.7 only `data()` is fallible (it rejects raw
                // newlines/carriage returns); event/id/comment/retry setters
                // return `Self`.
                SseFrame::Comment(text) => axum::response::sse::Event::default().comment(text),
                SseFrame::Event {
                    id,
                    retry,
                    event,
                    data,
                } => {
                    // axum 0.7 appends fields in call order and splits `data`
                    // on '\n' itself; match the core encoder's wire order
                    // (retry, id, event, data). Payloads are compact JSON.
                    let mut builder = axum::response::sse::Event::default();
                    if let Some(retry) = retry {
                        builder = builder.retry(retry);
                    }
                    if let Some(id) = id {
                        builder = builder.id(id.to_string());
                    }
                    builder.event(event).data(data)
                }
            };
            Ok(event)
        }));
    Ok(axum::response::sse::Sse::new(body))
}

/// `POST /agent/stream`: JSON [`AgentSseRequest`] in, `text/event-stream` out.
#[cfg(feature = "sse-server")]
pub async fn agent_sse_handler(
    axum::extract::State(state): axum::extract::State<AgentSseState>,
    headers: axum::http::HeaderMap,
    axum::Json(request): axum::Json<AgentSseRequest>,
) -> Result<AgentSseBody, (axum::http::StatusCode, String)> {
    sse_response(state, headers, request.input, request.last_event_id).await
}

/// `GET /agent/stream?input=...`: same run, reachable from native
/// `EventSource` (which cannot POST).
#[cfg(feature = "sse-server")]
pub async fn agent_sse_get_handler(
    axum::extract::State(state): axum::extract::State<AgentSseState>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(query): axum::extract::Query<AgentSseQuery>,
) -> Result<AgentSseBody, (axum::http::StatusCode, String)> {
    sse_response(state, headers, query.input, query.last_event_id).await
}

/// Conservative CORS layer, mirrored from lc-a2a: only localhost development
/// origins are allowed cross-origin; other browser origins get no allow
/// header and are blocked. Restrict or replace for real deployments.
#[cfg(feature = "sse-server")]
fn cors_layer() -> tower_http::cors::CorsLayer {
    use axum::http::{HeaderValue, Method};
    use tower_http::cors::{AllowOrigin, Any, CorsLayer};

    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin: &HeaderValue, _| {
            origin
                .to_str()
                .map(|s| s.starts_with("http://localhost") || s.starts_with("http://127.0.0.1"))
                .unwrap_or(false)
        }))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any)
}

/// Serves the default SSE router on `0.0.0.0:{port}`.
///
/// Returns when the server shuts down. Bind a listener yourself and use
/// [`serve_agent_sse_on`] for custom addresses, TLS, unix sockets, or
/// ephemeral test ports.
#[cfg(feature = "sse-server")]
pub async fn serve_agent_sse(
    factory: Arc<dyn AgentStreamFactory>,
    port: u16,
) -> std::io::Result<()> {
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    serve_agent_sse_on(factory, listener).await
}

/// Serves the default SSE router on an already-bound listener.
#[cfg(feature = "sse-server")]
pub async fn serve_agent_sse_on(
    factory: Arc<dyn AgentStreamFactory>,
    listener: tokio::net::TcpListener,
) -> std::io::Result<()> {
    axum::serve(listener, agent_sse_router(factory)).await
}

/// Builds a router exposing `POST|GET /agent/stream` with the default
/// [`DEFAULT_SSE_HEARTBEAT`] keep-alive and a localhost-only CORS layer.
#[cfg(feature = "sse-server")]
pub fn agent_sse_router(factory: Arc<dyn AgentStreamFactory>) -> axum::Router {
    agent_sse_router_with(factory, AgentSseServerConfig::default())
}

/// Builds a router with explicit [`AgentSseServerConfig`] (heartbeat, retry).
#[cfg(feature = "sse-server")]
pub fn agent_sse_router_with(
    factory: Arc<dyn AgentStreamFactory>,
    config: AgentSseServerConfig,
) -> axum::Router {
    use axum::routing::post;
    let state = AgentSseState { factory, config };
    axum::Router::new()
        .route(
            "/agent/stream",
            post(agent_sse_handler).get(agent_sse_get_handler),
        )
        .layer(cors_layer())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    fn sample_events() -> Vec<AgentStreamEvent> {
        vec![
            AgentStreamEvent::Text {
                content: "hi".to_string(),
            },
            AgentStreamEvent::ToolStart {
                name: "calc".to_string(),
                input: "1+1".to_string(),
            },
            AgentStreamEvent::ToolCall {
                state: ToolCallState::Completed {
                    tool_name: "calc".to_string(),
                    call_id: "c1".to_string(),
                    result: "2".to_string(),
                },
            },
            AgentStreamEvent::PipelineStep {
                step: "generating".to_string(),
                detail: None,
            },
            AgentStreamEvent::FinalAnswer {
                content: "答案".to_string(),
            },
        ]
    }

    #[tokio::test]
    async fn frames_are_numbered_named_and_terminated_by_done() {
        let frames: Vec<SseFrame> =
            agent_sse_frames(stream::iter(sample_events()), SseOptions::default())
                .collect()
                .await;

        // 5 numbered content frames + unnumbered done.
        assert_eq!(frames.len(), 6);
        for (index, frame) in frames.iter().take(5).enumerate() {
            match frame {
                SseFrame::Event {
                    id: Some(id),
                    retry: None,
                    event: _,
                    data: _,
                } => assert_eq!(*id, index as u64 + 1),
                other => panic!("expected numbered event, got {other:?}"),
            }
        }
        assert_eq!(
            frames[5],
            SseFrame::Event {
                id: None,
                retry: None,
                event: "done",
                data: "{\"status\":\"done\"}".to_string(),
            }
        );
    }

    #[test]
    fn text_frame_wire_snapshot() {
        let frame = SseFrame::event(
            7,
            "text",
            serde_json::to_string(&serde_json::json!({"content": "hello"})).unwrap(),
        );
        assert_eq!(
            encode_sse_frame(&frame),
            "id: 7\nevent: text\ndata: {\"content\":\"hello\"}\n\n"
        );
    }

    #[test]
    fn retry_hint_is_rendered_in_milliseconds() {
        let frame = SseFrame::Event {
            id: Some(1),
            retry: Some(Duration::from_millis(2500)),
            event: "text",
            data: "{}".to_string(),
        };
        assert_eq!(
            encode_sse_frame(&frame),
            "retry: 2500\nid: 1\nevent: text\ndata: {}\n\n"
        );
    }

    #[test]
    fn comment_frame_wire_snapshot() {
        assert_eq!(
            encode_sse_frame(&SseFrame::Comment("keep-alive".into())),
            ": keep-alive\n\n"
        );
    }

    #[test]
    fn multiline_data_gets_one_data_prefix_per_line() {
        let frame = SseFrame::event(1, "text", "line1\nline2".to_string());
        assert_eq!(
            encode_sse_frame(&frame),
            "id: 1\nevent: text\ndata: line1\ndata: line2\n\n"
        );
    }

    #[test]
    fn event_names_and_payload_shapes_cover_all_variants() {
        assert_eq!(
            sse_event_name(&AgentStreamEvent::Text {
                content: "x".into()
            }),
            "text"
        );
        let payload = sse_event_payload(&AgentStreamEvent::ToolCall {
            state: ToolCallState::ArgumentsStreaming {
                tool_name: "search".into(),
                call_id: "c9".into(),
                partial_args: "{\"q\"".into(),
            },
        });
        assert_eq!(payload["state"], "arguments_streaming");
        assert_eq!(payload["tool_name"], "search");
        assert_eq!(payload["call_id"], "c9");
        assert_eq!(payload["partial_args"], "{\"q\"");

        let started = sse_event_payload(&AgentStreamEvent::ToolCall {
            state: ToolCallState::Started {
                tool_name: "t".into(),
                call_id: "c".into(),
            },
        });
        assert_eq!(started["state"], "started");
        assert!(started.get("partial_args").is_none());

        let failed = sse_event_payload(&AgentStreamEvent::ToolCall {
            state: ToolCallState::Failed {
                tool_name: "t".into(),
                call_id: "c".into(),
                error: "boom".into(),
            },
        });
        assert_eq!(failed["state"], "failed");
        assert_eq!(failed["error"], "boom");

        assert_eq!(
            sse_event_payload(&AgentStreamEvent::ToolEnd {
                name: "x".into(),
                output: "y".into()
            })["output"],
            "y"
        );
        assert_eq!(
            sse_event_payload(&AgentStreamEvent::Error {
                message: "bad".into()
            })["message"],
            "bad"
        );
    }

    #[tokio::test]
    async fn resume_from_suppresses_frames_at_or_below_id() {
        let frames: Vec<SseFrame> = agent_sse_frames(
            stream::iter(sample_events()),
            SseOptions::default().with_resume_from(3),
        )
        .collect()
        .await;
        let ids: Vec<u64> = frames
            .iter()
            .filter_map(|f| match f {
                SseFrame::Event { id: Some(id), .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![4, 5]);
    }

    #[tokio::test]
    async fn retry_is_attached_to_the_first_frame_only() {
        let frames: Vec<SseFrame> = agent_sse_frames(
            stream::iter(sample_events()),
            SseOptions::default().with_retry(Duration::from_secs(3)),
        )
        .collect()
        .await;
        let retries: Vec<Option<Duration>> = frames
            .iter()
            .map(|f| match f {
                SseFrame::Event { retry, .. } => *retry,
                SseFrame::Comment(_) => None,
            })
            .collect();
        assert_eq!(retries[0], Some(Duration::from_secs(3)));
        assert!(retries[1..].iter().all(|slot| slot.is_none()));
    }

    #[tokio::test]
    async fn heartbeat_emits_comments_while_idle() {
        // A stream that never yields and never ends. The heartbeat interval
        // is a real, short wall-clock interval so the spawned framer task is
        // driven like it is in production (virtual time across spawned tasks
        // is flakier than it is worth here).
        let frames = agent_sse_frames(
            stream::pending::<AgentStreamEvent>(),
            SseOptions::default().with_heartbeat(Duration::from_millis(20)),
        );
        let mut frames = Box::pin(frames);

        for expected in 1..=2u64 {
            let frame = tokio::time::timeout(Duration::from_secs(2), frames.next())
                .await
                .expect("heartbeat should arrive")
                .expect("stream should stay open");
            assert_eq!(
                frame,
                SseFrame::Comment("keep-alive".into()),
                "tick {expected}"
            );
        }
    }

    #[tokio::test]
    async fn dropping_the_frame_stream_cancels_the_producer() {
        let (tx, rx) = mpsc::channel::<AgentStreamEvent>(4);
        let mut frames = agent_sse_frames(ReceiverStream::new(rx), SseOptions::default());

        // Feed one event through the whole pipeline.
        tx.send(AgentStreamEvent::Text {
            content: "first".into(),
        })
        .await
        .unwrap();
        let first = frames.next().await.expect("frame");
        assert!(matches!(first, SseFrame::Event { id: Some(1), .. }));

        // Simulate the HTTP client going away: the response stream is dropped.
        drop(frames);

        // The framer drains queued events into its dead output channel, then
        // stops polling; the agent producer's subsequent sends must fail (run
        // cancellation), not block or succeed forever.
        let closed = tokio::time::timeout(Duration::from_secs(5), async {
            let mut failures = 0;
            for i in 0..32 {
                if tx
                    .send(AgentStreamEvent::Text {
                        content: format!("{i}"),
                    })
                    .await
                    .is_err()
                {
                    failures += 1;
                }
            }
            failures
        })
        .await
        .expect("producer should observe disconnect promptly");
        assert!(closed > 0, "channel must close after consumer drop");
    }

    #[tokio::test]
    async fn error_event_still_precedes_done() {
        let frames: Vec<SseFrame> = agent_sse_frames(
            stream::iter(vec![AgentStreamEvent::Error {
                message: "boom".into(),
            }]),
            SseOptions::default(),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 2);
        assert!(matches!(
            frames[0],
            SseFrame::Event {
                id: Some(1),
                event: "error",
                ..
            }
        ));
        assert!(matches!(
            frames[1],
            SseFrame::Event {
                id: None,
                event: "done",
                ..
            }
        ));
    }
}

#[cfg(all(test, feature = "sse-server"))]
mod axum_tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::extract::Request;
    use axum::Router;
    use futures_util::stream;
    use tower::ServiceExt;

    fn stub_factory(events: Vec<AgentStreamEvent>) -> Arc<dyn AgentStreamFactory> {
        Arc::new(move |_input: String| {
            let events = events.clone();
            async move {
                let stream: AgentEventStream = Box::pin(stream::iter(events));
                stream
            }
        })
    }

    async fn send_request(
        router: Router,
        method: &str,
        uri: &str,
        body: Option<serde_json::Value>,
        header: Option<&str>,
    ) -> axum::response::Response {
        let mut builder = Request::builder().method(method).uri(uri);
        if body.is_some() {
            builder = builder.header("content-type", "application/json");
        }
        if let Some(value) = header {
            builder = builder.header("last-event-id", value);
        }
        let body = match body {
            Some(value) => axum::body::Body::from(value.to_string()),
            None => axum::body::Body::empty(),
        };
        router.oneshot(builder.body(body).unwrap()).await.unwrap()
    }

    async fn body_text(response: axum::response::Response) -> String {
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn post_endpoint_streams_numbered_frames_for_posted_input() {
        let events = vec![
            AgentStreamEvent::Text {
                content: "hello".into(),
            },
            AgentStreamEvent::FinalAnswer {
                content: "hello".into(),
            },
        ];
        let router = agent_sse_router(stub_factory(events));
        let response = send_request(
            router,
            "POST",
            "/agent/stream",
            Some(serde_json::json!({"input": "hi"})),
            None,
        )
        .await;
        let body = body_text(response).await;
        assert!(body.contains("id: 1\nevent: text\ndata: {\"content\":\"hello\"}"));
        assert!(body.contains("event: final_answer"));
        assert!(body.ends_with("event: done\ndata: {\"status\":\"done\"}\n\n"));
    }

    #[tokio::test]
    async fn get_endpoint_streams_frames_for_native_eventsource_clients() {
        let events = vec![AgentStreamEvent::Text {
            content: "ping".into(),
        }];
        let router = agent_sse_router(stub_factory(events));
        let response = send_request(
            router,
            "GET",
            "/agent/stream?input=hello%20world",
            None,
            None,
        )
        .await;
        let body = body_text(response).await;
        assert!(body.contains("event: text"));
        assert!(body.contains("{\"content\":\"ping\"}"));
    }

    #[tokio::test]
    async fn last_event_id_header_suppresses_earlier_frames() {
        let events = vec![
            AgentStreamEvent::Text {
                content: "one".into(),
            },
            AgentStreamEvent::Text {
                content: "two".into(),
            },
        ];
        let router = agent_sse_router(stub_factory(events));
        let response = send_request(
            router,
            "POST",
            "/agent/stream",
            Some(serde_json::json!({"input": "hi", "last_event_id": 99})),
            Some("1"),
        )
        .await;
        let body = body_text(response).await;
        // Header wins over the body field: only frame id 2 survives.
        assert!(!body.contains("id: 1\n"));
        assert!(body.contains("id: 2\n"));
    }

    #[tokio::test]
    async fn empty_input_is_rejected_with_a_400() {
        let router = agent_sse_router(stub_factory(vec![]));
        let response = send_request(
            router,
            "POST",
            "/agent/stream",
            Some(serde_json::json!({"input": "   "})),
            None,
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn malformed_body_is_rejected_before_opening_the_stream() {
        let router = agent_sse_router(stub_factory(vec![]));
        let response = send_request(
            router,
            "POST",
            "/agent/stream",
            Some(serde_json::json!({"oops": true})),
            None,
        )
        .await;
        assert!(
            response.status().is_client_error(),
            "missing input should be a 4xx, got {}",
            response.status()
        );
    }

    /// Factory whose producer pushes text immediately, pauses, then pushes the
    /// final answer — lets a real socket test observe frames arriving at
    /// distinct moments rather than one buffered body.
    fn delayed_factory() -> Arc<dyn AgentStreamFactory> {
        Arc::new(|_input: String| async {
            let (tx, rx) = mpsc::channel::<AgentStreamEvent>(8);
            tokio::spawn(async move {
                tx.send(AgentStreamEvent::Text {
                    content: "first".into(),
                })
                .await
                .ok();
                tokio::time::sleep(Duration::from_millis(80)).await;
                tx.send(AgentStreamEvent::FinalAnswer {
                    content: "first".into(),
                })
                .await
                .ok();
            });
            Box::pin(ReceiverStream::new(rx)) as AgentEventStream
        })
    }

    #[tokio::test]
    async fn served_connection_pushes_frames_incrementally_over_tcp() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = serve_agent_sse_on(delayed_factory(), listener).await;
        });

        let mut conn = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        // Content-Length must be computed from the body: hyper waits for the
        // declared number of body bytes before dispatching, so a length that is
        // off by one silently hangs the request.
        let body = r#"{"input":"hi"}"#;
        let request = format!(
            "POST /agent/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        conn.write_all(request.as_bytes()).await.unwrap();

        // First read must already carry the early text frame but not the
        // answer — i.e. frames hit the wire as they happen, no buffering.
        let mut first = Vec::new();
        let mut chunk = [0u8; 4096];
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let n = tokio::time::timeout(remaining, conn.read(&mut chunk))
                .await
                .expect("early frame should arrive promptly")
                .unwrap();
            if n == 0 {
                break;
            }
            first.extend_from_slice(&chunk[..n]);
            // Stop as soon as one complete frame (blank-line terminator) is in.
            if first.windows(2).any(|w| w == b"\n\n") {
                break;
            }
        }
        let first = String::from_utf8(first).unwrap();
        assert!(first.contains("event: text"), "early frame: {first:?}");
        assert!(
            first.contains("{\"content\":\"first\"}"),
            "early frame: {first:?}"
        );
        assert!(
            !first.contains("final_answer"),
            "answer must not be buffered with the first frame: {first:?}"
        );

        // After the producer's pause, drain the remainder and expect the
        // final answer plus the terminal done frame.
        let mut rest = String::new();
        let mut rest_buf = [0u8; 4096];
        loop {
            match tokio::time::timeout(Duration::from_secs(5), conn.read(&mut rest_buf)).await {
                Ok(Ok(0)) => break,
                Ok(Ok(n)) => rest.push_str(&String::from_utf8_lossy(&rest_buf[..n])),
                Ok(Err(_)) => break,
                Err(_) => panic!("trailing frames never arrived: {rest:?}"),
            }
        }
        assert!(rest.contains("event: final_answer"), "tail: {rest:?}");
        assert!(rest.contains("event: done"), "tail: {rest:?}");
    }

    #[tokio::test]
    async fn retry_config_is_sent_on_the_first_frame() {
        let events = vec![AgentStreamEvent::Text {
            content: "x".into(),
        }];
        let config = AgentSseServerConfig::default()
            .with_retry(Duration::from_secs(3))
            .with_heartbeat(Duration::from_secs(60));
        let router = agent_sse_router_with(stub_factory(events), config);
        let response = send_request(
            router,
            "POST",
            "/agent/stream",
            Some(serde_json::json!({"input": "hi"})),
            None,
        )
        .await;
        let body = body_text(response).await;
        // The space after the colon is optional in the SSE spec: axum 0.7
        // hand-rolled `retry:` without it, axum 0.8 emits `retry: 3000` (our own
        // encoder also includes the space). Accept both spellings.
        assert!(
            body.starts_with("retry:3000\nid: 1\n") || body.starts_with("retry: 3000\nid: 1\n"),
            "unexpected stream start: {body:?}"
        );
    }
}
