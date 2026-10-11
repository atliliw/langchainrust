//! A2A Client - connects to remote A2A agents over HTTP.
//!
//! The client uses `reqwest` (already in the project dependencies) to
//! communicate with A2A servers. It supports:
//!
//! - Fetching an agent card (`GET /.well-known/agent-card.json`), with
//!   URL-consistency and optional HMAC signature verification (P1-3)
//! - Sending a task (`tasks/send`), optionally idempotent via `message_id`
//!   (P1-6) and carrying a distributed `trace_id` (P1-5)
//! - Polling a task to completion (`tasks/get` via `send_task_and_wait`),
//!   surfacing the `input-required` state to the caller (P2-3)
//! - Resuming an `input-required` task with the client's answer
//!   (`resume_task`)
//! - Cancelling a task (`tasks/cancel`)
//! - Streaming task progress over SSE (`send_task_streaming` / `connect_sse`,
//!   P2-1)
//!
//! Requests carry a per-request timeout by default, and the builder can
//! enforce HTTPS for production deployments.
//!
//! # Fleet routing (T3)
//!
//! Attach an [`AgentFleet`] with [`A2AClientBuilder::fleet`] and the routing
//! policy decides the target agent: [`A2AClient::send_task_routed`] ranks the
//! fleet by skill relevance, skips agents whose circuit breaker is open, and
//! posts to the winner; [`A2AClient::send_task_routed_sticky`] additionally
//! pins a conversation key to one agent, and [`A2AClient::delegate_task`] gates
//! the hop on the fleet's hierarchy rule.
//!
//! # Example
//!
//! ```ignore
//! use lc_a2a::{A2AClient, A2AMessage};
//!
//! let client = A2AClient::new("https://agent.example.com".to_string()).unwrap();
//! let card = client.get_agent_card().await?;
//! let task = client.send_task(A2AMessage::user("hello")).await?;
//! ```

// `pub(crate)` so the server can reuse `constant_time_eq` for bearer-token
// checks (0.22.0 audit fix).
pub(crate) mod signing;
mod sse;
#[cfg(feature = "ws")]
pub mod ws;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskDetails, A2ATaskResult,
    AgentCard, TaskStatus, TraceContext,
};

pub use signing::{
    canonical_json, sign_agent_card, sign_card_jws, verify_card_jws, verify_card_signature,
};
pub use sse::A2ASseStream;
use sse::SseReconnect;
#[cfg(feature = "ws")]
pub use ws::A2AWebSocketStream;

use crate::protocol::TaskPushNotification;
use crate::scale::{AgentFleet, ScaleError};
use crate::security::{AccessRequest, SandboxConfig, TrustRegistry};

/// A live A2A task subscription established via `tasks/sendSubscribe`.
///
/// Holds the task acknowledged by the server (the subscription confirmation)
/// plus the open stream that carries incremental [`TaskPushNotification`]s for
/// that task's lifecycle, so the caller consumes pushes instead of polling
/// `tasks/get`.
///
/// Each subscription demultiplexes the shared stream to its **own** task (V0.27,
/// #94): if several subscriptions are multiplexed over one connection, or the
/// server broadcasts multiple owned tasks on the same stream, a subscription
/// never surfaces another task's notifications — `next()` yields only pushes
/// whose task id matches the acknowledged task.
pub struct A2ASubscription {
    task: A2ATask,
    stream: A2ASseStream,
}

impl A2ASubscription {
    fn new(task: A2ATask, stream: A2ASseStream) -> Self {
        Self { task, stream }
    }

    /// The task as acknowledged by the server (initially `submitted`).
    pub fn task(&self) -> &A2ATask {
        &self.task
    }

    /// Consume the subscription into its acknowledged task.
    pub fn into_task(self) -> A2ATask {
        self.task
    }

    /// Wait for the next incremental push, or `None` when the stream ends.
    ///
    /// Surfaces the same errors as [`A2ASseStream::next`], and (V0.27, #94)
    /// filters the stream down to this subscription's own task so a connection
    /// shared by many subscriptions never leaks another task's notifications.
    pub async fn next(&mut self) -> Option<Result<TaskPushNotification, A2AError>> {
        let task_id = self.task.id.clone();
        loop {
            let item = self.stream.next().await;
            match item {
                // Skip notifications for other tasks on a shared stream, so a
                // multiplexed connection does not leak a sibling subscription.
                Some(Ok(notification)) if notification.id() != task_id => continue,
                Some(_) => return item,
                None => return None,
            }
        }
    }
}

/// Errors that can occur during A2A client operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum A2AError {
    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    Http(String),

    /// JSON parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// API-level error (returned by the remote agent).
    #[error("API error [{code}]: {message}")]
    Api {
        /// The JSON-RPC error code.
        code: i32,
        /// Human-readable error message.
        message: String,
    },

    /// Request timed out.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Agent card signature verification failed, or a signed card could not be
    /// verified (P1-3).
    #[error("Agent card signature: {0}")]
    Signature(String),

    /// The agent needs more information before it can continue (P2-3).
    ///
    /// Resume the conversation with [`A2AClient::resume_task`].
    #[error("Task {task_id} requires more input: {prompt}")]
    InputRequired {
        /// ID of the task requiring more input.
        task_id: String,
        /// Prompt describing what additional input is needed.
        prompt: String,
    },

    /// Fleet routing could not pick an agent: no fleet router is configured,
    /// no agent is relevant to the query, or every candidate's circuit breaker
    /// is open (T3 scale wiring).
    #[error("No route: {0}")]
    NoRoute(String),
}

impl From<ScaleError> for A2AError {
    fn from(err: ScaleError) -> Self {
        A2AError::NoRoute(err.to_string())
    }
}

/// A task accepted by a fleet-routed agent (T3 scale wiring).
///
/// Returned by [`A2AClient::send_task_routed`] and friends so the caller can
/// see *which* agent the routing policy selected, not just the task.
#[derive(Debug, Clone)]
pub struct RoutedTask {
    /// URL of the agent that accepted the task.
    pub agent_url: String,
    /// The task as acknowledged by that agent.
    pub task: A2ATask,
}

impl From<reqwest::Error> for A2AError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            A2AError::Timeout(err.to_string())
        } else {
            A2AError::Http(err.to_string())
        }
    }
}

impl From<A2AErrorData> for A2AError {
    fn from(err: A2AErrorData) -> Self {
        A2AError::Api {
            code: err.code,
            message: err.message,
        }
    }
}

/// A2A Client - communicates with remote A2A agents.
pub struct A2AClient {
    /// Base URL of the remote agent (e.g. "http://localhost:8080").
    base_url: String,
    /// HTTP client for regular RPC requests (bounded total timeout).
    http: reqwest::Client,
    /// HTTP client for SSE/streaming requests: no total timeout so a
    /// long-lived stream is not cut off (0.22.0 audit fix H-P4). Connect
    /// timeout still applies.
    stream_http: reqwest::Client,
    /// Monotonic request ID counter.
    next_id: AtomicU64,
    /// Optional bearer token sent on every request.
    auth_token: Option<String>,
    /// Distributed trace id attached to every request's metadata (P1-5).
    trace_id: Option<String>,
    /// W3C trace context sent as a `traceparent` header (P2-8).
    trace_context: Option<TraceContext>,
    /// Optional secret used to verify `AgentCard` signatures (P1-3).
    card_secret: Option<Vec<u8>>,
    /// Reject signed cards that cannot be verified (P1-3).
    require_card_signature: bool,
    /// Trust registry a fetched card must belong to and verify against
    /// (0.25.0: `TrustRegistry` used to have no production call site).
    trust_registry: Option<std::sync::Arc<TrustRegistry>>,
    /// Opt-in SSE self-heal: max reconnect attempts per stream (0 = disabled,
    /// the default). See [`A2AClientBuilder::sse_reconnect_max_attempts`].
    sse_reconnect_max_attempts: usize,
    /// Idle budget for a single SSE read (B3 default, tunable via
    /// [`A2AClientBuilder::sse_idle_timeout`]).
    sse_idle_timeout: std::time::Duration,
    /// Optional fleet router (T3): when attached, `send_task_routed` and friends
    /// let the fleet's policy pick the target agent.
    fleet: Option<Arc<AgentFleet>>,
}

impl A2AClient {
    /// Create a new client targeting the given base URL.
    ///
    /// Uses a 30s per-request timeout and a 10s connect timeout. The client
    /// is safe to share and call concurrently; each request gets its own ID.
    ///
    /// Returns an error if the HTTP client cannot be built (e.g. the TLS
    /// backend fails to initialize). For full configuration, use
    /// [`builder`](Self::builder) instead.
    pub fn new(base_url: impl Into<String>) -> Result<Self, A2AError> {
        let base_url = base_url.into();
        if !base_url.starts_with("https://") {
            log::warn!(
                "A2A client connecting over non-HTTPS URL: {} (use TLS in production)",
                base_url
            );
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| A2AError::Http(format!("failed to build HTTP client: {e}")))?;
        // 0.22.0 audit fix (H-P4): the 30s total timeout above covers the whole
        // body read, so an SSE stream would be cut off at 30s. Streaming uses
        // a client with only a connect timeout; fall back to the RPC client if
        // this one cannot be built.
        let stream_http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| http.clone());
        Ok(Self::with_http_client(base_url, http).with_stream_client(stream_http))
    }

    /// Create a client with a custom `reqwest::Client` (for timeouts, etc.).
    ///
    /// The given client is used for both RPC and streaming requests; if it
    /// carries a total timeout, long-lived SSE streams will be cut off — the
    /// caller owns that trade-off.
    pub fn with_http_client(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            stream_http: http.clone(),
            http,
            next_id: AtomicU64::new(1),
            auth_token: None,
            trace_id: None,
            trace_context: None,
            card_secret: None,
            require_card_signature: false,
            trust_registry: None,
            sse_reconnect_max_attempts: 0,
            sse_idle_timeout: sse::DEFAULT_SSE_READ_IDLE_TIMEOUT,
            fleet: None,
        }
    }

    /// Replace the client used for SSE/streaming requests (0.22.0 audit fix
    /// H-P4). Typically a timeout-free client with a connect timeout.
    fn with_stream_client(mut self, stream_http: reqwest::Client) -> Self {
        self.stream_http = stream_http;
        self
    }

    /// Start building a client with full configuration.
    pub fn builder(base_url: impl Into<String>) -> A2AClientBuilder {
        A2AClientBuilder::new(base_url)
    }

    /// Allocate the next request ID.
    fn alloc_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Fetch the agent card from `GET /.well-known/agent-card.json`.
    ///
    /// Performs two integrity checks (P1-3):
    ///
    /// - **URL consistency**: if the card advertises a `url` that differs from
    ///   the base URL this client was pointed at, a warning is logged. The card
    ///   is still returned — a load-balanced deployment legitimately advertises
    ///   a public URL different from the node you reached.
    /// - **Signature**: if the card carries a `signature` and a verification
    ///   secret is configured, the signature is verified and a mismatch is a
    ///   hard error. With `require_card_signature`, a signed card with no
    ///   secret configured is also rejected. Unsigned cards pass through.
    pub async fn get_agent_card(&self) -> Result<AgentCard, A2AError> {
        // B7 wire-compat fix: fetch the standard location first, then fall back
        // to the `/.well-known/agent.json` alias some discovery tooling walks.
        let card_path = "/.well-known/agent-card.json";
        let mut last_err: Option<A2AError> = None;
        for path in [card_path, "/.well-known/agent.json"] {
            match self.fetch_agent_card_at(path).await {
                Ok(card) => {
                    self.verify_fetched_card(&card)?;
                    return Ok(card);
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| A2AError::Http("agent card request failed".to_string())))
    }

    /// GET the agent card from a specific discovery path.
    async fn fetch_agent_card_at(&self, path: &str) -> Result<AgentCard, A2AError> {
        let url = format!("{}{path}", self.base_url);
        let resp = self.with_traceparent(self.http.get(&url)).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(A2AError::Http(format!(
                "Agent card request to {path} failed with status {}",
                status
            )));
        }
        resp.json()
            .await
            .map_err(|e| A2AError::Parse(format!("Failed to parse agent card: {e}")))
    }

    /// Apply the URL-consistency, signature, and trust-registry checks to a
    /// freshly fetched card.
    fn verify_fetched_card(&self, card: &AgentCard) -> Result<(), A2AError> {
        // URL consistency check (warn-only; see doc comment).
        if !card.url.trim_end_matches('/').is_empty()
            && card.url.trim_end_matches('/') != self.base_url.trim_end_matches('/')
        {
            log::warn!(
                "Agent card URL mismatch: card.url={}, base_url={}",
                card.url,
                self.base_url
            );
        }

        // Signature verification.
        if card.signature.is_some() {
            match &self.card_secret {
                Some(secret) => {
                    verify_card_signature(card, secret)?;
                }
                None if self.require_card_signature => {
                    return Err(A2AError::Signature(
                        "agent card is signed but no verification secret is configured".to_string(),
                    ));
                }
                None => {
                    log::warn!(
                        "agent card is signed but no verification secret is configured; \
                         skipping signature verification"
                    );
                }
            }
        }

        // 0.25.0 trust-registry wiring: when a registry is configured the
        // fetched card MUST belong to a known, non-revoked agent and verify
        // against the key the directory attests for the card's own URL.
        if let Some(registry) = &self.trust_registry {
            registry
                .verify_card(card)
                .map_err(|e| A2AError::Signature(e.to_string()))?;
        }

        Ok(())
    }

    /// Send a task to the remote agent (`tasks/send`).
    ///
    /// The request carries the client's `trace_id`, if configured (P1-5).
    pub async fn send_task(&self, message: A2AMessage) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = self.with_context(A2ARequest::send_task(id, &message));
        self.send_task_req(req).await
    }

    /// Send a task with an explicit `message_id` so a retried call returns the
    /// already-created task instead of running the chain twice (P1-6).
    pub async fn send_task_with_message_id(
        &self,
        message: A2AMessage,
        message_id: &str,
    ) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = self.with_context(A2ARequest::send_task_with_message_id(
            id, &message, message_id,
        ));
        self.send_task_req(req).await
    }

    /// Send a message to continue an existing `input-required` task, resuming
    /// it back to `working` (P2-3).
    ///
    /// Equivalent to `tasks/send` carrying a `taskId`.
    pub async fn resume_task(
        &self,
        task_id: &str,
        message: A2AMessage,
    ) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = self.with_context(A2ARequest::continue_task(id, task_id, &message));
        self.send_task_req(req).await
    }

    /// Continue a task with an explicit `message_id`, so a retried/failed-over
    /// resume is deduplicated by the server instead of being applied twice
    /// (P1-6 idempotency covers `tasks/send` carrying a `taskId` too).
    pub async fn resume_task_with_message_id(
        &self,
        task_id: &str,
        message: A2AMessage,
        message_id: &str,
    ) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = self.with_context(
            A2ARequest::continue_task(id, task_id, &message).with_message_id(message_id),
        );
        self.send_task_req(req).await
    }

    /// Get a task by ID (`tasks/get`).
    pub async fn get_task(&self, task_id: &str) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = self.with_context(A2ARequest::get_task(id, task_id));
        let resp = self.post_request(req).await?;
        self.task_from_response(resp)
    }

    /// Get a task by ID including its result and error (`tasks/get`).
    pub async fn get_task_details(&self, task_id: &str) -> Result<A2ATaskDetails, A2AError> {
        let id = self.alloc_id();
        let req = self.with_context(A2ARequest::get_task(id, task_id));
        let resp = self.post_request(req).await?;

        let result = resp.into_result().map_err(A2AError::from)?;
        let task: A2ATask = result
            .get("task")
            .ok_or_else(|| A2AError::Parse("Missing 'task' in response".to_string()))
            .and_then(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| A2AError::Parse(format!("Failed to parse task: {}", e)))
            })?;
        let task_result: Option<A2ATaskResult> = result
            .get("result")
            .map(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| A2AError::Parse(format!("Failed to parse task result: {}", e)))
            })
            .transpose()?;
        let error = result
            .get("error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(A2ATaskDetails {
            task,
            result: task_result,
            error,
        })
    }

    /// Cancel a task by ID (`tasks/cancel`).
    pub async fn cancel_task(&self, task_id: &str) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = self.with_context(A2ARequest::cancel_task(id, task_id));
        let resp = self.post_request(req).await?;
        self.task_from_response(resp)
    }

    /// Attach the client's trace context to a request (P1-5).
    fn with_context(&self, req: A2ARequest) -> A2ARequest {
        match &self.trace_id {
            Some(tid) => req.with_trace_id(tid.as_str()),
            None => req,
        }
    }

    /// Apply the W3C `traceparent` header to a request builder, when a trace
    /// context is configured (P2-8).
    fn with_traceparent(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.trace_context {
            Some(ctx) => request.header("traceparent", ctx.to_traceparent()),
            None => request,
        }
    }

    /// POST a `tasks/send`-family request and extract the returned task.
    async fn send_task_req(&self, req: A2ARequest) -> Result<A2ATask, A2AError> {
        let resp = self.post_request(req).await?;
        self.task_from_response(resp)
    }

    /// Extract the `task` from a successful A2A response.
    fn task_from_response(&self, resp: A2AResponse) -> Result<A2ATask, A2AError> {
        let result = resp.into_result().map_err(A2AError::from)?;
        result
            .get("task")
            .ok_or_else(|| A2AError::Parse("Missing 'task' in response".to_string()))
            .and_then(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| A2AError::Parse(format!("Failed to parse task: {}", e)))
            })
    }

    /// Send a task and poll `tasks/get` until it reaches a terminal state.
    ///
    /// Returns the task result on `completed`, an error on `failed` /
    /// `cancelled` / `rejected` / `expired`, an [`A2AError::InputRequired`]
    /// when the agent asks for more information (resume with
    /// [`A2AClient::resume_task`]), or a [`A2AError::Timeout`] if the task does
    /// not complete within `timeout`.
    pub async fn send_task_and_wait(
        &self,
        message: A2AMessage,
        timeout: Duration,
    ) -> Result<A2ATaskResult, A2AError> {
        let task = self.send_task(message).await?;
        self.wait_for_task(&task.id, timeout).await
    }

    /// Idempotent variant of [`A2AClient::send_task_and_wait`]: sends the task
    /// with a `message_id` (P1-6) so retries never create duplicate tasks.
    pub async fn send_task_and_wait_with_message_id(
        &self,
        message: A2AMessage,
        message_id: &str,
        timeout: Duration,
    ) -> Result<A2ATaskResult, A2AError> {
        let task = self.send_task_with_message_id(message, message_id).await?;
        self.wait_for_task(&task.id, timeout).await
    }

    /// Poll `tasks/get` until the task reaches a terminal state, surfacing
    /// `input-required` to the caller (P2-3).
    async fn wait_for_task(
        &self,
        task_id: &str,
        timeout: Duration,
    ) -> Result<A2ATaskResult, A2AError> {
        let start = std::time::Instant::now();
        let poll_interval = Duration::from_secs(1);

        loop {
            // 0.22.0 audit fix: a single transient network hiccup used to fail
            // the whole poll. Retry each GET a few times with a small backoff
            // before giving up; only a task in a terminal state ends the poll.
            let mut details = None;
            let mut last_err: Option<A2AError> = None;
            for attempt in 0..3u32 {
                match self.get_task_details(task_id).await {
                    Ok(d) => {
                        details = Some(d);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        if attempt < 2 {
                            tokio::time::sleep(Duration::from_millis(100 << attempt)).await;
                        }
                    }
                }
            }
            let details = match details {
                Some(d) => d,
                None => {
                    return Err(last_err.unwrap_or_else(|| {
                        A2AError::Http("task poll failed without an error".to_string())
                    }))
                }
            };
            match details.task.status {
                TaskStatus::Completed => {
                    return details.result.ok_or_else(|| {
                        A2AError::Parse(format!("Task {} completed without a result", task_id))
                    })
                }
                TaskStatus::Failed => {
                    return Err(A2AError::Api {
                        code: -32000,
                        message: details.error.unwrap_or_else(|| "Task failed".to_string()),
                    })
                }
                TaskStatus::Cancelled => {
                    return Err(A2AError::Api {
                        code: -32000,
                        message: format!("Task {} was cancelled", task_id),
                    })
                }
                TaskStatus::Rejected => {
                    return Err(A2AError::Api {
                        code: -32000,
                        message: format!("Task {} was rejected", task_id),
                    })
                }
                TaskStatus::Expired => {
                    return Err(A2AError::Api {
                        code: -32000,
                        message: format!("Task {} expired", task_id),
                    })
                }
                TaskStatus::AuthRequired => {
                    return Err(A2AError::Api {
                        code: 401,
                        message: format!("Task {} requires authentication", task_id),
                    })
                }
                TaskStatus::InputRequired => {
                    // P2-3: the agent needs more information — surface it so the
                    // caller can answer via `resume_task` instead of polling forever.
                    return Err(A2AError::InputRequired {
                        task_id: task_id.to_string(),
                        prompt: details
                            .error
                            .unwrap_or_else(|| "Input required".to_string()),
                    });
                }
                TaskStatus::Submitted | TaskStatus::Working => {
                    if start.elapsed() > timeout {
                        return Err(A2AError::Timeout(format!(
                            "Task {} did not complete within {:?}",
                            task_id, timeout
                        )));
                    }
                    tokio::time::sleep(poll_interval).await;
                }
            }
        }
    }

    /// Send a raw A2A request via POST to the agent endpoint.
    pub async fn post_request(&self, req: A2ARequest) -> Result<A2AResponse, A2AError> {
        let url = format!("{}/", self.base_url);
        let mut request = self.with_traceparent(self.http.post(&url).json(&req));
        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }
        let resp = request.send().await?;
        let status = resp.status();
        if !status.is_success() {
            // 0.22.0 audit fix: servers now surface auth failures as HTTP 401;
            // prefer the JSON-RPC error body (if any) so API-level errors keep
            // flowing through non-2xx statuses.
            if let Ok(a2a_resp) = resp.json::<A2AResponse>().await {
                if let Some(err) = a2a_resp.error {
                    return Err(A2AError::from(err));
                }
            }
            return Err(A2AError::Http(format!(
                "A2A request failed with status {}",
                status
            )));
        }
        let a2a_resp: A2AResponse = resp
            .json()
            .await
            .map_err(|e| A2AError::Parse(format!("Failed to parse A2A response: {}", e)))?;
        Ok(a2a_resp)
    }

    /// Open an SSE stream from `sse_url`, yielding [`TaskPushNotification`]
    /// events as they arrive (P2-1).
    ///
    /// The stream is useful for observing task progress without polling
    /// `tasks/get`. Events carry a `task.id`, so a caller receiving
    /// notifications for multiple tasks can filter by the id it cares about.
    pub async fn connect_sse(&self, sse_url: &str) -> Result<A2ASseStream, A2AError> {
        // 0.22.0 audit fix (H-P4): use the streaming client (no total timeout)
        // so a long-lived SSE stream is not terminated by the RPC client's
        // 30s whole-body timeout.
        let mut request = self.with_traceparent(self.stream_http.get(sse_url));
        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }
        let resp = request.send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(A2AError::Http(format!(
                "SSE request failed with status {}",
                status
            )));
        }
        // 0.27.0 T3 #95: attach self-heal only when the caller opted in, so the
        // default path is byte-for-byte the pre-0.27 behavior.
        let reconnect = (self.sse_reconnect_max_attempts > 0).then(|| SseReconnect {
            sse_url: sse_url.to_string(),
            auth_token: self.auth_token.clone(),
            trace_context: self.trace_context.clone(),
            http: self.stream_http.clone(),
            max_attempts: self.sse_reconnect_max_attempts,
        });
        Ok(A2ASseStream::new(resp, reconnect, self.sse_idle_timeout))
    }

    /// Send a task and stream its progress notifications over SSE (P2-1).
    ///
    /// Opens the SSE subscription at `sse_url` first (so no early events are
    /// missed), then sends the task via `tasks/send`. The returned stream
    /// yields [`TaskPushNotification`] events for the task.
    pub async fn send_task_streaming(
        &self,
        sse_url: &str,
        message: A2AMessage,
    ) -> Result<A2ASseStream, A2AError> {
        let stream = self.connect_sse(sse_url).await?;
        let _ = self.send_task(message).await?;
        Ok(stream)
    }

    /// Open a WebSocket subscription from `ws_url`, yielding
    /// [`TaskPushNotification`] frames as they arrive (V0.27, #186).
    ///
    /// This is the WebSocket twin of [`A2AClient::connect_sse`]. The server
    /// applies identical auth and task filtering on the `/ws` upgrade, so the
    /// two transports carry the same notification semantics; pick whichever
    /// fits the deployment (e.g. a long-lived connection that does not want
    /// HTTP/SSE buffering).
    #[cfg(feature = "ws")]
    pub async fn connect_ws(&self, ws_url: &str) -> Result<A2AWebSocketStream, A2AError> {
        A2AWebSocketStream::connect(
            ws_url,
            self.auth_token.as_deref(),
            self.trace_context.as_ref().map(|t| t.to_traceparent()),
        )
        .await
    }

    /// Send a task and stream its progress notifications over WebSocket
    /// (V0.27, #186).
    ///
    /// Opens the WS subscription at `ws_url` first (so no early frames are
    /// missed), then sends the task via `tasks/send`. The returned stream
    /// yields [`TaskPushNotification`] frames for the task.
    #[cfg(feature = "ws")]
    pub async fn send_task_streaming_ws(
        &self,
        ws_url: &str,
        message: A2AMessage,
    ) -> Result<A2AWebSocketStream, A2AError> {
        let stream = self.connect_ws(ws_url).await?;
        let _ = self.send_task(message).await?;
        Ok(stream)
    }

    /// Subscribe to a task's lifecycle over `tasks/sendSubscribe` (P2-9).
    ///
    /// Subscribes in two steps: first the SSE stream at `sse_url` is opened
    /// (so no early increments are missed), then `tasks/sendSubscribe` is
    /// posted with `message`. The server's JSON-RPC result acknowledges the
    /// task (initially `submitted`) — the subscription confirmation — and the
    /// returned [`A2ASubscription`] then yields each incremental
    /// [`TaskPushNotification`] over the stream, so the caller receives pushes
    /// instead of polling `tasks/get`.
    ///
    /// > **Note**: a fully spec-faithful A2A sendSubscribe multiplexes the RPC
    /// > ack and the update events over a single HTTP/2 stream. Here the ack is
    /// > one POST and the updates are a separate SSE connection, matching this
    /// > crate's existing streaming architecture.
    pub async fn subscribe(
        &self,
        sse_url: &str,
        message: A2AMessage,
    ) -> Result<A2ASubscription, A2AError> {
        self.subscribe_with_context(sse_url, message, None, None).await
    }

    /// Subscribe to a task's lifecycle over `tasks/sendSubscribe`, optionally
    /// carrying a client-chosen `subscription_id` and `context_id` (V0.27, #94).
    ///
    /// The IDs are forwarded in the request params (`id` / `contextId`); a
    /// server that honors them echoes the context on the pushes it sends for
    /// this task, so a caller can demultiplex many subscriptions over one
    /// stream. The returned [`A2ASubscription`] already filters its `next()`
    /// down to its own task, so even without server cooperation a connection
    /// shared across subscriptions stays leak-free.
    pub async fn subscribe_with_context(
        &self,
        sse_url: &str,
        message: A2AMessage,
        subscription_id: Option<String>,
        context_id: Option<String>,
    ) -> Result<A2ASubscription, A2AError> {
        let stream = self.connect_sse(sse_url).await?;
        let id = self.alloc_id();
        let req = self.with_context(A2ARequest::send_subscribe_with_context(
            id,
            &message,
            subscription_id,
            context_id,
        ));
        let resp = self.send_task_req(req).await?;
        Ok(A2ASubscription::new(resp, stream))
    }

    // ---- T3: fleet routing (scale.rs wired into the send path) ----

    /// The fleet router attached to this client, if any.
    pub fn fleet(&self) -> Option<&Arc<AgentFleet>> {
        self.fleet.as_ref()
    }

    /// The agent the routing policy would pick for `query`, without sending
    /// anything.
    ///
    /// Errors with [`A2AError::NoRoute`] when no fleet is configured or no
    /// agent is relevant/healthy.
    pub fn route_task(&self, query: &str) -> Result<AgentCard, A2AError> {
        let fleet = self.fleet.as_ref().ok_or_else(|| {
            A2AError::NoRoute(
                "no fleet router is configured on this client (use \
                 A2AClientBuilder::fleet)"
                    .to_string(),
            )
        })?;
        Ok(fleet.route(query)?)
    }

    /// Send a task to the fleet agent the routing policy selects for `query`.
    ///
    /// The selected agent's circuit breaker records the outcome, so an agent
    /// that starts failing is skipped by later routed sends.
    pub async fn send_task_routed(
        &self,
        query: &str,
        message: A2AMessage,
    ) -> Result<RoutedTask, A2AError> {
        let agent = self.route_task(query)?;
        self.send_to_agent(&agent, message).await
    }

    /// Sticky variant of [`A2AClient::send_task_routed`]: `conversation_key`
    /// (e.g. a session or user id) prefers the same agent for the life of the
    /// conversation, so a stateful agent stays pinned.
    pub async fn send_task_routed_sticky(
        &self,
        query: &str,
        conversation_key: &str,
        message: A2AMessage,
    ) -> Result<RoutedTask, A2AError> {
        let fleet = self.fleet.as_ref().ok_or_else(|| {
            A2AError::NoRoute("no fleet router is configured on this client".to_string())
        })?;
        let agent = fleet.route_sticky(query, conversation_key)?;
        self.send_to_agent(&agent, message).await
    }

    /// Task-stable variant of [`A2AClient::send_task_routed`]: the same
    /// `task_id` prefers the same agent among the relevant ones, so the task's
    /// state stays collocated.
    pub async fn send_task_routed_for_task(
        &self,
        task_id: &str,
        query: &str,
        message: A2AMessage,
    ) -> Result<RoutedTask, A2AError> {
        let fleet = self.fleet.as_ref().ok_or_else(|| {
            A2AError::NoRoute("no fleet router is configured on this client".to_string())
        })?;
        let agent = fleet.route_for_task(task_id, query)?;
        self.send_to_agent(&agent, message).await
    }

    /// Delegate a sub-task from `from_agent`: the fleet enforces the
    /// Orchestrator→Worker hierarchy rule and the hop budget before routing,
    /// and records the resulting `parent_task_id -> child task id` edge in its
    /// delegation graph (a cycle is logged, not fatal — the task has already
    /// been accepted by then).
    pub async fn delegate_task(
        &self,
        parent_task_id: &str,
        from_agent: &str,
        hops: usize,
        query: &str,
        message: A2AMessage,
    ) -> Result<RoutedTask, A2AError> {
        let fleet = self.fleet.as_ref().ok_or_else(|| {
            A2AError::NoRoute("no fleet router is configured on this client".to_string())
        })?;
        let agent = fleet.route_from(from_agent, hops, query)?;
        let routed = self.send_to_agent(&agent, message).await?;
        if let Err(e) = fleet.link_delegation(parent_task_id, &routed.task.id) {
            log::warn!(
                "delegation {} -> {} was accepted but the delegation graph rejected the edge: {e}",
                parent_task_id,
                routed.task.id
            );
        }
        Ok(routed)
    }

    /// POST a task to `agent` and feed the outcome back into the fleet's
    /// circuit breaker.
    async fn send_to_agent(
        &self,
        agent: &AgentCard,
        message: A2AMessage,
    ) -> Result<RoutedTask, A2AError> {
        let client = self.client_for_url(&agent.url);
        let outcome = client.send_task(message).await;
        if let Some(fleet) = &self.fleet {
            match &outcome {
                Ok(_) => fleet.record_success(&agent.url),
                Err(_) => fleet.record_failure(&agent.url),
            }
        }
        Ok(RoutedTask {
            agent_url: agent.url.clone(),
            task: outcome?,
        })
    }

    /// A client for `url` that shares this client's transport, auth and trace
    /// configuration, used to reach a fleet-selected agent.
    fn client_for_url(&self, url: &str) -> A2AClient {
        A2AClient {
            base_url: url.trim_end_matches('/').to_string(),
            http: self.http.clone(),
            stream_http: self.stream_http.clone(),
            next_id: AtomicU64::new(1),
            auth_token: self.auth_token.clone(),
            trace_id: self.trace_id.clone(),
            trace_context: self.trace_context.clone(),
            card_secret: self.card_secret.clone(),
            require_card_signature: self.require_card_signature,
            trust_registry: self.trust_registry.clone(),
            sse_reconnect_max_attempts: self.sse_reconnect_max_attempts,
            sse_idle_timeout: self.sse_idle_timeout,
            // A routed sub-client never routes again: the fleet already picked
            // the target, so a nested `send_task_routed` cannot recurse.
            fleet: None,
        }
    }
}

/// Builder for [`A2AClient`] with explicit timeouts, TLS enforcement, and auth.
pub struct A2AClientBuilder {
    base_url: String,
    http_client: Option<reqwest::Client>,
    bearer_token: Option<String>,
    enforce_https: bool,
    timeout: Duration,
    connect_timeout: Duration,
    trace_id: Option<String>,
    trace_context: Option<TraceContext>,
    card_secret: Option<Vec<u8>>,
    require_card_signature: bool,
    trust_registry: Option<std::sync::Arc<TrustRegistry>>,
    sandbox: Option<std::sync::Arc<SandboxConfig>>,
    sse_reconnect_max_attempts: usize,
    /// Idle budget for a single SSE read (default 120s, the B3 value).
    sse_idle_timeout: Duration,
    /// Optional fleet router (T3).
    fleet: Option<Arc<AgentFleet>>,
}

impl A2AClientBuilder {
    /// Start a builder for the given base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http_client: None,
            bearer_token: None,
            enforce_https: false,
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            trace_id: None,
            trace_context: None,
            card_secret: None,
            require_card_signature: false,
            trust_registry: None,
            sandbox: None,
            sse_reconnect_max_attempts: 0,
            sse_idle_timeout: sse::DEFAULT_SSE_READ_IDLE_TIMEOUT,
            fleet: None,
        }
    }

    /// Use a custom `reqwest::Client` (overrides the timeouts configured below).
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Send an `Authorization: Bearer <token>` header on every request.
    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Reject non-HTTPS base URLs at build time (default: off, warn only).
    pub fn enforce_https(mut self, enforce: bool) -> Self {
        self.enforce_https = enforce;
        self
    }

    /// Per-request timeout (default 30s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Connect timeout (default 10s).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Attach a distributed `trace_id` to every request's metadata (P1-5).
    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Attach a W3C trace context (P2-8).
    ///
    /// Every request carries the context as a standard `traceparent` header,
    /// and the context's trace id is also attached to request metadata (P1-5)
    /// so a single configuration populates both channels.
    pub fn with_traceparent(mut self, context: TraceContext) -> Self {
        self.trace_context = Some(context.clone());
        self.trace_id = Some(context.trace_id.clone());
        self
    }

    /// Configure a shared HMAC secret used to verify agent-card signatures
    /// (P1-3).
    pub fn card_verification_secret(mut self, secret: impl Into<Vec<u8>>) -> Self {
        self.card_secret = Some(secret.into());
        self
    }

    /// Reject signed agent cards that cannot be verified (P1-3).
    ///
    /// When enabled, a card carrying a `signature` is refused unless the
    /// configured [`Self::card_verification_secret`] verifies it. Defaults to
    /// `false` (signed cards are logged, not rejected, when no secret is set).
    pub fn require_card_signature(mut self, require: bool) -> Self {
        self.require_card_signature = require;
        self
    }

    /// Require fetched agent cards to verify against a trust registry
    /// (0.25.0): the card's URL must name a known, non-revoked agent and its
    /// signature must verify against the registered key. Failures surface as
    /// [`A2AError::Signature`] from `get_agent_card`.
    pub fn trust_registry(mut self, registry: std::sync::Arc<TrustRegistry>) -> Self {
        self.trust_registry = Some(registry);
        self
    }

    /// Gate network egress through a sandbox (0.25.0): the agent host in the
    /// base URL must be allowed by [`SandboxConfig`]'s domain rules or
    /// [`Self::build`] fails.
    pub fn sandbox(mut self, sandbox: std::sync::Arc<SandboxConfig>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Enable SSE self-heal (0.27.0 T3 #95): a transport break or idle timeout
    /// on a task stream is retried up to `max_attempts` times, re-issuing the
    /// SSE `GET` against the same URL (same bearer token and `traceparent`)
    /// with exponential backoff, before the error is surfaced.
    ///
    /// Default `0` = disabled (a break surfaces immediately, the pre-0.27
    /// behavior). A clean server close is always terminal and never retried.
    ///
    /// The budget is per stream (not per break), and a reconnect does **not**
    /// replay missed events — see the caveat on the reconnect config.
    pub fn sse_reconnect_max_attempts(mut self, max_attempts: usize) -> Self {
        self.sse_reconnect_max_attempts = max_attempts;
        self
    }

    /// Sets the idle budget for a single SSE read (0.27.0, B3 default 120s).
    ///
    /// `A2ASseStream::next` turns a silent gap longer than this into
    /// [`A2AError::Timeout`] (with self-heal enabled,
    /// into a reconnect) instead of parking the caller forever. Raise it for a
    /// task that legitimately goes quiet for long stretches; the timer resets on
    /// every received chunk, so it bounds silence, not the stream's lifetime.
    pub fn sse_idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.sse_idle_timeout = idle_timeout;
        self
    }

    /// Attach a fleet router (T3 scale wiring).
    ///
    /// With a fleet attached, [`A2AClient::send_task_routed`] and its sticky /
    /// task-stable / delegated variants let the fleet's policy choose the target
    /// agent; the plain `send_task` family keeps talking to this client's own
    /// `base_url` and is unaffected.
    pub fn fleet(mut self, fleet: Arc<AgentFleet>) -> Self {
        self.fleet = Some(fleet);
        self
    }

    /// Build the client, enforcing HTTPS when configured.
    pub fn build(self) -> Result<A2AClient, A2AError> {
        if !self.base_url.starts_with("https://") {
            if self.enforce_https {
                return Err(A2AError::Http(format!(
                    "HTTPS is required for A2A, got insecure URL: {}",
                    self.base_url
                )));
            }
            log::warn!(
                "A2A client connecting over non-HTTPS URL: {} (use TLS in production)",
                self.base_url
            );
        }
        if let Some(sandbox) = &self.sandbox {
            let host = reqwest::Url::parse(&self.base_url)
                .ok()
                .and_then(|u| u.host_str().map(str::to_string));
            if let Some(host) = host {
                sandbox
                    .check(&AccessRequest::Network(host))
                    .map_err(|e| A2AError::Http(format!("sandbox denied egress: {e}")))?;
            }
        }
        let (http, stream_http) = match self.http_client {
            Some(client) => (client.clone(), client),
            None => {
                let http = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .connect_timeout(self.connect_timeout)
                    .build()
                    .map_err(|e| A2AError::Http(format!("failed to build HTTP client: {}", e)))?;
                // 0.22.0 audit fix (H-P4): a separate timeout-free client for
                // SSE streams, mirroring `A2AClient::new`.
                let stream_http = reqwest::Client::builder()
                    .connect_timeout(self.connect_timeout)
                    .build()
                    .unwrap_or_else(|_| http.clone());
                (http, stream_http)
            }
        };
        Ok(A2AClient {
            base_url: self.base_url,
            http,
            stream_http,
            next_id: AtomicU64::new(1),
            auth_token: self.bearer_token,
            trace_id: self.trace_id,
            trace_context: self.trace_context,
            card_secret: self.card_secret,
            require_card_signature: self.require_card_signature,
            trust_registry: self.trust_registry,
            sse_reconnect_max_attempts: self.sse_reconnect_max_attempts,
            sse_idle_timeout: self.sse_idle_timeout,
            fleet: self.fleet,
        })
    }
}

#[cfg(test)]
mod tests;
