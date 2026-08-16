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
//! # Example
//!
//! ```ignore
//! use lc_a2a::{A2AClient, A2AMessage};
//!
//! let client = A2AClient::new("https://agent.example.com".to_string()).unwrap();
//! let card = client.get_agent_card().await?;
//! let task = client.send_task(A2AMessage::user("hello")).await?;
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use hmac::{Hmac, Mac};
use sha2::Sha256;

use super::protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskDetails, A2ATaskResult,
    AgentCard, TaskPushNotification, TaskStatus, TraceContext,
};

/// Errors that can occur during A2A client operations.
#[derive(Debug, thiserror::Error)]
pub enum A2AError {
    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    Http(String),

    /// JSON parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// API-level error (returned by the remote agent).
    #[error("API error [{code}]: {message}")]
    Api { code: i32, message: String },

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
    InputRequired { task_id: String, prompt: String },
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
    /// HTTP client.
    http: reqwest::Client,
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
}

impl A2AClient {
    /// Create a new client targeting the given base URL.
    ///
    /// Uses a 30s per-request timeout and a 10s connect timeout. The client
    /// is safe to share and call concurrently; each request gets its own ID.
    ///
    /// Returns an error if the HTTP client cannot be built (e.g. the TLS
    /// backend fails to initialize). For full configuration, use [`builder`]
    /// (Self::builder) instead.
    pub fn new(base_url: String) -> Result<Self, A2AError> {
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
        Ok(Self::with_http_client(base_url, http))
    }

    /// Create a client with a custom `reqwest::Client` (for timeouts, etc.).
    pub fn with_http_client(base_url: String, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
            next_id: AtomicU64::new(1),
            auth_token: None,
            trace_id: None,
            trace_context: None,
            card_secret: None,
            require_card_signature: false,
        }
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
        let url = format!("{}/.well-known/agent-card.json", self.base_url);
        let resp = self.with_traceparent(self.http.get(&url)).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(A2AError::Http(format!(
                "Agent card request failed with status {}",
                status
            )));
        }
        let card: AgentCard = resp
            .json()
            .await
            .map_err(|e| A2AError::Parse(format!("Failed to parse agent card: {}", e)))?;

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
                    verify_card_signature(&card, secret)?;
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

        Ok(card)
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
            let details = self.get_task_details(task_id).await?;
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
        let mut request = self.with_traceparent(self.http.get(sse_url));
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
        Ok(A2ASseStream::new(resp))
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
}

/// A live SSE connection to an A2A server (P2-1).
///
/// Consume it with [`A2ASseStream::next`], which yields one
/// [`TaskPushNotification`] per complete SSE event until the server closes the
/// stream.
pub struct A2ASseStream {
    response: reqwest::Response,
    parser: A2aSseParser,
    pending: VecDeque<TaskPushNotification>,
}

impl A2ASseStream {
    fn new(response: reqwest::Response) -> Self {
        Self {
            response,
            parser: A2aSseParser::new(),
            pending: VecDeque::new(),
        }
    }

    /// Wait for the next event, or `None` when the stream ends.
    ///
    /// Returns an error if the connection breaks or an event fails to parse.
    pub async fn next(&mut self) -> Option<Result<TaskPushNotification, A2AError>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            match self.response.chunk().await {
                Ok(Some(chunk)) => {
                    let text = String::from_utf8_lossy(&chunk);
                    match self.parser.feed(&text) {
                        Ok(events) => {
                            self.pending.extend(events);
                            // Loop so queued events are returned even if a chunk
                            // carried none (or we keep reading on an empty chunk).
                            continue;
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Ok(None) => return None,
                Err(e) => return Some(Err(A2AError::from(e))),
            }
        }
    }
}

/// Incremental parser for A2A SSE event frames.
///
/// Mirrors the SSE parsing in `lc-providers/src/openai/sse.rs`: events are
/// terminated by a blank line (`\n\n`), and the payload is the `data:` field
/// (multi-line `data:` fields are joined with newlines per the SSE spec).
struct A2aSseParser {
    buffer: String,
}

impl A2aSseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Feed a chunk of the response body; returns any complete notifications.
    fn feed(&mut self, chunk: &str) -> Result<Vec<TaskPushNotification>, A2AError> {
        self.buffer.push_str(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_text: String = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos + 1);

            let data = event_text
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.trim_start().trim_end())
                .collect::<Vec<_>>()
                .join("\n");
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let notification: TaskPushNotification = serde_json::from_str(&data).map_err(|e| {
                A2AError::Parse(format!("Failed to parse SSE event `{data}`: {}", e))
            })?;
            out.push(notification);
        }
        Ok(out)
    }
}

// ---- P1-3: Agent Card HMAC signatures ----

/// Sign an agent card with a shared HMAC-SHA256 secret (P1-3).
///
/// The signature is computed over the canonical JSON of the card with the
/// `signature` field stripped, and stored hex-encoded in `card.signature`.
/// Verify with [`verify_card_signature`].
pub fn sign_agent_card(card: &mut AgentCard, secret: &[u8]) -> Result<(), A2AError> {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .map_err(|_| A2AError::Signature("invalid signing secret length".to_string()))?;
    mac.update(&canonical_card_bytes(card)?);
    let tag = mac.finalize().into_bytes();
    card.signature = Some(hex_encode(&tag));
    Ok(())
}

/// Verify an agent card's HMAC-SHA256 signature (P1-3).
///
/// Returns `Ok(())` for unsigned cards (nothing to verify). A card whose
/// signature does not match `secret` (or is malformed) yields a
/// [`A2AError::Signature`].
pub fn verify_card_signature(card: &AgentCard, secret: &[u8]) -> Result<(), A2AError> {
    let sig = match card.signature.as_deref() {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(()),
    };
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .map_err(|_| A2AError::Signature("invalid verification secret length".to_string()))?;
    mac.update(&canonical_card_bytes(card)?);
    let expected = hex_encode(&mac.finalize().into_bytes());
    if constant_time_eq(sig, &expected) {
        Ok(())
    } else {
        Err(A2AError::Signature(
            "agent card signature verification failed".to_string(),
        ))
    }
}

/// Canonical bytes of a card for signing: the card JSON with `signature`
/// removed so signatures don't cover themselves.
fn canonical_card_bytes(card: &AgentCard) -> Result<Vec<u8>, A2AError> {
    let mut value = serde_json::to_value(card)
        .map_err(|e| A2AError::Parse(format!("Failed to serialize agent card: {}", e)))?;
    if let Some(obj) = value.as_object_mut() {
        obj.remove("signature");
    }
    serde_json::to_vec(&value)
        .map_err(|e| A2AError::Parse(format!("Failed to serialize agent card: {}", e)))
}

/// Lowercase hex encoding.
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{:02x}", b).expect("writing to a String cannot fail");
    }
    s
}

/// Constant-time string comparison (avoids leaking the expected signature).
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let a = a.as_bytes();
    let b = b.as_bytes();
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
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
        let http = match self.http_client {
            Some(client) => client,
            None => reqwest::Client::builder()
                .timeout(self.timeout)
                .connect_timeout(self.connect_timeout)
                .build()
                .map_err(|e| A2AError::Http(format!("failed to build HTTP client: {}", e)))?,
        };
        Ok(A2AClient {
            base_url: self.base_url,
            http,
            next_id: AtomicU64::new(1),
            auth_token: self.bearer_token,
            trace_id: self.trace_id,
            trace_context: self.trace_context,
            card_secret: self.card_secret,
            require_card_signature: self.require_card_signature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Spawn a minimal HTTP/1.1 server that passes the request head (first
    /// line) and body to `handler` and writes the returned string as the body.
    async fn spawn_http_server(
        handler: impl Fn(&str, &str) -> String + Send + Sync + 'static,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = std::sync::Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => break,
                };
                let handler = handler.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 1024];
                    loop {
                        let n = match socket.read(&mut tmp).await {
                            Ok(n) => n,
                            Err(_) => return,
                        };
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let content_length: usize = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse().ok())
                        })
                        .unwrap_or(0);
                    while buf.len() < head_end + content_length {
                        let n = match socket.read(&mut tmp).await {
                            Ok(n) => n,
                            Err(_) => break,
                        };
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                    }
                    let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                    let response_body = handler(&head, &body);
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        format!("http://{}", addr)
    }

    /// Spawn a server that answers one GET request with an SSE event stream.
    ///
    /// Each event is written as `event`/`data` lines followed by a blank line;
    /// the connection is closed after all events are written.
    async fn spawn_sse_server(events: &'static [&'static str]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut tmp = [0u8; 4096];
                let _ = socket.read(&mut tmp).await; // consume the request
                let mut body =
                    String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
                for ev in events {
                    body.push_str(ev);
                    body.push_str("\n\n");
                }
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        format!("http://{}", addr)
    }

    /// Spawn a server that answers JSON-RPC `POST /` requests and serves SSE
    /// events on `GET /sse` (used by `send_task_streaming` tests).
    async fn spawn_sse_rpc_server(events: &'static [&'static str]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => break,
                };
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 1024];
                    loop {
                        let n = match socket.read(&mut tmp).await {
                            Ok(n) => n,
                            Err(_) => return,
                        };
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();

                    if head.starts_with("GET") {
                        let mut body = String::from(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n",
                        );
                        for ev in events {
                            body.push_str(ev);
                            body.push_str("\n\n");
                        }
                        let _ = socket.write_all(body.as_bytes()).await;
                    } else {
                        let content_length: usize = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|v| v.trim().parse().ok())
                            })
                            .unwrap_or(0);
                        while buf.len() < head_end + content_length {
                            let n = match socket.read(&mut tmp).await {
                                Ok(n) => n,
                                Err(_) => break,
                            };
                            if n == 0 {
                                break;
                            }
                            buf.extend_from_slice(&tmp[..n]);
                        }
                        let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                        let req: A2ARequest = serde_json::from_str(&body).unwrap();
                        let json = serde_json::to_string(&A2AResponse::ok(
                            req.id,
                            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
                        ))
                        .unwrap();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            json.len(),
                            json
                        );
                        let _ = socket.write_all(resp.as_bytes()).await;
                    }
                    let _ = socket.shutdown().await;
                });
            }
        });
        format!("http://{}", addr)
    }

    /// Spawn a server that delays its response by `delay` (for timeout tests).
    async fn spawn_slow_server(delay: Duration) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut tmp = [0u8; 1024];
                let _ = socket.read(&mut tmp).await; // read request headers
                tokio::time::sleep(delay).await;
                let resp = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(resp.as_bytes()).await;
            }
        });
        format!("http://{}", addr)
    }

    #[test]
    fn a2a_error_from_reqwest_timeout() {
        // We can't easily create a reqwest::Error, so test the variant exists.
        let err = A2AError::Timeout("connection timed out".to_string());
        assert!(err.to_string().contains("Timeout"));
    }

    #[test]
    fn a2a_error_from_reqwest_http() {
        let err = A2AError::Http("404 not found".to_string());
        assert!(err.to_string().contains("HTTP error"));
    }

    #[test]
    fn a2a_error_from_error_data() {
        let data = A2AErrorData::method_not_found();
        let err: A2AError = data.into();
        match err {
            A2AError::Api { code, message } => {
                assert_eq!(code, -32601);
                assert!(message.contains("Method not found"));
            }
            _ => panic!("Expected Api variant"),
        }
    }

    #[test]
    fn a2a_error_parse() {
        let err = A2AError::Parse("bad json".to_string());
        assert!(err.to_string().contains("Parse error"));
    }

    #[test]
    fn client_new_trims_trailing_slash() {
        let client = A2AClient::new("http://localhost:8080/".to_string()).unwrap();
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn client_alloc_id_increments() {
        let client = A2AClient::new("http://localhost:8080".to_string()).unwrap();
        assert_eq!(client.alloc_id(), 1);
        assert_eq!(client.alloc_id(), 2);
        assert_eq!(client.alloc_id(), 3);
    }

    #[test]
    fn client_with_custom_http() {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        let client = A2AClient::with_http_client("http://localhost:8080".to_string(), http);
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn builder_rejects_insecure_url_when_enforcing_https() {
        let result = A2AClient::builder("http://localhost:8080")
            .enforce_https(true)
            .build();
        match result {
            Err(A2AError::Http(msg)) => assert!(msg.contains("HTTPS is required")),
            _ => panic!("expected an HTTPS enforcement error"),
        }
    }

    #[tokio::test]
    async fn get_agent_card_uses_agent_card_json_path() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();
        let base = spawn_http_server(move |head, _body| {
            seen_clone
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(head.to_string());
            serde_json::to_string(&AgentCard::new("agent", "desc", "http://localhost")).unwrap()
        })
        .await;
        let client = A2AClient::new(base).unwrap();
        let card = client.get_agent_card().await.unwrap();
        assert_eq!(card.name, "agent");

        let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            lines
                .iter()
                .any(|l| l.contains("/.well-known/agent-card.json")),
            "expected request to use /.well-known/agent-card.json, got: {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("/.well-known/agent.json")),
            "must not use the legacy path: {lines:?}"
        );
    }

    #[tokio::test]
    async fn send_task_and_wait_polls_until_completed() {
        let base = spawn_http_server(|_head, body| {
            let req: A2ARequest = serde_json::from_str(body).unwrap();
            match req.method.as_str() {
                "tasks/send" => serde_json::to_string(&A2AResponse::ok(
                    req.id,
                    json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
                ))
                .unwrap(),
                "tasks/get" => serde_json::to_string(&A2AResponse::ok(
                    req.id,
                    json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "completed"}, "result": {"output": "done"}}),
                ))
                .unwrap(),
                _ => serde_json::to_string(&A2AResponse::error(req.id, -32601, "Method not found"))
                    .unwrap(),
            }
        })
        .await;
        let client = A2AClient::new(base).unwrap();
        let result = client
            .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(10))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().output, "done");
    }

    #[tokio::test]
    async fn send_task_and_wait_returns_error_on_failed() {
        let base = spawn_http_server(|_head, body| {
            let req: A2ARequest = serde_json::from_str(body).unwrap();
            match req.method.as_str() {
                "tasks/send" => serde_json::to_string(&A2AResponse::ok(
                    req.id,
                    json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
                ))
                .unwrap(),
                "tasks/get" => serde_json::to_string(&A2AResponse::ok(
                    req.id,
                    json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "failed"}, "error": "boom"}),
                ))
                .unwrap(),
                _ => serde_json::to_string(&A2AResponse::error(req.id, -32601, "Method not found"))
                    .unwrap(),
            }
        })
        .await;
        let client = A2AClient::new(base).unwrap();
        let result = client
            .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(10))
            .await;
        match result {
            Err(A2AError::Api { code, message }) => {
                assert_eq!(code, -32000);
                assert!(message.contains("boom"));
            }
            other => panic!("expected Api error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn send_task_and_wait_times_out() {
        let base = spawn_http_server(|_head, _body| {
            // Always report `submitted`, so the poll never terminates.
            serde_json::to_string(&A2AResponse::ok(
                0,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap()
        })
        .await;
        let client = A2AClient::new(base).unwrap();
        let result = client
            .send_task_and_wait(A2AMessage::user("hi"), Duration::from_millis(1500))
            .await;
        assert!(matches!(result, Err(A2AError::Timeout(_))));
    }

    #[tokio::test]
    async fn client_sends_bearer_token() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();
        let base = spawn_http_server(move |head, _body| {
            seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(head.to_string());
            serde_json::to_string(&A2AResponse::ok(
                0,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap()
        })
        .await;
        let client = A2AClient::builder(base)
            .bearer_token("s3cret")
            .build()
            .unwrap();
        let _ = client.send_task(A2AMessage::user("hi")).await;

        let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            lines.iter().any(|l| l
                .to_ascii_lowercase()
                .contains("authorization: bearer s3cret")),
            "expected Authorization: Bearer s3cret, got: {lines:?}"
        );
    }

    #[tokio::test]
    async fn client_enforces_per_request_timeout() {
        let base = spawn_slow_server(Duration::from_secs(5)).await;
        let client = A2AClient::builder(base)
            .timeout(Duration::from_millis(300))
            .connect_timeout(Duration::from_millis(300))
            .build()
            .unwrap();
        let result = client.send_task(A2AMessage::user("hi")).await;
        assert!(matches!(result, Err(A2AError::Timeout(_))));
    }

    #[tokio::test]
    async fn get_agent_card_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
        let result = client.get_agent_card().await;
        assert!(result.is_err());
        match result.unwrap_err() {
            A2AError::Http(_) | A2AError::Timeout(_) => {} // expected
            other => panic!("Expected Http or Timeout error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn send_task_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
        let result = client.send_task(A2AMessage::user("hello")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_task_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
        let result = client.get_task("task-123").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn cancel_task_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
        let result = client.cancel_task("task-123").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn post_request_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
        let req = A2ARequest::new(1, "test", None);
        let result = client.post_request(req).await;
        assert!(result.is_err());
    }

    // ---- P1-3: card signature ----

    #[test]
    fn sign_and_verify_card_signature_roundtrip() {
        let mut card = AgentCard::new("agent", "desc", "http://localhost");
        sign_agent_card(&mut card, b"secret").unwrap();
        assert!(card.signature.is_some());
        assert!(verify_card_signature(&card, b"secret").is_ok());
    }

    #[test]
    fn verify_card_signature_rejects_tampered_card() {
        let mut card = AgentCard::new("agent", "desc", "http://localhost");
        sign_agent_card(&mut card, b"secret").unwrap();
        card.name = "evil".to_string();
        assert!(matches!(
            verify_card_signature(&card, b"secret"),
            Err(A2AError::Signature(_))
        ));
    }

    #[test]
    fn verify_card_signature_rejects_wrong_secret() {
        let mut card = AgentCard::new("agent", "desc", "http://localhost");
        sign_agent_card(&mut card, b"secret").unwrap();
        assert!(matches!(
            verify_card_signature(&card, b"other"),
            Err(A2AError::Signature(_))
        ));
    }

    #[test]
    fn verify_card_signature_unsigned_is_ok() {
        let card = AgentCard::new("agent", "desc", "http://localhost");
        assert!(verify_card_signature(&card, b"secret").is_ok());
    }

    #[tokio::test]
    async fn get_agent_card_verifies_signed_card() {
        let mut card = AgentCard::new("agent", "desc", "http://localhost");
        sign_agent_card(&mut card, b"secret").unwrap();
        let card_json = serde_json::to_string(&card).unwrap();
        let base = spawn_http_server(move |_head, _body| card_json.clone()).await;

        // Correct secret -> verified.
        let client = A2AClient::builder(base.clone())
            .card_verification_secret(b"secret")
            .build()
            .unwrap();
        let got = client.get_agent_card().await.unwrap();
        assert_eq!(got.name, "agent");

        // Wrong secret -> hard error.
        let client = A2AClient::builder(base)
            .card_verification_secret(b"wrong")
            .build()
            .unwrap();
        assert!(matches!(
            client.get_agent_card().await,
            Err(A2AError::Signature(_))
        ));
    }

    #[tokio::test]
    async fn get_agent_card_requires_signature_without_secret() {
        let mut card = AgentCard::new("agent", "desc", "http://localhost");
        sign_agent_card(&mut card, b"secret").unwrap();
        let card_json = serde_json::to_string(&card).unwrap();
        let base = spawn_http_server(move |_head, _body| card_json.clone()).await;

        let client = A2AClient::builder(base)
            .require_card_signature(true)
            .build()
            .unwrap();
        assert!(matches!(
            client.get_agent_card().await,
            Err(A2AError::Signature(_))
        ));
    }

    #[tokio::test]
    async fn get_agent_card_unsigned_passes_with_require_signature() {
        let base = spawn_http_server(|_head, _body| {
            serde_json::to_string(&AgentCard::new("agent", "desc", "http://localhost")).unwrap()
        })
        .await;
        let client = A2AClient::builder(base)
            .require_card_signature(true)
            .build()
            .unwrap();
        assert!(client.get_agent_card().await.is_ok());
    }

    // ---- P1-5: trace propagation ----

    #[tokio::test]
    async fn client_attaches_trace_id_to_requests() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();
        let base = spawn_http_server(move |_head, body| {
            seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(body.to_string());
            serde_json::to_string(&A2AResponse::ok(
                0,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap()
        })
        .await;
        let client = A2AClient::builder(base)
            .trace_id("trace-123")
            .build()
            .unwrap();
        let _ = client.send_task(A2AMessage::user("hi")).await;

        let bodies = seen.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!bodies.is_empty());
        let parsed: A2ARequest = serde_json::from_str(&bodies[0]).unwrap();
        assert_eq!(parsed.trace_id(), Some("trace-123"));
    }

    // ---- P2-8: W3C traceparent header ----

    #[tokio::test]
    async fn client_sends_traceparent_header() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();
        let base = spawn_http_server(move |head, _body| {
            seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(head.to_string());
            serde_json::to_string(&A2AResponse::ok(
                0,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap()
        })
        .await;

        let trace =
            TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").sampled();
        let client = A2AClient::builder(base)
            .with_traceparent(trace)
            .build()
            .unwrap();
        let _ = client.send_task(A2AMessage::user("hi")).await;

        let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!lines.is_empty());
        let head = &lines[0];
        assert!(
            head.to_ascii_lowercase()
                .contains("traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
            "expected W3C traceparent header, got: {head}"
        );
    }

    #[tokio::test]
    async fn client_without_trace_context_sends_no_traceparent() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();
        let base = spawn_http_server(move |head, _body| {
            seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(head.to_string());
            serde_json::to_string(&A2AResponse::ok(
                0,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap()
        })
        .await;

        let client = A2AClient::new(base).unwrap();
        let _ = client.send_task(A2AMessage::user("hi")).await;

        let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            !lines[0].to_ascii_lowercase().contains("traceparent:"),
            "no traceparent expected, got: {:?}",
            lines[0]
        );
    }

    // ---- P1-6: idempotent send ----

    #[tokio::test]
    async fn client_sends_message_id() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();
        let base = spawn_http_server(move |_head, body| {
            seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(body.to_string());
            serde_json::to_string(&A2AResponse::ok(
                0,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap()
        })
        .await;
        let client = A2AClient::new(base).unwrap();
        let _ = client
            .send_task_with_message_id(A2AMessage::user("hi"), "idem-1")
            .await;

        let bodies = seen.lock().unwrap_or_else(|e| e.into_inner());
        let parsed: A2ARequest = serde_json::from_str(&bodies[0]).unwrap();
        assert_eq!(parsed.message_id(), Some("idem-1"));
    }

    // ---- P2-3: input-required handling ----

    #[tokio::test]
    async fn send_task_and_wait_surfaces_input_required() {
        let base = spawn_http_server(|_head, body| {
            let req: A2ARequest = serde_json::from_str(body).unwrap();
            match req.method.as_str() {
                "tasks/send" => serde_json::to_string(&A2AResponse::ok(
                    req.id,
                    json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
                ))
                .unwrap(),
                "tasks/get" => serde_json::to_string(&A2AResponse::ok(
                    req.id,
                    json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "input-required"}, "error": "please provide your name"}),
                ))
                .unwrap(),
                _ => serde_json::to_string(&A2AResponse::error(req.id, -32601, "Method not found"))
                    .unwrap(),
            }
        })
        .await;
        let client = A2AClient::new(base).unwrap();
        let result = client
            .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(10))
            .await;
        match result {
            Err(A2AError::InputRequired { task_id, prompt }) => {
                assert_eq!(task_id, "t1");
                assert!(prompt.contains("please provide your name"));
            }
            other => panic!("expected InputRequired error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn resume_task_carries_task_id_and_message() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();
        let base = spawn_http_server(move |_head, body| {
            seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(body.to_string());
            serde_json::to_string(&A2AResponse::ok(
                0,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "working"}}),
            ))
            .unwrap()
        })
        .await;
        let client = A2AClient::new(base).unwrap();
        let _ = client
            .resume_task("t1", A2AMessage::user("my name is alice"))
            .await;

        let bodies = seen.lock().unwrap_or_else(|e| e.into_inner());
        let parsed: A2ARequest = serde_json::from_str(&bodies[0]).unwrap();
        assert_eq!(parsed.method, "tasks/send");
        let params = parsed.params.as_ref().unwrap();
        assert_eq!(params["taskId"], "t1");
        assert_eq!(params["message"]["content"], "my name is alice");
    }

    // ---- P2-1: SSE streaming ----

    #[tokio::test]
    async fn connect_sse_parses_task_notifications() {
        let base = spawn_sse_server(&[
            "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}",
            "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"completed\"}",
        ])
        .await;
        let client = A2AClient::new("http://localhost:1".to_string()).unwrap(); // URL unused by connect_sse
        let mut stream = client.connect_sse(&base).await.unwrap();

        let first = stream.next().await.expect("first event").unwrap();
        assert_eq!(first.id(), "t1");
        assert_eq!(first.status_value(), Some(TaskStatus::Working));

        let second = stream.next().await.expect("second event").unwrap();
        assert_eq!(second.status_value(), Some(TaskStatus::Completed));

        assert!(stream.next().await.is_none(), "stream should end");
    }

    #[tokio::test]
    async fn send_task_streaming_sends_then_streams() {
        let base = spawn_sse_rpc_server(&[
            "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}",
            "event: artifact-update\ndata: {\"kind\":\"artifact-update\",\"id\":\"t1\",\"artifact\":{\"output\":\"hi\"}}",
        ])
        .await;
        let client = A2AClient::new(base.clone()).unwrap();
        let mut stream = client
            .send_task_streaming(&format!("{}/sse", base), A2AMessage::user("hi"))
            .await
            .unwrap();

        let first = stream.next().await.expect("first event").unwrap();
        assert_eq!(first.id(), "t1");
        assert_eq!(first.status_value(), Some(TaskStatus::Working));

        let second = stream.next().await.expect("second event").unwrap();
        assert!(
            second.status_value().is_none(),
            "artifact events carry no status"
        );
    }
}
