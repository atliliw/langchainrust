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

mod signing;
mod sse;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskDetails, A2ATaskResult,
    AgentCard, TaskStatus, TraceContext,
};

pub use signing::{sign_agent_card, verify_card_signature};
pub use sse::A2ASseStream;

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
        Ok(Self::with_http_client(base_url, http))
    }

    /// Create a client with a custom `reqwest::Client` (for timeouts, etc.).
    pub fn with_http_client(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
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
mod tests;
