//! P2-6: layered resilience for the A2A client.
//!
//! [`ResilientA2AClient`] wraps one or more [`A2AClient`]s and adds four
//! independent layers of fault tolerance, so a caller can trade a little extra
//! latency for far fewer hard failures:
//!
//! - **L1 — transport retry**: transient network failures ([`A2AError::Http`],
//!   [`A2AError::Timeout`]) are retried on the same agent with exponential
//!   backoff. API-level errors (a JSON-RPC `error` payload) are *not* retried —
//!   the server answered, it just refused.
//! - **L2 — connection re-establishment**: SSE subscriptions re-establish the
//!   connection a configurable number of times instead of failing on the first
//!   dropped connect.
//! - **L3 — task recovery**: an in-flight task is recovered by polling
//!   `tasks/get` via [`ResilientA2AClient::wait_for_task`] rather than
//!   re-created. A timeout surfaces the `task_id` so the caller can resume the
//!   wait later, or drive the task to a terminal state themselves.
//! - **L4 — degradation**: when the primary agent is unreachable (or refuses an
//!   idempotent operation), the request is retried against the configured
//!   fallback agents in order.
//!
//! L1/L2/L4 are applied automatically by the `send_*`/`get_*`/`cancel_*`
//! helpers; L3 is exposed as a public recovery method and used by the
//! `send_task_and_wait` variants.

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use crate::client::{A2AClient, A2ASseStream};
use crate::protocol::{A2AMessage, A2ATask, A2ATaskDetails, A2ATaskResult, AgentCard, TaskStatus};
use crate::A2AError;

/// A boxed future that borrows the client it operates on (`'a`).
type BoxedOp<'a, T> = Pin<Box<dyn Future<Output = Result<T, A2AError>> + Send + 'a>>;

/// Tuning knobs for [`ResilientA2AClient`].
///
/// All four layers are governed here; the defaults are conservative (small
/// backoff, a couple of retries) so the wrapper is safe to enable everywhere.
#[derive(Debug, Clone)]
pub struct ResilienceConfig {
    /// L1: transport retries per request, on the same agent.
    ///
    /// A retry happens only for transient [`A2AError::Http`]/[`A2AError::Timeout`]
    /// failures; API-level errors surface immediately.
    pub max_transport_retries: usize,
    /// L1: base backoff between retries. The delay grows exponentially
    /// (`base`, `2*base`, `4*base`, ...) per attempt.
    pub retry_base_delay: Duration,
    /// L2: how many times a dropped SSE connection is re-established before the
    /// error is surfaced.
    pub max_reconnect_attempts: usize,
    /// L3: overall deadline for [`ResilientA2AClient::wait_for_task`] (and the
    /// `send_task_and_wait` variants) before the task is reported timed out.
    ///
    /// A timed-out wait is recoverable — the caller keeps the `task_id` and can
    /// call [`ResilientA2AClient::wait_for_task`] again.
    pub task_timeout: Duration,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            max_transport_retries: 2,
            retry_base_delay: Duration::from_millis(50),
            max_reconnect_attempts: 2,
            task_timeout: Duration::from_secs(30),
        }
    }
}

/// A client that absorbs transient failures and degrades to backup agents.
///
/// Wraps a primary [`A2AClient`] plus an ordered list of fallback agents (L4).
/// Every operation is attempted against the primary first — with L1 transport
/// retries — and moves down the fallback chain only when the current agent
/// cannot serve the request. For exactly-once semantics across retries prefer
/// the `*_with_message_id` variants.
pub struct ResilientA2AClient {
    /// The preferred agent.
    primary: A2AClient,
    /// Backup agents tried in order when the primary is unreachable (L4).
    fallbacks: Vec<A2AClient>,
    /// Retry / reconnect / recovery tuning.
    config: ResilienceConfig,
}

impl ResilientA2AClient {
    /// Create a resilient client around a single primary agent.
    pub fn new(primary: A2AClient, config: ResilienceConfig) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
            config,
        }
    }

    /// Add a fallback agent tried after the primary (L4).
    ///
    /// Multiple fallbacks are consulted in the order they are added.
    pub fn with_fallback(mut self, fallback: A2AClient) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    /// The active resilience configuration.
    pub fn config(&self) -> &ResilienceConfig {
        &self.config
    }

    /// Fetch the primary agent's card, retrying transient failures and
    /// degrading to fallbacks (L1/L4).
    pub async fn get_agent_card(&self) -> Result<AgentCard, A2AError> {
        self.with_fallbacks(&|c| Box::pin(async move { c.get_agent_card().await }))
            .await
    }

    /// Send a task, retrying transient transport failures and degrading to
    /// fallback agents (L1/L4).
    ///
    /// For exactly-once delivery across fallback hops use
    /// [`ResilientA2AClient::send_task_with_message_id`].
    pub async fn send_task(&self, message: A2AMessage) -> Result<A2ATask, A2AError> {
        self.with_fallbacks(&|c| {
            let msg = message.clone();
            Box::pin(async move { c.send_task(msg).await })
        })
        .await
    }

    /// Idempotent `tasks/send` — retried calls return the already-created task
    /// instead of running the chain twice (P1-6).
    pub async fn send_task_with_message_id(
        &self,
        message: A2AMessage,
        message_id: &str,
    ) -> Result<A2ATask, A2AError> {
        self.with_fallbacks(&|c| {
            let msg = message.clone();
            let mid = message_id.to_string();
            Box::pin(async move { c.send_task_with_message_id(msg, &mid).await })
        })
        .await
    }

    /// Resume an `input-required` task (P2-3), retrying and degrading like
    /// [`ResilientA2AClient::send_task`].
    pub async fn resume_task(
        &self,
        task_id: &str,
        message: A2AMessage,
    ) -> Result<A2ATask, A2AError> {
        self.with_fallbacks(&|c| {
            let tid = task_id.to_string();
            let msg = message.clone();
            Box::pin(async move { c.resume_task(&tid, msg).await })
        })
        .await
    }

    /// Read a task by ID (idempotent; safe to retry across fallbacks).
    pub async fn get_task(&self, task_id: &str) -> Result<A2ATask, A2AError> {
        self.with_fallbacks(&|c| {
            let tid = task_id.to_string();
            Box::pin(async move { c.get_task(&tid).await })
        })
        .await
    }

    /// Read a task's details and result (idempotent; safe to retry).
    pub async fn get_task_details(&self, task_id: &str) -> Result<A2ATaskDetails, A2AError> {
        self.with_fallbacks(&|c| {
            let tid = task_id.to_string();
            Box::pin(async move { c.get_task_details(&tid).await })
        })
        .await
    }

    /// Cancel a task by ID (idempotent; safe to retry).
    pub async fn cancel_task(&self, task_id: &str) -> Result<A2ATask, A2AError> {
        self.with_fallbacks(&|c| {
            let tid = task_id.to_string();
            Box::pin(async move { c.cancel_task(&tid).await })
        })
        .await
    }

    /// Send a task and block until it reaches a terminal state (L1/L4 + L3).
    ///
    /// The wait is recovered by polling `tasks/get` on the agent that owns the
    /// task — never by re-sending it. If the wait times out, the error carries
    /// the `task_id` so the caller can recover later with
    /// [`ResilientA2AClient::wait_for_task`].
    pub async fn send_task_and_wait(
        &self,
        message: A2AMessage,
        timeout: Duration,
    ) -> Result<A2ATaskResult, A2AError> {
        let task = self.send_task(message).await?;
        self.wait_for_task(&task.id, timeout).await
    }

    /// Idempotent variant of [`ResilientA2AClient::send_task_and_wait`].
    pub async fn send_task_and_wait_with_message_id(
        &self,
        message: A2AMessage,
        message_id: &str,
        timeout: Duration,
    ) -> Result<A2ATaskResult, A2AError> {
        let task = self.send_task_with_message_id(message, message_id).await?;
        self.wait_for_task(&task.id, timeout).await
    }

    /// L3: recover an in-flight task by polling `tasks/get` until it reaches a
    /// terminal state, without ever re-creating it.
    ///
    /// Polls the agent that owns the task (the primary — the `task_id` was
    /// issued there), applying L1 transport retries to each individual `get` so
    /// a transient network blip does not abort the wait. Returns the result on
    /// completion, a mapped error for `failed`/`cancelled`/`rejected`/`expired`
    /// and `auth-required`, an [`A2AError::InputRequired`] when the agent asks
    /// for more input, and an [`A2AError::Timeout`] carrying the `task_id` when
    /// `timeout` elapses.
    pub async fn wait_for_task(
        &self,
        task_id: &str,
        timeout: Duration,
    ) -> Result<A2ATaskResult, A2AError> {
        let start = Instant::now();
        let poll_interval = Duration::from_secs(1);

        loop {
            let details = retry_on(
                &self.primary,
                self.config.max_transport_retries,
                self.config.retry_base_delay,
                &|c| {
                    let tid = task_id.to_string();
                    Box::pin(async move { c.get_task_details(&tid).await })
                },
            )
            .await?;

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
                            "Task {} did not complete within {:?}; \
                             recover by calling wait_for_task again",
                            task_id, timeout
                        )));
                    }
                    tokio::time::sleep(poll_interval).await;
                }
            }
        }
    }

    /// Open an SSE subscription, re-establishing a dropped connection up to
    /// [`ResilienceConfig::max_reconnect_attempts`] times (L2).
    ///
    /// The `sse_url` pins the endpoint, so no fallback hopping is performed
    /// here; reconnection happens against the same agent. Events are also
    /// recoverable via [`ResilientA2AClient::wait_for_task`] if the stream is
    /// lost mid-flight.
    pub async fn connect_sse(&self, sse_url: &str) -> Result<A2ASseStream, A2AError> {
        retry_on(
            &self.primary,
            self.config.max_reconnect_attempts,
            self.config.retry_base_delay,
            &|c| {
                let url = sse_url.to_string();
                Box::pin(async move { c.connect_sse(&url).await })
            },
        )
        .await
    }

    /// Send a task and stream its notifications, reconnecting the subscription
    /// as needed (L2) and retrying the send (L1/L4).
    pub async fn send_task_streaming(
        &self,
        sse_url: &str,
        message: A2AMessage,
    ) -> Result<A2ASseStream, A2AError> {
        let stream = self.connect_sse(sse_url).await?;
        self.send_task(message).await?;
        Ok(stream)
    }

    /// L4: run `op` against the primary — with L1 retries — then against each
    /// fallback in order until one succeeds.
    async fn with_fallbacks<F, T>(&self, op: &F) -> Result<T, A2AError>
    where
        F: for<'a> Fn(&'a A2AClient) -> BoxedOp<'a, T>,
    {
        let mut last_err: Option<A2AError> = None;
        for client in std::iter::once(&self.primary).chain(self.fallbacks.iter()) {
            match retry_on(
                client,
                self.config.max_transport_retries,
                self.config.retry_base_delay,
                op,
            )
            .await
            {
                Ok(value) => return Ok(value),
                // A different agent might still serve this request — degrade.
                Err(err) if should_fallback(&err) => last_err = Some(err),
                // Parse/signature/input-required are caller-side problems; the
                // request itself is bad and would fail identically anywhere.
                Err(err) => return Err(err),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            A2AError::Http("no A2A endpoints configured for resilient client".to_string())
        }))
    }
}

/// L1: is this error transient enough to retry on the same agent?
fn is_retryable(err: &A2AError) -> bool {
    matches!(err, A2AError::Http(_) | A2AError::Timeout(_))
}

/// L4: could a different agent plausibly succeed where this one failed?
///
/// Transport and API errors might, so they trigger degradation; parse,
/// signature, and input-required errors are intrinsic to the request/agent and
/// surface immediately.
fn should_fallback(err: &A2AError) -> bool {
    !matches!(
        err,
        A2AError::Parse(_) | A2AError::Signature(_) | A2AError::InputRequired { .. }
    )
}

/// Run `op` against a single client, retrying transient failures with
/// exponential backoff up to `max_retries` (L1).
async fn retry_on<F, T>(
    client: &A2AClient,
    max_retries: usize,
    base_delay: Duration,
    op: &F,
) -> Result<T, A2AError>
where
    F: for<'a> Fn(&'a A2AClient) -> BoxedOp<'a, T>,
{
    for attempt in 0..=max_retries {
        match op(client).await {
            Ok(value) => return Ok(value),
            Err(err) if attempt < max_retries && is_retryable(&err) => {
                let delay = base_delay.saturating_mul(2u32.saturating_pow(attempt as u32));
                tokio::time::sleep(delay).await;
            }
            // Non-retryable error, or retries exhausted — surface it.
            Err(err) => return Err(err),
        }
    }
    unreachable!("loop covers attempts 0..={max_retries}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    /// Request handler: given the request path and JSON body, return the
    /// HTTP status and response body.
    type Handler = Arc<dyn Fn(&str, &str) -> (u16, String) + Send + Sync>;

    async fn read_request(stream: &mut TcpStream) -> (String, String) {
        let mut buf = vec![0u8; 4096];
        let mut request = Vec::new();
        let mut head_end = None;
        while head_end.is_none() {
            let n = stream.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            head_end = request.windows(4).position(|w| w == b"\r\n\r\n");
        }
        let head_end = head_end.expect("request head terminator");
        let head = String::from_utf8_lossy(&request[..head_end]).to_string();
        let body_len = head
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length:"))
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let mut body = request[head_end + 4..].to_vec();
        while body.len() < body_len {
            let n = stream.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            body.extend_from_slice(&buf[..n]);
        }
        let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
        (path, String::from_utf8_lossy(&body).to_string())
    }

    async fn write_response(stream: &mut TcpStream, status: u16, body: &str) {
        let reason = match status {
            200 => "OK",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Status",
        };
        let head = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(head.as_bytes()).await;
        let _ = stream.write_all(body.as_bytes()).await;
        let _ = stream.shutdown().await;
    }

    /// Spawn a raw HTTP server on an ephemeral port; returns its base URL.
    async fn spawn_server(handler: Handler) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let handler = handler.clone();
                tokio::spawn(async move {
                    let mut stream = stream;
                    let (path, body) = read_request(&mut stream).await;
                    let (status, response) = handler(&path, &body);
                    write_response(&mut stream, status, &response).await;
                });
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    /// A `tasks/send` or `tasks/get` success payload with a completed task.
    fn completed_task_response(task_id: &str, output: &str) -> String {
        format!(
            r#"{{"jsonrpc":"2.0","id":1,"result":{{"task":{{"id":"{task_id}","message":{{"role":"user","content":"hi"}},"status":"completed"}},"result":{{"output":"{output}"}}}}}}"#
        )
    }

    /// A `tasks/send` success payload with a still-working task.
    fn working_task_response(task_id: &str) -> String {
        format!(
            r#"{{"jsonrpc":"2.0","id":1,"result":{{"task":{{"id":"{task_id}","message":{{"role":"user","content":"hi"}},"status":"working"}}}}}}"#
        )
    }

    /// A JSON-RPC error payload (HTTP 200 with an `error` member).
    fn api_error_response() -> String {
        r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"denied"}}"#.to_string()
    }

    fn fast_config() -> ResilienceConfig {
        ResilienceConfig {
            max_transport_retries: 2,
            retry_base_delay: Duration::from_millis(10),
            max_reconnect_attempts: 0,
            task_timeout: Duration::from_secs(5),
        }
    }

    #[tokio::test]
    async fn transport_retry_recovers_from_transient_failures() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let handler: Handler = Arc::new(move |_path, _body| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                (500, "boom".to_string())
            } else {
                (200, completed_task_response("task-retry", "ok"))
            }
        });
        let base = spawn_server(handler).await;

        let client = ResilientA2AClient::new(A2AClient::new(base).unwrap(), fast_config());
        let task = client.send_task(A2AMessage::user("hi")).await.unwrap();
        assert_eq!(task.id, "task-retry");
        // Two transport failures + one success = three attempts on the primary.
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn api_error_is_not_retried() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let handler: Handler = Arc::new(move |_path, _body| {
            counter.fetch_add(1, Ordering::SeqCst);
            (200, api_error_response())
        });
        let base = spawn_server(handler).await;

        let client = ResilientA2AClient::new(A2AClient::new(base).unwrap(), fast_config());
        let err = client.send_task(A2AMessage::user("hi")).await.unwrap_err();
        assert!(matches!(err, A2AError::Api { code: -32000, .. }));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn falls_back_to_alternate_agent_when_primary_is_down() {
        let primary_hits = Arc::new(AtomicUsize::new(0));
        let hits = primary_hits.clone();
        let primary: Handler = Arc::new(move |_path, _body| {
            hits.fetch_add(1, Ordering::SeqCst);
            (500, "down".to_string())
        });
        let primary_base = spawn_server(primary).await;

        let fallback: Handler =
            Arc::new(|_path, _body| (200, completed_task_response("from-fallback", "ok")));
        let fallback_base = spawn_server(fallback).await;

        let config = ResilienceConfig {
            max_transport_retries: 1,
            retry_base_delay: Duration::from_millis(5),
            ..fast_config()
        };
        let client = ResilientA2AClient::new(A2AClient::new(primary_base).unwrap(), config)
            .with_fallback(A2AClient::new(fallback_base).unwrap());

        let task = client.send_task(A2AMessage::user("hi")).await.unwrap();
        assert_eq!(task.id, "from-fallback");
        assert_eq!(primary_hits.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_fall_back_on_parse_error() {
        let primary: Handler = Arc::new(|_path, _body| (200, "not-json{{{".to_string()));
        let primary_base = spawn_server(primary).await;

        let fallback_hits = Arc::new(AtomicUsize::new(0));
        let hits = fallback_hits.clone();
        let fallback: Handler = Arc::new(move |_path, _body| {
            hits.fetch_add(1, Ordering::SeqCst);
            (200, completed_task_response("fallback", "ok"))
        });
        let fallback_base = spawn_server(fallback).await;

        let client = ResilientA2AClient::new(A2AClient::new(primary_base).unwrap(), fast_config())
            .with_fallback(A2AClient::new(fallback_base).unwrap());

        let err = client.send_task(A2AMessage::user("hi")).await.unwrap_err();
        assert!(matches!(err, A2AError::Parse(_)));
        assert_eq!(fallback_hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn wait_for_task_recovers_a_pending_task_by_id() {
        let handler: Handler = Arc::new(|_path, body| {
            if body.contains("tasks/get") {
                (200, completed_task_response("t1", "recovered"))
            } else {
                (200, working_task_response("t1"))
            }
        });
        let base = spawn_server(handler).await;

        let client = ResilientA2AClient::new(A2AClient::new(base).unwrap(), fast_config());
        let result = client
            .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(result.output, "recovered");
    }

    #[tokio::test]
    async fn wait_for_task_times_out_and_exposes_task_id_for_recovery() {
        let handler: Handler = Arc::new(|_path, _body| (200, working_task_response("t-stuck")));
        let base = spawn_server(handler).await;

        let client = ResilientA2AClient::new(A2AClient::new(base).unwrap(), fast_config());
        let err = client
            .wait_for_task("t-stuck", Duration::from_millis(100))
            .await
            .unwrap_err();
        assert!(matches!(err, A2AError::Timeout(_)));
        assert!(err.to_string().contains("t-stuck"));
    }

    #[tokio::test]
    async fn connect_sse_reconnects_after_transient_failure() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let handler: Handler = Arc::new(move |_path, _body| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                (500, "down".to_string())
            } else {
                (200, String::new())
            }
        });
        let base = spawn_server(handler).await;

        let config = ResilienceConfig {
            max_reconnect_attempts: 1,
            retry_base_delay: Duration::from_millis(5),
            ..fast_config()
        };
        let client = ResilientA2AClient::new(A2AClient::new(base.clone()).unwrap(), config);
        let _stream = client.connect_sse(&format!("{base}/events")).await.unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }
}
