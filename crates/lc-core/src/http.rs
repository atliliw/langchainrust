//! Unified outbound HTTP client.
//!
//! Every provider/API call in the workspace should eventually go through
//! [`HttpClient`] so that timeout, retry, body-size and cancellation policies
//! are defined in one place instead of drifting across 20 crates.
//!
//! # Policies
//!
//! - Two profiles: [`Profile::Api`] (connect 10s, total 60s, body capped at
//!   10 MiB) and [`Profile::Sse`] (connect 10s, total 300s, response body is
//!   streamed, not buffered).
//! - Retries are attempted **only** for statuses `408/429/500/502/503/504` and
//!   for connection/timeout transport errors. `501` and `505` are returned
//!   immediately — retrying a "not implemented"/"version unsupported" response
//!   just hammers a server that can never succeed.
//! - `Retry-After` is honored in both its forms (delta-seconds **and**
//!   IMF-fixdate, per RFC 9110) and is never truncated by the backoff cap.
//! - Backoff uses full jitter; retry sleeps are cancellation-aware.
//! - For SSE, retries happen only while establishing the stream (before the
//!   first successful response). A stream that breaks mid-flight is **not**
//!   reconnected, because that would silently replay already-delivered events.
//!
//! # SSRF
//!
//! This client is for public provider endpoints and does **not** perform
//! private-address pinning. Callers that fetch user-supplied URLs (tools, web
//! loaders) must keep using [`crate::ssrf::guarded_get`] and friends.

use std::io;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, RETRY_AFTER};
use reqwest::{Method, StatusCode};
use serde_json::Value;

use crate::runnables::CancellationToken;
use crate::ssrf::read_body_bounded;

/// Default TCP connect timeout for both profiles.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Default total (connect + send + response headers) budget for API calls.
pub const DEFAULT_API_TOTAL_TIMEOUT: Duration = Duration::from_secs(60);
/// Default response body cap for API calls: 10 MiB.
pub const DEFAULT_API_MAX_BYTES: usize = 10 * 1024 * 1024;
/// Default total budget for establishing an SSE stream.
pub const DEFAULT_SSE_TOTAL_TIMEOUT: Duration = Duration::from_secs(300);
/// Upper bound read from an error response body so messages stay bounded.
pub const ERROR_BODY_BYTES: usize = 64 * 1024;

/// Errors produced by the unified HTTP layer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HttpError {
    /// The underlying reqwest client could not be constructed.
    #[error("failed to build HTTP client: {0}")]
    Build(String),
    /// A URL failed validation.
    #[error("invalid URL {url:?}: {reason}")]
    InvalidUrl {
        /// The rejected URL.
        url: String,
        /// Why it was rejected.
        reason: String,
    },
    /// A transport-level failure (DNS, connection, body read, ...).
    #[error("HTTP transport error: {0}")]
    Transport(String),
    /// The total timeout elapsed before a response/body was produced.
    #[error("HTTP operation timed out after {0:?}")]
    Timeout(Duration),
    /// The operation was cancelled via a [`CancellationToken`].
    #[error("HTTP operation cancelled")]
    Cancelled,
    /// The response body exceeded the configured byte limit.
    #[error("response body exceeded the {limit} byte limit")]
    BodyTooLarge {
        /// The configured limit.
        limit: usize,
    },
    /// A non-success HTTP status. Error bodies are read with a small bound.
    #[error("HTTP status {status}: {body}")]
    Status {
        /// Numeric status code.
        status: u16,
        /// Bounded response body.
        body: String,
    },
}

/// Which transport errors are eligible for retry.
///
/// A request whose body may already have been dispatched to the server can be
/// ambiguous: retrying may double-charge or duplicate a side effect. This
/// mirrors the A14 boundary documented in lc-providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportRetryMode {
    /// Retry on connect, timeout and generic request errors. Better
    /// availability; carries a small double-dispatch risk.
    AllTransportErrors,
    /// Retry only on errors known to have happened before the request was
    /// dispatched (connect failures). Safe for non-idempotent calls.
    PreDispatchOnly,
}

/// Retry policy: attempt count, backoff bounds and transport classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    /// Total attempts including the first one (1 = no retries).
    pub max_attempts: usize,
    /// Base delay for exponential backoff.
    pub base_delay: Duration,
    /// Cap on *computed* backoff; a server-specified `Retry-After` is not
    /// truncated by this value.
    pub max_delay: Duration,
    /// Which transport errors are retriable.
    pub transport: TransportRetryMode,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            transport: TransportRetryMode::AllTransportErrors,
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries.
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            ..Default::default()
        }
    }
}

/// Selects the built-in timeout/body profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Buffered JSON API call: bounded total time and bounded response body.
    Api,
    /// Streaming response: longer establishment budget, body never buffered.
    Sse,
}

/// Per-request overrides applied on top of the client defaults.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    bearer: Option<String>,
    headers: HeaderMap,
    /// Per-request override of the retry-transport classification. When
    /// `None` (the default), non-idempotent methods (POST/...) are retried
    /// only on pre-dispatch (connect) failures.
    retry_mode: Option<TransportRetryMode>,
}

impl RequestOptions {
    /// Creates empty options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sends `Authorization: Bearer <token>` for this request only.
    pub fn bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Adds one header for this request only. Replaces a client default
    /// header with the same name.
    pub fn header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(name, value);
        self
    }

    /// Overrides which transport errors make THIS request retry.
    ///
    /// By default POST (and other non-idempotent methods) use
    /// [`TransportRetryMode::PreDispatchOnly`] so a request already
    /// dispatched to the server cannot be duplicated by a retry; safe
    /// methods (GET/HEAD) follow the client policy. Set
    /// [`TransportRetryMode::AllTransportErrors`] here only when the call is
    /// idempotent or deduplicated server side.
    pub fn retry_mode(mut self, mode: TransportRetryMode) -> Self {
        self.retry_mode = Some(mode);
        self
    }
}

/// Builder for [`HttpClient`]. Construct via [`HttpClient::api`] /
/// [`HttpClient::sse`] or [`HttpClient::builder`].
#[derive(Debug, Clone)]
pub struct HttpClientBuilder {
    profile: Profile,
    connect_timeout: Duration,
    total_timeout: Duration,
    max_bytes: usize,
    retry: RetryPolicy,
    bearer: Option<String>,
    user_agent: Option<String>,
    default_headers: HeaderMap,
    use_system_proxy: bool,
    cancel: Option<CancellationToken>,
}

impl HttpClientBuilder {
    fn new(profile: Profile) -> Self {
        let (total_timeout, max_bytes) = match profile {
            Profile::Api => (DEFAULT_API_TOTAL_TIMEOUT, DEFAULT_API_MAX_BYTES),
            Profile::Sse => (DEFAULT_SSE_TOTAL_TIMEOUT, usize::MAX),
        };
        Self {
            profile,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            total_timeout,
            max_bytes,
            retry: RetryPolicy::default(),
            bearer: None,
            user_agent: None,
            default_headers: HeaderMap::new(),
            // Default to no_proxy(): local proxy software (Clash, corporate
            // proxies) otherwise intercepts loopback/LAN base URLs. This was
            // a production incident (T4).
            use_system_proxy: false,
            cancel: None,
        }
    }

    /// Overrides connect and total timeouts.
    pub fn timeouts(mut self, connect: Duration, total: Duration) -> Self {
        self.connect_timeout = connect;
        self.total_timeout = total;
        self
    }

    /// Overrides the response body byte cap (Api profile only).
    pub fn max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Sets the retry policy.
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    /// Sets a default bearer token for every request.
    pub fn bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Sets the `User-Agent` header.
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Adds a default header sent on every request.
    pub fn default_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.default_headers.insert(name, value);
        self
    }

    /// Honors system proxy environment variables (default: ignores them).
    pub fn use_system_proxy(mut self) -> Self {
        self.use_system_proxy = true;
        self
    }

    /// Attaches a cancellation token observed during retries.
    pub fn cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Builds the client.
    pub fn build(self) -> Result<HttpClient, HttpError> {
        let mut builder = reqwest::Client::builder()
            .connect_timeout(self.connect_timeout)
            // No reqwest-level timeout: the client enforces a total deadline
            // that spans retries; an SSE stream must not be cut mid-flight.
            .redirect(reqwest::redirect::Policy::none());
        if !self.use_system_proxy {
            builder = builder.no_proxy();
        }
        if let Some(ua) = &self.user_agent {
            builder = builder.user_agent(ua);
        }
        let client = builder
            .build()
            .map_err(|e| HttpError::Build(e.to_string()))?;

        Ok(HttpClient {
            client,
            profile: self.profile,
            total_timeout: self.total_timeout,
            max_bytes: self.max_bytes,
            retry: self.retry,
            bearer: self.bearer,
            default_headers: self.default_headers,
            cancel: self.cancel,
        })
    }
}

/// A configured outbound HTTP client. Cheap to clone (wraps an Arc'd pool).
#[derive(Debug, Clone)]
pub struct HttpClient {
    client: reqwest::Client,
    profile: Profile,
    total_timeout: Duration,
    max_bytes: usize,
    retry: RetryPolicy,
    bearer: Option<String>,
    default_headers: HeaderMap,
    cancel: Option<CancellationToken>,
}

impl HttpClient {
    /// Builder for an API-style buffered client.
    pub fn api() -> HttpClientBuilder {
        HttpClientBuilder::new(Profile::Api)
    }

    /// Builder for an SSE streaming client.
    pub fn sse() -> HttpClientBuilder {
        HttpClientBuilder::new(Profile::Sse)
    }

    /// Builder for an explicit profile.
    pub fn builder(profile: Profile) -> HttpClientBuilder {
        HttpClientBuilder::new(profile)
    }

    /// The profile this client was built with.
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// GET returning a buffered, size-bounded response (error statuses are
    /// returned as [`HttpError::Status`]).
    pub async fn get(&self, url: &str) -> Result<BoundedResponse, HttpError> {
        self.get_with(url, RequestOptions::new()).await
    }

    /// GET with per-request options.
    pub async fn get_with(
        &self,
        url: &str,
        opts: RequestOptions,
    ) -> Result<BoundedResponse, HttpError> {
        let (resp, deadline) = self.execute(Method::GET, url, None, &opts).await?;
        self.read_bounded_within(resp, deadline).await
    }

    /// POST a JSON body returning a buffered, size-bounded response.
    pub async fn post_json(&self, url: &str, body: &Value) -> Result<BoundedResponse, HttpError> {
        self.post_json_with(url, body, RequestOptions::new()).await
    }

    /// POST a JSON body with per-request options.
    pub async fn post_json_with(
        &self,
        url: &str,
        body: &Value,
        opts: RequestOptions,
    ) -> Result<BoundedResponse, HttpError> {
        let (resp, deadline) = self.execute(Method::POST, url, Some(body), &opts).await?;
        self.read_bounded_within(resp, deadline).await
    }

    /// POSTs a `multipart/form-data` body with per-request options.
    ///
    /// A multipart form cannot be replayed (parts may stream from non-cloneable
    /// sources), so this performs exactly ONE send attempt: no retries, not
    /// even on a connect failure or retriable status. The total deadline,
    /// cancellation token and response size bound still apply. The response
    /// body is buffered like [`HttpClient::post_json_with`].
    pub async fn post_multipart_with(
        &self,
        url: &str,
        form: reqwest::multipart::Form,
        opts: RequestOptions,
    ) -> Result<BoundedResponse, HttpError> {
        if self
            .cancel
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(HttpError::Cancelled);
        }
        let deadline = tokio::time::Instant::now() + self.total_timeout;
        let request = self
            .client
            .request(Method::POST, url)
            .multipart(form)
            .headers(self.compose_headers(&opts));
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .ok_or(HttpError::Timeout(self.total_timeout))?;
        let resp = match tokio::time::timeout(remaining, request.send()).await {
            Err(_) => return Err(HttpError::Timeout(self.total_timeout)),
            Ok(Ok(resp)) => resp,
            Ok(Err(err)) => {
                if err.is_timeout() {
                    return Err(HttpError::Timeout(self.total_timeout));
                }
                return Err(HttpError::Transport(err.to_string()));
            }
        };
        self.read_bounded_within(resp, deadline).await
    }

    /// Opens an SSE byte stream. Retries cover only stream establishment;
    /// once a successful response header set is returned the stream runs to
    /// completion without reconnecting.
    ///
    /// Named `open_sse` (not `sse`) because [`HttpClient::sse`] is the
    /// SSE-profile builder constructor.
    pub async fn open_sse(
        &self,
        url: &str,
        body: Option<&Value>,
        opts: RequestOptions,
    ) -> Result<SseStream, HttpError> {
        let method = if body.is_some() {
            Method::POST
        } else {
            Method::GET
        };
        let (resp, deadline) = self.execute(method, url, body, &opts).await?;
        let status = resp.status();
        if !status.is_success() {
            // Bound the error-body read with the same deadline `execute` used
            // for the request: `read_body_bounded` is only size-capped, so a
            // server that answers e.g. `500` with a small body it never closes
            // would otherwise hang this caller forever (the buffered Api path
            // fixes this via `read_bounded_within`; the SSE path must too).
            let remaining = match deadline.checked_duration_since(tokio::time::Instant::now()) {
                Some(rem) => rem,
                None => return Err(HttpError::Timeout(self.total_timeout)),
            };
            let (text, truncated) =
                match tokio::time::timeout(remaining, read_body_bounded(resp, ERROR_BODY_BYTES))
                    .await
                {
                    Ok(Ok(pair)) => pair,
                    Ok(Err(e)) => {
                        return Err(HttpError::Transport(format!("read error body: {e}")));
                    }
                    Err(_elapsed) => return Err(HttpError::Timeout(self.total_timeout)),
                };
            let body = if truncated {
                truncate_for_error(&text)
            } else {
                text
            };
            return Err(HttpError::Status {
                status: status.as_u16(),
                body,
            });
        }
        let stream = resp
            .bytes_stream()
            .map(|result| result.map_err(|e| io::Error::other(e.to_string())));
        Ok(Box::pin(stream))
    }

    /// Read the buffered body subject to the overall deadline.
    ///
    /// Only capping `send()` left a server trickling response headers/body
    /// bytes forever after the headers arrived able to hang the caller
    /// indefinitely; the total timeout must cover the body read too (SSE is
    /// the deliberate exception and uses `bytes_stream()` directly).
    async fn read_bounded_within(
        &self,
        resp: reqwest::Response,
        deadline: tokio::time::Instant,
    ) -> Result<BoundedResponse, HttpError> {
        let remaining = match deadline.checked_duration_since(tokio::time::Instant::now()) {
            Some(remaining) => remaining,
            None => return Err(HttpError::Timeout(self.total_timeout)),
        };
        match tokio::time::timeout(remaining, self.read_bounded(resp)).await {
            Ok(result) => result,
            Err(_elapsed) => Err(HttpError::Timeout(self.total_timeout)),
        }
    }

    async fn read_bounded(&self, resp: reqwest::Response) -> Result<BoundedResponse, HttpError> {
        let status = resp.status();
        let headers = resp.headers().clone();
        let (body, truncated) = read_body_bounded(resp, self.max_bytes)
            .await
            .map_err(|e| HttpError::Transport(format!("read response body: {e}")))?;
        if truncated {
            return Err(HttpError::BodyTooLarge {
                limit: self.max_bytes,
            });
        }
        // Buffered calls (`get`/`post_json`) document non-2xx responses as
        // HttpError::Status; convert here so callers never mistake an error
        // body for a success payload to deserialize. Callers that need the raw
        // status can use the request builder directly (SSE checks inline).
        BoundedResponse {
            status,
            headers,
            body,
        }
        .error_for_status()
    }

    /// Merges client-default headers, per-request headers and the resolved
    /// bearer token (per-request override wins over the client default).
    fn compose_headers(&self, opts: &RequestOptions) -> HeaderMap {
        // Client defaults first, per-request headers override same names. Note:
        // `HeaderMap::extend`/`append` would ADD a second value for an existing
        // name (default + override both sent) — we `insert` so the per-request
        // header replaces the default, honoring the documented override contract.
        let mut headers = self.default_headers.clone();
        for (k, v) in opts.headers.iter() {
            headers.insert(k.clone(), v.clone());
        }
        let bearer = opts.bearer.as_deref().or(self.bearer.as_deref());
        if let Some(token) = bearer {
            if let Ok(value) = HeaderValue::from_str(&format!("Bearer {token}")) {
                headers.insert(reqwest::header::AUTHORIZATION, value);
            }
        }
        headers
    }

    fn prepare_request(
        &self,
        method: Method,
        url: &str,
        json: Option<&Value>,
        opts: &RequestOptions,
    ) -> reqwest::RequestBuilder {
        let mut builder = self.client.request(method, url);
        if let Some(json) = json {
            builder = builder.json(json);
        }
        builder = builder.headers(self.compose_headers(opts));
        builder
    }

    async fn execute(
        &self,
        method: Method,
        url: &str,
        json: Option<&Value>,
        opts: &RequestOptions,
    ) -> Result<(reqwest::Response, tokio::time::Instant), HttpError> {
        let deadline = tokio::time::Instant::now() + self.total_timeout;
        // Non-idempotent methods default to pre-dispatch-only retries
        // unless the caller explicitly opts in, so a POST already on the
        // wire cannot be duplicated by a transport timeout (0.25.0).
        let retry_mode = effective_retry_mode(&method, opts, self.retry.transport);
        let mut attempt: usize = 0;
        loop {
            if self
                .cancel
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err(HttpError::Cancelled);
            }
            let remaining = match deadline.checked_duration_since(tokio::time::Instant::now()) {
                Some(remaining) => remaining,
                None => return Err(HttpError::Timeout(self.total_timeout)),
            };
            let request = self.prepare_request(method.clone(), url, json, opts);
            match tokio::time::timeout(remaining, request.send()).await {
                Err(_) => return Err(HttpError::Timeout(self.total_timeout)),
                Ok(Ok(resp)) => {
                    let status = resp.status();
                    if attempt + 1 < self.retry.max_attempts && is_retryable_status(status) {
                        let retry_after = parse_retry_after(resp.headers());
                        drop(resp);
                        self.wait(attempt, retry_after, deadline).await?;
                        attempt += 1;
                        continue;
                    }
                    return Ok((resp, deadline));
                }
                Ok(Err(err)) => {
                    if attempt + 1 < self.retry.max_attempts
                        && is_retryable_transport(&err, retry_mode)
                    {
                        self.wait(attempt, None, deadline).await?;
                        attempt += 1;
                        continue;
                    }
                    if err.is_timeout() {
                        return Err(HttpError::Timeout(self.total_timeout));
                    }
                    return Err(HttpError::Transport(err.to_string()));
                }
            }
        }
    }

    async fn wait(
        &self,
        attempt: usize,
        retry_after: Option<Duration>,
        deadline: tokio::time::Instant,
    ) -> Result<(), HttpError> {
        let backoff = compute_backoff(attempt, self.retry, random_entropy());
        let delay = retry_after.map_or(backoff, |ra| ra.max(backoff));
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .unwrap_or(Duration::ZERO);
        if remaining.is_zero() {
            return Err(HttpError::Timeout(self.total_timeout));
        }
        let sleep_for = delay.min(remaining);
        match &self.cancel {
            Some(token) => {
                tokio::select! {
                    () = tokio::time::sleep(sleep_for) => {}
                    () = token.cancelled() => return Err(HttpError::Cancelled),
                }
            }
            None => tokio::time::sleep(sleep_for).await,
        }
        Ok(())
    }
}

/// A fully buffered response with status and headers.
#[derive(Debug, Clone)]
pub struct BoundedResponse {
    /// HTTP status code.
    pub status: StatusCode,
    /// Response headers.
    pub headers: HeaderMap,
    /// UTF-8 response body.
    pub body: String,
}

impl BoundedResponse {
    /// Returns true for 2xx statuses.
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Converts a non-2xx response into [`HttpError::Status`].
    pub fn error_for_status(self) -> Result<Self, HttpError> {
        if self.is_success() {
            Ok(self)
        } else {
            Err(HttpError::Status {
                status: self.status.as_u16(),
                body: truncate_for_error(&self.body),
            })
        }
    }
}

/// Boxed SSE byte stream.
pub type SseStream = Pin<Box<dyn Stream<Item = io::Result<bytes::Bytes>> + Send>>;

/// Returns true for the closed set of retriable statuses:
/// 408, 429, 500, 502, 503, 504. Notably 501/505 are NOT retriable.
pub fn is_retryable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504)
}

/// Selects the retry classification for one attempt: an explicit per-request
/// override wins; otherwise safe methods (GET/HEAD) keep the client policy
/// while non-idempotent methods (POST/PUT/PATCH/DELETE) fall back to
/// pre-dispatch-only so a request already on the wire is never duplicated
/// by a retry (0.25.0).
fn effective_retry_mode(
    method: &Method,
    opts: &RequestOptions,
    client_mode: TransportRetryMode,
) -> TransportRetryMode {
    match opts.retry_mode {
        Some(mode) => mode,
        None if method.is_safe() => client_mode,
        None => TransportRetryMode::PreDispatchOnly,
    }
}

fn is_retryable_transport(err: &reqwest::Error, mode: TransportRetryMode) -> bool {
    if err.is_connect() {
        return true;
    }
    matches!(mode, TransportRetryMode::AllTransportErrors) && (err.is_timeout() || err.is_request())
}

/// Parses a `Retry-After` header, accepting both delta-seconds and an
/// IMF-fixdate. A date in the past resolves to [`Duration::ZERO`].
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let date = httpdate::parse_http_date(value).ok()?;
    Some(
        date.duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO),
    )
}

/// Full-jitter backoff: uniform in `[0, min(base * 2^attempt, max_delay)]`.
pub fn compute_backoff(attempt: usize, policy: RetryPolicy, entropy_nanos: u64) -> Duration {
    // Duration::checked_mul takes u32; attempts past 31 saturate the multiplier.
    let shift = (attempt as u32).min(31);
    let multiplier = 1u32.checked_shl(shift).unwrap_or(u32::MAX);
    let uncapped = policy
        .base_delay
        .checked_mul(multiplier)
        .unwrap_or(policy.max_delay);
    let cap = uncapped.min(policy.max_delay);
    let span = cap.as_nanos();
    if span == 0 {
        return Duration::ZERO;
    }
    // u128 -> u64 is safe: cap is bounded by max_delay (seconds, not ages).
    let jitter = entropy_nanos % (span as u64);
    Duration::from_nanos(jitter)
}

fn random_entropy() -> u64 {
    // B1: full-jitter's sample window spans `[0, min(base*2^attempt, max_delay)]`.
    // Using only `subsec_nanos` caps the entropy at <1e9, so a `max_delay` above 1s
    // could never be reached by the jitter. Mix the whole timestamp (seconds ^ nanos)
    // into the entropy so it spans the u64 domain, while still avoiding pulling a
    // `rand` dependency through the core crate.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| (d.as_secs() ^ d.subsec_nanos() as u64) as u64)
        .unwrap_or(0)
}

fn truncate_for_error(body: &str) -> String {
    const MAX_ERROR_CHARS: usize = 2_000;
    if body.chars().count() <= MAX_ERROR_CHARS {
        body.to_string()
    } else {
        let truncated: String = body.chars().take(MAX_ERROR_CHARS).collect();
        format!("{truncated}…(truncated)")
    }
}

/// Canonical base-URL normalization used by every provider config.
///
/// Trims whitespace and a single trailing slash, rejects non-http(s) schemes
/// and empty hosts.
pub fn normalize_base_url(raw: &str) -> Result<String, HttpError> {
    let trimmed = raw.trim().trim_end_matches('/');
    let parsed = url::Url::parse(trimmed).map_err(|e| HttpError::InvalidUrl {
        url: raw.to_string(),
        reason: e.to_string(),
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(HttpError::InvalidUrl {
            url: raw.to_string(),
            reason: format!("unsupported scheme {}", parsed.scheme()),
        });
    }
    if parsed.host_str().is_none_or(str::is_empty) {
        return Err(HttpError::InvalidUrl {
            url: raw.to_string(),
            reason: "missing host".to_string(),
        });
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    fn fast_retry() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            ..Default::default()
        }
    }

    // ---- pure unit tests ------------------------------------------------

    #[test]
    fn retryable_status_set() {
        for code in [408u16, 429, 500, 502, 503, 504] {
            assert!(is_retryable_status(StatusCode::from_u16(code).unwrap()));
        }
        for code in [400, 401, 403, 404, 409, 422, 501, 505] {
            assert!(!is_retryable_status(StatusCode::from_u16(code).unwrap()));
        }
    }

    #[test]
    fn retry_after_delta_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("42"));
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(42)));
    }

    #[test]
    fn retry_after_http_date() {
        let mut headers = HeaderMap::new();
        // Fixed date far in the past → ZERO, no waiting in tests.
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&headers), Some(Duration::ZERO));
    }

    #[test]
    fn retry_after_future_http_date_is_some() {
        let mut headers = HeaderMap::new();
        // Roughly one hour in the future, formatted as IMF-fixdate.
        let future = SystemTime::now() + Duration::from_secs(3600);
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_str(&httpdate::fmt_http_date(future)).unwrap(),
        );
        let parsed = parse_retry_after(&headers).unwrap();
        assert!(parsed >= Duration::from_secs(3500) && parsed <= Duration::from_secs(3600));
    }

    #[test]
    fn backoff_always_bounded_by_cap() {
        let policy = fast_retry();
        for attempt in 0..10u32 {
            for entropy in [0u64, 1, 7, u64::MAX] {
                let d = compute_backoff(attempt as usize, policy, entropy);
                assert!(d <= policy.max_delay, "attempt {attempt}");
            }
        }
    }

    #[test]
    fn normalizes_base_urls() {
        assert_eq!(
            normalize_base_url("https://api.example.com/v1/").unwrap(),
            "https://api.example.com/v1"
        );
        assert_eq!(
            normalize_base_url("  http://localhost:8080  ").unwrap(),
            "http://localhost:8080"
        );
        assert!(normalize_base_url("ftp://api.example.com").is_err());
        assert!(normalize_base_url("not a url").is_err());
        assert!(normalize_base_url("https://").is_err());
    }

    // ---- raw TCP loopback stubs -----------------------------------------
    //
    // A plain TcpListener is used (not hyper): the sandbox blocks real HTTP
    // servers but raw loopback sockets work, and hand-written HTTP/1.1 is
    // enough to exercise status/retry/header behavior.

    /// One scripted response. The last entry is repeated for any extra
    /// request. Each response closes the connection.
    #[derive(Debug, Clone)]
    struct StubResponse {
        status: u16,
        reason: &'static str,
        extra_headers: &'static str,
        body: Vec<u8>,
    }

    fn spawn_stub(script: Vec<StubResponse>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let count_task = count.clone();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let idx = count_task.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Drain the request head (and any body) so the client can
                // finish sending.
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let resp = script
                    .get(idx)
                    .unwrap_or_else(|| script.last().expect("empty stub script"))
                    .clone();
                let head = format!(
                    "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
                    resp.status,
                    resp.reason,
                    resp.body.len(),
                    resp.extra_headers
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&resp.body);
                let _ = stream.flush();
            }
        });
        (format!("http://127.0.0.1:{}", addr.port()), count)
    }

    fn body_ok(text: &str) -> StubResponse {
        StubResponse {
            status: 200,
            reason: "OK",
            extra_headers: "Content-Type: application/json\r\n",
            body: text.as_bytes().to_vec(),
        }
    }

    fn status_response(status: u16, reason: &'static str) -> StubResponse {
        StubResponse {
            status,
            reason,
            extra_headers: "",
            body: b"{}".to_vec(),
        }
    }

    // ---- loopback integration tests -------------------------------------

    #[tokio::test]
    async fn success_is_not_retried() {
        let (url, count) = spawn_stub(vec![body_ok("{\"ok\":true}")]);
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_secs(10))
            .retry(fast_retry())
            .build()
            .unwrap();
        let resp = client.get(&url).await.unwrap();
        assert!(resp.is_success());
        assert_eq!(resp.body, "{\"ok\":true}");
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retries_503_twice_then_succeeds() {
        let script = vec![
            status_response(503, "Service Unavailable"),
            status_response(503, "Service Unavailable"),
            body_ok("{}"),
        ];
        let (url, count) = spawn_stub(script);
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_secs(10))
            .retry(fast_retry())
            .build()
            .unwrap();
        let resp = client.get(&url).await.unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn status_501_returns_immediately() {
        let (url, count) = spawn_stub(vec![status_response(501, "Not Implemented")]);
        let client = HttpClient::api().retry(fast_retry()).build().unwrap();
        // Buffered calls document non-2xx as HttpError::Status (error bodies
        // must never reach a success-payload deserializer).
        let err = client.get(&url).await.expect_err("501 is an error");
        assert!(
            matches!(err, HttpError::Status { status: 501, .. }),
            "expected HttpError::Status 501, got {err:?}"
        );
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn status_505_returns_immediately() {
        let (url, count) = spawn_stub(vec![status_response(505, "HTTP Version Not Supported")]);
        let client = HttpClient::api().retry(fast_retry()).build().unwrap();
        let err = client.get(&url).await.expect_err("505 is an error");
        assert!(
            matches!(err, HttpError::Status { status: 505, .. }),
            "expected HttpError::Status 505, got {err:?}"
        );
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_after_zero_does_not_delay() {
        let script = vec![
            StubResponse {
                status: 429,
                reason: "Too Many Requests",
                extra_headers: "Retry-After: 0\r\n",
                body: b"{}".to_vec(),
            },
            body_ok("{}"),
        ];
        let (url, count) = spawn_stub(script);
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_secs(10))
            // Even a large configured delay must be irrelevant when the
            // server says Retry-After: 0 (max with backoff ≥ 0).
            .retry(RetryPolicy {
                max_attempts: 3,
                base_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(2),
                ..Default::default()
            })
            .build()
            .unwrap();
        let start = std::time::Instant::now();
        let resp = client.get(&url).await.unwrap();
        assert_eq!(resp.status, 200);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_after_past_http_date_does_not_delay() {
        let script = vec![
            StubResponse {
                status: 503,
                reason: "Service Unavailable",
                extra_headers: "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT\r\n",
                body: b"{}".to_vec(),
            },
            body_ok("{}"),
        ];
        let (url, _count) = spawn_stub(script);
        let client = HttpClient::api()
            .retry(RetryPolicy {
                max_attempts: 3,
                base_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(5),
                ..Default::default()
            })
            .build()
            .unwrap();
        let start = std::time::Instant::now();
        let resp = client.get(&url).await.unwrap();
        assert_eq!(resp.status, 200);
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn body_over_limit_errors() {
        let big = vec![b'x'; 200_000];
        let (url, _count) = spawn_stub(vec![StubResponse {
            status: 200,
            reason: "OK",
            extra_headers: "",
            body: big,
        }]);
        let client = HttpClient::api()
            .retry(RetryPolicy::none())
            .max_bytes(1024)
            .build()
            .unwrap();
        let err = client.get(&url).await.unwrap_err();
        assert!(
            matches!(err, HttpError::BodyTooLarge { limit: 1024 }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn total_timeout_fires_against_blackhole() {
        // Accept but never respond.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                // Hold the connection open without writing until the client
                // gives up; the socket drops when the thread ends the loop.
                let _owned = stream;
                std::thread::sleep(Duration::from_secs(10));
            }
        });
        let url = format!("http://127.0.0.1:{}", addr.port());
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_millis(200))
            .retry(fast_retry())
            .build()
            .unwrap();
        let err = client.get(&url).await.unwrap_err();
        assert!(matches!(err, HttpError::Timeout(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn pre_cancelled_token_aborts_before_request() {
        let (url, count) = spawn_stub(vec![body_ok("{}")]);
        let token = CancellationToken::new();
        token.cancel();
        let client = HttpClient::api()
            .retry(fast_retry())
            .cancellation_token(token)
            .build()
            .unwrap();
        let err = client.get(&url).await.unwrap_err();
        assert!(matches!(err, HttpError::Cancelled));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cancel_during_retry_wait_aborts() {
        let script = vec![
            StubResponse {
                status: 503,
                reason: "Service Unavailable",
                extra_headers: "Retry-After: 30\r\n",
                body: b"{}".to_vec(),
            },
            body_ok("{}"),
        ];
        let (url, count) = spawn_stub(script);
        let token = CancellationToken::new();
        let canceller = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            canceller.cancel();
        });
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_secs(30))
            .retry(RetryPolicy {
                max_attempts: 3,
                base_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(5),
                ..Default::default()
            })
            .cancellation_token(token)
            .build()
            .unwrap();
        let start = std::time::Instant::now();
        let err = client.get(&url).await.unwrap_err();
        assert!(matches!(err, HttpError::Cancelled), "got {err:?}");
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn sse_retries_until_stream_opens() {
        let script = vec![
            status_response(503, "Service Unavailable"),
            StubResponse {
                status: 200,
                reason: "OK",
                extra_headers: "Content-Type: text/event-stream\r\n",
                body: b"data: one\n\ndata: two\n\n".to_vec(),
            },
        ];
        let (url, count) = spawn_stub(script);
        let client = HttpClient::sse()
            .timeouts(Duration::from_secs(5), Duration::from_secs(10))
            .retry(fast_retry())
            .build()
            .unwrap();
        let mut stream = client
            .open_sse(&url, None, RequestOptions::new())
            .await
            .unwrap();
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(collected, b"data: one\n\ndata: two\n\n");
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn sse_mid_stream_close_does_not_reconnect() {
        let script = vec![StubResponse {
            status: 200,
            reason: "OK",
            extra_headers: "Content-Type: text/event-stream\r\n",
            body: b"data: only-one-event\n\n".to_vec(),
        }];
        let (url, count) = spawn_stub(script);
        let client = HttpClient::sse().retry(fast_retry()).build().unwrap();
        let mut stream = client
            .open_sse(&url, None, RequestOptions::new())
            .await
            .unwrap();
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(collected, b"data: only-one-event\n\n");
        // Give a potentially-wrong reconnect implementation a moment to act.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "mid-stream close must not trigger a reconnect"
        );
    }

    #[tokio::test]
    async fn sse_error_status_is_reported_with_body() {
        let (url, _count) = spawn_stub(vec![StubResponse {
            status: 400,
            reason: "Bad Request",
            extra_headers: "",
            body: b"{\"error\":\"bad payload\"}".to_vec(),
        }]);
        let client = HttpClient::sse()
            .retry(RetryPolicy::none())
            .build()
            .unwrap();
        // Use let-else rather than unwrap_err(): the success value is a boxed
        // stream with no Debug impl.
        let Err(HttpError::Status { status, body }) =
            client.open_sse(&url, None, RequestOptions::new()).await
        else {
            panic!("expected an HttpError::Status");
        };
        assert_eq!(status, 400);
        assert_eq!(body, "{\"error\":\"bad payload\"}");
    }

    #[tokio::test]
    async fn per_request_bearer_header_is_sent() {
        // The stub reads the head into its buffer but discards it; verify
        // header assembly with a dedicated echo stub instead.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                while let Ok(n) = stream.read(&mut tmp) {
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                // hyper 1.x emits header names in lowercase on the wire.
                let echo = String::from_utf8_lossy(&buf).to_string().to_lowercase();
                let answer = format!(
                    "{{\"seen\":\"{}\"}}",
                    echo.contains("authorization: bearer secret")
                );
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    answer.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(answer.as_bytes());
            }
        });
        let url = format!("http://127.0.0.1:{}", addr.port());
        let client = HttpClient::api()
            .retry(RetryPolicy::none())
            .build()
            .unwrap();
        let resp = client
            .post_json_with(
                &url,
                &serde_json::json!({"a": 1}),
                RequestOptions::new().bearer("secret"),
            )
            .await
            .unwrap();
        assert_eq!(resp.body, "{\"seen\":\"true\"}");
    }

    // ---- 0.25.0 MEDIUM regressions: body deadline + POST retry safety ----

    #[test]
    fn effective_retry_mode_is_method_aware() {
        let client_policy = TransportRetryMode::AllTransportErrors;
        // Safe methods follow the client policy.
        assert_eq!(
            effective_retry_mode(&Method::GET, &RequestOptions::new(), client_policy),
            TransportRetryMode::AllTransportErrors
        );
        assert_eq!(
            effective_retry_mode(&Method::HEAD, &RequestOptions::new(), client_policy),
            TransportRetryMode::AllTransportErrors
        );
        // Non-idempotent methods default to pre-dispatch-only, regardless of
        // the client policy, so a request already on the wire is not duplicated.
        for method in [Method::POST, Method::PUT, Method::PATCH, Method::DELETE] {
            assert_eq!(
                effective_retry_mode(&method, &RequestOptions::new(), client_policy),
                TransportRetryMode::PreDispatchOnly,
                "{method} must default to PreDispatchOnly"
            );
        }
        // An explicit per-request override wins for every method.
        assert_eq!(
            effective_retry_mode(
                &Method::POST,
                &RequestOptions::new().retry_mode(TransportRetryMode::AllTransportErrors),
                TransportRetryMode::PreDispatchOnly,
            ),
            TransportRetryMode::AllTransportErrors
        );
        assert_eq!(
            effective_retry_mode(
                &Method::GET,
                &RequestOptions::new().retry_mode(TransportRetryMode::PreDispatchOnly),
                TransportRetryMode::AllTransportErrors,
            ),
            TransportRetryMode::PreDispatchOnly
        );
    }

    /// Stub that fully drains one request per connection then closes without
    /// sending any response. reqwest surfaces this as a post-dispatch request
    /// error (connection closed before message completed), not a connect
    /// error: the TCP connection was accepted and the bytes were sent.
    fn spawn_close_after_dispatch_stub() -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let count_task = count.clone();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                count_task.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Drain the whole request (head + body) so the client has
                // definitely dispatched before the connection drops.
                // Bound the drain: the client is waiting for a response, so a
                // blocking read for an unsent body byte would deadlock until
                // the client deadline. Loopback coalesces the tiny request.
                let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
                let mut buf = [0u8; 4096];
                loop {
                    match stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
                // Drop without writing: client sees an incomplete response.
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
        });
        (format!("http://127.0.0.1:{}", addr.port()), count)
    }

    #[tokio::test]
    async fn post_transport_error_after_dispatch_is_not_retried_by_default() {
        // 0.25.0 MEDIUM: a POST whose bytes reached the server must not be
        // blindly retried (double-charge / duplicate side effect risk).
        let (url, count) = spawn_close_after_dispatch_stub();
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_secs(10))
            .retry(fast_retry())
            .build()
            .unwrap();
        let err = client
            .post_json(&url, &serde_json::json!({"id": 1}))
            .await
            .unwrap_err();
        assert!(matches!(err, HttpError::Transport(_)), "got {err:?}");
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "dispatched POST must not be retried under PreDispatchOnly"
        );
    }

    #[tokio::test]
    async fn post_retries_after_dispatch_only_with_explicit_opt_in() {
        // The escape hatch: a known-idempotent or server-deduplicated POST can
        // restore AllTransportErrors per request.
        let (url, count) = spawn_close_after_dispatch_stub();
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_secs(10))
            .retry(fast_retry())
            .build()
            .unwrap();
        let result = client
            .post_json_with(
                &url,
                &serde_json::json!({"id": 1}),
                RequestOptions::new().retry_mode(TransportRetryMode::AllTransportErrors),
            )
            .await;
        assert!(result.is_err());
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "explicit AllTransportErrors must retry the dispatched POST"
        );
    }

    #[tokio::test]
    async fn get_still_retries_after_dispatch_under_default_policy() {
        // Contrast case: safe methods keep the client-level retry policy.
        let (url, count) = spawn_close_after_dispatch_stub();
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_secs(10))
            .retry(fast_retry())
            .build()
            .unwrap();
        assert!(client.get(&url).await.is_err());
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn buffered_body_read_respects_total_timeout() {
        // 0.25.0 MEDIUM: a server that sends headers then trickles body bytes
        // forever must not hang the caller past the total deadline. The
        // deadline covers send() AND the buffered body read (not just send).
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let head = "HTTP/1.1 200 OK
Content-Length: 64
Connection: close

";
                let _ = stream.write_all(head.as_bytes());
                // Only 5 of the 64 promised bytes...
                let _ = stream.write_all(b"hello");
                let _ = stream.flush();
                // ...then withhold the remaining bytes until the client exits.
                std::thread::sleep(Duration::from_secs(5));
            }
        });
        let url = format!("http://127.0.0.1:{}", addr.port());
        let client = HttpClient::api()
            .timeouts(Duration::from_secs(5), Duration::from_millis(300))
            .retry(RetryPolicy::none())
            .build()
            .unwrap();
        let start = std::time::Instant::now();
        let err = client.get(&url).await.unwrap_err();
        assert!(matches!(err, HttpError::Timeout(_)), "got {err:?}");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "body read must be bounded by the total deadline"
        );
    }
}
