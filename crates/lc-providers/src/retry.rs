// lc-providers/src/retry.rs
//! Exponential backoff retry for provider HTTP calls (0.22.0 audit H-P2).
//!
//! Providers previously errored on the first transient failure (429 rate
//! limiting, 5xx server errors, transport blips), turning a single network
//! hiccup into a hard failure. This module provides [`send_with_retry`]:
//! 429 / 5xx responses and connect-level transport errors are retried with
//! exponential backoff + bounded jitter, while other 4xx (auth, invalid
//! parameters — permanent failures) return immediately without masking config
//! errors. A server-provided `Retry-After` (seconds form) raises the delay
//! when larger than the computed backoff.
//!
//! Intentionally **not** wired into streaming request paths: a stream cannot
//! be safely resumed mid-flight, so only non-streaming (`chat_internal`-style)
//! requests are retried.
//!
//! # Non-idempotent request boundary (A14)
//!
//! Chat/completion calls are **non-idempotent POSTs**: replaying one after the
//! request has already reached the provider can execute (and bill) the
//! generation twice. Retries are therefore safe only when the failure
//! provably happened *before dispatch*:
//!
//! - DNS / TCP / TLS connect failures ([`reqwest::Error::is_connect`]) mean
//!   the TCP/TLS handshake never completed, so the request could not have
//!   reached the application → always safe to replay.
//! - A **timeout** or other `is_request` error is ambiguous: with only
//!   `connect_timeout` configured a timeout is usually a connect-phase
//!   failure, but a per-request/overall timeout can fire *after* the server
//!   accepted the request and started generating — and, in reqwest 0.12,
//!   `is_request()` is `true` even for that post-dispatch timeout, so it is
//!   not a trustworthy pre-dispatch signal.
//! - Response-body / decode errors ([`reqwest::Error::is_body`] /
//!   [`reqwest::Error::is_decode`]) are never retried: the server already
//!   produced the response.
//!
//! The historical default ([`TransportRetryMode::AllTransportErrors`], used by
//! [`DEFAULT_RETRY`]) also retries timeouts — this is a deliberate
//! availability-vs-double-billing tradeoff and is **unchanged**. Deployments
//! that prefer strict at-most-once semantics can opt into
//! [`TransportRetryMode::PreDispatchOnly`] via [`SAFE_RETRY`]. The same
//! ambiguity applies to 429/5xx responses: a 502 can be returned *after* the
//! upstream processed the request, so HTTP-status retries are likewise
//! best-effort, not guaranteed safe.
//!
//! The backoff pattern mirrors `retry.rs` in lc-embeddings / lc-agents
//! (`base_delay * 2^attempt`, capped at `max_delay`).

use std::time::Duration;

/// Selects which transport-level errors are retried for a (non-idempotent)
/// request. See the module docs ("Non-idempotent request boundary (A14)") for
/// the double-billing tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransportRetryMode {
    /// Retry every transport error the implementation has historically
    /// retried: connect failure, request-builder error, **and timeout**.
    ///
    /// Availability-first (the historical default). A timeout can fire after
    /// the request was dispatched, so a retry may execute the generation twice.
    AllTransportErrors,
    /// Retry only errors that provably occur during connection establishment
    /// ([`reqwest::Error::is_connect`]: DNS/TCP/TLS failure — including a
    /// connect timeout). Request-phase errors and timeouts are not retried:
    /// `is_request`/`is_timeout` can fire after the server accepted the
    /// request (in reqwest 0.12 a post-dispatch timeout has both flags set).
    ///
    /// Opt-in via [`SAFE_RETRY`]; exercised in tests even though no default
    /// provider call site selects it (A14 keeps historical behavior).
    #[allow(dead_code)]
    PreDispatchOnly,
}

/// Exponential backoff retry configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryConfig {
    /// Total attempts including the first (3 = 1 initial + 2 retries).
    pub max_attempts: usize,
    /// Initial delay before the first retry.
    pub base_delay: Duration,
    /// Upper bound for backoff delay.
    pub max_delay: Duration,
    /// Which transport errors are eligible for retry (A14 boundary).
    pub transport: TransportRetryMode,
}

/// Default retry policy: at most 3 attempts, base 500ms, cap 8s.
///
/// Transport mode is [`TransportRetryMode::AllTransportErrors`] — the
/// historical behavior, kept unchanged by A14 (timeouts are retried).
pub(crate) const DEFAULT_RETRY: RetryConfig = RetryConfig {
    max_attempts: 3,
    base_delay: Duration::from_millis(500),
    max_delay: Duration::from_secs(8),
    transport: TransportRetryMode::AllTransportErrors,
};

/// Strict retry policy for non-idempotent POSTs: same bounds as
/// [`DEFAULT_RETRY`], but transport retries are limited to provably
/// pre-dispatch errors ([`TransportRetryMode::PreDispatchOnly`]). Opt in at a
/// call site (`send_with_retry(builder, &SAFE_RETRY)`) where double-billing a
/// generation outweighs a lost retry. Not referenced by default provider
/// paths by design (A14 preserves historical behavior), hence the allow.
#[allow(dead_code)]
pub(crate) const SAFE_RETRY: RetryConfig = RetryConfig {
    max_attempts: 3,
    base_delay: Duration::from_millis(500),
    max_delay: Duration::from_secs(8),
    transport: TransportRetryMode::PreDispatchOnly,
};

/// The shared default HTTP client for all providers (0.22.0 audit H-P1).
///
/// A connect timeout guards against hanging TCP handshakes. There is
/// deliberately **no total `.timeout()`** — it would kill long-running SSE
/// streams mid-generation.
pub(crate) fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

/// Sends a (non-streaming) request with exponential backoff, returning the
/// first non-transient response or the last response once retries are exhausted.
///
/// - 429 / 5xx: retry (bounded jitter, capped at `max_delay`, `Retry-After`
///   honored when larger);
/// - other 4xx: return immediately (permanent failure, retrying is pointless);
/// - transport errors: retried according to `retry.transport` — see
///   [`TransportRetryMode`] and the module-level A14 boundary discussion; the
///   final error is returned as-is.
///
/// The caller keeps handling status codes and bodies after receiving the
/// response, so error semantics are unchanged from the un-retried path.
pub(crate) async fn send_with_retry(
    build_request: impl Fn() -> reqwest::RequestBuilder,
    retry: &RetryConfig,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut attempt = 0usize;
    loop {
        match build_request().send().await {
            Ok(response) => {
                let status = response.status();
                if attempt + 1 < retry.max_attempts && is_transient(&status) {
                    // `Retry-After` (seconds form) is honored when larger than
                    // the computed backoff. Read before dropping the response.
                    let retry_after_secs = response
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.trim().parse::<u64>().ok());
                    let delay = next_backoff(attempt, retry_after_secs, retry, entropy());
                    log::warn!(
                        "provider HTTP {} (attempt {}/{}), retrying in {:?}",
                        status,
                        attempt + 1,
                        retry.max_attempts,
                        delay
                    );
                    drop(response);
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Ok(response);
            }
            Err(e) => {
                if attempt + 1 < retry.max_attempts && is_retryable_error(&e, retry.transport) {
                    let delay = next_backoff(attempt, None, retry, entropy());
                    log::warn!(
                        "provider transport error: {e} (attempt {}/{}), retrying in {:?}",
                        attempt + 1,
                        retry.max_attempts,
                        delay
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(e);
            }
        }
    }
}

/// Whether the status is a transient failure (retryable): 429 rate limit or 5xx server error.
fn is_transient(status: &reqwest::StatusCode) -> bool {
    status.as_u16() == 429 || status.as_u16() >= 500
}

/// Whether a transport error is worth retrying under `mode`.
///
/// Only connect-phase failures (DNS/TCP/TLS handshake, `is_connect`) are
/// provably pre-dispatch: the request could not have reached the server.
/// `is_request` is deliberately excluded from the strict mode — in reqwest
/// 0.12 a timeout while awaiting response headers reports `is_request() ==
/// true`, so it cannot be treated as "never dispatched". Timeouts and other
/// request-phase errors are retried only under
/// [`TransportRetryMode::AllTransportErrors`] (the historical default).
/// Response-body/decoding errors are never retried under either mode.
fn is_retryable_error(e: &reqwest::Error, mode: TransportRetryMode) -> bool {
    match mode {
        // Only the connect phase (DNS/TCP/TLS handshake) is provably
        // pre-dispatch. `is_request` is NOT: in reqwest 0.12 a timeout while
        // waiting for response headers reports `is_request() == true` even
        // though the request may already have been processed.
        TransportRetryMode::PreDispatchOnly => e.is_connect(),
        TransportRetryMode::AllTransportErrors => {
            e.is_connect() || e.is_timeout() || e.is_request()
        }
    }
}

/// Computes the delay before the next retry attempt (pure, unit-testable).
///
/// - base: `base_delay * 2^attempt`, capped at `max_delay`;
/// - bounded jitter: + 0-25% of the base (derived from `entropy`), before the cap;
/// - `retry_after_secs` (server's `Retry-After`, seconds form) raises the delay
///   when larger than the jittered backoff; everything is capped at `max_delay`.
fn next_backoff(
    attempt: usize,
    retry_after_secs: Option<u64>,
    retry: &RetryConfig,
    entropy: u64,
) -> Duration {
    let shift = 1u32.checked_shl(attempt as u32).unwrap_or(u32::MAX);
    let base = retry.base_delay.saturating_mul(shift).min(retry.max_delay);
    let jitter = base.mul_f64((entropy % 25) as f64 / 100.0);
    let backoff = (base + jitter).min(retry.max_delay);

    let server_delay = retry_after_secs
        .map(|s| Duration::from_secs(s).min(retry.max_delay))
        .unwrap_or(Duration::ZERO);
    backoff.max(server_delay)
}

/// Cheap entropy source for jitter: sub-second nanos of the wall clock.
fn entropy() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    #[test]
    fn transient_status_classification() {
        assert!(is_transient(&reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(is_transient(&reqwest::StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_transient(&reqwest::StatusCode::SERVICE_UNAVAILABLE));
        assert!(is_transient(&reqwest::StatusCode::BAD_GATEWAY));
        assert!(!is_transient(&reqwest::StatusCode::BAD_REQUEST));
        assert!(!is_transient(&reqwest::StatusCode::UNAUTHORIZED));
        assert!(!is_transient(&reqwest::StatusCode::OK));
    }

    #[test]
    fn next_backoff_zero_entropy_is_plain_base() {
        assert_eq!(
            next_backoff(0, None, &DEFAULT_RETRY, 0),
            Duration::from_millis(500)
        );
        assert_eq!(
            next_backoff(2, None, &DEFAULT_RETRY, 0),
            Duration::from_millis(2000)
        );
    }

    #[test]
    fn next_backoff_jitter_is_bounded() {
        // Any entropy keeps the delay within [base, base * 1.25).
        for entropy in [0u64, 1, 7, 13, 24, 999, u64::MAX] {
            let delay = next_backoff(0, None, &DEFAULT_RETRY, entropy);
            assert!(delay >= Duration::from_millis(500));
            assert!(delay < Duration::from_millis(625));
        }
    }

    #[test]
    fn next_backoff_caps_at_max_delay() {
        // attempt 10 → 500ms * 2^10 = ~512s, capped at 8s (jitter 0).
        assert_eq!(
            next_backoff(10, None, &DEFAULT_RETRY, 0),
            Duration::from_secs(8)
        );
    }

    #[test]
    fn next_backoff_honors_larger_retry_after() {
        // Server says wait 3600s; cap applies → 8s (still above the backoff).
        assert_eq!(
            next_backoff(0, Some(3600), &DEFAULT_RETRY, 0),
            Duration::from_secs(8)
        );
    }

    #[test]
    fn next_backoff_ignores_smaller_retry_after() {
        assert_eq!(
            next_backoff(0, Some(0), &DEFAULT_RETRY, 0),
            Duration::from_millis(500)
        );
    }

    /// HTTP stub that serves the `plan` statuses in order (the last entry
    /// repeats for extra requests) and counts served requests.
    /// Returns `(base_url, request_count)`.
    async fn spawn_status_stub(
        plan: Vec<(u16, Option<u64>)>,
    ) -> (String, std::sync::Arc<AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = requests.clone();
        tokio::spawn(async move {
            let mut served: usize = 0;
            while let Ok((mut socket, _)) = listener.accept().await {
                // Drain the request head + body so reqwest completes its POST.
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                loop {
                    if socket.read_exact(&mut byte).await.is_err() {
                        return;
                    }
                    head.push(byte[0]);
                    if head.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let head_str = String::from_utf8_lossy(&head).to_lowercase();
                let content_length: usize = head_str
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                if content_length > 0 {
                    let mut body = vec![0u8; content_length];
                    if socket.read_exact(&mut body).await.is_err() {
                        return;
                    }
                }
                let (status, retry_after) = plan
                    .get(served)
                    .or_else(|| plan.last())
                    .copied()
                    .unwrap_or((200, None));
                served += 1;
                counter.fetch_add(1, Ordering::SeqCst);
                let retry_after_line = retry_after
                    .map(|s| format!("Retry-After: {s}\r\n"))
                    .unwrap_or_default();
                let response = format!(
                    "HTTP/1.1 {status} Stub\r\n{retry_after_line}Connection: close\r\nContent-Length: 2\r\n\r\nok"
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}"), requests)
    }

    fn fast_retry() -> RetryConfig {
        RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            transport: TransportRetryMode::AllTransportErrors,
        }
    }

    /// [`SAFE_RETRY`] with near-zero delays for tests.
    fn fast_safe_retry() -> RetryConfig {
        RetryConfig {
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            ..SAFE_RETRY
        }
    }

    #[tokio::test]
    async fn transient_429s_are_retried_until_success() {
        let (base_url, requests) =
            spawn_status_stub(vec![(429, None), (429, None), (200, None)]).await;
        let client = reqwest::Client::new();

        let resp = send_with_retry(
            || client.post(&base_url).json(&serde_json::json!({"m": 1})),
            &fast_retry(),
        )
        .await
        .expect("should succeed after transient 429s");
        assert_eq!(resp.status().as_u16(), 200);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");
    }

    #[tokio::test]
    async fn retries_are_capped_and_last_transient_response_is_returned() {
        let (base_url, requests) = spawn_status_stub(vec![(503, None)]).await;
        let client = reqwest::Client::new();

        let resp = send_with_retry(|| client.post(&base_url), &fast_retry())
            .await
            .expect("after retries are exhausted, the last response is returned");
        assert_eq!(resp.status().as_u16(), 503);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "3 total attempts");
    }

    #[tokio::test]
    async fn permanent_4xx_is_not_retried() {
        let (base_url, requests) = spawn_status_stub(vec![(400, None)]).await;
        let client = reqwest::Client::new();

        let resp = send_with_retry(|| client.post(&base_url), &fast_retry())
            .await
            .expect("4xx responses are returned as-is");
        assert_eq!(resp.status().as_u16(), 400);
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "permanent failure must not be retried"
        );
    }

    #[tokio::test]
    async fn retry_after_header_raises_retry_delay() {
        // 429 + `Retry-After: 1` with a 5ms cap: each retry waits the full
        // cap → ≥ 10ms total (plain ~1ms backoff would finish far earlier).
        let (base_url, requests) =
            spawn_status_stub(vec![(429, Some(1)), (429, Some(1)), (200, None)]).await;
        let client = reqwest::Client::new();

        let start = Instant::now();
        let resp = send_with_retry(
            || client.post(&base_url).json(&serde_json::json!({"m": 1})),
            &fast_retry(),
        )
        .await
        .expect("should succeed after transient 429s");
        let elapsed = start.elapsed();

        assert_eq!(resp.status().as_u16(), 200);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");
        assert!(
            elapsed >= Duration::from_millis(10),
            "Retry-After should raise each retry delay to the 5ms cap (elapsed {elapsed:?})"
        );
    }

    /// A14: stub that completes the TCP handshake and drains the request, but
    /// never sends a response. A per-request timeout then fires *after* the
    /// request was dispatched — the ambiguous case the retry boundary is about.
    /// Returns `(base_url, accepted_connections)`.
    async fn spawn_blackhole_stub() -> (String, std::sync::Arc<AtomicUsize>) {
        use tokio::io::AsyncReadExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let accepted = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = accepted.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                // Hold the connection open without ever responding; cancelled
                // when the test runtime shuts down.
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    while head.len() < 64 * 1024 {
                        if socket.read_exact(&mut byte).await.is_err() {
                            return;
                        }
                        head.push(byte[0]);
                        if head.ends_with(b"\r\n\r\n") {
                            break;
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    drop(socket);
                });
            }
        });
        (format!("http://{addr}"), accepted)
    }

    #[tokio::test]
    async fn post_dispatch_timeout_is_retried_by_default_mode() {
        // Historical behavior (unchanged by A14): timeouts are retried even
        // though the request may already have reached the server.
        let (base_url, accepted) = spawn_blackhole_stub().await;
        let client = reqwest::Client::new();

        let result = send_with_retry(
            || {
                client
                    .post(&base_url)
                    .timeout(Duration::from_millis(100))
                    .json(&serde_json::json!({"m": 1}))
            },
            &fast_retry(),
        )
        .await;

        assert!(result.is_err(), "all attempts time out");
        assert_eq!(
            accepted.load(Ordering::SeqCst),
            3,
            "default mode replays the timed-out (possibly dispatched) request"
        );
    }

    #[tokio::test]
    async fn post_dispatch_timeout_is_not_retried_in_predispatch_only_mode() {
        // Strict A14 mode: once the request is in flight, a timeout is terminal.
        let (base_url, accepted) = spawn_blackhole_stub().await;
        let client = reqwest::Client::new();

        let result = send_with_retry(
            || {
                client
                    .post(&base_url)
                    .timeout(Duration::from_millis(100))
                    .json(&serde_json::json!({"m": 1}))
            },
            &fast_safe_retry(),
        )
        .await;

        assert!(result.is_err(), "the single attempt times out");
        assert_eq!(
            accepted.load(Ordering::SeqCst),
            1,
            "PreDispatchOnly must not replay a request that may have been processed"
        );
    }

    #[tokio::test]
    async fn connect_failure_is_still_retried_in_predispatch_only_mode() {
        // A closed local port refuses the TCP handshake: the request never
        // reached a server, so even the strict mode retries all attempts.
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let url = format!("http://{addr}");
        // `no_proxy` so an ambient corporate HTTP proxy cannot answer for a
        // refused localhost connection (which would look like a 5xx response).
        let client = reqwest::Client::builder().no_proxy().build().unwrap();

        let result = send_with_retry(
            || client.post(&url).json(&serde_json::json!({"m": 1})),
            &fast_safe_retry(),
        )
        .await;

        assert!(result.is_err(), "connect refused every time");
        // No server-side counter exists; the retry loop completing within the
        // test budget asserts attempts == max_attempts implicitly. Assert the
        // classifier contract directly instead:
        let err = result.unwrap_err();
        assert!(
            is_retryable_error(&err, TransportRetryMode::PreDispatchOnly),
            "connect refused must be retryable in PreDispatchOnly (connect={}, request={}, timeout={})",
            err.is_connect(),
            err.is_request(),
            err.is_timeout()
        );
        assert!(
            is_retryable_error(&err, TransportRetryMode::AllTransportErrors),
            "connect refused must be retryable under the default mode too"
        );
    }
}
