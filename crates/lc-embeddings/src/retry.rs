// lc-embeddings/src/retry.rs
//! Exponential backoff retry for embedding HTTP calls (P2-5).
//!
//! This module provides a unified [`post_json_with_retry`] with exponential backoff + bounded
//! jitter and an at-most-once replay policy for the **non-idempotent** embedding POSTs
//! (replaying one after the provider has already embedded can double-bill the text).
//!
//! 0.21.0 S3.3 backoff alignment: each backoff adds 0-25% pseudo-random slack (wall-clock
//! entropy) so concurrent clients do not re-collide; a server `Retry-After` (seconds form)
//! raises the delay when larger than the computed backoff, still capped at `max_delay`;
//! delay computation is a pure function ([`next_backoff`]) so the policy is unit-testable.
//! The backoff shape matches `retry.rs` in lc-agents / lc-providers
//! (`base_delay * 2^attempt`, capped at `max_delay`).
//!
//! # At-most-once replay boundary (§3.3 / lc-providers SAFE_RETRY, A14)
//!
//! A POST that has already reached the provider may have been embedded and billed, so it must
//! not be blindly replayed. Only three outcomes are provably pre-dispatch and safe to replay:
//! - a **429** (rate limit) — the provider rejected the request *before* processing it;
//! - a **connect-phase transport failure** ([`reqwest::Error::is_connect`]) — the DNS/TCP/TLS
//!   handshake never completed, so the request could not have reached the application.
//!
//! Everything else is returned as-is and not replayed:
//! - **5xx** is ambiguous — it can be returned *after* the provider embedded and billed the
//!   request, so replaying risks double-billing (in reqwest 0.12,
//!   [`reqwest::Error::is_timeout`]/[`reqwest::Error::is_request`] can likewise fire after
//!   dispatch, so timeouts are not replayed either);
//! - other 4xx (auth, invalid parameters) are permanent failures.
//!
//! This mirrors lc-providers' `SAFE_RETRY` / `TransportRetryMode::PreDispatchOnly` decision.

use std::time::Duration;

/// Exponential backoff retry configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryConfig {
    /// Maximum retries after the first failure.
    pub max_retries: usize,
    /// Initial delay before the first retry.
    pub base_delay: Duration,
    /// Upper bound for backoff delay.
    pub max_delay: Duration,
}

/// Default retry config: at most 3 retries, base 1s, cap 30s.
pub(crate) const DEFAULT_RETRY: RetryConfig = RetryConfig {
    max_retries: 3,
    base_delay: Duration::from_secs(1),
    max_delay: Duration::from_secs(30),
};

/// Retries a POST JSON request with exponential backoff under the at-most-once policy.
///
/// - **429 only** among HTTP statuses: retry (exponential backoff with bounded jitter, capped
///   at `max_delay`, `Retry-After` honored when larger). A 429 is provably pre-dispatch
///   (rate-limited before processing), so replaying a non-idempotent embedding POST is safe.
/// - **5xx**: return immediately — a 5xx can arrive *after* the provider embedded and billed
///   the request, so it is not blindly replayed (avoids double-billing; see module docs §3.3).
/// - other 4xx: return immediately (permanent failure, retrying is pointless);
/// - transport errors: retry only a **connect-phase** failure ([`reqwest::Error::is_connect`],
///   provably pre-dispatch); timeouts/request-phase errors are returned as-is (ambiguous —
///   may have already been processed).
///
/// The caller handles the status code and body after receiving the response (P1-4: error body not swallowed).
pub(crate) async fn post_json_with_retry(
    client: &reqwest::Client,
    url: &str,
    bearer_token: &str,
    body: &serde_json::Value,
    retry: &RetryConfig,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut attempt = 0usize;
    loop {
        let response = match client
            .post(url)
            .header("Authorization", format!("Bearer {bearer_token}"))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(err) => {
                // §3.3: only a connect-phase failure is provably pre-dispatch (the handshake
                // never completed, so nothing was embedded/billed) and safe to replay. A
                // timeout / request-phase error in reqwest 0.12 can fire *after* the provider
                // accepted and billed the request, so it is returned immediately.
                if attempt < retry.max_retries && err.is_connect() {
                    let delay = next_backoff(attempt, None, retry, entropy());
                    log::warn!(
                        "embedding connect failure (attempt {}), retrying in {:?}",
                        attempt + 1,
                        delay
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(err);
            }
        };

        let status = response.status();
        // §3.3: among HTTP statuses only 429 (rate limit, provably pre-dispatch) is replayed
        // for a non-idempotent embedding POST. `is_transient` also classifies 5xx, but the
        // replay *decision* deliberately narrows to 429 — a 5xx may be a post-dispatch / already
        // billed failure and must not be blindly replayed.
        if is_transient(&status) && status.as_u16() == 429 && attempt < retry.max_retries {
            // `Retry-After` (seconds form) is honored when larger than the computed backoff.
            let retry_after_secs = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            let delay = next_backoff(attempt, retry_after_secs, retry, entropy());
            log::warn!(
                "embedding HTTP {} (attempt {}), retrying in {:?}",
                status,
                attempt + 1,
                delay
            );
            tokio::time::sleep(delay).await;
            attempt += 1;
            continue;
        }
        return Ok(response);
    }
}

/// Computes the delay before the next retry attempt (pure, unit-testable).
///
/// - base: `base_delay * 2^attempt`, capped at `max_delay`;
/// - bounded jitter: + 0-25% of the base (derived from `entropy`), before the cap —
///   smooths thundering-herd retries without unbounded waits;
/// - `retry_after_secs` (server's `Retry-After`, seconds form) raises the delay when
///   larger than the jittered backoff; everything is capped at `max_delay`.
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

/// Whether the status belongs to the *transient* failure class: 429 rate limit or 5xx server error.
///
/// This is a **classification** predicate, not the replay decision. For a non-idempotent
/// embedding POST the actual replay decision in [`post_json_with_retry`] deliberately narrows
/// this to the provably pre-dispatch 429 (a 5xx may already have billed the request; see
/// module docs §3.3). Kept as a class predicate so callers/tests can reason about transient
/// statuses without implying they are all safe to replay.
fn is_transient(status: &reqwest::StatusCode) -> bool {
    status.as_u16() == 429 || status.as_u16() >= 500
}

/// Request-level deadline for embedding HTTP calls (A4/A5/A6, and `openai.rs`).
///
/// Without this, a hanging / non-responsive provider keeps the awaiting task (and the whole
/// pipeline behind it) pending forever. `180s` aligns with the unified `HttpClient::api()`
/// deadline used by lc-providers, so one standard applies across embedding calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

/// Builds a shared embedding [`reqwest::Client`] carrying the request-level deadline.
///
/// Uses [`reqwest::Client::builder`] so every request through `post_json_with_retry` and the
/// direct-send providers inherits the timeout. Falls back to the bare default only when the
/// static builder config cannot build (in practice unreachable).
pub(crate) fn embedding_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .unwrap_or_else(|e| {
            log::warn!(
                "failed to build embedding HTTP client with timeout, falling back to default: {e}"
            );
            reqwest::Client::new()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::spawn_status_stub;
    use std::sync::atomic::Ordering;

    fn cfg() -> RetryConfig {
        RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        }
    }

    #[tokio::test]
    async fn retry_succeeds_after_transient_429s() {
        let (base_url, requests) = spawn_status_stub(429, 2, 200, "{\"ok\":true}").await;
        let client = reqwest::Client::new();
        let body = serde_json::json!({"model": "m", "input": ["a", "b"]});

        let resp = post_json_with_retry(&client, &base_url, "test-key", &body, &cfg()).await;
        assert!(
            resp.is_ok(),
            "should retry successfully after 429: {:?}",
            resp.err()
        );
        assert_eq!(resp.unwrap().status().as_u16(), 200);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");
    }

    #[tokio::test]
    async fn does_not_replay_transient_5xx() {
        // §3.3: a 5xx may be a post-dispatch / already-billed failure, so a
        // non-idempotent embedding POST is NOT blindly replayed here — the
        // single 503 is returned as-is even though retry attempts remain.
        let (base_url, requests) = spawn_status_stub(503, 1, 200, "{\"ok\":true}").await;
        let client = reqwest::Client::new();
        let body = serde_json::json!({"model": "m", "input": ["a"]});

        let resp = post_json_with_retry(&client, &base_url, "test-key", &body, &cfg()).await;
        assert!(resp.is_ok());
        assert_eq!(resp.unwrap().status().as_u16(), 503);
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "5xx must not be replayed"
        );
    }

    #[tokio::test]
    async fn retry_exhausts_and_returns_last_transient_response() {
        let (base_url, requests) = spawn_status_stub(429, 100, 200, "{\"ok\":true}").await;
        let client = reqwest::Client::new();
        let body = serde_json::json!({"model": "m", "input": ["a"]});
        let retry = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        };

        let resp = post_json_with_retry(&client, &base_url, "test-key", &body, &retry).await;
        assert!(resp.is_ok(), "after retries are exhausted, should return the last response rather than a transport error");
        assert_eq!(resp.unwrap().status().as_u16(), 429);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");
    }

    #[tokio::test]
    async fn does_not_retry_permanent_4xx() {
        let (base_url, requests) = spawn_status_stub(400, 100, 200, "{\"ok\":true}").await;
        let client = reqwest::Client::new();
        let body = serde_json::json!({"model": "m", "input": ["a"]});

        let resp = post_json_with_retry(&client, &base_url, "test-key", &body, &cfg()).await;
        assert!(resp.is_ok());
        assert_eq!(resp.unwrap().status().as_u16(), 400);
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "4xx is a permanent failure, should not be retried"
        );
    }

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

    /// 0.21.0 S3.3: entropy 0 → zero jitter, exactly the exponential base.
    #[test]
    fn next_backoff_zero_entropy_is_plain_base() {
        let cfg = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
        };
        assert_eq!(next_backoff(0, None, &cfg, 0), Duration::from_millis(1000));
        assert_eq!(next_backoff(2, None, &cfg, 0), Duration::from_millis(4000));
    }

    /// 0.21.0 S3.3: jitter is bounded — 0-25% of the base, monotone in entropy.
    #[test]
    fn next_backoff_jitter_is_bounded() {
        let cfg = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
        };
        // entropy 24 → +24%; entropy 25 wraps to 0 → plain base.
        assert_eq!(next_backoff(0, None, &cfg, 24), Duration::from_millis(1240));
        assert_eq!(next_backoff(0, None, &cfg, 25), Duration::from_millis(1000));

        // Any entropy keeps the delay within [base, base * 1.25).
        for entropy in [0u64, 1, 7, 13, 24, 999, u64::MAX] {
            let delay = next_backoff(0, None, &cfg, entropy);
            assert!(delay >= Duration::from_millis(1000));
            assert!(delay < Duration::from_millis(1250));
        }
    }

    /// 0.21.0 S3.3: the exponential base is still capped at `max_delay` before jitter.
    #[test]
    fn next_backoff_caps_at_max_delay() {
        let cfg = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
        };
        // attempt 10 → 1s * 2^10 = 1024s, capped at 5s (jitter 0).
        assert_eq!(next_backoff(10, None, &cfg, 0), Duration::from_secs(5));
    }

    /// 0.21.0 S3.3: a server `Retry-After` larger than the computed backoff raises
    /// the delay (still capped at `max_delay`).
    #[test]
    fn next_backoff_honors_larger_retry_after() {
        let cfg = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        };
        // Server says wait 3600s; cap applies → 5ms (still above the 1ms backoff).
        assert_eq!(
            next_backoff(0, Some(3600), &cfg, 0),
            Duration::from_millis(5)
        );
    }

    /// 0.21.0 S3.3: a `Retry-After` smaller than the computed backoff is ignored —
    /// the exponential backoff wins.
    #[test]
    fn next_backoff_ignores_smaller_retry_after() {
        let cfg = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
        };
        assert_eq!(
            next_backoff(0, Some(0), &cfg, 0),
            Duration::from_millis(1000)
        );
    }

    /// 0.21.0 S3.3 integration: with a `Retry-After: 1` response header and a 5ms cap,
    /// each retry waits the full cap — measurable vs the ~1ms plain backoff.
    #[tokio::test]
    async fn retry_after_header_raises_retry_delay() {
        use crate::test_support::spawn_retry_after_stub;
        use std::time::Instant;

        let (base_url, requests) = spawn_retry_after_stub(1, 2).await;
        let client = reqwest::Client::new();
        let body = serde_json::json!({"model": "m", "input": ["a"]});
        // base 1ms, cap 5ms: with Retry-After honored each retry waits 5ms → ≥ 12ms total;
        // without the header the 3 attempts would take ~4ms.
        let retry = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        };

        let start = Instant::now();
        let resp = post_json_with_retry(&client, &base_url, "test-key", &body, &retry).await;
        let elapsed = start.elapsed();

        assert!(resp.is_ok());
        assert_eq!(resp.unwrap().status().as_u16(), 200);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");
        assert!(
            elapsed >= Duration::from_millis(12),
            "Retry-After should raise each retry delay to the 5ms cap (elapsed {:?})",
            elapsed
        );
    }
}
