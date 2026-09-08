// lc-embeddings/src/retry.rs
//! Exponential backoff retry for embedding HTTP calls (P2-5).
//!
//! Providers currently error on the first transient failure (429 rate limiting, 5xx server
//! errors), turning a single network blip into a hard failure. This module provides a unified
//! [`post_json_with_retry`]: 429 / 5xx are retried with exponential backoff, while other 4xx
//! (auth, invalid parameters — permanent failures) return immediately without masking config
//! errors.
//!
//! 0.21.0 S3.3 alignment with current retry practice:
//! - **bounded jitter**: each backoff adds 0-25% pseudo-random slack (entropy from the wall
//!   clock) so concurrent clients do not re-collide on the same schedule;
//! - **`Retry-After` honored**: a server-provided `Retry-After` (seconds form) raises the
//!   delay when larger than the computed backoff, still capped at `max_delay`;
//! - delay computation is a pure function ([`next_backoff`]) so the policy is unit-testable.
//!
//! The backoff pattern matches `retry.rs` in lc-agents (`base_delay * 2^attempt`,
//! capped at `max_delay`).

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

/// Retries a POST JSON request with exponential backoff, returning the first non-transient response.
///
/// - 429 / 5xx: retry (exponential backoff with bounded jitter, capped at `max_delay`,
///   `Retry-After` honored when larger);
/// - other 4xx: return immediately (permanent failure, retrying is pointless);
/// - transport errors: return as-is (outside HTTP status semantics, left to the caller).
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
        let response = client
            .post(url)
            .header("Authorization", format!("Bearer {bearer_token}"))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await?;

        let status = response.status();
        if is_transient(&status) && attempt < retry.max_retries {
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

/// Whether the status is a transient failure (retryable): 429 rate limit or 5xx server error.
fn is_transient(status: &reqwest::StatusCode) -> bool {
    status.as_u16() == 429 || status.as_u16() >= 500
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
    async fn retry_succeeds_after_transient_5xx() {
        let (base_url, _requests) = spawn_status_stub(503, 1, 200, "{\"ok\":true}").await;
        let client = reqwest::Client::new();
        let body = serde_json::json!({"model": "m", "input": ["a"]});

        let resp = post_json_with_retry(&client, &base_url, "test-key", &body, &cfg()).await;
        assert!(resp.is_ok());
        assert_eq!(resp.unwrap().status().as_u16(), 200);
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
