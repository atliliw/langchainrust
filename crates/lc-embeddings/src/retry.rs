// lc-embeddings/src/retry.rs
//! Exponential backoff retry for embedding HTTP calls (P2-5).
//!
//! Providers currently error on the first transient failure (429 rate limiting, 5xx server
//! errors), turning a single network blip into a hard failure. This module provides a unified
//! [`post_json_with_retry`]: 429 / 5xx are retried with exponential backoff, while other 4xx
//! (auth, invalid parameters — permanent failures) return immediately without masking config
//! errors. The backoff pattern matches `retry.rs` in lc-agents (`base_delay * 2^attempt`,
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
/// - 429 / 5xx: retry (exponential backoff, capped at `max_delay`);
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
            // Exponential backoff: base_delay * 2^attempt, capped at max_delay.
            let shift = 1u32.checked_shl(attempt as u32).unwrap_or(u32::MAX);
            let delay = retry.base_delay.saturating_mul(shift).min(retry.max_delay);
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
}
