// lc-embeddings/src/retry.rs
//! 嵌入 HTTP 调用的指数退避重试（P2-5）。
//!
//! provider 对瞬时故障（429 限流、5xx 服务端错误）目前一次失败即抛错，
//! 把一次网络抖动变成硬失败。这里提供统一的 [`post_json_with_retry`]：
//! 对 429 / 5xx 做指数退避重试，其余 4xx（鉴权、参数错误等永久性失败）
//! 立即返回，不掩盖配置错误。退避模式与 lc-agents 的 `retry.rs` 一致
//! （`base_delay * 2^attempt`，封顶 `max_delay`）。

use std::time::Duration;

/// 指数退避重试配置。
#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryConfig {
    /// 首次失败后的最大重试次数。
    pub max_retries: usize,
    /// 首次重试前的初始延迟。
    pub base_delay: Duration,
    /// 退避延迟上限。
    pub max_delay: Duration,
}

/// 默认重试配置：最多 3 次重试，base 1s，上限 30s。
pub(crate) const DEFAULT_RETRY: RetryConfig = RetryConfig {
    max_retries: 3,
    base_delay: Duration::from_secs(1),
    max_delay: Duration::from_secs(30),
};

/// 对 POST JSON 请求做指数退避重试，返回首个非瞬时失败的响应。
///
/// - 429 / 5xx：重试（指数退避，封顶 `max_delay`）；
/// - 其余 4xx：立即返回（永久性失败，重试无意义）；
/// - 传输层错误：直接返回（不在 HTTP 状态语义内，由调用方判定）。
///
/// 调用方拿到响应后自行处理状态码与 body（P1-4 错误体不吞错）。
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
            // 指数退避：base_delay * 2^attempt，封顶 max_delay。
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

/// 是否瞬时失败（可重试）：429 限流或 5xx 服务端错误。
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
        assert!(resp.is_ok(), "429 后应重试成功: {:?}", resp.err());
        assert_eq!(resp.unwrap().status().as_u16(), 200);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 次初始 + 2 次重试");
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
        assert!(resp.is_ok(), "重试耗尽后应返回最后响应而非传输错误");
        assert_eq!(resp.unwrap().status().as_u16(), 429);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 次初始 + 2 次重试");
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
            "4xx 是永久性失败,不应重试"
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
