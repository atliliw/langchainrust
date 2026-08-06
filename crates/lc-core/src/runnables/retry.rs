// lc-core/src/runnables/retry.rs
//! RunnableRetry - retry a Runnable with exponential backoff.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_core::runnables::{RetryConfig, RetryOn, RunnableExt};
//! use std::time::Duration;
//!
//! let chain = prompt.pipe(llm).pipe(parser)
//!     .with_retry(RetryConfig {
//!         max_retries: 3,
//!         initial_delay: Duration::from_millis(500),
//!         max_delay: Duration::from_secs(10),
//!         backoff_multiplier: 2.0,
//!         retry_on: RetryOn::TransientErrors,
//!     });
//! ```

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};

use super::any::RunnableAny;
use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;

/// Determines which errors should trigger a retry.
#[derive(Clone)]
pub enum RetryOn {
    /// Retry on any error.
    AllErrors,
    /// Retry only on transient errors (rate limits, timeouts, server errors).
    /// Specifically: HTTP 429, 500, 502, 503, 504 and timeout errors.
    TransientErrors,
    /// Retry only when a custom predicate returns true.
    Custom(Arc<dyn Fn(&str) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RetryOn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllErrors => write!(f, "AllErrors"),
            Self::TransientErrors => write!(f, "TransientErrors"),
            Self::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
}

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not counting the initial call).
    pub max_retries: usize,

    /// Initial delay before the first retry.
    pub initial_delay: Duration,

    /// Maximum delay between retries (caps exponential growth).
    pub max_delay: Duration,

    /// Multiplier applied to the delay after each attempt.
    /// A value of 2.0 means each retry waits twice as long as the previous.
    pub backoff_multiplier: f64,

    /// Which errors should trigger a retry.
    pub retry_on: RetryOn,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            retry_on: RetryOn::TransientErrors,
        }
    }
}

impl RetryConfig {
    /// Creates a new RetryConfig with the specified max retries and defaults for other fields.
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Sets the initial delay.
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Sets the maximum delay.
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Sets the backoff multiplier.
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Sets which errors should trigger a retry.
    pub fn with_retry_on(mut self, retry_on: RetryOn) -> Self {
        self.retry_on = retry_on;
        self
    }

    /// Checks if an error should trigger a retry.
    fn should_retry(&self, error: &str) -> bool {
        match &self.retry_on {
            RetryOn::AllErrors => true,
            RetryOn::TransientErrors => is_transient_error(error),
            RetryOn::Custom(predicate) => predicate(error),
        }
    }

    /// Calculates the delay for a given attempt number (0-based).
    fn delay_for_attempt(&self, attempt: usize) -> Duration {
        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let delay = self.initial_delay.as_secs_f64() * multiplier;
        let delay = delay.min(self.max_delay.as_secs_f64());
        Duration::from_secs_f64(delay)
    }
}

/// Check if an error looks like a transient (retriable) error.
fn is_transient_error(error: &str) -> bool {
    let error_lower = error.to_lowercase();

    // HTTP status codes that are retriable
    for code in &["429", "500", "502", "503", "504"] {
        if error_lower.contains(code) {
            return true;
        }
    }

    // Common transient error patterns
    let transient_patterns = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "temporary failure",
        "service unavailable",
        "internal server error",
        "overloaded",
        "capacity",
    ];

    for pattern in &transient_patterns {
        if error_lower.contains(pattern) {
            return true;
        }
    }

    false
}

/// A Runnable wrapper that retries the inner Runnable on failure.
pub struct RunnableRetry<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    runnable: Arc<dyn RunnableAny>,
    retry_config: RetryConfig,
    _marker: std::marker::PhantomData<(I, O)>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> RunnableRetry<I, O> {
    /// Creates a new RunnableRetry from a boxed RunnableAny.
    pub fn new(runnable: Box<dyn RunnableAny>, retry_config: RetryConfig) -> Self {
        Self {
            runnable: Arc::from(runnable),
            retry_config,
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<I: Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O> for RunnableRetry<I, O>
where
    I: Clone,
{
    type Error = LcelError;

    async fn invoke(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<O, Self::Error> {
        // Check cancellation before starting
        if config.as_ref().is_some_and(|c| c.is_cancelled()) {
            return Err(LcelError::Other("Operation cancelled".to_string()));
        }

        let mut last_error = None;

        for attempt in 0..=self.retry_config.max_retries {
            // Check cancellation before each attempt
            if attempt > 0 && config.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(LcelError::Other("Operation cancelled".to_string()));
            }

            // Delay before retry (not on first attempt)
            if attempt > 0 {
                let delay = self.retry_config.delay_for_attempt(attempt - 1);
                tokio::time::sleep(delay).await;
            }

            match self.runnable.invoke_any(Box::new(input.clone()), config.clone()).await {
                Ok(result) => {
                    return result
                        .downcast::<O>()
                        .map(|boxed| *boxed)
                        .map_err(|_| LcelError::Other("Type mismatch in retry result".to_string()));
                }
                Err(e) => {
                    let error_str = e.to_string();
                    if attempt < self.retry_config.max_retries && self.retry_config.should_retry(&error_str) {
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| LcelError::Other("Retry exhausted with no error recorded".to_string())))
    }

    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, Self::Error>> + Send>>, Self::Error> {
        // For stream, we retry the stream setup (construction), not individual tokens.
        // Once the stream is established, token-level errors propagate normally.
        let mut last_error = None;

        for attempt in 0..=self.retry_config.max_retries {
            if attempt > 0 && config.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(LcelError::Other("Operation cancelled".to_string()));
            }

            if attempt > 0 {
                let delay = self.retry_config.delay_for_attempt(attempt - 1);
                tokio::time::sleep(delay).await;
            }

            match self.runnable.stream_any(Box::new(input.clone()), config.clone()).await {
                Ok(stream) => {
                    // Convert the type-erased stream to a typed stream
                    let typed_stream = stream.map(|result| {
                        result
                            .and_then(|boxed| {
                                boxed
                                    .downcast::<O>()
                                    .map(|b| *b)
                                    .map_err(|_| LcelError::Other("Type mismatch in retry stream".to_string()))
                            })
                    });
                    return Ok(Box::pin(typed_stream));
                }
                Err(e) => {
                    let error_str = e.to_string();
                    if attempt < self.retry_config.max_retries && self.retry_config.should_retry(&error_str) {
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| LcelError::Other("Retry exhausted with no error recorded".to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::{CancellationToken, RunnableConfig, RunnableLambda, RunnableExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(10));
        assert!(config.backoff_multiplier - 2.0 < f64::EPSILON);
    }

    #[test]
    fn test_delay_for_attempt() {
        let config = RetryConfig::default();
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(500));
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(1));
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(2));
        // Should be capped at max_delay
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(10));
    }

    #[test]
    fn test_is_transient_error() {
        assert!(is_transient_error("HTTP 429: Too Many Requests"));
        assert!(is_transient_error("HTTP 503: Service Unavailable"));
        assert!(is_transient_error("rate limit exceeded"));
        assert!(is_transient_error("Connection timeout"));
        assert!(is_transient_error("internal server error"));

        assert!(!is_transient_error("HTTP 401: Unauthorized"));
        assert!(!is_transient_error("HTTP 403: Forbidden"));
        assert!(!is_transient_error("invalid API key"));
        assert!(!is_transient_error("model not found"));
    }

    #[tokio::test]
    async fn test_retry_succeeds_on_second_attempt() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = call_count.clone();

        let runnable = RunnableLambda::new_async(move |_: i32| {
            let count = count_clone.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(LcelError::Other("HTTP 503: Service Unavailable".to_string()))
                } else {
                    Ok(42)
                }
            }
        });

        let retry = runnable.with_retry(RetryConfig::new(2));
        let result: Result<i32, _> = retry.invoke(1, None).await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_exhausts_all_attempts() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = call_count.clone();

        let runnable = RunnableLambda::new_async(move |_: i32| {
            let count = count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err(LcelError::Other("HTTP 503: Service Unavailable".to_string()))
            }
        });

        let retry = runnable.with_retry(RetryConfig::new(2));
        let result: Result<i32, _> = retry.invoke(1, None).await;
        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_non_retriable_error_fails_immediately() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = call_count.clone();

        let runnable = RunnableLambda::new_async(move |_: i32| {
            let count = count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err(LcelError::Other("HTTP 401: Unauthorized".to_string()))
            }
        });

        let retry = runnable.with_retry(RetryConfig::new(3));
        let result: Result<i32, _> = retry.invoke(1, None).await;
        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // No retry for 401
    }

    #[tokio::test]
    async fn test_retry_succeeds_on_first_attempt() {
        let runnable = RunnableLambda::new_sync(|x: i32| x * 2);
        let retry = runnable.with_retry(RetryConfig::new(3));
        let result: Result<i32, _> = retry.invoke(5, None).await;
        assert_eq!(result.unwrap(), 10);
    }

    #[tokio::test]
    async fn test_retry_respects_cancellation() {
        let token = CancellationToken::new();
        token.cancel();

        let runnable = RunnableLambda::new_sync(|x: i32| x * 2);
        let retry = runnable.with_retry(RetryConfig::new(3));

        let config = RunnableConfig::new().with_cancellation_token(token);
        let result: Result<i32, _> = retry.invoke(5, Some(config)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }
}
