//! Rate limiting for A2A requests.
//!
//! `RateLimiter` combines two independent limits:
//!
//! - **Concurrency limit**: a `tokio::sync::Semaphore` bounding how many
//!   requests are in-flight at once.
//! - **Window rate limit**: a rolling one-minute token count bounding how
//!   many requests may be admitted per window.
//!
//! Acquiring a permit returns a `RateLimitPermit` guard that releases the
//! concurrency slot when dropped. Pass `0` for either dimension to disable
//! that limit.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

/// Error returned when a request is not admitted by the rate limiter.
#[derive(Debug, thiserror::Error)]
pub enum RateLimitError {
    /// The per-window request budget has been exhausted.
    #[error("request rate limit exceeded")]
    TooManyRequests,
    /// The maximum number of concurrent requests is already in flight.
    #[error("too many concurrent requests")]
    ConcurrencyLimitExceeded,
}

struct WindowState {
    window_start: Instant,
    count: usize,
}

/// A rate limiter combining a concurrency cap and a per-minute request cap.
pub struct RateLimiter {
    /// Semaphore guarding the maximum number of concurrent requests.
    semaphore: Arc<Semaphore>,
    /// Length of the rate window.
    window: Duration,
    /// Maximum requests admitted per window (`0` = unlimited).
    max_requests: usize,
    /// Rolling window counter.
    state: Mutex<WindowState>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// `max_concurrent` bounds in-flight requests (0 = unlimited) and
    /// `max_requests_per_minute` bounds the per-minute admission rate
    /// (0 = unlimited).
    pub fn new(max_concurrent: usize, max_requests_per_minute: usize) -> Self {
        let permits = if max_concurrent == 0 {
            Semaphore::MAX_PERMITS
        } else {
            max_concurrent
        };
        Self {
            semaphore: Arc::new(Semaphore::new(permits)),
            window: Duration::from_secs(60),
            max_requests: max_requests_per_minute,
            state: Mutex::new(WindowState {
                window_start: Instant::now(),
                count: 0,
            }),
        }
    }

    /// Try to acquire a permit to process one request.
    ///
    /// Returns a guard that releases the concurrency slot when dropped, or a
    /// `RateLimitError` if the request would exceed either limit.
    pub async fn try_acquire(&self) -> Result<RateLimitPermit, RateLimitError> {
        // 1. Window rate check.
        if self.max_requests > 0 {
            let mut state = self.state.lock().await;
            let now = Instant::now();
            if now.duration_since(state.window_start) >= self.window {
                state.window_start = now;
                state.count = 0;
            }
            if state.count >= self.max_requests {
                return Err(RateLimitError::TooManyRequests);
            }
            state.count += 1;
        }

        // 2. Concurrency slot.
        let permit = self
            .semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| RateLimitError::ConcurrencyLimitExceeded)?;

        Ok(RateLimitPermit { _permit: permit })
    }
}

/// Guard that releases a rate-limiter concurrency slot when dropped.
pub struct RateLimitPermit {
    _permit: OwnedSemaphorePermit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_acquires() {
        let limiter = RateLimiter::new(0, 0);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let permit = limiter.try_acquire().await;
            assert!(permit.is_ok());
            drop(permit);
            assert!(limiter.try_acquire().await.is_ok());
        });
    }

    #[test]
    fn enforces_per_minute_limit() {
        let limiter = RateLimiter::new(0, 2);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            assert!(limiter.try_acquire().await.is_ok());
            assert!(limiter.try_acquire().await.is_ok());
            let err = limiter.try_acquire().await;
            assert!(matches!(err, Err(RateLimitError::TooManyRequests)));
        });
    }

    #[test]
    fn enforces_concurrency_limit() {
        let limiter = RateLimiter::new(1, 0);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let p1 = limiter.try_acquire().await.unwrap();
            let err = limiter.try_acquire().await;
            assert!(matches!(err, Err(RateLimitError::ConcurrencyLimitExceeded)));
            drop(p1);
            // Slot freed -> acquire succeeds again.
            assert!(limiter.try_acquire().await.is_ok());
        });
    }
}
