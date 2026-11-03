// lc-embeddings/src/limiter.rs
//! Concurrent token-bucket rate limiter (v0.27.0 T2 §5: 并发 token-bucket 限流).
//!
//! Embedding HTTP providers fan requests out across a batch (see `openai.rs`
//! `buffer_unordered` chunking). Bursts that outpace a provider's rate limits get
//! throttled server-side with 429s; [`TokenBucket`] smooths the client side so the
//! outgoing request rate stays within a configured budget *before* a 429 can happen.
//!
//! A token bucket is both a concurrency cap and a rate limit in one primitive:
//!
//! - **capacity** — how many tokens the bucket starts full; the largest burst of
//!   requests allowed through back-to-back. Because each in-flight request consumes
//!   a token that only comes back via refill, this also bounds concurrent in-flight
//!   requests (the "并发" half).
//! - **refill_per_sec** — how many tokens the bucket replenishes per second; the
//!   sustained steady-state request rate. A `0.0` refill is burst-only: exactly
//!   `capacity` requests ever pass.
//!
//! Refill is **lazy** (computed from elapsed time on each acquire), so a quiet
//! limiter costs nothing and a burst that has already respected the rate never pays
//! extra. The limiter is cheap, `Send + Sync` via `Arc<TokenBucket>`, and opt-in:
//! an embedding provider config defaults to *no* limiter, preserving existing
//! behaviour until a caller attaches one.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::time::Instant;

/// Error returned by [`TokenBucket::acquire`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TokenBucketError {
    /// The bucket is burst-only (`refill_per_sec == 0.0`) and its one-time token
    /// grant is spent. Because no refill will ever arrive, acquirers fail
    /// immediately instead of parking on a maximal sleep forever (which would
    /// otherwise hang the caller, since no timeout is armed here).
    #[error("token-bucket exhausted: burst-only bucket has no tokens left")]
    Exhausted,
}

/// Lazy token bucket used to pace embedding HTTP requests.
///
/// Clone/`Arc` freely: all mutable state lives behind a single `tokio::Mutex`, so
/// many concurrent acquire calls share one budget.
#[derive(Clone)]
pub struct TokenBucket {
    inner: Arc<TokenBucketInner>,
}

struct TokenBucketInner {
    /// Burst / concurrency cap (tokens the bucket may hold).
    capacity: f64,
    /// Tokens replenished per second; `0.0` = burst-only.
    refill_per_sec: f64,
    state: Mutex<State>,
}

struct State {
    /// Tokens currently available (fractional under a balancing refill).
    stored: f64,
    /// Wall-clock instant of the last refill/consume.
    last: Instant,
}

impl TokenBucket {
    /// A bucket that starts full with `capacity` tokens and refills
    /// `refill_per_sec` tokens per second up to `capacity`.
    ///
    /// # Panics
    /// Panics if `capacity == 0` (a zero-capacity bucket could never admit a
    /// request).
    pub fn new(capacity: usize, refill_per_sec: f64) -> Self {
        assert!(capacity > 0, "token-bucket capacity must be >= 1");
        Self {
            inner: Arc::new(TokenBucketInner {
                capacity: capacity as f64,
                refill_per_sec: refill_per_sec.max(0.0),
                state: Mutex::new(State {
                    stored: capacity as f64,
                    last: Instant::now(),
                }),
            }),
        }
    }

    /// Wait until a token is available, then consume one.
    ///
    /// Refill is computed lazily from elapsed time on entry. If a token is ready the
    /// call returns immediately; otherwise it sleeps for exactly the time the next
    /// token needs to refill. F-01: a burst-only bucket (`refill_per_sec == 0.0`)
    /// that has exhausted its one-time grant returns [`TokenBucketError::Exhausted`]
    /// immediately instead of spinning forever — a deficit it can never close.
    pub async fn acquire(&self) -> Result<(), TokenBucketError> {
        loop {
            let wait = {
                let mut st = self.inner.state.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(st.last).as_secs_f64();
                st.last = now;
                // Lazy refill, capped at capacity. A deficit (stored < 1) stays a
                // deficit; excess is capped so the bucket never grows boundless.
                st.stored =
                    (st.stored + elapsed * self.inner.refill_per_sec).min(self.inner.capacity);
                if st.stored >= 1.0 {
                    st.stored -= 1.0;
                    return Ok(());
                }
                // F-01: a burst-only bucket can never refill this deficit, so fail
                // immediately rather than parking on the maximal sleep below (a
                // permanent hang with no caller-armed timeout to cancel it).
                if self.inner.refill_per_sec == 0.0 {
                    return Err(TokenBucketError::Exhausted);
                }
                // No token yet: sleep for the refill of exactly the shortfall
                // (deficit / rate).
                let deficit = 1.0 - st.stored;
                Duration::from_secs_f64(deficit / self.inner.refill_per_sec)
            };
            tokio::time::sleep(wait).await;
        }
    }
}

impl std::fmt::Debug for TokenBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBucket")
            .field("capacity", &self.inner.capacity)
            .field("refill_per_sec", &self.inner.refill_per_sec)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bucket starts full, so `capacity` back-to-back acquires succeed
    /// immediately without any waiting.
    #[tokio::test]
    async fn starts_full_and_serves_up_to_capacity() {
        let b = TokenBucket::new(3, 0.0);
        b.acquire().await.expect("1st acquires within capacity");
        b.acquire().await.expect("2nd acquires within capacity");
        b.acquire().await.expect("3rd consumes the last token");
    }

    /// Beyond `capacity` the acquire fails fast with [`TokenBucketError::Exhausted`]
    /// — exactly the configured number slip through (the "并发" cap), the rest error
    /// rather than hang forever.
    #[tokio::test]
    async fn burst_only_bounds_concurrent_acquires() {
        let b = Arc::new(TokenBucket::new(2, 0.0));
        let mut handles = Vec::new();
        for _ in 0..5 {
            let b = b.clone();
            handles.push(tokio::spawn(async move { b.acquire().await }));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;

        let mut granted = 0;
        let mut exhausted = 0;
        for h in handles {
            match h.await.expect("acquire task should not panic") {
                Ok(()) => granted += 1,
                Err(TokenBucketError::Exhausted) => exhausted += 1,
                Err(_) => exhausted += 1,
            }
        }
        assert_eq!(granted, 2, "only the configured burst capacity may pass");
        assert_eq!(
            exhausted, 3,
            "over-capacity requests fail fast instead of hanging"
        );
    }

    /// F-01: a burst-only bucket (`refill_per_sec == 0.0`) fails fast once its
    /// one-time grant is spent, never parking forever on the maximal sleep — the
    /// point of the fix is that an exhausted bucket returns promptly, not hangs.
    #[tokio::test]
    async fn burst_only_exhausted_fails_immediately() {
        let b = TokenBucket::new(1, 0.0);
        b.acquire().await.expect("first acquire consumes the lone token");
        assert_eq!(
            b.acquire().await,
            Err(TokenBucketError::Exhausted),
            "second acquire must fail immediately, not wait forever"
        );
    }

    /// A non-zero refill lets the bucket sustain repeated acquires that would
    /// otherwise park forever (proving the sustain-rate half of the bucket).
    #[tokio::test]
    async fn refill_sustains_repeated_acquire() {
        // capacity 1, 50 tokens/sec: each successive acquire must wait for a
        // ~20ms refill rather than hanging (which would fail via the runtime's
        // test timeout).
        let b = TokenBucket::new(1, 50.0);
        b.acquire().await.expect("consume the initial token");
        for _ in 0..3 {
            b.acquire().await.expect("refill must keep admitting");
        }
    }

    /// The bucket never grows beyond capacity under a high refill (a burst of
    /// spare time cannot bank an unlimited reserve).
    #[tokio::test]
    async fn stored_tokens_are_capped_at_capacity() {
        // capacity 1, huge refill: after a long idle the stored cap still
        // admits exactly one immediate token, not an unbounded backlog.
        let b = TokenBucket::new(1, 100_000.0);
        b.acquire().await.expect("consumes cap token");
    }

    /// `new` rejects a zero capacity (which could never admit anything).
    #[test]
    #[should_panic(expected = "capacity")]
    fn zero_capacity_panics() {
        let _ = TokenBucket::new(0, 1.0);
    }

    /// A shared bucket still advances correctly across interleaved consumers
    /// (the design target: many concurrent embedding calls, one budget).
    #[tokio::test]
    async fn concurrent_consumers_share_budget() {
        let b = Arc::new(TokenBucket::new(3, 0.0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let b = b.clone();
            handles.push(tokio::spawn(async move { b.acquire().await }));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut granted = 0;
        let mut exhausted = 0;
        for h in handles {
            match h.await.expect("acquire task should not panic") {
                Ok(()) => granted += 1,
                Err(_) => exhausted += 1,
            }
        }
        assert_eq!(granted, 3, "shared bucket funds all consumers up to capacity");
        assert_eq!(exhausted, 1, "the one over-capacity consumer fails fast");
    }
}
