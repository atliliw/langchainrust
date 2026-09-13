//! Per-model admission control for [`super::RouterLLM`] (B13, 0.22.4).
//!
//! Two independent limits are applied to every routed call:
//!
//! - **Request rate**: at most `requests_per_window` calls may start in any
//!   sliding `window`. Admitted requests borrow a permit that is returned
//!   exactly one window later (each permit is scheduled back with a timer
//!   task), so a burst of `N` is accepted immediately and the next caller
//!   queues until the oldest permit comes home.
//! - **Concurrency**: at most `max_concurrent` calls may be in flight at once
//!   (including the time the caller spends streaming the response, since the
//!   router keeps the [`GatePermit`] alive inside the returned stream).
//!
//! When a limit is saturated, callers **queue** instead of being rejected:
//! `tokio::sync::Semaphore` wakes waiters FIFO, which is exactly the queue
//! fairness the router needs (the oldest blocked caller is admitted first).
//! The queue is optionally bounded (`max_queue`) and each waiter waits at
//! most `wait_timeout`; both produce a
//! [`RouterError::RateLimited`](super::RouterError::RateLimited) that the
//! router treats like a model failure and uses to fall through to the next
//! candidate model.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::RouterError;

/// Default admission queue timeout when callers configure nothing else.
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Per-model rate / concurrency limits attached to one router slot.
///
/// Construct with [`ModelRateLimit::per_minute`] (or
/// [`ModelRateLimit::per_second`]) and tighten the optional dimensions with
/// the `with_*` builders. Pass to
/// [`RouterLLM::with_model_rate_limited`](super::RouterLLM::with_model_rate_limited)
/// or [`RouterLLM::with_last_rate_limit`](super::RouterLLM::with_last_rate_limit).
#[derive(Debug, Clone)]
pub struct ModelRateLimit {
    /// Max calls whose windows overlap (0 disables the rate dimension).
    pub(crate) requests_per_window: usize,
    /// Length of the rate window.
    pub(crate) window: Duration,
    /// Max simultaneously in-flight calls (0 disables the dimension).
    pub(crate) max_concurrent: usize,
    /// Max callers allowed to wait (0 = unbounded).
    pub(crate) max_queue: usize,
    /// Max time one caller waits for admission.
    pub(crate) wait_timeout: Duration,
}

impl ModelRateLimit {
    /// At most `requests_per_minute` admitted per 60-second window, unlimited
    /// concurrency, unbounded queue, 60-second wait timeout.
    pub fn per_minute(requests_per_minute: usize) -> Self {
        Self {
            requests_per_window: requests_per_minute,
            window: Duration::from_secs(60),
            max_concurrent: 0,
            max_queue: 0,
            wait_timeout: DEFAULT_WAIT_TIMEOUT,
        }
    }

    /// At most `requests_per_second` admitted per one-second window.
    pub fn per_second(requests_per_second: usize) -> Self {
        Self {
            requests_per_window: requests_per_second,
            window: Duration::from_secs(1),
            max_concurrent: 0,
            max_queue: 0,
            wait_timeout: Duration::from_secs(1),
        }
    }

    /// Overrides the rate window length (e.g. a provider quota stated per day).
    /// The wait timeout is intentionally left untouched: a queued caller may
    /// legitimately wait several windows deep, so configure
    /// [`ModelRateLimit::with_wait_timeout`] separately when shortening.
    pub fn with_window(mut self, window: Duration) -> Self {
        self.window = window;
        self
    }

    /// Bounds in-flight calls. The slot stays occupied until the whole
    /// response (including a streamed body) has finished. `0` (the default)
    /// disables the concurrency dimension.
    pub fn with_max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.max_concurrent = max_concurrent;
        self
    }

    /// Bounds how many callers may wait for admission; `0` (the default)
    /// means an unbounded FIFO queue. A full queue rejects immediately with
    /// [`RouterError::RateLimited`](super::RouterError::RateLimited) so the
    /// router can fall through to a fallback model without delay.
    pub fn with_max_queue(mut self, max_queue: usize) -> Self {
        self.max_queue = max_queue;
        self
    }

    /// Caps how long one caller waits before the router gives up on this
    /// slot and tries the next model. Defaults to one rate window.
    pub fn with_wait_timeout(mut self, wait_timeout: Duration) -> Self {
        self.wait_timeout = wait_timeout;
        self
    }

    /// Configured requests-per-window.
    pub fn requests_per_window(&self) -> usize {
        self.requests_per_window
    }

    /// Configured rate window.
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Configured concurrency cap (0 = unlimited).
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Configured queue depth (0 = unbounded).
    pub fn max_queue(&self) -> usize {
        self.max_queue
    }

    /// Configured admission wait timeout.
    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout
    }
}

/// Admission gate built from a [`ModelRateLimit`] and shared (`Arc`) by every
/// call routed to one slot.
pub(super) struct ModelGate {
    /// `None` when the rate dimension is disabled.
    rate: Option<RateGate>,
    /// Concurrency permits; [`Semaphore::MAX_PERMITS`] when unlimited.
    concurrency: Arc<Semaphore>,
    /// Max wait for a free concurrency slot (same configured timeout as the
    /// rate queue, so a slot saturated by in-flight streams is skipped).
    wait_timeout: Duration,
}

impl ModelGate {
    pub(super) fn new(config: &ModelRateLimit) -> Arc<Self> {
        let concurrency = if config.max_concurrent == 0 {
            Semaphore::MAX_PERMITS
        } else {
            config.max_concurrent
        };
        Arc::new(Self {
            rate: if config.requests_per_window > 0 {
                Some(RateGate::new(config))
            } else {
                None
            },
            concurrency: Arc::new(Semaphore::new(concurrency)),
            wait_timeout: config.wait_timeout,
        })
    }

    /// Acquires admission for one call.
    ///
    /// Rate admission happens **first** so the FIFO rate queue determines
    /// global order; the concurrency slot is taken afterwards and held in the
    /// returned guard until the call (and its streamed body) completes.
    /// Either dimension parks at most `wait_timeout`, then the router skips
    /// the slot.
    pub(super) async fn acquire(&self, model: &str) -> Result<GatePermit, RouterError> {
        if let Some(rate) = &self.rate {
            rate.acquire(model).await?;
        }
        // An unlimited-configured semaphore cannot park or close; a bounded
        // one waits up to the configured timeout so the caller can fall
        // through to the next candidate instead of blocking indefinitely.
        let acquired =
            tokio::time::timeout(self.wait_timeout, self.concurrency.clone().acquire_owned()).await;
        let concurrency = match acquired {
            Ok(Ok(permit)) => permit,
            Ok(Err(_closed)) => panic!("router concurrency semaphore is never closed"),
            Err(_elapsed) => {
                return Err(RouterError::RateLimited {
                    model: model.to_string(),
                    reason: RateLimitReason::Timeout {
                        waited: self.wait_timeout,
                    },
                });
            }
        };
        Ok(GatePermit {
            _concurrency: concurrency,
        })
    }
}

/// Held while one call is in flight; dropping it frees the concurrency slot.
pub(super) struct GatePermit {
    _concurrency: OwnedSemaphorePermit,
}

impl std::fmt::Debug for GatePermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatePermit").finish_non_exhaustive()
    }
}

/// Sliding-window request-rate gate with a bounded FIFO wait queue.
struct RateGate {
    /// One permit per admissible call; returned `window` after acquisition.
    semaphore: Arc<Semaphore>,
    window: Duration,
    /// Callers currently trying to acquire (waiting or about to be admitted).
    queued: AtomicUsize,
    max_queue: usize,
    wait_timeout: Duration,
}

impl RateGate {
    fn new(config: &ModelRateLimit) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.requests_per_window)),
            window: config.window,
            queued: AtomicUsize::new(0),
            max_queue: config.max_queue,
            wait_timeout: config.wait_timeout,
        }
    }

    async fn acquire(&self, model: &str) -> Result<(), RouterError> {
        // Queue accounting brackets only the wait: a call that already holds
        // a rate permit must not consume queue capacity.
        let position = self.queued.fetch_add(1, Ordering::AcqRel);
        if self.max_queue > 0 && position >= self.max_queue {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            return Err(RouterError::RateLimited {
                model: model.to_string(),
                reason: RateLimitReason::QueueFull {
                    max_queue: self.max_queue,
                },
            });
        }

        let acquired =
            tokio::time::timeout(self.wait_timeout, self.semaphore.clone().acquire_owned()).await;
        self.queued.fetch_sub(1, Ordering::AcqRel);

        let permit = match acquired {
            Ok(Ok(permit)) => permit,
            Ok(Err(_closed)) => panic!("router rate semaphore is never closed"),
            Err(_elapsed) => {
                return Err(RouterError::RateLimited {
                    model: model.to_string(),
                    reason: RateLimitReason::Timeout {
                        waited: self.wait_timeout,
                    },
                });
            }
        };

        // Return the permit exactly one window after this acquisition: the
        // semaphore itself is the sliding-window log, and its FIFO wake order
        // is the admission queue.
        let window = self.window;
        tokio::spawn(async move {
            tokio::time::sleep(window).await;
            drop(permit);
        });
        Ok(())
    }
}

/// Why a queued call could not be admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitReason {
    /// The bounded waiting queue was already full.
    QueueFull {
        /// Configured queue capacity.
        max_queue: usize,
    },
    /// No permit became available within the configured wait timeout.
    Timeout {
        /// Time waited.
        waited: Duration,
    },
}

impl std::fmt::Display for RateLimitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitReason::QueueFull { max_queue } => {
                write!(f, "admission queue full (max_queue={max_queue})")
            }
            RateLimitReason::Timeout { waited } => {
                write!(f, "no permit within {} ms", waited.as_millis())
            }
        }
    }
}

impl std::error::Error for RateLimitReason {}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(rl: &ModelRateLimit) -> Arc<ModelGate> {
        ModelGate::new(rl)
    }

    #[tokio::test(start_paused = true)]
    async fn admits_burst_then_queues_fifo_across_window() {
        // 1 call / 100 ms, concurrency 1 so admissions serialize and the
        // completion order is the admission order.
        let rl = ModelRateLimit::per_second(1)
            .with_window(Duration::from_millis(100))
            .with_max_concurrent(1);
        let g = gate(&rl);

        let finished = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for id in 0..3u8 {
            let g = g.clone();
            let finished = finished.clone();
            handles.push(tokio::spawn(async move {
                let _permit = g.acquire("m").await.unwrap();
                // Simulate in-call work while holding both permits.
                tokio::time::sleep(Duration::from_millis(10)).await;
                finished.lock().await.push(id);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // FIFO: spawn order 0,1,2 must be completion order even though
        // callers 1 and 2 had to wait for sliding-window replenishment.
        assert_eq!(*finished.lock().await, vec![0, 1, 2]);
    }

    #[tokio::test(start_paused = true)]
    async fn full_queue_rejects_instead_of_waiting() {
        let rl = ModelRateLimit::per_minute(1).with_max_queue(1);
        let g = gate(&rl);

        let _p0 = g.acquire("m").await.unwrap();
        // One caller is allowed to park in the queue ...
        let g1 = g.clone();
        let waiter = tokio::spawn(async move { g1.acquire("m").await });
        // Let the waiter park.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        // ... the next caller hits the bounded queue and fails immediately so
        // the router can fall through to a fallback.
        let err = g.acquire("m").await.unwrap_err();
        assert!(
            matches!(
                err,
                RouterError::RateLimited {
                    reason: RateLimitReason::QueueFull { max_queue: 1 },
                    ..
                }
            ),
            "got {err:?}"
        );
        // The parked waiter is unaffected and gets admitted once a permit
        // returns (one window later).
        tokio::time::sleep(Duration::from_secs(60)).await;
        assert!(waiter.await.unwrap().is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn wait_timeout_falls_through() {
        let rl = ModelRateLimit::per_minute(1).with_wait_timeout(Duration::from_secs(5));
        let g = gate(&rl);
        let _p0 = g.acquire("m").await.unwrap();
        let started = tokio::time::Instant::now();
        let err = g.acquire("primary").await.unwrap_err();
        assert_eq!(started.elapsed(), Duration::from_secs(5));
        assert!(
            matches!(
                err,
                RouterError::RateLimited {
                    reason: RateLimitReason::Timeout { .. },
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn permit_returns_after_window_and_keeps_rate_stable() {
        // 2 / 100 ms: two bursts of two, separated by a window, must admit.
        let rl = ModelRateLimit::per_second(2).with_window(Duration::from_millis(100));
        let g = gate(&rl);
        let _p1 = g.acquire("m").await.unwrap();
        let _p2 = g.acquire("m").await.unwrap();
        // Third call queues; must be admitted shortly after the window rolls.
        let g2 = g.clone();
        let h = tokio::spawn(async move { g2.acquire("m").await });
        tokio::time::sleep(Duration::from_millis(101)).await;
        assert!(h.await.unwrap().is_ok());
    }

    #[test]
    fn disabled_dimensions_are_represented_as_zero() {
        let rl = ModelRateLimit::per_minute(10);
        assert_eq!(rl.max_concurrent(), 0);
        assert_eq!(rl.max_queue(), 0);
        assert_eq!(rl.requests_per_window(), 10);
        let rl2 = rl
            .with_window(Duration::from_secs(30))
            .with_wait_timeout(Duration::from_secs(30));
        // Window and wait timeout are configured independently (a queued
        // caller may wait more than one window).
        assert_eq!(rl2.window(), Duration::from_secs(30));
        assert_eq!(rl2.wait_timeout(), Duration::from_secs(30));
    }
}
