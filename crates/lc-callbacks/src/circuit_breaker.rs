//! Handler-level circuit breaker (T2 §5: 熔断/降级).
//!
//! A [`CircuitBreaker`] turns a flaky side channel (network trace upload, file
//! write) into a bounded local buffer instead of an unbounded stall or a long
//! tail of doomed requests. It models the classic closed → open → half-open
//! state machine, driven by two events the caller already has:
//!
//! - [`CircuitBreaker::record_success`] — one call completed from the
//!   destination's point of view.
//! - [`CircuitBreaker::allow`] — consult before doing the (possibly failing)
//!   work; returns `false` while the breaker is open so the caller can degrade
//!   to buffering/sampling instead of attempting another call that is almost
//!   certainly going to fail the same way.
//!
//! # Semantics
//!
//! - **Closed**: calls flow. N consecutive failures trip the breaker open
//!   (`CircuitBreaker::trip_threshold`). A single success resets the failure
//!   count.
//! - **Open**: [`allow`](CircuitBreaker::allow) returns `false` — the caller
//!   buffers locally. After [`cooldown`](CircuitBreaker::new) elapses, the next
//!   [`allow`](CircuitBreaker::allow) becomes a half-open probe: exactly one call
//!   is admitted.
//! - **Half-open**: one trial admitted. A success closes the breaker (and resets
//!   the count); a failure trips it back open with a fresh cooldown.
//!
//! The breaker is cheap and process-local: a single `Mutex` over a small enum,
//! no timers of its own. Timeouts are read from a fixed `Instant` captured at
//! trip time, so logic is fully unit-testable without sleeping.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Consecutive-failure → trip threshold shared by the hand-rolled tests. Inline
/// it into [`CircuitBreaker::with_trip_threshold`] call sites; this default is
/// used only when the caller does not configure one.
const DEFAULT_TRIP_THRESHOLD: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    /// Calls flow; `consecutive_failures` counts how many in a row it takes to trip.
    Closed { consecutive_failures: u32 },
    /// Calls are refused; the cooldown window started at `since`.
    Open { since: Instant },
    /// A single trial is admitted; the outcome decides closed vs open.
    HalfOpen,
}

/// Trip stats surfaced for observability / tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BreakerStatus {
    /// Number of calls currently refused because the breaker is open.
    pub refused: u64,
    /// Number of `allow` calls since the breaker was constructed.
    pub total_calls: u64,
    /// Number of times the breaker tripped open out of closed.
    pub trips: u64,
}

/// A per-handler circuit breaker for degrading flaky side channels.
pub struct CircuitBreaker {
    trip_threshold: u32,
    cooldown: Duration,
    state: Mutex<BreakerInner>,
}

struct BreakerInner {
    state: State,
    status: BreakerStatus,
}

impl CircuitBreaker {
    /// Create a closed breaker that trips after `trip_threshold` consecutive
    /// failures and reopens after `cooldown`.
    pub fn new(trip_threshold: u32, cooldown: Duration) -> Self {
        Self {
            trip_threshold: trip_threshold.max(1),
            cooldown,
            state: Mutex::new(BreakerInner {
                state: State::Closed {
                    consecutive_failures: 0,
                },
                status: BreakerStatus::default(),
            }),
        }
    }

    /// Shortcut with the default trip threshold and a caller-provided cooldown.
    pub fn with_cooldown(cooldown: Duration) -> Self {
        Self::new(DEFAULT_TRIP_THRESHOLD, cooldown)
    }

    /// True when calls may proceed; false while the breaker is open (degrade).
    ///
    /// A closed breaker always admits. An open breaker refuses until the cooldown
    /// has elapsed, at which point the next `allow` becomes a single half-open
    /// probe. Note the probe does not gate on anything else, so a burst arriving
    /// right at the half-open transition may slip more than one call through;
    /// that is acceptable for an idempotent-side-channel use case.
    pub fn allow(&self) -> bool {
        let mut inner = self.state.lock().unwrap_or_else(|p| p.into_inner());
        inner.status.total_calls += 1;
        match inner.state {
            State::Closed { .. } => true,
            State::Open { since } => {
                if since.elapsed() >= self.cooldown {
                    // Half-open probe: admit exactly this transition.
                    inner.state = State::HalfOpen;
                    true
                } else {
                    inner.status.refused += 1;
                    false
                }
            }
            State::HalfOpen => true,
        }
    }

    /// Record a successful call. Closes a half-open breaker and clears a closed
    /// failure streak; a refused call stacked in an open breaker is unaffected.
    pub fn record_success(&self) {
        let mut inner = self.state.lock().unwrap_or_else(|p| p.into_inner());
        match inner.state {
            State::Closed { .. } => {
                inner.state = State::Closed {
                    consecutive_failures: 0,
                }
            }
            State::HalfOpen => {
                inner.state = State::Closed {
                    consecutive_failures: 0,
                }
            }
            State::Open { .. } => {}
        }
    }

    /// Record a failed call. Trips a closed breaker once the streak reaches the
    /// threshold, and re-opens a half-open probe on its failure.
    pub fn record_failure(&self) {
        let mut inner = self.state.lock().unwrap_or_else(|p| p.into_inner());
        match inner.state {
            State::Closed {
                consecutive_failures,
            } => {
                let next = consecutive_failures + 1;
                if next >= self.trip_threshold {
                    inner.status.trips += 1;
                    inner.state = State::Open {
                        since: Instant::now(),
                    };
                } else {
                    inner.state = State::Closed {
                        consecutive_failures: next,
                    };
                }
            }
            State::HalfOpen => {
                inner.status.trips += 1;
                inner.state = State::Open {
                    since: Instant::now(),
                };
            }
            // Already open: a resetting failure keeps the cooldown running.
            State::Open { .. } => {}
        }
    }

    /// Whether calls are currently being refused.
    pub fn is_open(&self) -> bool {
        matches!(
            self.state.lock().unwrap_or_else(|p| p.into_inner()).state,
            State::Open { .. }
        )
    }

    /// Aggregated throughput/refusal stats (for logging or dashboards).
    pub fn status(&self) -> BreakerStatus {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).status
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::with_cooldown(Duration::from_secs(30))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_closed_and_admits() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(1));
        assert!(!cb.is_open());
        assert!(cb.allow());
        cb.record_success(); // success resets, stays closed
        assert!(!cb.is_open());
    }

    #[test]
    fn trips_open_after_threshold_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        assert!(cb.allow());
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open(), "below threshold stays closed");
        assert!(cb.allow());
        cb.record_failure(); // 3 consecutive → open
        assert!(cb.is_open());
        assert!(!cb.allow(), "open breaker refuses calls");
        assert_eq!(cb.status().trips, 1);
        assert_eq!(cb.status().refused, 1);
    }

    #[test]
    fn a_success_resets_the_failure_streak() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.allow();
        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // resets streak to 0
        cb.allow();
        cb.record_failure(); // only 1 again — must not trip
        assert!(!cb.is_open());
    }

    #[test]
    fn half_open_probe_trips_open_on_failure() {
        // Zero cooldown means the very next `allow` is a half-open probe, so the
        // probe-failure path is observable through `is_open`/`trips` instead of a
        // refused `allow` (which a zero cooldown never returns).
        let cb = CircuitBreaker::new(1, Duration::ZERO); // trip after 1 failure
        assert!(cb.allow());
        cb.record_failure(); // → open
        assert!(cb.is_open());
        assert!(cb.allow(), "zero cooldown admits a probe immediately");
        cb.record_failure(); // probe fails → back open
        assert!(cb.is_open(), "probe failure reopens the breaker");
        assert_eq!(cb.status().trips, 2, "initial trip + probe trip");
    }

    #[test]
    fn half_open_probe_closes_on_success() {
        let cb = CircuitBreaker::new(1, Duration::ZERO);
        assert!(cb.allow());
        cb.record_failure(); // open
        assert!(cb.is_open());
        assert!(cb.allow(), "probe admitted after zero cooldown");
        assert!(!cb.is_open(), "half-open is not reported open");
        cb.record_success(); // probe succeeds → closed, streak reset
        assert!(!cb.is_open());
        assert!(cb.allow());
    }

    #[test]
    fn open_breaker_stays_open_until_cooldown_elapses() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(3600));
        assert!(cb.allow());
        cb.record_failure(); // open
                             // Immediately refused; the pending open has `since`=now, far from cooldown end.
        assert!(!cb.allow());
        assert!(!cb.allow());
        assert_eq!(cb.status().refused, 2);
    }

    #[test]
    fn unrelated_breakers_are_independent() {
        let a = CircuitBreaker::new(1, Duration::from_secs(60));
        a.allow();
        a.record_failure();
        assert!(a.is_open());
        let b = CircuitBreaker::new(1, Duration::from_secs(60));
        assert!(!b.is_open());
        assert!(b.allow());
    }
}
