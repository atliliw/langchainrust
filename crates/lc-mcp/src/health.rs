//! MCP Server health checks + circuit breaker (P2-5).
//!
//! Any one of 100+ servers may die at any moment. This module provides:
//!
//! - **Heartbeat probe**: `list_tools` is the probe — being able to list tools means the server is alive;
//! - **Breaker removal**: after `N` consecutive failures `CircuitBreaker` goes Open (trips), and `client()`
//!   fails fast without hitting the unhealthy server;
//! - **Exponential backoff reconnect**: after tripping, back off 0.5s → 1s → 2s → … (cap 30s); when the
//!   backoff ends it enters a half-open probe window — `allow_request` lets one through; success recovers,
//!   failure advances to the next backoff step.
//!
//! [`ServerHealth`] is a single health snapshot, returned by `ConnectionManager::health(name)`;
//! [`CircuitBreaker`] is the persistent breaker state machine, one per managed server.

use std::time::{Duration, Instant};

use crate::client::MCPClient;
use crate::protocol::MCPError;

/// Exponential backoff base (0.5s).
const BASE_BACKOFF: Duration = Duration::from_millis(500);
/// Backoff cap: prevents the wait time from growing without bound.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Server health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Healthy (no failures).
    Healthy,
    /// Has seen failures but is still usable (below the removal threshold).
    Degraded,
    /// Consecutive failures hit the threshold; tripped and removed by the breaker.
    Down,
}

/// A single server's health snapshot (P2-5).
///
/// Returned by [`ConnectionManager::health`](crate::connection_manager::ConnectionManager::health); `last_check` is the most recent
/// probe time, `failures` the consecutive failure count, `max_failures` the threshold that triggers Down.
#[derive(Debug, Clone)]
pub struct ServerHealth {
    /// Current health status.
    pub status: HealthStatus,
    /// Consecutive failure count (accumulated from probes + connection failures).
    pub failures: u32,
    /// Most recent probe time (`None` if never probed).
    pub last_check: Option<Instant>,
    /// N consecutive failures trips Down (removal threshold).
    pub max_failures: u32,
}

impl ServerHealth {
    /// Builds an initial health snapshot.
    pub fn new(max_failures: u32) -> Self {
        Self {
            status: HealthStatus::Healthy,
            failures: 0,
            last_check: None,
            max_failures,
        }
    }

    /// Records a successful probe: restores health.
    pub fn record_success(&mut self) {
        self.status = HealthStatus::Healthy;
        self.failures = 0;
        self.last_check = Some(Instant::now());
    }

    /// Records a failed probe: increments the failure count; at the threshold goes Down, otherwise Degraded.
    pub fn record_failure(&mut self) {
        self.failures += 1;
        self.last_check = Some(Instant::now());
        self.status = if self.failures >= self.max_failures.max(1) {
            HealthStatus::Down
        } else {
            HealthStatus::Degraded
        };
    }

    /// Whether it has been tripped and removed.
    pub fn is_down(&self) -> bool {
        self.status == HealthStatus::Down
    }
}

/// Breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Normal, requests allowed.
    Closed,
    /// Tripped: requests are rejected until the backoff time elapses (enters a half-open probe window).
    Open,
}

/// Exponential backoff: 0.5s → 1s → 2s → 4s → … capped at 30s.
fn backoff_delay(step: u32) -> Duration {
    let ms = (BASE_BACKOFF.as_millis() as u64)
        .checked_shl(step.min(6))
        .unwrap_or(MAX_BACKOFF.as_millis() as u64)
        .min(MAX_BACKOFF.as_millis() as u64);
    Duration::from_millis(ms)
}

/// Circuit breaker (P2-5): trips on consecutive failures + exponential backoff + half-open probe.
///
/// One per `ManagedServer` inside [`ConnectionManager`](crate::connection_manager::ConnectionManager). Request allow rules:
///
/// - `Closed`: allowed (can connect/call normally);
/// - `Open` before the backoff time: rejected (fail fast, don't hammer the bad server);
/// - `Open` after the backoff time (half-open window): one probe allowed; success recovers, failure advances the backoff.
pub struct CircuitBreaker {
    /// Trips after N consecutive failures.
    max_failures: u32,
    /// Current consecutive failure count.
    failures: u32,
    state: BreakerState,
    /// Next time a probe is allowed (backoff deadline).
    next_retry_at: Option<Instant>,
    /// Backoff duration armed by the most recent trip (exponential backoff).
    backoff: Duration,
    /// Next backoff step (increments on each trip, growing exponentially).
    backoff_step: u32,
}

impl CircuitBreaker {
    /// Creates a circuit breaker.
    pub fn new(max_failures: u32) -> Self {
        Self {
            max_failures: max_failures.max(1),
            failures: 0,
            state: BreakerState::Closed,
            next_retry_at: None,
            backoff: Duration::ZERO,
            backoff_step: 0,
        }
    }

    /// Current breaker state.
    pub fn state(&self) -> BreakerState {
        self.state
    }

    /// Current consecutive failure count.
    pub fn failures(&self) -> u32 {
        self.failures
    }

    /// Whether a request may be sent.
    ///
    /// `Closed` always allows; `Open` allows once only after the backoff time has passed (half-open probe window).
    pub fn allow_request(&self) -> bool {
        match self.state {
            BreakerState::Closed => true,
            BreakerState::Open => self
                .next_retry_at
                .map(|t| Instant::now() >= t)
                .unwrap_or(true),
        }
    }

    /// Records a success: back to `Closed`, resets the failure count and backoff steps.
    pub fn record_success(&mut self) {
        self.state = BreakerState::Closed;
        self.failures = 0;
        self.backoff = Duration::ZERO;
        self.backoff_step = 0;
        self.next_retry_at = None;
    }

    /// Records a failure: increments the failure count; at the threshold → Open + schedules the next backoff retry time.
    ///
    /// On a failed half-open probe (already Open), re-opens and advances the backoff step (exponential growth).
    pub fn record_failure(&mut self) {
        self.failures += 1;
        if self.failures >= self.max_failures {
            self.state = BreakerState::Open;
            self.backoff = backoff_delay(self.backoff_step);
            self.next_retry_at = Some(Instant::now() + self.backoff);
            self.backoff_step += 1;
        }
    }

    /// Currently armed backoff duration (the exponential backoff used by the most recent trip).
    pub fn backoff(&self) -> Duration {
        self.backoff
    }

    /// Derives the health status from the current breaker state.
    pub fn health_status(&self) -> HealthStatus {
        to_status(self.state, self.failures)
    }
}

/// Heartbeat probe: `list_tools` is the probe (P2-5).
///
/// Being able to list tools means the server can at least respond to requests; a failed probe is
/// recorded by the caller ([`CircuitBreaker::record_failure`]) into the breaker count.
pub async fn probe_health(client: &MCPClient) -> Result<(), MCPError> {
    client.list_tools().await.map(|_| ())
}

/// Derives the health status from the breaker state.
fn to_status(state: BreakerState, failures: u32) -> HealthStatus {
    match state {
        BreakerState::Open => HealthStatus::Down,
        BreakerState::Closed if failures > 0 => HealthStatus::Degraded,
        BreakerState::Closed => HealthStatus::Healthy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::MCPConfig;

    /// `ServerHealth` state flow: Healthy → Degraded → Down → back to Healthy.
    #[test]
    fn test_server_health_transitions() {
        let mut health = ServerHealth::new(2);
        assert_eq!(health.status, HealthStatus::Healthy);
        health.record_failure();
        assert_eq!(health.status, HealthStatus::Degraded);
        assert_eq!(health.failures, 1);
        assert!(health.last_check.is_some());
        health.record_failure();
        assert_eq!(health.status, HealthStatus::Down);
        assert!(health.is_down());
        health.record_success();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.failures, 0);
    }

    /// Breaker: only opens at the threshold; once Open, requests are rejected before the backoff time.
    #[test]
    fn test_circuit_breaker_trips_and_blocks() {
        let mut cb = CircuitBreaker::new(2);
        assert!(cb.allow_request());
        cb.record_failure();
        // Below the threshold, still Closed.
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow_request());
        cb.record_failure();
        // Threshold reached → Open, rejected within the backoff period.
        assert_eq!(cb.state(), BreakerState::Open);
        assert!(
            !cb.allow_request(),
            "should reject requests while circuit is open"
        );
    }

    /// Exponential backoff: doubles per step, capped at 30s.
    #[test]
    fn test_circuit_breaker_exponential_backoff() {
        let mut cb = CircuitBreaker::new(1);
        cb.record_failure();
        assert_eq!(cb.backoff(), Duration::from_millis(500));
        cb.record_failure(); // failed half-open probe → next step
        assert_eq!(cb.backoff(), Duration::from_millis(1000));
        cb.record_failure();
        assert_eq!(cb.backoff(), Duration::from_millis(2000));
        cb.record_failure();
        assert_eq!(cb.backoff(), Duration::from_millis(4000));
        // step 6 → 500 << 6 = 32000 → capped at 30000.
        for _ in 0..4 {
            cb.record_failure();
        }
        assert_eq!(cb.backoff(), MAX_BACKOFF);
    }

    /// Recovery: a successful call closes the breaker and allows requests.
    #[test]
    fn test_circuit_breaker_recovers_on_success() {
        let mut cb = CircuitBreaker::new(1);
        cb.record_failure();
        assert_eq!(cb.state(), BreakerState::Open);
        cb.record_success();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow_request());
        assert_eq!(cb.failures(), 0);
    }

    /// The minimum threshold floor is 1.
    #[test]
    fn test_circuit_breaker_max_failures_min_one() {
        let mut cb = CircuitBreaker::new(0);
        cb.record_failure();
        assert_eq!(
            cb.state(),
            BreakerState::Open,
            "max_failures must be at least 1"
        );
    }

    /// Probe: list_tools against a fake SSE server succeeds.
    #[tokio::test]
    async fn test_probe_health_with_fake_server() {
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        probe_health(&client)
            .await
            .expect("list_tools probe should succeed");
    }
}
