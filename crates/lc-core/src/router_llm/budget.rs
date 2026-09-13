//! Router-level spend/token budget circuit breaker (B13, 0.22.4).
//!
//! A [`RouterBudget`] is shared (`Arc`) by every call through one
//! [`super::RouterLLM`]. Before a candidate model is attempted the router
//! asks [`RouterBudget::precheck`] whether the **projected** spend (the
//! prompt priced through the slot's [`ModelPrice`](crate::cost::ModelPrice))
//! still fits; after a successful call it records the model-reported
//! [`TokenUsage`](crate::language_models::TokenUsage), which is what actually
//! trips the breaker once cumulative spend crosses the cap.
//!
//! Crucially, the breaker does not make the router fail: it makes the router
//! **skip** the candidate that would exceed the budget and continue down the
//! fallback chain. A slot without a price (a local / free model) projects
//! zero cost, so after the paid tier trips, traffic rolls over to free
//! fallbacks — the mainstream "cost guard" pattern. Only when every
//! candidate is skipped does the caller see
//! [`RouterError::BudgetExceeded`](super::RouterError::BudgetExceeded).
//!
//! All state lives in atomics, so the precheck is a cheap synchronous read
//! usable on both the chat and streaming paths.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Floating-point tolerance for "already exactly at the limit" comparisons.
const COST_EPSILON: f64 = 1e-9;

/// Which dimension of a [`RouterBudget`] was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetKind {
    /// Cumulative USD spend crossed the configured cost cap.
    CostUsd,
    /// Cumulative token usage crossed the configured token cap.
    Tokens,
}

impl std::fmt::Display for BudgetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetKind::CostUsd => f.write_str("cost budget (USD)"),
            BudgetKind::Tokens => f.write_str("token budget"),
        }
    }
}

/// A circuit breaker over cumulative spend and/or token usage.
///
/// Construct with one of the constructors, wrap in an `Arc`, and attach via
/// [`super::RouterLLM::with_budget`]. Cloning is not supported; share by
/// `Arc` so every call of one run observes the same totals.
pub struct RouterBudget {
    max_cost_usd: Option<f64>,
    max_tokens: Option<u64>,
    /// Cumulative spend as `f64` bits, updated with a CAS loop.
    spent_bits: AtomicU64,
    tokens: AtomicUsize,
    trips: AtomicUsize,
}

impl std::fmt::Debug for RouterBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterBudget")
            .field("max_cost_usd", &self.max_cost_usd)
            .field("max_tokens", &self.max_tokens)
            .field("spent_usd", &self.spent_usd())
            .field("tokens", &self.tokens())
            .field("trips", &self.trips())
            .finish()
    }
}

impl RouterBudget {
    /// Only a USD spend cap.
    pub fn with_cost_limit(max_cost_usd: f64) -> Self {
        Self::new(Some(max_cost_usd), None)
    }

    /// Only a cumulative token cap.
    pub fn with_token_limit(max_tokens: u64) -> Self {
        Self::new(None, Some(max_tokens))
    }

    /// Both a USD cap and a token cap; either one tripping skips the slot.
    pub fn with_cost_and_token_limits(max_cost_usd: f64, max_tokens: u64) -> Arc<Self> {
        Arc::new(Self::new(Some(max_cost_usd), Some(max_tokens)))
    }

    fn new(max_cost_usd: Option<f64>, max_tokens: Option<u64>) -> Self {
        Self {
            max_cost_usd,
            max_tokens,
            spent_bits: AtomicU64::new(0.0f64.to_bits()),
            tokens: AtomicUsize::new(0),
            trips: AtomicUsize::new(0),
        }
    }

    /// Configured USD cap, if any.
    pub fn max_cost_usd(&self) -> Option<f64> {
        self.max_cost_usd
    }

    /// Configured token cap, if any.
    pub fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }

    /// Cumulative recorded USD spend.
    pub fn spent_usd(&self) -> f64 {
        f64::from_bits(self.spent_bits.load(Ordering::Acquire))
    }

    /// Cumulative recorded token usage.
    pub fn tokens(&self) -> u64 {
        self.tokens.load(Ordering::Acquire) as u64
    }

    /// Number of times a precheck or a post-call record crossed a cap.
    pub fn trips(&self) -> usize {
        self.trips.load(Ordering::Acquire)
    }

    /// Whether either cap is already crossed.
    pub fn is_tripped(&self) -> bool {
        if let Some(limit) = self.max_cost_usd {
            if self.spent_usd() > limit + COST_EPSILON {
                return true;
            }
        }
        if let Some(limit) = self.max_tokens {
            if self.tokens() > limit {
                return true;
            }
        }
        false
    }

    /// Projects one pending call and returns the dimension that would be
    /// exceeded, if any.
    ///
    /// `projected_cost_usd` is the prompt priced against the slot's price
    /// (`0.0` for free/unpriced slots); `projected_tokens` is the prompt-token
    /// estimate. A projected **zero-cost** call always passes the *cost*
    /// dimension even while the breaker is tripped — that is what lets free
    /// fallbacks keep serving. The token dimension is price-agnostic.
    pub fn precheck(
        &self,
        projected_cost_usd: f64,
        projected_tokens: u64,
    ) -> Result<(), BudgetExceeded> {
        if let Some(limit) = self.max_cost_usd {
            if projected_cost_usd > 0.0
                && self.spent_usd() + projected_cost_usd > limit + COST_EPSILON
            {
                self.trips.fetch_add(1, Ordering::AcqRel);
                return Err(BudgetExceeded {
                    kind: BudgetKind::CostUsd,
                    used: self.spent_usd(),
                    limit,
                });
            }
        }
        if let Some(limit) = self.max_tokens {
            let used = self.tokens();
            if used + projected_tokens > limit {
                self.trips.fetch_add(1, Ordering::AcqRel);
                return Err(BudgetExceeded {
                    kind: BudgetKind::Tokens,
                    used: used as f64,
                    limit: limit as f64,
                });
            }
        }
        Ok(())
    }

    /// Records the measured cost/tokens of one finished call.
    ///
    /// Returns the first dimension this record pushed over its cap (the
    /// breaker latches — later prechecks keep skipping paid candidates).
    pub fn record(&self, cost_usd: f64, total_tokens: u64) -> Option<BudgetExceeded> {
        if cost_usd != 0.0 {
            let mut cur = self.spent_bits.load(Ordering::Acquire);
            loop {
                let next = f64::from_bits(cur) + cost_usd;
                match self.spent_bits.compare_exchange(
                    cur,
                    next.to_bits(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(actual) => cur = actual,
                }
            }
        }
        if total_tokens != 0 {
            self.tokens
                .fetch_add(total_tokens as usize, Ordering::AcqRel);
        }

        if let Some(limit) = self.max_cost_usd {
            if self.spent_usd() > limit + COST_EPSILON {
                self.trips.fetch_add(1, Ordering::AcqRel);
                return Some(BudgetExceeded {
                    kind: BudgetKind::CostUsd,
                    used: self.spent_usd(),
                    limit,
                });
            }
        }
        if let Some(limit) = self.max_tokens {
            if self.tokens() > limit {
                self.trips.fetch_add(1, Ordering::AcqRel);
                return Some(BudgetExceeded {
                    kind: BudgetKind::Tokens,
                    used: self.tokens() as f64,
                    limit: limit as f64,
                });
            }
        }
        None
    }

    /// Zeroes accumulated spend/tokens/trip count (start a new run reusing
    /// the same budget configuration).
    pub fn reset(&self) {
        self.spent_bits.store(0.0f64.to_bits(), Ordering::Release);
        self.tokens.store(0, Ordering::Release);
        self.trips.store(0, Ordering::Release);
    }
}

/// Snapshot carried by [`RouterError::BudgetExceeded`](super::RouterError::BudgetExceeded).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetExceeded {
    /// Which dimension tripped.
    pub kind: BudgetKind,
    /// Used amount at the moment of the trip (USD or tokens, per `kind`).
    pub used: f64,
    /// Configured limit in the same unit.
    pub limit: f64,
}

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} exceeded: used {:.6}, limit {:.6}",
            self.kind, self.used, self.limit
        )
    }
}

impl std::error::Error for BudgetExceeded {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_within_cost_cap_and_blocks_projected_overrun() {
        let b = RouterBudget::with_cost_limit(1.0);
        b.precheck(0.4, 10).unwrap();
        b.record(0.4, 10);
        assert_eq!(b.spent_usd(), 0.4);
        // 0.4 + 0.7 > 1.0 → blocked.
        let e = b.precheck(0.7, 0).unwrap_err();
        assert_eq!(e.kind, BudgetKind::CostUsd);
        assert_eq!(e.used, 0.4);
        assert_eq!(e.limit, 1.0);
        assert!(b.trips() >= 1);
    }

    #[test]
    fn free_call_passes_cost_dimension_even_when_tripped() {
        let b = RouterBudget::with_cost_limit(1.0);
        b.record(2.0, 0);
        assert!(b.is_tripped());
        // A zero-cost fallback must still be reachable.
        b.precheck(0.0, 0).unwrap();
        // But a paid call stays blocked.
        assert!(b.precheck(0.01, 0).is_err());
    }

    #[test]
    fn token_cap_counts_estimates_independent_of_price() {
        let b = RouterBudget::with_token_limit(100);
        b.precheck(0.0, 60).unwrap();
        b.record(0.0, 60);
        let e = b.precheck(0.0, 50).unwrap_err();
        assert_eq!(e.kind, BudgetKind::Tokens);
        assert_eq!(e.used, 60.0);
        assert_eq!(e.limit, 100.0);
    }

    #[test]
    fn record_latches_breaker_on_overshoot() {
        let b = RouterBudget::with_cost_limit(1.0);
        // Projected zero output cost, but the measured call overshoots.
        b.precheck(0.9, 0).unwrap();
        let trip = b.record(1.5, 100).expect("should trip on record");
        assert_eq!(trip.kind, BudgetKind::CostUsd);
        assert!(b.is_tripped());
    }

    #[test]
    fn concurrent_record_sums_without_losing_updates() {
        let b = Arc::new(RouterBudget::with_cost_limit(f64::INFINITY));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let b = b.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    b.record(0.001, 1);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!((b.spent_usd() - 8.0).abs() < 1e-9);
        assert_eq!(b.tokens(), 8000);
    }

    #[test]
    fn reset_clears_totals_and_trip() {
        let b = RouterBudget::with_cost_limit(1.0);
        b.record(2.0, 50);
        assert!(b.is_tripped());
        b.reset();
        assert!(!b.is_tripped());
        assert_eq!(b.spent_usd(), 0.0);
        assert_eq!(b.tokens(), 0);
        assert_eq!(b.trips(), 0);
    }
}
