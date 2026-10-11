//! Span sampling (T2 §5: span 采样率与限流，按 run_type/name).
//!
//! A [`Sampler`] decides whether to publish a span, letting high-volume streams
//! (every retriever hit, every token-heavy trace) be thinned to a configurable
//! rate without changing the `Tracer` API. It supports:
//!
//! - a global [`rate`](Sampler::with_rate) in `0.0..=1.0` (default `1.0` = sample
//!   everything);
//! - an optional per-kind override ([`with_kind_rate`](Sampler::with_kind_rate))
//!   so, say, retriever spans can be sampled at 1/10th while `llm` stays full;
//! - a deterministic seeded PRNG (xorshift64*), injectable via
//!   [`with_seed`](Sampler::with_seed) so tests and reproducible repros need no
//!   wall-clock randomness.
//!
//! The decision is made once, at span *start*: a sampled-out span is a no-op guard
//! that never reaches the backend, so cost/aggregate math sees a clean tree.

use std::collections::HashMap;

use super::SpanKind;

/// Reusable sampling state for a [`Tracer`](super::Tracer).
///
/// Interior mutability is provided by the harness in the tracer (`Mutex<Sampler>`),
/// so the sampler itself holds no lock. Call [`should_sample`](Sampler::should_sample)
/// under whichever lock guards the decision.
#[derive(Debug, Clone)]
pub struct Sampler {
    rate: f64,
    /// Per-kind overrides keyed by `SpanKind`'s `Display` representation
    /// (`"llm"`, `"chain"`, `"tool"`, `"retriever"`, `"agent"`, `"custom:<name>"`).
    kind_rates: HashMap<String, f64>,
    /// xorshift64* state; advanced per decision for an independent draws.
    state: u64,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::with_rate(1.0)
    }
}

impl Sampler {
    /// A sampler that keeps `rate` of spans (`0.0` drops everything, `1.0` keeps
    /// everything). Seeded from `SystemTime` — deterministic for `rate` of 0 or 1,
    /// not for fractional rates (use [`with_seed`](Sampler::with_seed) for those).
    pub fn with_rate(rate: f64) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        Self {
            rate: rate.clamp(0.0, 1.0),
            kind_rates: HashMap::new(),
            state: seed.max(1),
        }
    }

    /// Deterministic sampler for tests and reproducible repros.
    pub fn with_seed(rate: f64, seed: u64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
            kind_rates: HashMap::new(),
            state: seed.max(1),
        }
    }

    /// Override the sampling rate for one span kind. `None`-worth overrides
    /// accumulate; a kind not present falls back to the global rate.
    pub fn with_kind_rate(mut self, kind: SpanKind, rate: f64) -> Self {
        self.kind_rates
            .insert(kind.to_string(), rate.clamp(0.0, 1.0));
        self
    }

    /// The effective rate for a span (kind override if present, else global).
    pub fn rate_for(&self, kind: &SpanKind) -> f64 {
        self.kind_rates
            .get(&kind.to_string())
            .copied()
            .unwrap_or(self.rate)
    }

    /// Decide whether a span of `kind` named `name` should be published.
    ///
    /// Edge rates (0.0 / 1.0) short-circuit so they cost no RNG draw and stay
    /// perfectly deterministic.
    pub fn should_sample(&mut self, kind: &SpanKind, _name: &str) -> bool {
        let rate = self.rate_for(kind);
        if rate <= 0.0 {
            false
        } else if rate >= 1.0 {
            true
        } else {
            self.next_f64() <= rate
        }
    }

    /// Next uniform `[0,1)` draw from the xorshift64* stream.
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let rand_u64 = (x as u128 * 0x2545_F491_4F6C_DD1D_u128) & u64::MAX as u128;
        (rand_u64 as u64 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLM: SpanKind = SpanKind::Llm;
    const RETRIEVER: SpanKind = SpanKind::Retriever;

    #[test]
    fn rate_zero_and_one_are_deterministic() {
        let mut all = Sampler::with_rate(1.0);
        for i in 0..64 {
            assert!(all.should_sample(&LLM, &format!("span{i}")));
        }
        let mut none = Sampler::with_rate(0.0);
        for i in 0..64 {
            assert!(!none.should_sample(&LLM, &format!("span{i}")));
        }
    }

    #[test]
    fn fractional_rate_keeps_a_mix_and_respects_bound() {
        let mut s = Sampler::with_seed(0.5, 42);
        let mut kept = 0;
        let n = 10_000;
        for i in 0..n {
            if s.should_sample(&LLM, &format!("s{i}")) {
                kept += 1;
            }
        }
        // ~5000 ± a wide margin; this only disproves "always" / "never".
        assert!((3_000..=7_000).contains(&kept), "kept={kept}");
    }

    #[test]
    fn kind_override_can_thin_one_kind_while_others_stay_full() {
        // Retriever at 10%, LLM at 100%.
        let mut s = Sampler::with_seed(0.1, 7).with_kind_rate(SpanKind::Llm, 1.0);
        let mut ret_kept = 0;
        let n = 5_000;
        for i in 0..n {
            if s.should_sample(&RETRIEVER, &format!("r{i}")) {
                ret_kept += 1;
            }
        }
        assert!((100..=1200).contains(&ret_kept), "ret_kept={ret_kept}");
        for i in 0..64 {
            assert!(
                s.should_sample(&LLM, &format!("llm{i}")),
                "llm must be unsampled"
            );
        }
    }

    #[test]
    fn rate_for_falls_back_to_global() {
        let s = Sampler::with_seed(0.3, 1).with_kind_rate(RETRIEVER, 0.0);
        assert_eq!(s.rate_for(&LLM), 0.3);
        assert_eq!(s.rate_for(&RETRIEVER), 0.0);
    }

    #[test]
    fn seed_makes_the_stream_reproducible() {
        let mut a = Sampler::with_seed(0.5, 1234);
        let mut b = Sampler::with_seed(0.5, 1234);
        for i in 0..100 {
            assert_eq!(
                a.should_sample(&LLM, &format!("d{i}")),
                b.should_sample(&LLM, &format!("d{i}")),
                "same seed must produce the same decision sequence"
            );
        }
    }
}
