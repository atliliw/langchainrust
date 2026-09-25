//! E1 — eval cost as a first-class citizen (v0.22.1 §S7).
//!
//! Evaluation runs burn tokens through the predictor (and any LLM-as-judge). This module gives
//! the eval report a real cost ledger: a per-`TokenUsage` meter plus a configurable USD `PriceBook`
//! whose [`PriceBook::estimate_cost`] is a pure function of token counts and a model name.
//!
//! Design notes:
//! - Rates are a static snapshot of list prices (per 1M tokens), deliberately approximate.
//! - `PriceBook` matches model names **exactly** (not substring) — an eval run configures the
//!   book it expects, so an unknown model means "I don't know this price", returning `None`
//!   rather than a guess.
//! - `TokenUsage` is additive: a predictor that has no token metering reports `None` and the
//!   report simply carries a zero/None cost ledger (zero behavior change).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// USD price of a model, per 1M tokens.
///
/// Values are **dollars** (USD) per 1M tokens, e.g. `input_per_1m = 3.0` means $3.00 per 1M
/// input tokens. `f64` (not `u64` cents) so fractional prices like gpt-4o-mini's $0.15 are
/// lossless and no `/100` re-scaling is needed at estimation time. `PartialEq` only (no `Eq`,
/// because `f64` does not implement `Eq`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    /// USD per 1M input (prompt) tokens.
    pub input_per_1m: f64,
    /// USD per 1M output (completion) tokens.
    pub output_per_1m: f64,
}

/// A token-usage report from one predictor step (an eval input).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    /// Prompt tokens consumed.
    pub prompt_tokens: usize,
    /// Completion tokens consumed.
    pub completion_tokens: usize,
    /// Model that produced the prediction, for price lookup. `None` when unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl TokenUsage {
    /// Total tokens consumed.
    pub fn total(&self) -> usize {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// Configurable USD price book for eval-run cost estimation.
#[derive(Debug, Clone, Default)]
pub struct PriceBook {
    rates: HashMap<String, Price>,
}

impl PriceBook {
    /// A price book with a few common models preloaded (approximate list prices, USD per 1M).
    pub fn default_set() -> Self {
        let mut book = Self::default();
        book.set(
            "claude-3-5-sonnet",
            Price {
                input_per_1m: 3.0,
                output_per_1m: 15.0,
            },
        );
        book.set(
            "claude-3-5-haiku",
            Price {
                input_per_1m: 0.8,
                output_per_1m: 4.0,
            },
        );
        book.set(
            "gpt-4o-mini",
            Price {
                input_per_1m: 0.15,
                output_per_1m: 0.6,
            },
        );
        book.set(
            "gpt-4o",
            Price {
                input_per_1m: 2.5,
                output_per_1m: 10.0,
            },
        );
        book.set(
            "deepseek-chat",
            Price {
                input_per_1m: 0.27,
                output_per_1m: 1.1,
            },
        );
        book
    }

    /// Sets (or overrides) the price for a model. Prices are in **USD per 1M tokens**: a price of
    /// `3.0` means $3.00 per 1M input tokens; `0.8` means $0.80 per 1M.
    pub fn set(&mut self, model: impl Into<String>, price: Price) {
        self.rates.insert(model.into(), price);
    }

    /// Looks up an exact model price.
    pub fn get(&self, model: &str) -> Option<Price> {
        self.rates.get(model).copied()
    }

    /// Estimates the USD cost of a usage report; `None` when the model isn't in the book.
    ///
    /// Pure function over the book + usage, so it is trivially unit-testable.
    pub fn estimate(&self, usage: &TokenUsage) -> Option<f64> {
        let model = usage.model.as_deref()?;
        self.estimate_cost(usage.prompt_tokens, usage.completion_tokens, model)
    }

    /// Estimates the USD cost of token counts for a named model.
    ///
    /// Direct USD per-1M math — rates are already dollars, so no cents `/100` re-scaling:
    /// `prompt / 1e6 * input_price + completion / 1e6 * output_price`. Returns `None` when the
    /// model is not in the book.
    pub fn estimate_cost(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        model: &str,
    ) -> Option<f64> {
        let p = self.rates.get(model)?;
        let usd = prompt_tokens as f64 / 1e6 * p.input_per_1m
            + completion_tokens as f64 / 1e6 * p.output_per_1m;
        Some(usd)
    }
}

/// Cost ledger carried on an eval `Report` (E1).
///
/// Token buckets accumulate every reported predictor usage; `cost_usd` is `Some` only when at
/// least one usage had a model that the configured price book knows. Default is all-zero + `None`
/// so old reports serialize compatibly under `#[serde(default)]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverallCost {
    /// Total prompt tokens across the run.
    pub prompt_tokens: usize,
    /// Total completion tokens across the run.
    pub completion_tokens: usize,
    /// Total tokens (prompt + completion).
    pub total_tokens: usize,
    /// Estimated USD cost of the run; `None` when no priced usage was reported.
    pub cost_usd: Option<f64>,
}

impl OverallCost {
    /// Adds a usage report, accumulating token totals and (when priced) the USD estimate.
    pub fn accumulate(&mut self, usage: &TokenUsage, book: &PriceBook) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(usage.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total());
        if let Some(usd) = book.estimate(usage) {
            self.cost_usd = Some(self.cost_usd.unwrap_or(0.0) + usd);
        }
    }
}

/// I3: per-evaluator accumulator for judge/tool LLM token usage that does not round-trip through
/// the `Evaluator` trait's `eval` return type.
///
/// A judge evaluator records each LLM call's usage on its shared ledger while scoring; the runner
/// then drains the ledger once per `eval` into `Report.cost` via
/// [`Evaluator::report_token_usage`](crate::Evaluator::report_token_usage). Interior-mutable so
/// concurrent judge calls (e.g. `buffered` contexts/claims) can add to it from `&self`.
///
/// Draining is destructive: each recorded usage is attributed exactly once (the runner drains
/// after every eval), so a failed judge call between drains is not double-counted across examples.
#[derive(Debug, Clone, Default)]
pub(crate) struct UsageLedger {
    total: Arc<Mutex<TokenUsage>>,
}

impl UsageLedger {
    /// Adds one judge/tool call's usage. `model` names the judge model so the price book can
    /// price it; a `None` usage (model did not report) records nothing.
    pub(crate) fn record(&self, usage: Option<lc_core::TokenUsage>, model: &str) {
        let Some(u) = usage else { return };
        if let Ok(mut t) = self.total.lock() {
            t.prompt_tokens = t.prompt_tokens.saturating_add(u.prompt_tokens);
            t.completion_tokens = t.completion_tokens.saturating_add(u.completion_tokens);
            // the last recorded non-empty model wins (all calls share the same judge here)
            if t.model.is_none() {
                t.model = Some(model.to_string());
            }
        }
    }

    /// Takes the accumulated usage, resetting the ledger to zero for the next eval.
    pub(crate) fn drain(&self) -> Option<TokenUsage> {
        let Ok(mut t) = self.total.lock() else {
            return None;
        };
        let drained = std::mem::take(&mut *t);
        if drained.prompt_tokens == 0 && drained.completion_tokens == 0 {
            None
        } else {
            Some(drained)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cheap_book() -> PriceBook {
        let mut b = PriceBook::default();
        b.set(
            "test-model",
            Price {
                input_per_1m: 3.0,
                output_per_1m: 15.0,
            },
        ); // $3.00 / $15.00 per 1M = USD dollars now
        b
    }

    #[test]
    fn estimate_is_pure_and_per_1m_scaled() {
        let b = cheap_book();
        // 1M in + 1M out on the $3/$15 model = $18.00
        let usd = b.estimate_cost(1_000_000, 1_000_000, "test-model").unwrap();
        assert!((usd - 18.0).abs() < 1e-9, "got {usd}");
        // 1k each = $0.018
        let small = b.estimate_cost(1000, 1000, "test-model").unwrap();
        assert!((small - 0.018).abs() < 1e-9, "got {small}");
    }

    #[test]
    fn unknown_model_yields_none_not_a_guess() {
        let b = cheap_book();
        assert!(b.estimate_cost(1, 1, "mystery-model").is_none());
        assert!(b.get("nonexistent").is_none());
    }

    #[test]
    fn exact_match_not_substring() {
        let mut b = cheap_book();
        b.set(
            "gpt-4o",
            Price {
                input_per_1m: 2.5,
                output_per_1m: 10.0,
            },
        );
        // "gpt-4o" and "gpt-4o-mini" are distinct entries; no substring matching
        assert!(b.get("gpt-4o-mini").is_none());
        assert!(b.get("gpt-4o").is_some());
    }

    #[test]
    fn accumulate_totals_and_priced_usd() {
        let b = cheap_book();
        let mut cost = OverallCost::default();
        cost.accumulate(
            &TokenUsage {
                prompt_tokens: 1000,
                completion_tokens: 1000,
                model: Some("test-model".into()),
            },
            &b,
        );
        cost.accumulate(
            &TokenUsage {
                prompt_tokens: 500,
                completion_tokens: 0,
                model: None,
            },
            &b,
        );
        assert_eq!(cost.prompt_tokens, 1500);
        assert_eq!(cost.completion_tokens, 1000);
        assert_eq!(cost.total_tokens, 2500);
        // only the priced usage contributes USD
        assert!(
            (cost.cost_usd.unwrap() - 0.018).abs() < 1e-9,
            "got {:?}",
            cost.cost_usd
        );
    }

    #[test]
    fn accumulate_without_any_priced_usage_keeps_cost_none() {
        let b = cheap_book();
        let mut cost = OverallCost::default();
        cost.accumulate(
            &TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
                model: None,
            },
            &b,
        );
        assert_eq!(cost.total_tokens, 20);
        assert!(cost.cost_usd.is_none());
    }

    #[test]
    fn default_set_has_common_models() {
        let b = PriceBook::default_set();
        assert!(b.get("gpt-4o-mini").is_some());
        assert!(b.get("deepseek-chat").is_some());
        // 1M in + 1M out on gpt-4o-mini: $0.15 + $0.60 = $0.75
        let usd = b
            .estimate_cost(1_000_000, 1_000_000, "gpt-4o-mini")
            .unwrap();
        assert!((usd - 0.75).abs() < 1e-9, "got {usd}");
    }

    /// B6: default_set sonnet cost is real USD dollars (not cents): $3 in + $15 out per 1M,
    /// and no `/100` re-scaling at estimate time.
    #[test]
    fn default_set_sonnet_is_usd_dollars_not_cents() {
        let b = PriceBook::default_set();
        let p = b.get("claude-3-5-sonnet").expect("sonnet in default set");
        // USD dollars per 1M: $3 input, $15 output (previously mispriced as cents there).
        assert!(
            (p.input_per_1m - 3.0).abs() < 1e-9,
            "got {:?}",
            p.input_per_1m
        );
        assert!(
            (p.output_per_1m - 15.0).abs() < 1e-9,
            "got {:?}",
            p.output_per_1m
        );

        // 1M in + 1M out = $3 + $15 = $18 (no /100 → previously $0.18).
        let usd = b
            .estimate_cost(1_000_000, 1_000_000, "claude-3-5-sonnet")
            .unwrap();
        assert!((usd - 18.0).abs() < 1e-9, "got {usd}");

        // 1k each on sonnet = $0.003 + $0.015 = $0.018 (dollars, not cents).
        let small = b.estimate_cost(1000, 1000, "claude-3-5-sonnet").unwrap();
        assert!((small - 0.018).abs() < 1e-9, "got {small}");
    }
}
