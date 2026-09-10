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

use serde::{Deserialize, Serialize};

/// USD price of a model, per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Price {
    /// USD per 1M input (prompt) tokens.
    pub input_per_1m: u64,
    /// USD per 1M output (completion) tokens.
    pub output_per_1m: u64,
}

/// A token-usage report from one predictor step (an eval input).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    /// A price book with a few common models preloaded (approximate list prices).
    pub fn default_set() -> Self {
        let mut book = Self::default();
        book.set(
            "claude-3-5-sonnet",
            Price {
                input_per_1m: 3,
                output_per_1m: 15,
            },
        );
        book.set(
            "claude-3-5-haiku",
            Price {
                input_per_1m: 80,
                output_per_1m: 400,
            },
        );
        book.set(
            "gpt-4o-mini",
            Price {
                input_per_1m: 15,
                output_per_1m: 60,
            },
        );
        book.set(
            "gpt-4o",
            Price {
                input_per_1m: 250,
                output_per_1m: 1000,
            },
        );
        book.set(
            "deepseek-chat",
            Price {
                input_per_1m: 27,
                output_per_1m: 110,
            },
        );
        book
    }

    /// Sets (or overrides) the price for a model. Prices are in "units per 1M": a price of `3`
    /// means $3.00 per 1M input tokens.
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
    /// Integer cents-per-1M math: `prompt / 1e6 * input_price/100`. Returns `None` when the
    /// model is not in the book.
    pub fn estimate_cost(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        model: &str,
    ) -> Option<f64> {
        let p = self.rates.get(model)?;
        let usd = prompt_tokens as f64 / 1e6 * (p.input_per_1m as f64 / 100.0)
            + completion_tokens as f64 / 1e6 * (p.output_per_1m as f64 / 100.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cheap_book() -> PriceBook {
        let mut b = PriceBook::default();
        b.set(
            "test-model",
            Price {
                input_per_1m: 300,
                output_per_1m: 1500,
            },
        ); // $3.00 / $15.00
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
                input_per_1m: 250,
                output_per_1m: 1000,
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
}
