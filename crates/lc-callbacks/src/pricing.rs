// lc-callbacks/src/pricing.rs
//! USD cost estimation for token usage (E3, v0.22.1 §S6).
//!
//! LLM providers bill per token, at rates that differ by model and by input-vs-output.
//! This module turns a `(prompt_tokens, completion_tokens, model)` triple into a USD
//! estimate, so traces and cost dashboards can answer "what did this agent run cost?"
//! without hardcoding floats in every caller.
//!
//! Rates are a static snapshot of the listed model prices (per 1M tokens) and are
//! deliberately approximate — list price, not the effective price after tiered volume
//! discounts or other negotiated terms. Unknown models yield `None` rather than a guess
//! (an invented rate is worse than no rate).
//!
//! ## Design
//!
//! [`price_for`] matches by model-name substring, longest/most-specific first, so
//! `claude-3-5-sonnet` resolves to its own rate instead of the `claude-3-sonnet` catch.
//! The table lives as an ordered slice so a single `find` returns the right entry.

/// USD price of a model, per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrice {
    /// USD per 1M input (prompt) tokens.
    pub input_usd_per_1m: f64,
    /// USD per 1M output (completion) tokens.
    pub output_usd_per_1m: f64,
}

/// `(model substring, price)`. Ordered most-specific first so `find` picks the best match.
const PRICE_TABLE: &[(&str, ModelPrice)] = &[
    ("claude-3-5-sonnet", ModelPrice { input_usd_per_1m: 3.0, output_usd_per_1m: 15.0 }),
    ("claude-3-5-haiku", ModelPrice { input_usd_per_1m: 0.80, output_usd_per_1m: 4.0 }),
    ("claude-3-opus", ModelPrice { input_usd_per_1m: 15.0, output_usd_per_1m: 75.0 }),
    ("claude-3-sonnet", ModelPrice { input_usd_per_1m: 3.0, output_usd_per_1m: 15.0 }),
    ("claude-3-haiku", ModelPrice { input_usd_per_1m: 0.25, output_usd_per_1m: 1.25 }),
    // generic claude fallback last among claude entries
    ("claude", ModelPrice { input_usd_per_1m: 3.0, output_usd_per_1m: 15.0 }),
    ("gpt-4o-mini", ModelPrice { input_usd_per_1m: 0.15, output_usd_per_1m: 0.60 }),
    ("gpt-4o", ModelPrice { input_usd_per_1m: 2.50, output_usd_per_1m: 10.0 }),
    ("gpt-4-turbo", ModelPrice { input_usd_per_1m: 10.0, output_usd_per_1m: 30.0 }),
    ("gpt-4", ModelPrice { input_usd_per_1m: 30.0, output_usd_per_1m: 60.0 }),
    ("gpt-3.5-turbo", ModelPrice { input_usd_per_1m: 0.50, output_usd_per_1m: 1.50 }),
    ("deepseek-reasoner", ModelPrice { input_usd_per_1m: 0.55, output_usd_per_1m: 2.19 }),
    ("deepseek-chat", ModelPrice { input_usd_per_1m: 0.27, output_usd_per_1m: 1.10 }),
    ("qwen", ModelPrice { input_usd_per_1m: 0.50, output_usd_per_1m: 2.0 }),
];

/// Looks up the list price for a model name by most-specific substring match.
///
/// Returns `None` for unknown models (the caller should treat an unknown rate as unknown
/// cost, not mint one).
pub fn price_for(model: &str) -> Option<ModelPrice> {
    let lower = model.to_ascii_lowercase();
    PRICE_TABLE
        .iter()
        .find(|(pattern, _)| lower.contains(pattern))
        .map(|(_, price)| *price)
}

/// Estimates the USD cost of a call from its token counts and model name.
///
/// `None` when the model is not in the price table.
pub fn estimate_cost_usd(prompt_tokens: usize, completion_tokens: usize, model: &str) -> Option<f64> {
    let price = price_for(model)?;
    Some(
        prompt_tokens as f64 / 1e6 * price.input_usd_per_1m
            + completion_tokens as f64 / 1e6 * price.output_usd_per_1m,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specific_beats_generic_claude() {
        let opus = price_for("claude-3-opus-20240229").unwrap();
        assert_eq!(opus.input_usd_per_1m, 15.0);
        // claude-3-sonnet must NOT match the generic "claude" fallback
        let sonnet = price_for("claude-3-sonnet-20240229").unwrap();
        assert_eq!(sonnet.input_usd_per_1m, 3.0);
    }

    #[test]
    fn gpt_4o_mini_beats_gpt_4o() {
        let mini = price_for("gpt-4o-mini").unwrap();
        assert_eq!(mini.input_usd_per_1m, 0.15);
        let full = price_for("gpt-4o").unwrap();
        assert_eq!(full.input_usd_per_1m, 2.50);
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(price_for("some-future-model").is_none());
        assert!(estimate_cost_usd(10, 10, "unknown").is_none());
    }

    #[test]
    fn estimate_is_per_1m_scaled() {
        // 1M input + 1M output vs claude-3-5-sonnet (3 / 15)
        let cost = estimate_cost_usd(1_000_000, 1_000_000, "claude-3-5-sonnet").unwrap();
        assert!((cost - 18.0).abs() < 1e-9, "got {cost}");
        // 1k tokens each
        let small = estimate_cost_usd(1000, 1000, "claude-3-5-sonnet").unwrap();
        assert!((small - 0.018).abs() < 1e-9, "got {small}");
    }
}