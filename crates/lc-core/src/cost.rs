//! Unified cost tracking (B3, 0.22.4).
//!
//! Per provider/model pricing, cumulative USD spend aggregation per run or
//! session, and hard budget enforcement. A single [`CostTracker`] is shared
//! (`Arc`) across every LLM call of a run — or across multiple runs of one
//! session — and [`CostTracker::record`] prices each call through its
//! [`PricingTable`], aggregates the totals, and optionally emits a
//! [`crate::observability::ObsEvent::Cost`] record.
//!
//! Design notes:
//! - Prices are quoted **USD per 1,000 tokens** (matching the legacy
//!   [`crate::token_counter::ModelPricing`]); zero is a valid price (local /
//!   OSS models).
//! - Calls to models absent from the table are still counted (calls/tokens)
//!   but priced at 0.0 — missing pricing data degrades to usage-only tracking,
//!   never a hard error, so attaching a tracker cannot break the agent loop.
//! - The tracker only *measures*. The hard stop lives next to the existing
//!   budget gates (`lc-agents::executor::BudgetConfig::max_cost_usd`), keeping
//!   enforcement policy out of the measurement primitive.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::language_models::TokenUsage;
use crate::observability::{MetricsSink, ObsEvent};

/// Price of one model, in USD per 1,000 tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    /// USD per 1,000 **input** (prompt) tokens.
    pub input_per_1k: f64,
    /// USD per 1,000 **output** (completion) tokens.
    pub output_per_1k: f64,
}

impl ModelPrice {
    /// Creates a price entry.
    pub fn new(input_per_1k: f64, output_per_1k: f64) -> Self {
        Self {
            input_per_1k,
            output_per_1k,
        }
    }

    /// Free entry (local / open-source / self-hosted model).
    pub fn free() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Prices one call. Pure function — the core of the whole tracker.
    pub fn cost_of(&self, prompt_tokens: usize, completion_tokens: usize) -> f64 {
        (prompt_tokens as f64 / 1000.0) * self.input_per_1k
            + (completion_tokens as f64 / 1000.0) * self.output_per_1k
    }

    /// Single comparable cost figure for routing weights, assuming a
    /// representative 3:1 prompt:completion traffic mix (75% input, 25%
    /// output). Callers that know their real mix should price calls directly.
    pub fn blended_per_1k(&self) -> f64 {
        0.75 * self.input_per_1k + 0.25 * self.output_per_1k
    }
}

/// Provider/model pricing table.
///
/// Lookup keys on `(provider, model)`; entries whose provider is `None` act as
/// model-only fallbacks (matched when no provider-qualified entry exists).
#[derive(Debug, Clone, Default)]
pub struct PricingTable {
    qualified: HashMap<(String, String), ModelPrice>,
    model_only: HashMap<String, ModelPrice>,
}

impl PricingTable {
    /// Empty table (every call prices at zero).
    pub fn new() -> Self {
        Self::default()
    }

    /// Built-in snapshot of common models (USD/1K, as of 2026-09).
    ///
    /// This is a convenience seed, not a maintained source of truth: provider
    /// prices change frequently. Fetch a current [`crate::model_registry::ModelRegistry`]
    /// remotely and convert it with [`PricingTable::from_registry`] for
    /// production accounting.
    pub fn builtin() -> Self {
        let mut t = Self::new();
        for (provider, model, input, output) in [
            ("openai", "gpt-4o", 2.5, 10.0),
            ("openai", "gpt-4o-mini", 0.15, 0.60),
            ("openai", "gpt-4.1", 2.0, 8.0),
            ("openai", "gpt-4.1-mini", 0.40, 1.60),
            ("openai", "o4-mini", 1.10, 4.40),
            ("anthropic", "claude-3-5-sonnet-latest", 3.0, 15.0),
            ("anthropic", "claude-3-5-haiku-latest", 0.80, 4.0),
            ("google", "gemini-1.5-pro", 1.25, 5.0),
            ("google", "gemini-1.5-flash", 0.075, 0.30),
            ("groq", "llama-3.3-70b-versatile", 0.59, 0.79),
            ("groq", "llama-3.1-8b-instant", 0.05, 0.08),
            ("deepseek", "deepseek-chat", 0.27, 1.10),
        ] {
            t.insert(Some(provider), model, ModelPrice::new(input, output));
        }
        t
    }

    /// Adds/replaces a qualified entry; returns the table for chaining.
    pub fn with(
        mut self,
        provider: impl Into<String>,
        model: impl Into<String>,
        price: ModelPrice,
    ) -> Self {
        self.insert(Some(provider), model, price);
        self
    }

    /// Adds/replaces a model-only fallback entry.
    pub fn with_model_only(mut self, model: impl Into<String>, price: ModelPrice) -> Self {
        self.insert(Option::<&str>::None, model, price);
        self
    }

    /// Inserts an entry. `provider = None` registers a model-only fallback.
    pub fn insert(
        &mut self,
        provider: Option<impl Into<String>>,
        model: impl Into<String>,
        price: ModelPrice,
    ) {
        let model = model.into();
        match provider {
            Some(provider) => {
                self.qualified.insert((provider.into(), model), price);
            }
            None => {
                self.model_only.insert(model, price);
            }
        }
    }

    /// Lookup: provider-qualified first, then the model-only fallback.
    pub fn get(&self, provider: Option<&str>, model: &str) -> Option<&ModelPrice> {
        if let Some(provider) = provider {
            if let Some(price) = self
                .qualified
                .get(&(provider.to_string(), model.to_string()))
            {
                return Some(price);
            }
        }
        self.model_only.get(model)
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.qualified.len() + self.model_only.len()
    }

    /// Whether no entry is registered.
    pub fn is_empty(&self) -> bool {
        self.qualified.is_empty() && self.model_only.is_empty()
    }

    /// Builds a table from a model registry (qualified entries only).
    pub fn from_registry(registry: &crate::model_registry::ModelRegistry) -> Self {
        let mut table = Self::new();
        for info in registry.models() {
            table.insert(Some(info.provider.clone()), info.id.clone(), info.price);
        }
        table
    }
}

/// One priced LLM call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostRecord {
    /// Provider slug (`"openai"`, `"anthropic"`, ...); `None` when the caller
    /// did not declare one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model id as reported by the model.
    pub model: String,
    /// Prompt tokens of the call.
    pub prompt_tokens: usize,
    /// Completion tokens of the call.
    pub completion_tokens: usize,
    /// Priced USD cost (0.0 when the table has no entry).
    pub cost_usd: f64,
}

/// Aggregated spend of one `(provider, model)` key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSpend {
    /// Number of calls.
    pub calls: usize,
    /// Cumulative prompt tokens.
    pub prompt_tokens: usize,
    /// Cumulative completion tokens.
    pub completion_tokens: usize,
    /// Cumulative USD spend.
    pub cost_usd: f64,
}

/// Point-in-time aggregate report of a [`CostTracker`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostReport {
    /// Optional run/session label the tracker was scoped with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Total calls recorded.
    pub calls: usize,
    /// Total prompt tokens.
    pub prompt_tokens: usize,
    /// Total completion tokens.
    pub completion_tokens: usize,
    /// Total USD spend across all models.
    pub total_cost_usd: f64,
    /// Per-model breakdown keyed `"<provider>/<model>"` (or `"<model>"` when no
    /// provider was declared).
    pub by_model: HashMap<String, ModelSpend>,
}

#[derive(Default)]
struct Inner {
    calls: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    total_cost_usd: f64,
    by_model: HashMap<String, ModelSpend>,
    records: Vec<CostRecord>,
}

/// Thread-safe cumulative cost tracker.
///
/// Construct once per run (or share one per session), attach it to a
/// [`crate::token_counter::TokenTrackingLLM`] with
/// `with_cost_tracker`, optionally share the same `Arc` with an
/// `AgentExecutor::with_cost_tracker` for hard budget enforcement.
pub struct CostTracker {
    table: Arc<PricingTable>,
    scope: Option<String>,
    sink: Option<Arc<dyn MetricsSink>>,
    inner: Mutex<Inner>,
}

impl CostTracker {
    /// Tracker over the given pricing table.
    pub fn new(table: impl Into<Arc<PricingTable>>) -> Self {
        Self {
            table: table.into(),
            scope: None,
            sink: None,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Tracker over the built-in price snapshot.
    pub fn with_builtin_prices() -> Self {
        Self::new(Arc::new(PricingTable::builtin()))
    }

    /// Labels this tracker with a run/session id (carried in reports and
    /// emitted `cost` events).
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Attaches an observability sink: every `record` emits one
    /// [`ObsEvent::Cost`]. Sink failures are warned, never propagated.
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// The pricing table backing this tracker.
    pub fn pricing(&self) -> &PricingTable {
        &self.table
    }

    /// Records one LLM call and returns its USD cost.
    ///
    /// Models absent from the table price at 0.0 but still count toward token /
    /// call aggregates. The observability event (when attached) is exported
    /// *after* aggregation; export failure only logs a warning.
    pub async fn record(
        &self,
        provider: Option<&str>,
        model: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> f64 {
        let cost = self
            .table
            .get(provider, model)
            .map(|p| p.cost_of(prompt_tokens, completion_tokens))
            .unwrap_or(0.0);

        let record = CostRecord {
            provider: provider.map(str::to_string),
            model: model.to_string(),
            prompt_tokens,
            completion_tokens,
            cost_usd: cost,
        };
        let key = match provider {
            Some(provider) => format!("{provider}/{model}"),
            None => model.to_string(),
        };

        {
            let mut inner = self.inner.lock().await;
            inner.calls += 1;
            inner.prompt_tokens += prompt_tokens;
            inner.completion_tokens += completion_tokens;
            inner.total_cost_usd += cost;
            let entry = inner.by_model.entry(key).or_default();
            entry.calls += 1;
            entry.prompt_tokens += prompt_tokens;
            entry.completion_tokens += completion_tokens;
            entry.cost_usd += cost;
            inner.records.push(record.clone());
        }

        if let Some(sink) = &self.sink {
            let evt = ObsEvent::Cost(crate::observability::CostEvent {
                scope: self.scope.clone(),
                provider: provider.map(str::to_string),
                model: model.to_string(),
                prompt_tokens,
                completion_tokens,
                cost_usd: cost,
            });
            if let Err(e) = sink.export(&evt).await {
                log::warn!(target: "lc_core::cost", "cost event export failed: {e}");
            }
        }

        cost
    }

    /// Records one call from a provider-annotated [`TokenUsage`].
    pub async fn record_usage(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &TokenUsage,
    ) -> f64 {
        self.record(
            provider,
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        .await
    }

    /// Cumulative USD spend (the value the budget gate reads).
    pub async fn total_cost_usd(&self) -> f64 {
        self.inner.lock().await.total_cost_usd
    }

    /// Snapshot report of everything recorded so far.
    pub async fn report(&self) -> CostReport {
        let inner = self.inner.lock().await;
        CostReport {
            scope: self.scope.clone(),
            calls: inner.calls,
            prompt_tokens: inner.prompt_tokens,
            completion_tokens: inner.completion_tokens,
            total_cost_usd: inner.total_cost_usd,
            by_model: inner.by_model.clone(),
        }
    }

    /// Every individual call recorded so far (oldest first).
    pub async fn records(&self) -> Vec<CostRecord> {
        self.inner.lock().await.records.clone()
    }

    /// Resets all accumulation (e.g. start a new run while keeping the same
    /// session-scoped tracker).
    pub async fn reset(&self) {
        *self.inner.lock().await = Inner::default();
    }
}

/// Errors from loading a price/model catalog (local JSON or remote fetch).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CostError {
    /// Transport/HTTP failure while fetching a remote catalog.
    #[error("cost catalog fetch failed: {0}")]
    Fetch(String),
    /// JSON payload did not match the catalog schema.
    #[error("cost catalog payload invalid: {0}")]
    Payload(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn price_calculation_is_exact() {
        let p = ModelPrice::new(2.0, 8.0);
        // 500 in * 2/1k + 250 out * 8/1k = 1.0 + 2.0 = 3.0
        assert_eq!(p.cost_of(500, 250), 3.0);
        assert_eq!(p.cost_of(0, 0), 0.0);
        // 1k/1k at full price
        assert_eq!(p.cost_of(1000, 1000), 10.0);
    }

    #[test]
    fn free_prices_remain_zero() {
        assert_eq!(ModelPrice::free().cost_of(10_000, 10_000), 0.0);
    }

    #[test]
    fn blended_mix_weights_input_three_quarters() {
        // 0.75*4 + 0.25*8 = 3 + 2 = 5
        assert_eq!(ModelPrice::new(4.0, 8.0).blended_per_1k(), 5.0);
    }

    #[test]
    fn table_qualified_entry_shadows_model_only() {
        let table = PricingTable::new()
            .with("openai", "gpt-x", ModelPrice::new(1.0, 2.0))
            .with_model_only("gpt-x", ModelPrice::new(9.0, 9.0));
        assert_eq!(
            table.get(Some("openai"), "gpt-x"),
            Some(&ModelPrice::new(1.0, 2.0))
        );
        // A different provider falls back to the model-only entry.
        assert_eq!(
            table.get(Some("proxy"), "gpt-x"),
            Some(&ModelPrice::new(9.0, 9.0))
        );
        // No provider at all also uses model-only.
        assert_eq!(table.get(None, "gpt-x"), Some(&ModelPrice::new(9.0, 9.0)));
        assert_eq!(table.get(Some("openai"), "missing"), None);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn builtin_table_covers_seeded_models() {
        let table = PricingTable::builtin();
        assert!(table.len() >= 10);
        assert_eq!(
            table.get(Some("openai"), "gpt-4o-mini"),
            Some(&ModelPrice::new(0.15, 0.60))
        );
    }

    #[tokio::test]
    async fn tracker_aggregates_per_model_and_total() {
        let tracker = CostTracker::new(Arc::new(
            PricingTable::new()
                .with("openai", "gpt-x", ModelPrice::new(2.0, 8.0))
                .with("anthropic", "c-x", ModelPrice::new(3.0, 15.0)),
        ));

        // call 1: 1000/500 on gpt-x -> 2 + 4 = 6
        let c1 = tracker.record(Some("openai"), "gpt-x", 1000, 500).await;
        assert_eq!(c1, 6.0);
        // call 2: 2000/0 on gpt-x -> 4
        tracker.record(Some("openai"), "gpt-x", 2000, 0).await;
        // call 3: 1000/1000 on c-x -> 3 + 15 = 18
        tracker.record(Some("anthropic"), "c-x", 1000, 1000).await;

        assert_eq!(tracker.total_cost_usd().await, 28.0);
        let report = tracker.report().await;
        assert_eq!(report.calls, 3);
        assert_eq!(report.prompt_tokens, 4000);
        assert_eq!(report.completion_tokens, 1500);
        assert_eq!(report.by_model["openai/gpt-x"].calls, 2);
        assert_eq!(report.by_model["openai/gpt-x"].cost_usd, 10.0);
        assert_eq!(report.by_model["anthropic/c-x"].cost_usd, 18.0);
        assert_eq!(tracker.records().await.len(), 3);
    }

    #[tokio::test]
    async fn unknown_model_prices_zero_but_still_counts() {
        let tracker = CostTracker::with_builtin_prices();
        let cost = tracker.record(Some("local"), "oss-model", 1000, 1000).await;
        assert_eq!(cost, 0.0);
        let report = tracker.report().await;
        assert_eq!(report.calls, 1);
        assert_eq!(report.total_cost_usd, 0.0);
        assert_eq!(report.by_model["local/oss-model"].prompt_tokens, 1000);
    }

    #[tokio::test]
    async fn reset_clears_accumulation() {
        let tracker = CostTracker::with_builtin_prices();
        tracker
            .record(Some("openai"), "gpt-4o-mini", 1000, 1000)
            .await;
        assert_eq!(tracker.total_cost_usd().await, 0.75);
        tracker.reset().await;
        assert_eq!(tracker.total_cost_usd().await, 0.0);
        assert_eq!(tracker.report().await.calls, 0);
    }

    #[tokio::test]
    async fn report_serializes_scope_and_totals() {
        let tracker = CostTracker::with_builtin_prices().with_scope("run-7");
        tracker
            .record(Some("openai"), "gpt-4o-mini", 1000, 1000)
            .await;
        let json = serde_json::to_value(tracker.report().await).unwrap();
        assert_eq!(json["scope"], "run-7");
        assert_eq!(json["total_cost_usd"], 0.75);
        assert_eq!(json["by_model"]["openai/gpt-4o-mini"]["calls"], 1);
    }
}
