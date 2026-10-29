// lc-core/src/observability/mod.rs
//! Unified, pluggable observability export.
//!
//! `TokenTrackingLLM` (token usage) and `AgentExecutor` (per-run metrics) both
//! write through the same [`MetricsSink`] interface with an [`ObsEvent`] payload.
//! The framework only provides the capability — no concrete sink is bundled here
//! (see the `lc-observability` crate for `JsonLinesSink`/`MongoSink`). Failures
//! are logged as `warn` and never interrupt the main flow.

mod agent_metrics;
mod error;

pub use agent_metrics::AgentMetrics;
pub use error::ObsError;

use crate::language_models::TokenUsage;
use serde::Serialize;

/// One observability record (the unified export payload).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObsEvent {
    /// Token usage of a single LLM call (exported as it happens).
    TokenUsage(TokenUsageEvent),
    /// Aggregated metrics of one agent run (exported once at the end).
    AgentMetrics(AgentMetrics),
    /// Priced USD cost of one LLM call (B3; exported as it happens).
    Cost(CostEvent),
    /// A failed run, exported as an error span so failures stay visible in every
    /// roll-up (jsonl / mongo / otlp). Carries the same run `trace_id` join key as
    /// [`AgentMetrics`]/[`CostEvent`] so the failure joins the run's other events.
    RunError(RunEvent),
}

/// Token-usage payload for a single LLM call ([`ObsEvent::TokenUsage`]).
///
/// Mirrors [`CostEvent`]/[`RunEvent`]: the run `trace_id` joins this call to the
/// run's other events for cross-backend roll-up, and `provider`/`model` attribute
/// the call to the model that served it. The shared
/// `lc_core::language_models::TokenUsage` is flattened so the serialized record
/// keeps `prompt_tokens`/`completion_tokens`/`total_tokens` at the top level
/// (the same JSON shape as the shared-type events). Unlike the shared `TokenUsage`
/// (constructed at ~77 sites across embeddings/rag/chains), this wrapper is
/// observability-specific, so it can carry join keys without a broad cross-crate
/// API churn.
#[derive(Debug, Clone, Serialize)]
pub struct TokenUsageEvent {
    /// The call's token counts (flattened into the JSON record).
    #[serde(flatten)]
    pub usage: TokenUsage,
    /// trace_id of the owning run (`None` when absent). Joins this call to the
    /// run's `AgentMetrics`/`Cost`/`RunError` events so token spend rolls up per
    /// trace and every event of a run shares one OTel trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Provider slug (`"openai"`, ...); `None` when undeclared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model id as reported by the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// A failed run (emitted by `AgentExecutor`/run executors on error).
#[derive(Debug, Clone, Serialize)]
pub struct RunEvent {
    /// trace_id of the failed run (`None` if absent). Joins this failure to the
    /// run's other events (`AgentMetrics`/`CostEvent`) for cross-backend roll-up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Provider slug (`"openai"`, ...); `None` when undeclared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model id as reported by the model (`None` when the failing stage has no model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The run's terminating error message.
    pub error: String,
}

/// Priced cost of one LLM call (emitted by `CostTracker`).
#[derive(Debug, Clone, Serialize)]
pub struct CostEvent {
    /// trace_id of the owning run (`None` when absent). Joins this cost to the
    /// run's `AgentMetrics`/`RunError` events so llm spend rolls up per trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Run/session label when the tracker was scoped with one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Provider slug (`"openai"`, ...); `None` when undeclared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model id as reported by the model.
    pub model: String,
    /// Prompt tokens of the call.
    pub prompt_tokens: usize,
    /// Completion tokens of the call.
    pub completion_tokens: usize,
    /// Priced USD cost (0.0 when no price entry matched).
    pub cost_usd: f64,
}

/// Pluggable observability sink. The framework only provides the interface and
/// binds no concrete plugin.
#[async_trait::async_trait]
pub trait MetricsSink: Send + Sync {
    /// Pushes one record. Implementations must contain their own failures (or
    /// let the framework `warn` on `Err`) — errors never propagate to the caller.
    async fn export(&self, event: &ObsEvent) -> Result<(), ObsError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn sample_usage() -> TokenUsage {
        TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..TokenUsage::default()
        }
    }

    #[test]
    fn obs_event_token_usage_serializes_with_kind_tag() {
        let json = serde_json::to_string(&ObsEvent::TokenUsage(TokenUsageEvent {
            usage: sample_usage(),
            trace_id: None,
            provider: None,
            model: None,
        }))
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "token_usage");
        assert_eq!(v["prompt_tokens"], 10);
        assert_eq!(v["completion_tokens"], 5);
        assert_eq!(v["total_tokens"], 15);
    }

    #[test]
    fn obs_event_agent_metrics_serializes_with_kind_tag() {
        let m = AgentMetrics {
            trace_id: Some("trace-x".to_string()),
            llm_calls: 2,
            cache_hits: 0,
            tool_calls: 1,
            compactions: 1,
            total_tokens: Some(30),
            duration: Duration::from_millis(100),
        };
        let json = serde_json::to_string(&ObsEvent::AgentMetrics(m)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "agent_metrics");
        assert_eq!(v["llm_calls"], 2);
        assert_eq!(v["tool_calls"], 1);
        assert_eq!(v["total_tokens"], 30);
    }

    #[test]
    fn obs_event_cost_serializes_with_kind_tag() {
        let json = serde_json::to_string(&ObsEvent::Cost(CostEvent {
            trace_id: Some("trace-x".to_string()),
            scope: Some("run-7".to_string()),
            provider: Some("openai".to_string()),
            model: "gpt-4o-mini".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 1000,
            cost_usd: 0.75,
        }))
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "cost");
        assert_eq!(v["scope"], "run-7");
        assert_eq!(v["provider"], "openai");
        assert_eq!(v["model"], "gpt-4o-mini");
        assert_eq!(v["cost_usd"], 0.75);
    }

    #[test]
    fn obs_event_run_error_serializes_with_trace_and_error() {
        let json = serde_json::to_string(&ObsEvent::RunError(RunEvent {
            trace_id: Some("trace-x".to_string()),
            provider: Some("openai".to_string()),
            model: Some("gpt-4o-mini".to_string()),
            error: "tool failed".to_string(),
        }))
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "run_error");
        assert_eq!(v["trace_id"], "trace-x");
        assert_eq!(v["provider"], "openai");
        assert_eq!(v["model"], "gpt-4o-mini");
        assert_eq!(v["error"], "tool failed");
    }

    #[test]
    fn obs_event_cost_serializes_trace_id_when_present() {
        let json = serde_json::to_string(&ObsEvent::Cost(CostEvent {
            trace_id: Some("trace-x".to_string()),
            scope: None,
            provider: Some("openai".to_string()),
            model: "gpt-4o-mini".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 1000,
            cost_usd: 0.75,
        }))
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "cost");
        assert_eq!(v["trace_id"], "trace-x");
    }
}
