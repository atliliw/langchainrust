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
    TokenUsage(TokenUsage),
    /// Aggregated metrics of one agent run (exported once at the end).
    AgentMetrics(AgentMetrics),
    /// Priced USD cost of one LLM call (B3; exported as it happens).
    Cost(CostEvent),
}

/// Priced cost of one LLM call (emitted by `CostTracker`).
#[derive(Debug, Clone, Serialize)]
pub struct CostEvent {
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
        }
    }

    #[test]
    fn obs_event_token_usage_serializes_with_kind_tag() {
        let json = serde_json::to_string(&ObsEvent::TokenUsage(sample_usage())).unwrap();
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
}
