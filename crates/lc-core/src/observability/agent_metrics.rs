// lc-core/src/observability/agent_metrics.rs
//! Aggregated metrics for a single agent execution.
//!
//! Moved from `lc-agents` (v0.20.2) so the unified observability payload
//! (`ObsEvent`) can reference it from `lc-core`. `lc-agents` keeps a one-line
//! re-export for path compatibility.

use crate::language_models::TokenUsage;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Aggregated metrics for a single agent execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// trace_id of this execution (`None` if absent).
    pub trace_id: Option<String>,
    /// Number of LLM planning calls (plan iterations).
    pub llm_calls: usize,
    /// Number of LLM result cache hits that skipped an LLM call.
    pub cache_hits: usize,
    /// Number of tool executions (including parallel).
    pub tool_calls: usize,
    /// Number of context compactions performed during the run (0.21.0 S6.1).
    /// `#[serde(default)]` keeps pre-0.21.0 payloads deserializable.
    #[serde(default)]
    pub compactions: usize,
    /// Cumulative token usage (`None` if the agent does not report tokens).
    pub total_tokens: Option<usize>,
    /// Total duration of a single invoke.
    pub duration: Duration,
}

impl AgentMetrics {
    /// Accumulates token usage from one LLM call.
    pub fn add_token_usage(&mut self, usage: &TokenUsage) {
        self.total_tokens = Some(self.total_tokens.unwrap_or(0) + usage.total_tokens);
    }

    /// Average tokens per LLM call.
    pub fn tokens_per_call(&self) -> Option<f64> {
        if self.llm_calls == 0 {
            return None;
        }
        self.total_tokens.map(|t| t as f64 / self.llm_calls as f64)
    }

    /// Emits the metrics audit-log line (target: `lc_agents::metrics`).
    pub fn log_summary(&self) {
        let trace = self.trace_id.as_deref().unwrap_or("-");
        let duration_ms = self.duration.as_millis();
        match self.total_tokens {
            Some(tokens) => log::info!(
                target: "lc_agents::metrics",
                "agent_exec summary trace_id={} llm_calls={} tool_calls={} total_tokens={} duration_ms={}",
                trace,
                self.llm_calls,
                self.tool_calls,
                tokens,
                duration_ms
            ),
            None => log::info!(
                target: "lc_agents::metrics",
                "agent_exec summary trace_id={} llm_calls={} tool_calls={} total_tokens=n/a duration_ms={}",
                trace,
                self.llm_calls,
                self.tool_calls,
                duration_ms
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_default() {
        let m = AgentMetrics::default();
        assert_eq!(m.llm_calls, 0);
        assert_eq!(m.tool_calls, 0);
        assert_eq!(m.total_tokens, None);
        assert_eq!(m.trace_id, None);
        assert_eq!(m.duration, Duration::ZERO);
    }

    #[test]
    fn test_add_token_usage() {
        let mut m = AgentMetrics::default();
        m.add_token_usage(&TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        });
        m.add_token_usage(&TokenUsage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
        });
        assert_eq!(m.total_tokens, Some(20));
    }

    #[test]
    fn test_tokens_per_call() {
        let mut m = AgentMetrics::default();
        assert_eq!(m.tokens_per_call(), None);
        m.llm_calls = 2;
        m.total_tokens = Some(20);
        assert_eq!(m.tokens_per_call(), Some(10.0));
    }

    #[test]
    fn test_tokens_per_call_zero_calls() {
        let m = AgentMetrics {
            total_tokens: Some(10),
            ..Default::default()
        };
        assert_eq!(m.tokens_per_call(), None);
    }

    #[test]
    fn test_serde_roundtrip() {
        let m = AgentMetrics {
            trace_id: Some("trace-x".to_string()),
            llm_calls: 3,
            cache_hits: 1,
            tool_calls: 2,
            compactions: 1,
            total_tokens: Some(42),
            duration: Duration::from_millis(1500),
        };
        let json = serde_json::to_string(&m).expect("serialize");
        let back: AgentMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.trace_id, m.trace_id);
        assert_eq!(back.llm_calls, m.llm_calls);
        assert_eq!(back.cache_hits, m.cache_hits);
        assert_eq!(back.tool_calls, m.tool_calls);
        assert_eq!(back.compactions, m.compactions);
        assert_eq!(back.total_tokens, m.total_tokens);
        assert_eq!(back.duration, m.duration);
    }

    /// 0.21.0 S6.1: pre-0.21.0 payloads (no `compactions` field) still deserialize
    /// — the field is `#[serde(default)]` for backward compatibility.
    #[test]
    fn test_serde_roundtrip_without_compactions_field() {
        let legacy = r#"{
            "trace_id": "trace-x",
            "llm_calls": 3,
            "cache_hits": 1,
            "tool_calls": 2,
            "total_tokens": 42,
            "duration": {"secs": 1, "nanos": 500000000}
        }"#;
        let back: AgentMetrics = serde_json::from_str(legacy).expect("legacy payload deserializes");
        assert_eq!(back.compactions, 0, "missing field defaults to 0");
    }

    #[test]
    fn test_log_summary_no_panic() {
        let m = AgentMetrics {
            trace_id: Some("trace-x".to_string()),
            ..Default::default()
        };
        m.log_summary();
        let m = AgentMetrics {
            total_tokens: Some(42),
            ..Default::default()
        };
        m.log_summary();
    }
}
