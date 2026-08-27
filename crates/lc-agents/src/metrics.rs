// lc-agents/src/metrics.rs
//! Agent execution metrics (P1-5)
//!
//! Tracks per-`AgentExecutor::invoke` LLM call count, tool call count, token
//! usage and total duration for cost/latency observability. Metrics are written
//! to `AgentExecutor::last_metrics()` at the end of every execution and emitted
//! as an audit log via `log_summary()`.

use lc_core::language_models::TokenUsage;
use std::time::Duration;

/// Aggregated metrics for a single agent execution.
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    /// trace_id of this execution (`None` if absent).
    pub trace_id: Option<String>,
    /// Number of LLM planning calls (plan iterations).
    pub llm_calls: usize,
    /// Number of LLM result cache hits that skipped an LLM call (P2-1).
    pub cache_hits: usize,
    /// Number of tool executions (including parallel).
    pub tool_calls: usize,
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
