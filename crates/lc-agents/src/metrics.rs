// lc-agents/src/metrics.rs
//! Agent 执行指标(P1-5)
//!
//! 统计单次 `AgentExecutor::invoke` 的 LLM 调用次数、工具调用次数、
//! token 用量与总耗时,供成本/延迟观测。指标在每次执行结束时写入
//! `AgentExecutor::last_metrics()`,并通过 `log_summary()` 输出审计日志。

use lc_core::language_models::TokenUsage;
use std::time::Duration;

/// 单次 Agent 执行的聚合指标。
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    /// 本次执行的 trace_id(无则 None)。
    pub trace_id: Option<String>,
    /// LLM 规划调用次数(plan 迭代次数)。
    pub llm_calls: usize,
    /// 命中 LLM 结果缓存、跳过 LLM 调用的次数(P2-1)。
    pub cache_hits: usize,
    /// 工具执行次数(含并行)。
    pub tool_calls: usize,
    /// 累计 token 用量(agent 不上报 token 则为 None)。
    pub total_tokens: Option<usize>,
    /// 单次 invoke 总耗时。
    pub duration: Duration,
}

impl AgentMetrics {
    /// 累加一次 LLM 调用的 token 用量。
    pub fn add_token_usage(&mut self, usage: &TokenUsage) {
        self.total_tokens = Some(self.total_tokens.unwrap_or(0) + usage.total_tokens);
    }

    /// 平均每次 LLM 调用的 token 数。
    pub fn tokens_per_call(&self) -> Option<f64> {
        if self.llm_calls == 0 {
            return None;
        }
        self.total_tokens.map(|t| t as f64 / self.llm_calls as f64)
    }

    /// 输出指标审计日志行(target: `lc_agents::metrics`)。
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
        let mut m = AgentMetrics::default();
        m.trace_id = Some("trace-x".to_string());
        m.log_summary();
        m.total_tokens = Some(42);
        m.log_summary();
    }
}
