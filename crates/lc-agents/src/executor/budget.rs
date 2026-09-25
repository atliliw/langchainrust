// lc-agents/src/executor/budget.rs
//! Budget gates (§4.2): `BudgetConfig` hard-limit configuration + `BudgetExceeded`
//! over-limit details + four gate-check functions (shared by the invoke / stream paths).
//!
//! The `AgentExecutor` control loop is all-off by default (`None` field = unlimited) and
//! existing behavior is unchanged; once enabled via `.with_budget(BudgetConfig { .. })`,
//! hitting any limit returns [`super::AgentError::BudgetExceeded`], letting the caller
//! distinguish a budget stop from the model not converging. The gate functions are called
//! with the same semantics in `run_agent_loop_from` (invoke) and `stream`, so the two
//! paths cannot diverge.

use super::AgentError;
use crate::metrics::AgentMetrics;
use std::time::{Duration, Instant};

/// Hard budget for the agent control loop. A `None` field means that item is unlimited.
#[derive(Debug, Clone, Default)]
pub struct BudgetConfig {
    /// Cumulative tool-call cap (parallel calls included; stops when exceeded).
    pub max_tool_calls: Option<usize>,
    /// Cumulative LLM output-token cap (reads `AgentMetrics.total_tokens`; has no effect
    /// when the agent does not report tokens).
    pub max_tokens: Option<usize>,
    /// Loop wall-clock cap (timed from `run_agent_loop`).
    pub max_duration: Option<Duration>,
    /// Iteration cap (tightens `AgentExecutor::max_iterations`; hitting it returns an error
    /// instead of the placeholder return path used at the iteration limit).
    pub max_iterations: Option<usize>,
    /// Cumulative USD spend cap (read from the shared
    /// `lc_core::cost::CostTracker`; has no effect when the executor carries no
    /// cost tracker). Measured after each LLM call.
    pub max_cost_usd: Option<f64>,
}

/// Details of a budget over-limit.
#[derive(Debug, Clone)]
pub enum BudgetExceeded {
    /// Cumulative tool-call count exceeded.
    ToolCalls {
        /// Configured limit.
        limit: usize,
        /// Actual cumulative count at trigger time.
        actual: usize,
    },
    /// Cumulative LLM output-token count exceeded.
    Tokens {
        /// Configured limit.
        limit: usize,
        /// Actual cumulative tokens at trigger time.
        actual: usize,
    },
    /// Loop wall-clock duration exceeded.
    Duration {
        /// Configured limit.
        limit: Duration,
        /// Actual elapsed time at trigger time.
        elapsed: Duration,
    },
    /// Iteration count exceeded.
    Iterations {
        /// Effective limit (already `min`'d with `AgentExecutor::max_iterations`).
        limit: usize,
    },
    /// Cumulative USD spend exceeded.
    Cost {
        /// Configured USD limit.
        limit: f64,
        /// Actual cumulative USD spend at trigger time.
        actual: f64,
    },
}

/// Budget gate (§4.2): iteration-level check (iteration count + wall-clock). Returns an
/// error when a limit is exceeded.
///
/// `max_iterations` uses `min(self.max_iterations, budget.max_iterations)` as the
/// effective limit — when the budget is tighter than the default it hard-stops on
/// exceeding; when looser, `max_iterations` backs it up without changing the original
/// placeholder return path. Shared by invoke / stream.
pub(crate) fn budget_iteration_gate(
    budget: Option<&BudgetConfig>,
    max_iterations: usize,
    iteration: usize,
    loop_start: Instant,
) -> Option<AgentError> {
    let budget = budget?;
    if let Some(limit) = budget.max_iterations {
        let effective = limit.min(max_iterations);
        if iteration >= effective {
            return Some(AgentError::BudgetExceeded(BudgetExceeded::Iterations {
                limit: effective,
            }));
        }
    }
    if let Some(limit) = budget.max_duration {
        let elapsed = loop_start.elapsed();
        if elapsed >= limit {
            return Some(AgentError::BudgetExceeded(BudgetExceeded::Duration {
                limit,
                elapsed,
            }));
        }
    }
    None
}

/// Budget gate (§4.2): cumulative-token check after an LLM call. No effect when the
/// agent does not report tokens.
///
/// Uses `> limit` — allowing up to and including `limit` tokens, tripping only once
/// the cumulative spend *exceeds* it (stage-G G5; `>=` would silently allow only
/// `limit - 1` tokens). Because token spend is only measurable post-LLM-call, the
/// over-limit trip is the correct boundary here — it permits exactly `limit` tokens,
/// the same "allow exactly the limit" outcome the tool-call gate achieves.
pub(crate) fn budget_token_gate(
    budget: Option<&BudgetConfig>,
    metrics: &AgentMetrics,
) -> Option<AgentError> {
    let budget = budget?;
    let limit = budget.max_tokens?;
    let actual = metrics.total_tokens.unwrap_or(0);
    if actual > limit {
        return Some(AgentError::BudgetExceeded(BudgetExceeded::Tokens {
            limit,
            actual,
        }));
    }
    None
}

/// Budget gate (§4.2): cumulative USD-spend check after an LLM call. The caller
/// reads `CostTracker::total_cost_usd()` and passes the value in; when no
/// tracker is attached the measured spend stays `0.0`, so a cost limit without
/// a tracker simply never trips (measurement and enforcement stay explicit).
pub(crate) fn budget_cost_gate(
    budget: Option<&BudgetConfig>,
    current_cost_usd: f64,
) -> Option<AgentError> {
    let budget = budget?;
    let limit = budget.max_cost_usd?;
    if current_cost_usd >= limit {
        return Some(AgentError::BudgetExceeded(BudgetExceeded::Cost {
            limit,
            actual: current_cost_usd,
        }));
    }
    None
}

/// Budget gate (§4.2): checks cumulative call count and wall-clock before a tool runs.
///
/// H2: `metrics.tool_calls` counts tools that *actually executed* (parallel included)
/// — no pre-increment, since non-executed calls (approval `Deny`, `ToolNotFound`, hook
/// `Reject`) never enter the count. The check therefore uses `>= limit`: launching a
/// further round is only allowed while `tool_calls < limit`, so exactly `limit` tools
/// may execute and the `limit + 1`-th round is the hard stop. Previously the gate ran
/// on a pre-incremented "attempted" count with `> limit`, so a batch whose *attempt*
/// count crossed the cap but whose executed share stayed within it (denied / not-found
/// slots) was wrongly stopped before anything ran.
pub(crate) fn budget_tool_gate(
    budget: Option<&BudgetConfig>,
    metrics: &AgentMetrics,
    loop_start: Instant,
) -> Option<AgentError> {
    let budget = budget?;
    if let Some(limit) = budget.max_tool_calls {
        if metrics.tool_calls >= limit {
            return Some(AgentError::BudgetExceeded(BudgetExceeded::ToolCalls {
                limit,
                actual: metrics.tool_calls,
            }));
        }
    }
    if let Some(limit) = budget.max_duration {
        let elapsed = loop_start.elapsed();
        if elapsed >= limit {
            return Some(AgentError::BudgetExceeded(BudgetExceeded::Duration {
                limit,
                elapsed,
            }));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost_cfg(limit: f64) -> BudgetConfig {
        BudgetConfig {
            max_cost_usd: Some(limit),
            ..Default::default()
        }
    }

    #[test]
    fn cost_gate_without_budget_is_inert() {
        assert!(budget_cost_gate(None, 9_999.0).is_none());
    }

    #[test]
    fn cost_gate_without_limit_is_inert_even_with_budget() {
        assert!(budget_cost_gate(Some(&BudgetConfig::default()), 9_999.0).is_none());
    }

    #[test]
    fn cost_gate_below_limit_passes() {
        assert!(budget_cost_gate(Some(&cost_cfg(1.0)), 0.99).is_none());
    }

    #[test]
    fn cost_gate_at_and_above_limit_stops() {
        match budget_cost_gate(Some(&cost_cfg(1.0)), 1.0) {
            Some(AgentError::BudgetExceeded(BudgetExceeded::Cost { limit, actual })) => {
                assert_eq!(limit, 1.0);
                assert_eq!(actual, 1.0);
            }
            other => panic!("expected Cost stop at the limit, got {other:?}"),
        }
        assert!(budget_cost_gate(Some(&cost_cfg(1.0)), 1.5).is_some());
    }

    fn token_cfg(limit: usize) -> BudgetConfig {
        BudgetConfig {
            max_tokens: Some(limit),
            ..Default::default()
        }
    }

    fn metrics_with_tokens(tokens: usize) -> AgentMetrics {
        AgentMetrics {
            total_tokens: Some(tokens),
            ..Default::default()
        }
    }

    /// stage-G G5: token gate uses the same exact-limit-then-stop semantics as the
    /// tool-call gate — `max_tokens` allows up to and including the limit, and
    /// trips only once the cumulative spend exceeds it.
    #[test]
    fn token_gate_matches_tool_gate_equality_semantics() {
        // At exactly the limit: allowed (both gates permit exactly `limit` calls).
        assert!(budget_token_gate(Some(&token_cfg(100)), &metrics_with_tokens(100)).is_none());
        assert!(budget_token_gate(Some(&token_cfg(100)), &metrics_with_tokens(99)).is_none());
        // One past the limit: hard stop.
        match budget_token_gate(Some(&token_cfg(100)), &metrics_with_tokens(101)) {
            Some(AgentError::BudgetExceeded(BudgetExceeded::Tokens { limit, actual })) => {
                assert_eq!(limit, 100);
                assert_eq!(actual, 101);
            }
            other => panic!("expected Tokens stop past the limit, got {other:?}"),
        }
        // Inert without a budget or a limit.
        assert!(budget_token_gate(None, &metrics_with_tokens(101)).is_none());
        assert!(budget_token_gate(Some(&BudgetConfig::default()), &metrics_with_tokens(101)).is_none());
    }
}
