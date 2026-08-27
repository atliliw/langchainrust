// lc-agents/src/executor/budget.rs
//! Budget gates (§4.2): `BudgetConfig` hard-limit configuration + `BudgetExceeded`
//! over-limit details + three gate-check functions (shared by the invoke / stream paths).
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
pub(crate) fn budget_token_gate(
    budget: Option<&BudgetConfig>,
    metrics: &AgentMetrics,
) -> Option<AgentError> {
    let budget = budget?;
    let limit = budget.max_tokens?;
    let actual = metrics.total_tokens.unwrap_or(0);
    if actual >= limit {
        return Some(AgentError::BudgetExceeded(BudgetExceeded::Tokens {
            limit,
            actual,
        }));
    }
    None
}

/// Budget gate (§4.2): checks cumulative call count and wall-clock before a tool runs.
///
/// `metrics.tool_calls` is already incremented, so the check uses `> limit` — allowing
/// exactly `limit` tool executions, with the `limit + 1`-th triggering the hard stop.
pub(crate) fn budget_tool_gate(
    budget: Option<&BudgetConfig>,
    metrics: &AgentMetrics,
    loop_start: Instant,
) -> Option<AgentError> {
    let budget = budget?;
    if let Some(limit) = budget.max_tool_calls {
        if metrics.tool_calls > limit {
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
