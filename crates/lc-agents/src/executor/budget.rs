// lc-agents/src/executor/budget.rs
//! 预算门(§4.2):`BudgetConfig` 硬上限配置 + `BudgetExceeded` 超限详情。
//!
//! `AgentExecutor` 控制循环默认全关(两个字段均为 `None`),存量行为不变;
//! 通过 `.with_budget(BudgetConfig { .. })` 开启后,任一上限被触发即返回
//! [`super::AgentError::BudgetExceeded`],调用方可捕获该错误区分"预算截停"
//! 与"模型未收敛"。

use std::time::Duration;

/// Agent 控制循环的硬预算。`None` 字段 = 该项不限。
#[derive(Debug, Clone, Default)]
pub struct BudgetConfig {
    /// 累计工具调用次数上限(含并行,超过即停)。
    pub max_tool_calls: Option<usize>,
    /// 累计 LLM 输出 token 上限(读 `AgentMetrics.total_tokens`;agent 不上报
    /// token 时该项自然不生效)。
    pub max_tokens: Option<usize>,
    /// 循环总时长上限(从 `run_agent_loop` 起表)。
    pub max_duration: Option<Duration>,
    /// 迭代次数上限(收紧 `AgentExecutor::max_iterations`;超出返回错误而非
    /// 走到迭代上限的占位返回路径)。
    pub max_iterations: Option<usize>,
}

/// 预算超限详情。
#[derive(Debug, Clone)]
pub enum BudgetExceeded {
    /// 累计工具调用次数超限。
    ToolCalls {
        /// 配置的上限。
        limit: usize,
        /// 触发时的实际累计次数。
        actual: usize,
    },
    /// 累计 LLM 输出 token 超限。
    Tokens {
        /// 配置的上限。
        limit: usize,
        /// 触发时的实际累计 token。
        actual: usize,
    },
    /// 循环总时长超限。
    Duration {
        /// 配置的上限。
        limit: Duration,
        /// 触发时的实际已用时长。
        elapsed: Duration,
    },
    /// 迭代次数超限。
    Iterations {
        /// 生效的上限(已与 `AgentExecutor::max_iterations` 取小)。
        limit: usize,
    },
}
