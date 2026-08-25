//! 高层编排器公共 trait(P1-1)
//!
//! `PlanExecuteAgent` / `DeepResearchAgent` / `CorrectiveRAGAgent` / `AdaptiveRAG`
//! 此前各写各的 `run()`,签名互不兼容,无法组合、无法进 LCEL。这里统一收敛:
//!
//! - [`Orchestrator`] 定义 `run_with_context(input, ctx)`,错误统一到 [`AgentError`]。
//! - [`RunContext`] 携带 `trace_id`(P1-4 可观测性)与跨步骤共享工作区。
//! - [`crate::adapter::OrchestratorRunnable`] 让编排器能进 LCEL 管道。
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_agents::orchestration::{Orchestrator, RunContext};
//!
//! let plan_agent = PlanExecuteAgent::new(llm, tools);
//! let ctx = RunContext::new("trace-abc");
//! let output = plan_agent.run_with_context("目标".to_string(), &ctx).await?;
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use serde_json::Value;

use crate::AgentError;

mod fan_out_fan_in;
mod impls;
mod review;
mod sequential;
mod task_adapter;
#[cfg(test)]
mod tests;

pub use fan_out_fan_in::FanOutFanIn;
pub use review::{parse_review_verdict, review_envelope, ReviewOrchestrator, ReviewVerdict};
pub use sequential::SequentialPipeline;
pub use task_adapter::{task_adapter, TaskAdapter};

/// 高层编排器公共 trait。
///
/// 关联类型表达各编排器不同的输入/输出(PlanExecute→String,AdaptiveRAG→AdaptiveRAGResult 等),
/// `run_with_context` 统一签名 + 统一 [`AgentError`],让编排器可组合、可进 LCEL。
#[async_trait]
pub trait Orchestrator: Send + Sync {
    /// 输入类型(通常为 `String` 目标/问题)。
    type Input;
    /// 输出类型。
    type Output;

    /// 携带运行上下文的执行入口。
    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError>;
}

/// 编排器运行上下文。
///
/// `trace_id` 在多 Agent / 跨步骤调用链间传播(P1-4);`shared_state` 提供
/// 跨步骤共享的 JSON 工作区。
#[derive(Debug, Clone)]
pub struct RunContext {
    /// 追踪 ID:整条调用链共享,用于日志/审计/指标关联。
    pub trace_id: String,
    /// 跨步骤共享工作区(可选)。
    pub shared_state: Option<Arc<Mutex<Value>>>,
}

/// 生成一个轻量 trace_id(时间戳十六进制)。
pub fn generate_trace_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("trace-{:x}", nanos)
}

impl RunContext {
    /// 创建上下文,指定 `trace_id`。
    pub fn new(trace_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            shared_state: None,
        }
    }

    /// 创建上下文,自动生成 `trace_id`。
    pub fn new_random() -> Self {
        Self::new(generate_trace_id())
    }

    /// 携带共享工作区。
    pub fn with_shared_state(mut self, shared_state: Arc<Mutex<Value>>) -> Self {
        self.shared_state = Some(shared_state);
        self
    }

    /// 从 LCEL [`RunnableConfig`] 提取 `trace_id`(读 `metadata["trace_id"]`),
    /// 缺失则自动生成。用于把 LCEL 管道的 trace 贯通到编排器。
    pub fn from_config(config: &RunnableConfig) -> Self {
        let trace_id = config
            .metadata
            .get("trace_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(generate_trace_id);
        Self::new(trace_id)
    }
}
