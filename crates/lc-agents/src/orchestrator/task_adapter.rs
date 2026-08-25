//! `TaskAdapter`:把 `Input=String` 编排器桥接为消费 [`AgentTask`] 的子 Agent(P2-5)。

use async_trait::async_trait;
use std::sync::Arc;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// 把 `Input=String` 的编排器适配为消费 [`AgentTask`] 的子 Agent(P2-5)。
///
/// 桥接两类编排器:真实 Agent(PlanExecuteAgent / DeepResearchAgent 等,
/// `Input=String`)经此包装后,可放进 `FanOutFanIn` / `SequentialPipeline`
/// 接受 [`AgentTask`] 派发。取目标喂给底层编排器;任务声明了 `allowed_tools`
/// 时,由消费方(AgentExecutor 等)按白名单装配,此处不越界替其过滤。
pub struct TaskAdapter {
    inner: Arc<dyn Orchestrator<Input = String, Output = String>>,
}

impl TaskAdapter {
    /// 包装一个 `Input=String` 的编排器。
    pub fn new(inner: Arc<dyn Orchestrator<Input = String, Output = String>>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Orchestrator for TaskAdapter {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        task: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "TaskAdapter dispatch objective='{}' trace_id = {}",
            task.objective,
            ctx.trace_id
        );
        self.inner.run_with_context(task.objective, ctx).await
    }
}

/// 便捷包装:把 `Input=String` 编排器转成可接收 [`AgentTask`] 派发的 trait 对象。
pub fn task_adapter(
    inner: Arc<dyn Orchestrator<Input = String, Output = String>>,
) -> Arc<dyn Orchestrator<Input = AgentTask, Output = String>> {
    Arc::new(TaskAdapter::new(inner))
}
