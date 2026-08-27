//! `TaskAdapter`: bridges an `Input=String` orchestrator into a child agent consuming [`AgentTask`] (P2-5).

use async_trait::async_trait;
use std::sync::Arc;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// Adapts an `Input=String` orchestrator into a child agent consuming [`AgentTask`] (P2-5).
///
/// Bridges two orchestrator kinds: a real agent (PlanExecuteAgent /
/// DeepResearchAgent, etc., `Input=String`) wrapped this way can be placed in a
/// `FanOutFanIn` / `SequentialPipeline` that dispatches [`AgentTask`]. Takes the
/// objective and feeds it to the inner orchestrator; when the task declares
/// `allowed_tools`, the consumer (AgentExecutor, etc.) assembles the tool list
/// from that allowlist — this adapter does not filter on its behalf.
pub struct TaskAdapter {
    inner: Arc<dyn Orchestrator<Input = String, Output = String>>,
}

impl TaskAdapter {
    /// Wrap an `Input=String` orchestrator.
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

/// Convenience wrapper: converts an `Input=String` orchestrator into a trait object that accepts [`AgentTask`] dispatch.
pub fn task_adapter(
    inner: Arc<dyn Orchestrator<Input = String, Output = String>>,
) -> Arc<dyn Orchestrator<Input = AgentTask, Output = String>> {
    Arc::new(TaskAdapter::new(inner))
}
