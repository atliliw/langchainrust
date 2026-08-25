//! 顺序流水线编排器 `SequentialPipeline`(P2-3 / P2-5)。

use async_trait::async_trait;
use std::sync::Arc;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// 顺序流水线编排器(P2-3 / P2-5)。
///
/// 把 N 个阶段按序执行:前一阶段的输出作为后一阶段的目标,返回末阶段输出。
/// 任务级约束(预期输出 / 允许工具)沿链传递,各阶段保持一致。
/// 各阶段独立失败即整体失败,报错带阶段序号便于定位。
pub struct SequentialPipeline {
    stages: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>,
}

impl SequentialPipeline {
    /// 用一组顺序执行的阶段构造。
    pub fn new(stages: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>) -> Self {
        Self { stages }
    }

    /// 追加一个阶段(链式)。
    pub fn push_stage(
        mut self,
        stage: Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
    ) -> Self {
        self.stages.push(stage);
        self
    }
}

#[async_trait]
impl Orchestrator for SequentialPipeline {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        let mut current = input;
        for (i, stage) in self.stages.iter().enumerate() {
            log::debug!(
                target: "lc_agents::orchestrator",
                "SequentialPipeline stage {i} trace_id = {}",
                ctx.trace_id
            );
            let output = stage
                .run_with_context(current.clone(), ctx)
                .await
                .map_err(|e| AgentError::Other(format!("SequentialPipeline stage {i}: {e}")))?;
            // 阶段输出成为下一阶段的目标;任务级约束沿链传递(P2-5)。
            let mut next = AgentTask::new(output);
            if let Some(expected) = current.expected_output.clone() {
                next = next.with_expected_output(expected);
            }
            next = next.with_allowed_tools(current.allowed_tools.clone());
            current = next;
        }
        Ok(current.objective)
    }
}
