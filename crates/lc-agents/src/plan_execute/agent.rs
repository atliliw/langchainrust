//! PlanExecuteAgent - 规划-执行-重规划

use std::sync::Arc;

use crate::{AgentError, AgentExecutor, BaseAgent, FunctionCallingAgent};
use lc_core::language_models::BaseChatModel;
use lc_core::tools::BaseTool;
use lc_providers::ProviderError;

use super::plan::StepStatus;
use super::planner::Planner;

/// Plan-Execute Agent 错误类型
#[derive(Debug, thiserror::Error)]
pub enum PlanExecuteError {
    /// 规划失败
    #[error("Planning failed: {0}")]
    PlanningError(String),
    /// 步骤执行失败
    #[error("Step execution failed: {0}")]
    StepExecutionError(String),
    /// 达到最大重规划次数
    #[error("Max replans reached: step [{step}] failed: {reason}")]
    MaxReplansReached { step: String, reason: String },
    /// 计划未完成
    #[error("Plan incomplete after all replans")]
    PlanIncomplete,
}

impl From<AgentError> for PlanExecuteError {
    fn from(e: AgentError) -> Self {
        PlanExecuteError::StepExecutionError(e.to_string())
    }
}

/// Plan-Execute Agent:先规划,逐步执行,失败时重规划
///
/// 支持任何实现了 `BaseChatModel` 的 LLM Provider。
pub struct PlanExecuteAgent {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    tools: Vec<Arc<dyn BaseTool>>,
    max_replans: usize,
}

impl PlanExecuteAgent {
    /// 创建新的 Plan-Execute Agent
    ///
    /// # 参数
    /// * `llm` - LLM 客户端（任何实现了 `BaseChatModel` 的类型）
    /// * `tools` - 可用工具列表
    ///
    /// # 向后兼容
    /// 旧代码 `PlanExecuteAgent::new(openai_chat, tools)` 仍然可用。
    pub fn new<L>(llm: L, tools: Vec<Arc<dyn BaseTool>>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: lc_providers::wrap_chat_model(llm),
            tools,
            max_replans: 2,
        }
    }

    /// 从已包装的 `Arc<dyn BaseChatModel>` 创建 Agent
    pub fn from_arc(
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
        tools: Vec<Arc<dyn BaseTool>>,
    ) -> Self {
        Self {
            llm,
            tools,
            max_replans: 2,
        }
    }

    pub fn with_max_replans(mut self, n: usize) -> Self {
        self.max_replans = n;
        self
    }

    /// 运行完整任务:规划 -> 执行每步 -> 失败重规划 -> 汇总
    pub async fn run(&self, objective: &str) -> Result<String, PlanExecuteError> {
        let planner = Planner::new(self.llm.clone());
        let mut plan = planner
            .plan(objective)
            .await
            .map_err(PlanExecuteError::PlanningError)?;

        for replan_count in 0..=self.max_replans {
            let pending_ids: Vec<usize> = plan
                .steps
                .iter()
                .filter(|s| s.status == StepStatus::Pending)
                .map(|s| s.id)
                .collect();

            let mut failed = false;
            for step_id in pending_ids {
                let step = plan.steps.iter_mut().find(|s| s.id == step_id);
                let step_desc = match step {
                    Some(s) => {
                        s.status = StepStatus::Running;
                        s.description.clone()
                    }
                    None => {
                        // step_id does not correspond to any step; skip it
                        continue;
                    }
                };

                match self.execute_step(&step_desc).await {
                    Ok(result) => plan.mark_completed(step_id, result),
                    Err(e) => {
                        let error_msg = e.to_string();
                        plan.mark_failed(step_id, error_msg.clone());
                        if replan_count < self.max_replans {
                            plan = planner
                                .replan(objective, &step_desc, &error_msg)
                                .await
                                .map_err(PlanExecuteError::PlanningError)?;
                            failed = true;
                            break;
                        } else {
                            return Err(PlanExecuteError::MaxReplansReached {
                                step: step_desc,
                                reason: error_msg,
                            });
                        }
                    }
                }
            }

            if !failed && plan.is_complete() {
                let summary: Vec<String> = plan
                    .steps
                    .iter()
                    .map(|s| {
                        format!(
                            "{}. {}: {}",
                            s.id + 1,
                            s.description,
                            s.result.as_deref().unwrap_or("无结果")
                        )
                    })
                    .collect();
                return Ok(summary.join("\n"));
            }
        }
        Err(PlanExecuteError::PlanIncomplete)
    }

    /// 执行单步:用 FunctionCallingAgent + tools
    async fn execute_step(&self, step: &str) -> Result<String, PlanExecuteError> {
        let agent = FunctionCallingAgent::from_arc(self.llm.clone(), self.tools.clone(), None);
        let executor =
            AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, self.tools.clone())
                .with_max_iterations(5);
        executor
            .invoke(step.to_string())
            .await
            .map_err(|e| PlanExecuteError::StepExecutionError(e.to_string()))
    }
}
