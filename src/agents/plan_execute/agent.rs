//! PlanExecuteAgent - 规划-执行-重规划

use std::sync::Arc;

use crate::agents::{AgentExecutor, BaseAgent, FunctionCallingAgent};
use crate::language_models::OpenAIChat;
use crate::BaseTool;

use super::plan::StepStatus;
use super::planner::Planner;

/// Plan-Execute Agent:先规划,逐步执行,失败时重规划
pub struct PlanExecuteAgent {
    llm: OpenAIChat,
    tools: Vec<Arc<dyn BaseTool>>,
    max_replans: usize,
}

impl PlanExecuteAgent {
    pub fn new(llm: OpenAIChat, tools: Vec<Arc<dyn BaseTool>>) -> Self {
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
    pub async fn run(&self, objective: &str) -> Result<String, String> {
        let planner = Planner::new(self.llm.clone());
        let mut plan = planner.plan(objective).await?;

        for replan_count in 0..=self.max_replans {
            let pending_ids: Vec<usize> = plan
                .steps
                .iter()
                .filter(|s| s.status == StepStatus::Pending)
                .map(|s| s.id)
                .collect();

            let mut failed = false;
            for step_id in pending_ids {
                let step_desc = plan.steps[step_id].description.clone();
                plan.steps[step_id].status = StepStatus::Running;

                match self.execute_step(&step_desc).await {
                    Ok(result) => plan.mark_completed(step_id, result),
                    Err(e) => {
                        plan.mark_failed(step_id, e.clone());
                        if replan_count < self.max_replans {
                            plan = planner.replan(objective, &step_desc, &e).await?;
                            failed = true;
                            break;
                        } else {
                            return Err(format!(
                                "步骤 [{}] 失败,已达最大重规划次数: {}",
                                step_desc, e
                            ));
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
        Err("计划未完成".to_string())
    }

    /// 执行单步:用 FunctionCallingAgent + tools
    async fn execute_step(&self, step: &str) -> Result<String, String> {
        let agent = FunctionCallingAgent::new(self.llm.clone(), self.tools.clone(), None);
        let executor = AgentExecutor::new(
            Arc::new(agent) as Arc<dyn BaseAgent>,
            self.tools.clone(),
        )
        .with_max_iterations(5);
        executor
            .invoke(step.to_string())
            .await
            .map_err(|e| format!("{:?}", e))
    }
}
