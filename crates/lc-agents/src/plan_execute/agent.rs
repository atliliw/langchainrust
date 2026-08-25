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
#[non_exhaustive]
pub enum PlanExecuteError {
    /// 规划失败
    #[error("Planning failed: {0}")]
    PlanningError(String),
    /// 步骤执行失败
    #[error("Step execution failed: {0}")]
    StepExecutionError(String),
    /// 达到最大重规划次数
    #[error("Max replans reached: step [{step}] failed: {reason}")]
    MaxReplansReached {
        /// 失败的步骤
        step: String,
        /// 失败原因
        reason: String,
    },
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
    /// 每步执行 Agent 的工厂(P1-2)。缺省回落 `FunctionCallingAgent`。
    agent_factory: Option<Arc<dyn Fn() -> Arc<dyn BaseAgent> + Send + Sync>>,
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
            agent_factory: None,
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
            agent_factory: None,
        }
    }

    /// 设置最大重规划次数。
    pub fn with_max_replans(mut self, n: usize) -> Self {
        self.max_replans = n;
        self
    }

    /// 自定义每步执行 Agent 的工厂(P1-2)。
    ///
    /// 工厂返回的 `BaseAgent` 将用于 PlanExecute 的每个 `execute_step`。
    /// 缺省回落 `FunctionCallingAgent`。适用于需要给执行 Agent 注入
    /// ReAct / Streaming / 自研 Agent 的场景。
    pub fn with_agent_factory(
        mut self,
        factory: Arc<dyn Fn() -> Arc<dyn BaseAgent> + Send + Sync>,
    ) -> Self {
        self.agent_factory = Some(factory);
        self
    }

    /// 运行完整任务:规划 -> 执行每步 -> 失败重规划 -> 汇总
    pub async fn run(&self, objective: &str) -> Result<String, PlanExecuteError> {
        let planner = Planner::new(self.llm.clone());
        let mut plan = planner
            .plan(objective)
            .await
            .map_err(|e| PlanExecuteError::PlanningError(e.to_string()))?;

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
                                .map_err(|e| PlanExecuteError::PlanningError(e.to_string()))?;
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

    /// 执行单步:优先用 `agent_factory` 产出的 Agent,缺省回落 FunctionCallingAgent(P1-2)
    async fn execute_step(&self, step: &str) -> Result<String, PlanExecuteError> {
        let agent: Arc<dyn BaseAgent> = match &self.agent_factory {
            Some(factory) => factory(),
            None => Arc::new(FunctionCallingAgent::from_arc(
                self.llm.clone(),
                self.tools.clone(),
                None,
            )) as Arc<dyn BaseAgent>,
        };
        let executor = AgentExecutor::new(agent, self.tools.clone()).with_max_iterations(5);
        executor
            .invoke(step.to_string())
            .await
            .map_err(|e| PlanExecuteError::StepExecutionError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentFinish, AgentOutput, AgentStep};
    use crate::AgentError;
    use async_trait::async_trait;
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use std::collections::HashMap;

    /// 可离线的 mock agent:直接返回 Finish,用于验证 agent_factory 生效(P1-2)。
    struct FakeAgent;

    #[async_trait]
    impl BaseAgent for FakeAgent {
        async fn plan(
            &self,
            _intermediate_steps: &[AgentStep],
            _inputs: &HashMap<String, String>,
        ) -> Result<AgentOutput, AgentError> {
            Ok(AgentOutput::Finish(AgentFinish::new(
                "executed by factory".to_string(),
                String::new(),
            )))
        }
    }

    /// P1-2:execute_step 应使用 agent_factory 产出的 Agent,而非硬编码 FunctionCallingAgent。
    #[tokio::test]
    async fn test_execute_step_uses_agent_factory() {
        let agent = PlanExecuteAgent::new(OpenAIChat::new(OpenAIConfig::default()), vec![])
            .with_agent_factory(Arc::new(|| Arc::new(FakeAgent) as Arc<dyn BaseAgent>));
        let result = agent.execute_step("step 1").await.unwrap();
        assert_eq!(result, "executed by factory");
    }
}
