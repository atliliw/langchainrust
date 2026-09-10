//! PlanExecuteAgent - plan - execute - replan

use std::sync::Arc;

use crate::{AgentError, AgentExecutor, BaseAgent, FunctionCallingAgent};
use lc_core::language_models::BaseChatModel;
use lc_core::tools::BaseTool;
use lc_providers::ProviderError;

use super::plan::{Plan, PlanStep, StepStatus};
use super::planner::Planner;

/// Plan-Execute Agent error type
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PlanExecuteError {
    /// Planning failed
    #[error("Planning failed: {0}")]
    PlanningError(String),
    /// Step execution failed
    #[error("Step execution failed: {0}")]
    StepExecutionError(String),
    /// Max replan count reached
    #[error("Max replans reached: step [{step}] failed: {reason}")]
    MaxReplansReached {
        /// The failed step
        step: String,
        /// Failure reason
        reason: String,
    },
    /// The plan is incomplete
    #[error("Plan incomplete after all replans")]
    PlanIncomplete,
}

impl From<AgentError> for PlanExecuteError {
    fn from(e: AgentError) -> Self {
        PlanExecuteError::StepExecutionError(e.to_string())
    }
}

/// Plan-Execute Agent: plans first, executes step by step, and replans on failure.
///
/// Supports any LLM provider that implements `BaseChatModel`.
pub struct PlanExecuteAgent {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    tools: Vec<Arc<dyn BaseTool>>,
    max_replans: usize,
    /// Factory for the per-step execution agent (P1-2). Defaults to `FunctionCallingAgent`.
    agent_factory: Option<Arc<dyn Fn() -> Arc<dyn BaseAgent> + Send + Sync>>,
}

impl PlanExecuteAgent {
    /// Creates a new Plan-Execute Agent
    ///
    /// # Parameters
    /// * `llm` - LLM client (any type implementing `BaseChatModel`)
    /// * `tools` - available tools
    ///
    /// # Backward compatibility
    /// Legacy code `PlanExecuteAgent::new(openai_chat, tools)` still works.
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

    /// Creates an agent from an already-wrapped `Arc<dyn BaseChatModel>`
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

    /// Sets the maximum number of replans.
    pub fn with_max_replans(mut self, n: usize) -> Self {
        self.max_replans = n;
        self
    }

    /// Custom factory for the per-step execution agent (P1-2).
    ///
    /// The `BaseAgent` returned by the factory is used for each `execute_step`
    /// of the PlanExecute. Defaults to `FunctionCallingAgent`. Useful for
    /// injecting ReAct / Streaming / custom agents as the execution agent.
    pub fn with_agent_factory(
        mut self,
        factory: Arc<dyn Fn() -> Arc<dyn BaseAgent> + Send + Sync>,
    ) -> Self {
        self.agent_factory = Some(factory);
        self
    }

    /// Runs the full task: plan -> execute each step -> replan on failure -> summarize
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
                            // 0.22.0 C4 fix: carry the already-completed steps
                            // (and their results) into the replan — the prompt
                            // asks for the *remaining* work, and the completed
                            // steps are spliced back so the summary keeps the
                            // history instead of re-executing everything.
                            let completed_steps: Vec<(String, String)> = plan
                                .steps
                                .iter()
                                .filter(|s| s.status == StepStatus::Completed)
                                .filter_map(|s| {
                                    s.result
                                        .as_ref()
                                        .map(|r| (s.description.clone(), r.clone()))
                                })
                                .collect();
                            let completed_block = if completed_steps.is_empty() {
                                String::new()
                            } else {
                                completed_steps
                                    .iter()
                                    .map(|(d, r)| format!("- {d} → {r}"))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            };
                            plan = planner
                                .replan(objective, &step_desc, &error_msg, &completed_block)
                                .await
                                .map_err(|e| PlanExecuteError::PlanningError(e.to_string()))?;
                            // Splice the completed steps (with results) back at
                            // the front so they show up in the summary and are
                            // not re-executed (only Pending steps run).
                            let mut merged: Vec<PlanStep> = Vec::new();
                            let mut next_id = 0usize;
                            for (desc, result) in &completed_steps {
                                let mut st = PlanStep::new(next_id, desc.clone());
                                st.status = StepStatus::Completed;
                                st.result = Some(result.clone());
                                merged.push(st);
                                next_id += 1;
                            }
                            for s in plan.steps {
                                if s.status != StepStatus::Completed {
                                    merged.push(PlanStep::new(next_id, s.description));
                                    next_id += 1;
                                }
                            }
                            plan = Plan::new(objective, merged);
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

    /// Executes a single step: prefers the agent from `agent_factory`, falling back to FunctionCallingAgent (P1-2)
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

    /// Offline-capable mock agent: returns Finish directly, to verify that agent_factory takes effect (P1-2).
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

    /// P1-2: execute_step should use the agent from agent_factory, not a hardcoded FunctionCallingAgent.
    #[tokio::test]
    async fn test_execute_step_uses_agent_factory() {
        let agent = PlanExecuteAgent::new(OpenAIChat::new(OpenAIConfig::default()), vec![])
            .with_agent_factory(Arc::new(|| Arc::new(FakeAgent) as Arc<dyn BaseAgent>));
        let result = agent.execute_step("step 1").await.unwrap();
        assert_eq!(result, "executed by factory");
    }
}
