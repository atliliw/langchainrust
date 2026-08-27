//! Sequential pipeline orchestrator `SequentialPipeline` (P2-3 / P2-5).

use async_trait::async_trait;
use std::sync::Arc;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// Sequential pipeline orchestrator (P2-3 / P2-5).
///
/// Runs N stages in order: each stage's output becomes the next stage's
/// objective; returns the final stage's output. Task-level constraints
/// (expected output / allowed tools) propagate along the chain so all stages
/// stay consistent. Any stage failing independently fails the whole pipeline;
/// errors carry the stage index for easy diagnosis.
pub struct SequentialPipeline {
    stages: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>,
}

impl SequentialPipeline {
    /// Construct from a set of stages executed in sequence.
    pub fn new(stages: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>) -> Self {
        Self { stages }
    }

    /// Append a stage (chainable).
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
            // The stage output becomes the next stage's objective; task-level constraints propagate along the chain (P2-5).
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
