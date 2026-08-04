// crates/lc-langgraph/src/compiled/invoke.rs
//! CompiledGraph invoke, invoke_with_execution, resume, and invoke_from_node methods

use super::graph::CompiledGraph;
use super::types::{ExecutionStep, GraphExecution, GraphInvocation, ParallelBranch};
use crate::errors::{GraphError, GraphResult};
use crate::node::NodeConfig;
use crate::state::StateSchema;
use crate::END;
use std::collections::HashMap;

impl<S: StateSchema> CompiledGraph<S> {
    pub async fn invoke(&self, input: S) -> GraphResult<GraphInvocation<S>> {
        let mut state = input;
        let mut current_node = self.entry_point.clone();
        let mut steps: Vec<ExecutionStep> = Vec::new();
        let mut recursion_count = 0;

        if let Some(ref checkpointer) = self.checkpointer {
            let checkpoint_id = checkpointer.lock().await.save(&state).await?;
            steps.push(ExecutionStep::checkpoint(
                checkpoint_id,
                current_node.clone(),
            ));
        }

        while current_node != END && recursion_count < self.recursion_limit {
            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            // Check for FanOut edge — execute all branches in parallel and merge
            let fan_out_targets = self.find_fan_out_targets(&current_node).await;
            if let Some(targets) = fan_out_targets {
                recursion_count += 1;
                let mut parallel_branches: Vec<ParallelBranch<S>> = Vec::new();

                let branch_results = self.execute_parallel_branches(&targets, &state).await?;
                for (name, inv) in branch_results {
                    parallel_branches.push(ParallelBranch {
                        name: name.clone(),
                        final_state: inv.final_state.clone(),
                        steps: inv.steps.clone(),
                    });
                    steps.push(ExecutionStep::ParallelNode {
                        branch: name,
                        metadata: HashMap::new(),
                    });
                }

                let merge_target = self.find_fan_in_target(&targets).await;
                if let Some(merge_node) = merge_target {
                    state = self.merge_parallel_states(&parallel_branches)?;
                    current_node = merge_node;
                } else {
                    state = self.merge_parallel_states(&parallel_branches)?;
                    current_node = END.to_string();
                }

                if let Some(ref checkpointer) = self.checkpointer {
                    let checkpoint_id = checkpointer.lock().await.save(&state).await?;
                    steps.push(ExecutionStep::checkpoint(
                        checkpoint_id,
                        current_node.clone(),
                    ));
                }
                continue;
            }

            recursion_count += 1;

            let node = self.get_node(&current_node).await?;

            let config = NodeConfig {
                recursion_limit: self.recursion_limit,
                debug: false,
                metadata: HashMap::new(),
            };

            let update = node.execute(&state, Some(config)).await?;

            if let Some(new_state) = update.update {
                state = self.default_reducer.reduce(&state, &new_state);
            }

            steps.push(ExecutionStep::node(
                current_node.clone(),
                update.metadata.clone(),
            ));

            if self.interrupt_after.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(format!(
                    "after_{}",
                    current_node
                )));
            }

            let next_node = self.find_next_node(&current_node, &state).await?;

            if let Some(ref checkpointer) = self.checkpointer {
                let checkpoint_id = checkpointer.lock().await.save(&state).await?;
                steps.push(ExecutionStep::checkpoint(checkpoint_id, next_node.clone()));
            }

            current_node = next_node;
        }

        if recursion_count >= self.recursion_limit {
            return Err(GraphError::RecursionLimitReached(self.recursion_limit));
        }

        Ok(GraphInvocation {
            final_state: state,
            steps,
            recursion_count,
        })
    }

    pub async fn invoke_with_execution(
        &self,
        execution: GraphExecution<S>,
    ) -> GraphResult<GraphInvocation<S>> {
        let mut state = execution.state;
        let mut current_node = if execution.interrupted_at.starts_with("after_") {
            self.find_next_node(&execution.current_node, &state).await?
        } else {
            execution.current_node
        };
        let mut steps = execution.steps;
        let mut recursion_count = execution.recursion_count;
        let first_node = current_node.clone();

        while current_node != END && recursion_count < self.recursion_limit {
            if current_node != first_node && self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            recursion_count += 1;

            let node = self.get_node(&current_node).await?;

            let config = NodeConfig {
                recursion_limit: self.recursion_limit,
                debug: false,
                metadata: HashMap::new(),
            };

            let update = node.execute(&state, Some(config)).await?;

            if let Some(new_state) = update.update {
                state = self.default_reducer.reduce(&state, &new_state);
            }

            steps.push(ExecutionStep::node(
                current_node.clone(),
                update.metadata.clone(),
            ));

            if self.interrupt_after.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(format!(
                    "after_{}",
                    current_node
                )));
            }

            let next_node = self.find_next_node(&current_node, &state).await?;

            if let Some(ref checkpointer) = self.checkpointer {
                let checkpoint_id = checkpointer.lock().await.save(&state).await?;
                steps.push(ExecutionStep::checkpoint(checkpoint_id, next_node.clone()));
            }

            current_node = next_node;
        }

        if recursion_count >= self.recursion_limit {
            return Err(GraphError::RecursionLimitReached(self.recursion_limit));
        }

        Ok(GraphInvocation {
            final_state: state,
            steps,
            recursion_count,
        })
    }

    pub async fn resume(&self, execution: GraphExecution<S>) -> GraphResult<GraphInvocation<S>> {
        self.invoke_with_execution(execution).await
    }

    pub async fn invoke_from_node(
        &self,
        start_node: String,
        input: S,
    ) -> GraphResult<GraphInvocation<S>> {
        let mut state = input;
        let mut current_node = start_node;
        let mut steps: Vec<ExecutionStep> = Vec::new();
        let mut recursion_count = 0;

        while current_node != END && recursion_count < self.recursion_limit {
            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            recursion_count += 1;

            let node = self.get_node(&current_node).await?;

            let config = NodeConfig {
                recursion_limit: self.recursion_limit,
                debug: false,
                metadata: HashMap::new(),
            };

            let update = node.execute(&state, Some(config)).await?;

            if let Some(new_state) = update.update {
                state = self.default_reducer.reduce(&state, &new_state);
            }

            steps.push(ExecutionStep::node(
                current_node.clone(),
                update.metadata.clone(),
            ));

            if self.interrupt_after.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(format!(
                    "after_{}",
                    current_node
                )));
            }

            current_node = self.find_next_node(&current_node, &state).await?;
        }

        Ok(GraphInvocation {
            final_state: state,
            steps,
            recursion_count,
        })
    }
}
