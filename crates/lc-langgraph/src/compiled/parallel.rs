// crates/lc-langgraph/src/compiled/parallel.rs
//! CompiledGraph invoke_parallel, invoke_dynamic, and merge_parallel_states methods

use super::graph::CompiledGraph;
use super::types::{
    DynamicInjection, DynamicPlanner, ExecutionStep, GraphInvocation, ParallelBranch,
    ParallelInvocation,
};
use crate::errors::{GraphError, GraphResult};
use crate::node::NodeConfig;
use crate::state::StateSchema;
use crate::END;
use std::collections::HashMap;

impl<S: StateSchema> CompiledGraph<S> {
    /// Run the graph while capturing parallel fan-out branches into a [`ParallelInvocation`].
    pub async fn invoke_parallel(&self, input: S) -> GraphResult<ParallelInvocation<S>> {
        let mut state = input;
        let mut current_node = self.entry_point.clone();
        let mut steps: Vec<ExecutionStep> = Vec::new();
        let mut recursion_count = 0;
        let mut parallel_branches: Vec<ParallelBranch<S>> = Vec::new();

        if let Some(ref checkpointer) = self.checkpointer {
            let checkpoint_id = checkpointer.lock().await.save(&state, 0).await?;
            steps.push(ExecutionStep::checkpoint(
                checkpoint_id,
                current_node.clone(),
            ));
        }

        loop {
            if current_node == END {
                break;
            }

            // Q3: check the limit before executing a step (same guard as `invoke`).
            if recursion_count >= self.recursion_limit {
                return Err(GraphError::RecursionLimitReached(self.recursion_limit));
            }

            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            recursion_count += 1;

            let fan_out_targets = self.find_fan_out_targets(&current_node).await;

            if let Some(targets) = fan_out_targets {
                let branch_results = self
                    .execute_parallel_branches(&targets, &state, recursion_count)
                    .await?;

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
                    // H6: 与 invoke.rs 的主路径一致——所有分支结果经 reducer 合并,
                    // 而不是只保留 `parallel_branches.last()` 一个分支、丢弃其余分支。
                    state = self.merge_parallel_states(&parallel_branches)?;
                    current_node = END.to_string();
                }
            } else {
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
        }

        Ok(ParallelInvocation {
            final_state: state,
            steps,
            recursion_count,
            parallel_branches,
        })
    }

    /// Execute the graph with dynamic task injection support.
    ///
    /// After each node completes, checks the task_inbox for newly submitted tasks.
    /// If tasks are found, uses the provided `planner` to convert them into new
    /// nodes/edges and injects them into the runtime registries before continuing.
    pub async fn invoke_dynamic(
        &self,
        input: S,
        planner: &dyn DynamicPlanner<S>,
    ) -> GraphResult<GraphInvocation<S>> {
        let mut state = input;
        let mut current_node = self.entry_point.clone();
        let mut steps: Vec<ExecutionStep> = Vec::new();
        let mut recursion_count = 0;

        if let Some(ref checkpointer) = self.checkpointer {
            let checkpoint_id = checkpointer.lock().await.save(&state, 0).await?;
            steps.push(ExecutionStep::checkpoint(
                checkpoint_id,
                current_node.clone(),
            ));
        }

        loop {
            if current_node == END {
                break;
            }

            // Q3: check the limit before executing a step (same guard as `invoke`).
            if recursion_count >= self.recursion_limit {
                return Err(GraphError::RecursionLimitReached(self.recursion_limit));
            }

            // Check for pending dynamically-submitted tasks
            let pending = {
                let mut inbox = self.task_inbox.lock().await;
                inbox.drain(..).collect::<Vec<_>>()
            };
            if !pending.is_empty() {
                let injection: DynamicInjection<S> = planner
                    .plan(&pending, &state)
                    .await
                    .map_err(|e| GraphError::RuntimeError(e.to_string()))?;

                {
                    let mut rn = self.runtime_nodes.write().await;
                    for (name, node) in injection.nodes {
                        rn.insert(name, node);
                    }
                }
                {
                    let mut re = self.runtime_edges.write().await;
                    re.extend(injection.edges);
                }
            }

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

            let next_node = self.find_next_node(&current_node, &state).await?;

            if let Some(ref checkpointer) = self.checkpointer {
                let checkpoint_id = checkpointer
                    .lock()
                    .await
                    .save(&state, recursion_count)
                    .await?;
                steps.push(ExecutionStep::checkpoint(checkpoint_id, next_node.clone()));
            }

            current_node = next_node;
        }

        Ok(GraphInvocation {
            final_state: state,
            steps,
            recursion_count,
        })
    }

    pub(super) fn merge_parallel_states(&self, branches: &[ParallelBranch<S>]) -> GraphResult<S> {
        if branches.is_empty() {
            return Err(GraphError::StateError(
                "Cannot merge states: no parallel branches completed".to_string(),
            ));
        }

        let mut merged = branches[0].final_state.clone();
        for branch in branches.iter().skip(1) {
            merged = self.default_reducer.reduce(&merged, &branch.final_state);
        }
        Ok(merged)
    }
}
