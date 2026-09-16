// crates/lc-langgraph/src/compiled/invoke.rs
//! CompiledGraph invoke, invoke_with_execution, resume, and invoke_from_node methods

use super::graph::CompiledGraph;
use super::types::{ExecutionStep, GraphExecution, GraphInvocation, ParallelBranch};
use crate::errors::{GraphError, GraphResult};
use crate::node::{NodeConfig, INTERRUPT_RESUME_KEY};
use crate::state::{StateSchema, StateUpdate};
use crate::END;
use std::collections::HashMap;

impl<S: StateSchema> CompiledGraph<S> {
    /// Run a single node, translating a runtime [`GraphError::InterruptRequest`]
    /// into a producer-facing [`GraphError::DynamicInterrupt`]. Before surfacing
    /// the interrupt the current state is persisted (when a checkpointer is
    /// attached) so a crash between interrupt and resume cannot lose the run.
    async fn run_node(
        &self,
        state: &S,
        current_node: &str,
        recursion_count: usize,
        resume: Option<&serde_json::Value>,
    ) -> GraphResult<StateUpdate<S>> {
        let node = self.get_node(current_node).await?;
        let mut metadata = HashMap::new();
        if let Some(value) = resume {
            metadata.insert(INTERRUPT_RESUME_KEY.to_string(), value.clone());
        }
        let config = NodeConfig {
            recursion_limit: self.recursion_limit,
            debug: false,
            metadata,
        };
        match node.execute(state, Some(config)).await {
            Err(GraphError::InterruptRequest { ref payload }) => {
                if let Some(ref cp) = self.checkpointer {
                    cp.lock().await.save(state, recursion_count).await?;
                }
                Err(GraphError::DynamicInterrupt {
                    node: current_node.to_string(),
                    payload: payload.clone(),
                })
            }
            other => other,
        }
    }
    /// Run the graph from its entry point with the given input state.
    pub async fn invoke(&self, input: S) -> GraphResult<GraphInvocation<S>> {
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

            // Q3: check the limit BEFORE executing a step. A `loop` with this
            // guard means a graph that legitimately uses exactly `limit` steps
            // and then reaches END is NOT misreported as exceeding the limit
            // (the old `count >= limit` post-check fired at `count == limit`).
            if recursion_count >= self.recursion_limit {
                return Err(GraphError::RecursionLimitReached(self.recursion_limit));
            }

            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            // 0.22.0 C3 fix: the fan-out source node executes BEFORE branching
            // (previously the fan-out check ran first and the source node was
            // silently skipped). START is a virtual entry — never executed.
            if current_node != crate::START {
                recursion_count += 1;

                let update = self
                    .run_node(&state, &current_node, recursion_count, None)
                    .await?;

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
            }

            // FanOut: after the source node has run, execute all branches in
            // parallel and merge (on top of the pre-fan-out state).
            let fan_out_targets = self.find_fan_out_targets(&current_node).await;
            if let Some(targets) = fan_out_targets {
                recursion_count += 1;
                let mut parallel_branches: Vec<ParallelBranch<S>> = Vec::new();

                // Q6: branches share the main-path recursion budget so the limit
                // cannot be bypassed by fanning out into deep sub-executions.
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
                // C3: fold on top of the pre-fan-out main-path state.
                let base = state.clone();
                if let Some(merge_node) = merge_target {
                    state = self.merge_parallel_states(&parallel_branches, &base)?;
                    current_node = merge_node;
                } else {
                    state = self.merge_parallel_states(&parallel_branches, &base)?;
                    current_node = END.to_string();
                }

                if let Some(ref checkpointer) = self.checkpointer {
                    let checkpoint_id = checkpointer
                        .lock()
                        .await
                        .save(&state, recursion_count)
                        .await?;
                    steps.push(ExecutionStep::checkpoint(
                        checkpoint_id,
                        current_node.clone(),
                    ));
                }
                continue;
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

    /// Continue execution from a saved [`GraphExecution`] (e.g. after an interrupt).
    pub async fn invoke_with_execution(
        &self,
        execution: GraphExecution<S>,
    ) -> GraphResult<GraphInvocation<S>> {
        let mut state = execution.state;
        // A node-requested (runtime) interrupt always re-enters the interrupted
        // node with the human's decision. A compile-time `after_` interrupt skips
        // the node (it already ran) and continues at its successor.
        let rerun_interrupted = execution.pending_interrupt.is_some()
            || !execution.interrupted_at.starts_with("after_");
        let mut current_node = if rerun_interrupted {
            execution.current_node
        } else {
            self.find_next_node(&execution.current_node, &state).await?
        };
        let mut steps = execution.steps;
        let mut recursion_count = execution.recursion_count;
        let mut resume_value = execution.pending_interrupt.map(|p| p.value);
        let first_node = current_node.clone();

        loop {
            if current_node == END {
                break;
            }

            // Q3: check the limit before executing a step (same guard as `invoke`).
            if recursion_count >= self.recursion_limit {
                return Err(GraphError::RecursionLimitReached(self.recursion_limit));
            }

            if current_node != first_node && self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            recursion_count += 1;

            // Only the resume target node receives the injected decision.
            let inject = resume_value.take();
            let update = self
                .run_node(&state, &current_node, recursion_count, inject.as_ref())
                .await?;

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

    /// Resume execution from the given execution context.
    pub async fn resume(&self, execution: GraphExecution<S>) -> GraphResult<GraphInvocation<S>> {
        self.invoke_with_execution(execution).await
    }

    /// Run the graph starting from the given node with the given input state.
    pub async fn invoke_from_node(
        &self,
        start_node: String,
        input: S,
    ) -> GraphResult<GraphInvocation<S>> {
        self.invoke_from_node_with_count(start_node, input, 0).await
    }

    /// Like [`invoke_from_node`](Self::invoke_from_node), but continues from an
    /// existing recursion count.
    ///
    /// Q3/Q6: this is the single enforcement point for the recursion limit on
    /// the "start from an arbitrary node" path. Parallel FanOut branches call
    /// this with the main path's current count so their depth stays visible to
    /// the shared `recursion_limit` budget instead of restarting from zero.
    pub(super) async fn invoke_from_node_with_count(
        &self,
        start_node: String,
        input: S,
        mut recursion_count: usize,
    ) -> GraphResult<GraphInvocation<S>> {
        let mut state = input;
        let mut current_node = start_node;
        let mut steps: Vec<ExecutionStep> = Vec::new();

        loop {
            if current_node == END {
                break;
            }

            if recursion_count >= self.recursion_limit {
                return Err(GraphError::RecursionLimitReached(self.recursion_limit));
            }

            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            recursion_count += 1;

            let update = self
                .run_node(&state, &current_node, recursion_count, None)
                .await?;

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
