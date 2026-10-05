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

            // 0.22.0 C3 fix: the fan-out source node executes BEFORE branching.
            // Previously the fan-out check ran first and the source node was
            // silently skipped. START is virtual (never executed).
            if current_node != crate::START {
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
                    // 0.26.0 P0-3 (H2): persist the node's post-run state before
                    // interrupting on the parallel invoke / dynamic paths (the
                    // invoke_parallel and invoke_dynamic loops). The main `invoke`
                    // (M-15), resume and streaming paths already persist; these
                    // previously returned immediately, so the node's folded update
                    // never reached a checkpoint and a crash-then-rebuild resume
                    // re-ran the interrupted node (double side effect).
                    if let Some(ref checkpointer) = self.checkpointer {
                        let checkpoint_id = checkpointer
                            .lock()
                            .await
                            .save(&state, recursion_count)
                            .await?;
                        steps.push(ExecutionStep::checkpoint(
                            checkpoint_id,
                            format!("after_{}", current_node),
                        ));
                    }
                    return Err(GraphError::ExecutionInterrupted(format!(
                        "after_{}",
                        current_node
                    )));
                }
            }

            // FanOut: after the source node has run, execute all branches and merge.
            let fan_out_targets = self.find_fan_out_targets(&current_node).await;

            if let Some(targets) = fan_out_targets {
                recursion_count += 1;
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
                // H4: persist the post-merge state like `invoke` does, so resume from
                // this point covers the merged branches.
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
            } else {
                let next_node = self.find_next_node(&current_node, &state).await?;
                // H4: per-step checkpoint on the sequential path, mirroring `invoke`
                // (parallel.rs previously advanced without saving, so an interrupt /
                // crash mid-execution of a mixed parallel+sequential graph could not
                // resume past the last FanOut merge).
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

            // A8 fix: a dynamically-routed edge may enter a FanOut. Previously this
            // path only ran `find_next_node`, which returns `targets[0]` and silently
            // dropped every other branch (state/effects). Handle FanOut exactly like
            // `invoke`: run all branches, then merge on top of the pre-fan-out state.
            let fan_out_targets = self.find_fan_out_targets(&current_node).await;
            if let Some(targets) = fan_out_targets {
                recursion_count += 1;
                let mut parallel_branches: Vec<ParallelBranch<S>> = Vec::new();

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
                let base = state.clone();
                state = self.merge_parallel_states(&parallel_branches, &base)?;
                current_node = merge_target.unwrap_or_else(|| END.to_string());

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

    /// Merges parallel branch states on top of the main-path `base` state.
    ///
    /// 0.22.0 C3 fix: the fold used to start from `branches[0]`, so the last
    /// branch overwrote everything **and** any state the fan-out source node
    /// wrote before branching was dropped. Now `base` (the state at the
    /// fan-out point) is the fold's foundation.
    pub(super) fn merge_parallel_states(
        &self,
        branches: &[ParallelBranch<S>],
        base: &S,
    ) -> GraphResult<S> {
        if branches.is_empty() {
            return Err(GraphError::StateError(
                "Cannot merge states: no parallel branches completed".to_string(),
            ));
        }

        // 0.26.0 P0-4 / 0.22.0 C3: field-level union of parallel branches.
        //
        // The old fold (`default_reducer.reduce(base, b1); reduce(.., b2)`)
        // replaced the WHOLE state each step when no per-field reducer was
        // registered (the default `ReplaceReducer` returns `update.clone()`).
        // So with two branches writing different fields, only the LAST branch
        // survived and earlier branches' unique fields were silently dropped.
        //
        // Now: start from `base` (the state at the fan-out point), then for each
        // branch merge field-by-field — every top-level field the branch wrote
        // is folded in on top of the running union, so multi-branch writes to
        // DIFFERENT fields all survive. A field with a registered reducer is
        // folded through that reducer across ALL branches (concatenating
        // accumulating messages / steps), because dropping that would regress
        // the C3 add_messages semantics.
        //
        // Structure: gather every branch's top-level field set into one JSON
        // union (base ∪ b1 ∪ b2 …), then separately reducer-fold the registered
        // fields over all branches so accumulating fields don't just take the
        // last write.
        let mut merged = base.clone();
        for branch in branches {
            merged = union_states(base, &merged, &branch.final_state)?;
        }
        for (field, reducer) in &self.field_reducers {
            let mut acc = base.clone();
            for branch in branches {
                acc = reducer.reduce(&acc, &branch.final_state);
            }
            merged = patch_field(&merged, &acc, field)?;
        }
        Ok(merged)
    }
}

/// Field-level union of a parallel branch onto an accumulated merge, diffed
/// against `base`.
///
/// Parallelled branch `final_state`s carry the WHOLE state (they started from a
/// clone of `base`, so untouched fields equal base's). A plain "last insert
/// wins" union therefore still lets a later branch's untouched field clobber an
/// earlier branch's write. Instead: a field is only layered in when the branch
/// value DIFFERS from `base`'s value for that key — i.e. the branch actually
/// wrote it. Two branches writing DIFFERENT fields both survive; a branch
/// writing the same value we already carry (written by an earlier branch) keeps
/// the earlier value.
fn union_states<S: StateSchema>(
    base: &S,
    merged: &S,
    branch: &S,
) -> GraphResult<S> {
    let base_v = base.to_json()?;
    let Some(base_map) = base_v.as_object() else {
        // Non-map state can't be unioned field-by-field; fall back to the
        // downstream value (old behavior: last branch wins).
        return Ok(branch.clone());
    };
    let branch_v = branch.to_json()?;
    let Some(branch_map) = branch_v.as_object() else {
        return Ok(merged.clone());
    };
    let merged_v = merged.to_json()?;
    let merged_map = merged_v.as_object().cloned().unwrap_or_default();

    let mut result_map = merged_map;
    for (k, v) in branch_map {
        // Only layer in keys the branch actually changed relative to its input.
        let base_same = base_map
            .get(k)
            .map(|b| b == v)
            .unwrap_or(false);
        if !base_same {
            result_map.insert(k.clone(), v.clone());
        }
    }
    serde_json::from_value(serde_json::Value::Object(result_map)).map_err(|e| {
        GraphError::StateError(format!("Parallel branch merge failed to reassemble state: {e}"))
    })
}

/// Copy a single reducer-managed field from `out` (a reducer result) into a
/// running merged state, leaving every other field untouched.
fn patch_field<S: StateSchema>(state: &S, out: &S, field: &str) -> GraphResult<S> {
    let mut state_v = state.to_json()?;
    let Some(state_map) = state_v.as_object_mut() else {
        return Ok(state.clone());
    };
    let out_v = out.to_json()?;
    let value = out_v
        .as_object()
        .and_then(|m| m.get(field).cloned())
        .unwrap_or(serde_json::Value::Null);
    state_map.insert(field.to_string(), value);
    serde_json::from_value(state_v).map_err(|e| {
        GraphError::StateError(format!("Parallel reducer patch failed to reassemble state: {e}"))
    })
}
