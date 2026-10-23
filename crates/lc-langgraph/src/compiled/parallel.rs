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
    ///
    /// Interrupt and checkpoint semantics match [`invoke`](Self::invoke): nodes run
    /// through `run_node`, so a runtime interrupt becomes a
    /// checkpointed, `__checkpoint_id`-stamped [`GraphError::DynamicInterrupt`]
    /// rather than a bare [`GraphError::InterruptRequest`].
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

                // T2: run through the shared `run_node` (like `invoke`), not a raw
                // `node.execute`. A node that suspends itself with
                // `GraphError::InterruptRequest` must surface as a
                // `DynamicInterrupt` with the suspend state persisted and
                // `__checkpoint_id` stamped — otherwise this entry point breaks
                // the contract documented on `GraphError::InterruptRequest` and
                // the interrupt is unrecoverable.
                let update = self
                    .run_node(
                        &state,
                        &current_node,
                        recursion_count,
                        self.node_config(self.recursion_limit, &NodeConfig::default(), None),
                    )
                    .await?;

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
    /// At the top of every step — before the next node runs, and therefore after
    /// the previous one completed (including once before the entry node, so a
    /// task submitted before the run is planned too) — the task inbox is drained.
    /// When tasks are found, the provided `planner` converts them into new
    /// nodes/edges which are injected into the runtime registries before the step
    /// continues.
    ///
    /// # Interrupt and checkpoint semantics
    ///
    /// This entry point upholds the same invariant as [`invoke`](Self::invoke):
    /// **a checkpoint is persisted before any interrupt is surfaced**. Both
    /// interrupt kinds save first — a node-requested runtime
    /// [`GraphError::InterruptRequest`] is translated by
    /// `run_node` into a [`GraphError::DynamicInterrupt`] whose
    /// payload carries `__checkpoint_id`, and the configured
    /// `interrupt_before` / `interrupt_after` boundaries save the state at the
    /// boundary. A crash after an interrupt therefore resumes from the real
    /// boundary instead of re-running nodes that already produced side effects.
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
                // M-15 parity (HIGH, framework review §3.1): the dynamic-routing
                // path previously returned immediately without writing a checkpoint,
                // unlike `invoke`. A crash-then-rebuild resume then had no record of
                // the work completed before the interrupt boundary and re-ran nodes
                // that had already produced side effects. Persist before interrupting
                // exactly like the main `invoke` path does.
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
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            recursion_count += 1;

            // T2: share `run_node` with the main path so a runtime interrupt on
            // the dynamic-routing entry point is translated to a
            // `DynamicInterrupt`, checkpointed at the suspension point, and
            // stamped with `__checkpoint_id` — the same invariant every other
            // entry point upholds. Previously a node-requested interrupt escaped
            // here as a bare `InterruptRequest`: no checkpoint, no stamped id, so
            // `resume_from_interrupt` had nothing to resolve and a crash-resume
            // re-ran nodes that had already produced side effects.
            let update = self
                .run_node(
                    &state,
                    &current_node,
                    recursion_count,
                    self.node_config(self.recursion_limit, &NodeConfig::default(), None),
                )
                .await?;

            if let Some(new_state) = update.update {
                state = self.default_reducer.reduce(&state, &new_state);
            }
            steps.push(ExecutionStep::node(
                current_node.clone(),
                update.metadata.clone(),
            ));

            if self.interrupt_after.contains(&current_node) {
                // M-15 parity (HIGH, framework review §3.1): persist the node's
                // post-run state before interrupting, matching the main `invoke`
                // path. Without this the dynamic-routing path lost the interrupting
                // node's updates on a crash resume (double side effect).
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
            // H-1 (0.28 audit): fold each branch's *delta beyond `base`*, not its
            // whole final state. Every branch starts from a clone of the fan-out
            // `base`, so `final_state[field]` already begins with `base[field]`;
            // folding the whole array in every round re-copied `base` `1 +
            // branch_count` times for append-style reducers (messages/steps
            // silently ballooned). When every branch's field array has `base` as
            // a prefix, compute the suffix past base and concatenate
            // `base ++ Σ deltas` (base counted exactly once); otherwise fall back
            // to the base-seeded reducer fold for exotic / replace-style reducers.
            match delta_union_field(base, branches, field)? {
                Some(union) => {
                    let out = state_with_field(base, field, union)?;
                    merged = patch_field(&merged, &out, field)?;
                }
                None => {
                    let mut acc = base.clone();
                    for branch in branches {
                        acc = reducer.reduce(&acc, &branch.final_state);
                    }
                    merged = patch_field(&merged, &acc, field)?;
                }
            }
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
fn union_states<S: StateSchema>(base: &S, merged: &S, branch: &S) -> GraphResult<S> {
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
        let base_same = base_map.get(k).map(|b| b == v).unwrap_or(false);
        if !base_same {
            result_map.insert(k.clone(), v.clone());
        }
    }
    serde_json::from_value(serde_json::Value::Object(result_map)).map_err(|e| {
        GraphError::StateError(format!(
            "Parallel branch merge failed to reassemble state: {e}"
        ))
    })
}

/// Extract a top-level field's JSON value, or `None` if the state has no such
/// field (or fails to serialize).
fn field_value<S: StateSchema>(s: &S, field: &str) -> Option<serde_json::Value> {
    let v = s.to_json().ok()?;
    let obj = v.as_object()?;
    obj.get(field).cloned()
}

/// H-1 fold: given the fan-out `base` and the branch final states, produce the
/// reducible value for one accumulating field when every branch's array has
/// `base[field]` as a prefix (the common append-style reducer case, e.g.
/// messages/steps). Returns `Some(base[field] ++ Σ suffixes-past-base)`, so
/// `base` is counted exactly once across the fold; returns `None` when any
/// branch lacks the field, its value is not an array, or `base` is not a
/// prefix — the caller then falls back to the reducer-seeded fold.
fn delta_union_field<S: StateSchema>(
    base: &S,
    branches: &[ParallelBranch<S>],
    field: &str,
) -> GraphResult<Option<serde_json::Value>> {
    let Some(base_els) = field_value(base, field).and_then(|v| v.as_array().cloned()) else {
        return Ok(None);
    };
    let mut union: Vec<serde_json::Value> = base_els.clone();
    for branch in branches {
        let Some(arr) = field_value(&branch.final_state, field)
            .and_then(|v| v.as_array().cloned())
        else {
            return Ok(None);
        };
        if arr.len() < base_els.len() || arr[..base_els.len()] != base_els[..] {
            return Ok(None);
        }
        union.extend_from_slice(&arr[base_els.len()..]);
    }
    Ok(Some(serde_json::Value::Array(union)))
}

/// Rebuild `base` with a single top-level field replaced by `value`, so the
/// folded value can be handed to [`patch_field`].
fn state_with_field<S: StateSchema>(
    base: &S,
    field: &str,
    value: serde_json::Value,
) -> GraphResult<S> {
    let mut state_v = base.to_json()?;
    if let Some(map) = state_v.as_object_mut() {
        map.insert(field.to_string(), value);
    }
    serde_json::from_value(state_v).map_err(|e| {
        GraphError::StateError(format!(
            "Parallel reducer delta fold failed to reassemble state: {e}"
        ))
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
        GraphError::StateError(format!(
            "Parallel reducer patch failed to reassemble state: {e}"
        ))
    })
}
