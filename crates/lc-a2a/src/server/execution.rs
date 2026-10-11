use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tracing::Instrument;

use lc_chains::base::BaseChain;

use crate::protocol::{
    A2AArtifact, A2AMessage, A2ATaskResult, TaskFilter, TaskPushNotification, TaskStatus,
    WorkflowStep,
};
use crate::store::{StoredTask, TaskStore};

use super::handlers::{publish_artifact, publish_status};
use super::message::{
    build_chain_input, build_chain_input_from_history, extract_output, is_input_required,
};

/// Maximum number of steps a `workflow/run` request may carry (0.20.0 S4 G2).
///
/// A client-supplied workflow is unbounded input: without a cap, a single request
/// could drive an arbitrarily long serial chain run. The `tasks/runWorkflow`
/// handler rejects workflows beyond this limit with an `invalid_params` error.
pub(crate) const MAX_WORKFLOW_STEPS: usize = 50;

/// RAII guard releasing an in-flight resume claim on drop.
///
/// Uses `std::sync::Mutex` so the release is synchronous and works from `Drop`
/// (no await in drop); the critical section is a single set removal.
pub(crate) struct InflightResume {
    pub(crate) inner: Arc<std::sync::Mutex<HashSet<String>>>,
    pub(crate) task_id: String,
}

impl Drop for InflightResume {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.remove(&self.task_id);
        }
    }
}

/// Sweep tasks older than `ttl` (P1-2).
///
/// Terminal tasks are deleted to bound memory; live tasks that may transition
/// are marked `expired`. Shared by the lazy read-path cleanup and the
/// background [`super::A2AServer::with_background_cleanup`] sweeper.
pub(crate) async fn sweep_expired_tasks(store: &Arc<dyn TaskStore>, ttl: Duration) {
    let Ok(list) = store.list(&TaskFilter::new()).await else {
        return;
    };
    for stored in list {
        if stored.age() < ttl {
            continue;
        }
        if stored.task.status.is_terminal() {
            // Free memory for tasks that have been terminal for > ttl.
            let _ = store.delete(&stored.task.id).await;
        } else if stored.task.status.can_transition_to(&TaskStatus::Expired) {
            let mut expired = stored;
            expired.task.status = TaskStatus::Expired;
            expired.touch();
            let _ = store.upsert(expired).await;
        }
    }
}

/// Runs a compare-and-update transition, reporting conflicts instead of
/// silently dropping them (0.27.0 T2).
///
/// `stored` must already carry the intended next status (the atomic CAS swaps
/// it as-is). Returns `true` when this writer won and the caller should publish
/// its own transition event.
///
/// On `false` a concurrent writer moved the task before ours landed — most
/// commonly a `tasks/cancel` flipping it terminal in the 0.22.0 audit window.
/// That is no longer swallowed: a `tracing::warn` reports the lost race (the
/// winner already broadcast its own state, so we deliberately do *not* double
/// emit). A store `Err` is also surfaced via `tracing::error` instead of the
/// old `.unwrap_or(false)` that collapsed both failure modes into silence.
async fn cas_apply(store: &Arc<dyn TaskStore>, task_id: &str, stored: &StoredTask) -> bool {
    let intended = stored.task.status;
    match store.compare_and_update(task_id, stored.clone()).await {
        Ok(true) => true,
        Ok(false) => {
            tracing::warn!(
                task_id,
                %intended,
                "CAS transition lost to a concurrent writer (e.g. tasks/cancel); the winner already broadcast its own update — not re-publishing",
            );
            false
        }
        Err(e) => {
            tracing::error!(task_id, %intended, error = %e, "store rejected CAS transition");
            false
        }
    }
}

/// Execute a task in the background: `submitted -> working ->
/// completed/failed/input-required`, guarded by the task state machine so it
/// never clobbers a terminal status (e.g. a task cancelled while the chain
/// was still running).
///
/// Instrumented with a `tracing` span so a trace can follow a single task
/// through its transitions (0.27.0 T2: observability 埋点 / trace 贯通).
#[tracing::instrument(skip_all, fields(task_id = %task_id))]
pub(crate) async fn run_task(
    store: &Arc<dyn TaskStore>,
    chain: Arc<dyn BaseChain>,
    task_id: &str,
    history: Vec<A2AMessage>,
    event_bus: Option<Arc<broadcast::Sender<TaskPushNotification>>>,
    cancel: Arc<tokio::sync::Notify>,
) {
    // submitted / input-required -> working
    if let Ok(Some(mut stored)) = store.get(task_id).await {
        if stored.task.status.can_transition_to(&TaskStatus::Working) {
            stored.task.status = TaskStatus::Working;
            stored.touch();
            // 0.22.0 audit fix: compare-and-update so a terminal state written
            // concurrently (e.g. tasks/cancel) is not clobbered by this write.
            if cas_apply(store, task_id, &stored).await {
                tracing::debug!("task transitioning to working");
                publish_status(&event_bus, task_id, TaskStatus::Working, None);
            }
        }
    }

    let input = build_chain_input_from_history(&history, chain.as_ref());
    // B7 point 6: run the chain inside a select so a `tasks/cancel` that flips
    // the task to `cancelled` (and fires the per-task Notify) *terminates* the
    // in-flight execution by dropping the invoke future — it no longer just
    // marks a status and lets the chain keep burning cycles.
    let outcome = tokio::select! {
        result = chain.invoke(input) => result,
        _ = cancel.notified() => {
            // Already flipped to `cancelled` by tasks/cancel's CAS before this
            // notification fired; do not transition to a live state.
            tracing::debug!("task cancelled while chain in flight");
            return;
        }
    };
    match outcome {
        Ok(result) => {
            let output = extract_output(&result);
            if let Ok(Some(mut stored)) = store.get(task_id).await {
                if stored.task.status.can_transition_to(&TaskStatus::Completed) {
                    stored.task.status = TaskStatus::Completed;
                    stored.result = Some(A2ATaskResult::new(output.clone()));
                    stored.error = None;
                    stored.touch();
                    // 0.22.0 audit fix: CAS so a concurrent cancel/complete
                    // cannot be overwritten (check-then-act race).
                    if cas_apply(store, task_id, &stored).await {
                        tracing::debug!(output_len = output.chars().count(), "task completed");
                        publish_status(&event_bus, task_id, TaskStatus::Completed, None);
                        // The streamed artifact carries the output as a text
                        // part (A2A v0.3 TaskArtifact), not a bare string.
                        publish_artifact(&event_bus, task_id, A2AArtifact::text(task_id, output));
                    }
                }
            }
        }
        Err(e) => {
            if is_input_required(&e) {
                // P2-3: the chain is asking the client for more information.
                let prompt = e.to_string();
                if let Ok(Some(mut stored)) = store.get(task_id).await {
                    if stored
                        .task
                        .status
                        .can_transition_to(&TaskStatus::InputRequired)
                    {
                        stored.task.status = TaskStatus::InputRequired;
                        stored.error = Some(prompt.clone());
                        stored.touch();
                        // 0.22.0 audit fix: CAS transition (see above).
                        if cas_apply(store, task_id, &stored).await {
                            tracing::debug!("task awaiting more input");
                            publish_status(
                                &event_bus,
                                task_id,
                                TaskStatus::InputRequired,
                                Some(&prompt),
                            );
                        }
                    }
                }
            } else if let Ok(Some(mut stored)) = store.get(task_id).await {
                if stored.task.status.can_transition_to(&TaskStatus::Failed) {
                    stored.task.status = TaskStatus::Failed;
                    stored.error = Some(e.to_string());
                    stored.touch();
                    // 0.22.0 audit fix: CAS transition (see above).
                    if cas_apply(store, task_id, &stored).await {
                        tracing::error!(error = %e, "task failed");
                        publish_status(
                            &event_bus,
                            task_id,
                            TaskStatus::Failed,
                            Some(&e.to_string()),
                        );
                    }
                }
            }
        }
    }
}

/// Execute a workflow in the background: run each step's chain in order, aggregate
/// per-step outputs, and finalize the backing task (`working -> completed/failed`),
/// guarded by the task state machine so a cancelled task is not clobbered.
///
/// The handler acknowledges the `working` task immediately and spawns this; the
/// client polls `tasks/get` for the outcome (0.20.0 S4 G2).
///
/// Instrumented with a `tracing` span (0.27.0 T2: observability 埋点).
#[tracing::instrument(skip_all, fields(task_id = %task_id))]
pub(crate) async fn run_workflow(
    store: &Arc<dyn TaskStore>,
    steps: Vec<WorkflowStep>,
    chains: Vec<Arc<dyn BaseChain>>,
    task_id: &str,
    event_bus: Option<Arc<broadcast::Sender<TaskPushNotification>>>,
) {
    let mut results: Vec<(String, String)> = Vec::new();
    let mut failure: Option<(String, String)> = None; // (step_id, message)
    for (step, chain) in steps.iter().zip(chains.iter()) {
        let span = tracing::debug_span!("run_workflow.step", step_id = %step.id);
        let input = build_chain_input(&step.message.content, chain.as_ref());
        match chain.invoke(input).instrument(span).await {
            Ok(result) => {
                let output = extract_output(&result);
                tracing::debug!(step_id = %step.id, "workflow step completed");
                // B2 (0.26.0): aggregate as an ordered `Vec<(step_id, output)>`
                // keyed by execution index, NOT a serde_json::Map — with
                // `preserve_order` off the map is a BTreeMap that reorders by
                // step_id alphabetically, so `["s2","s1"]` became
                // `"second\nfirst"`, breaking the documented execution order.
                results.push((step.id.clone(), output));
            }
            Err(e) => {
                tracing::warn!(step_id = %step.id, error = %e, "workflow step failed");
                failure = Some((step.id.clone(), e.to_string()));
                break;
            }
        }
    }

    let aggregated = results
        .iter()
        .map(|(_, output)| output.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let Ok(Some(mut finalize)) = store.get(task_id).await else {
        return;
    };
    match failure {
        Some((step_id, message)) => {
            if finalize.task.status.can_transition_to(&TaskStatus::Failed) {
                finalize.task.status = TaskStatus::Failed;
                // B2 (0.26.0): a failing workflow no longer discards the outputs
                // of steps that already ran — routes.rs:432 documents that "the
                // results already aggregated are still returned" on failure, but
                // the code only ever wrote `error` and never `result`. Surface
                // the partial aggregation (may be empty when the first step
                // failed) so callers see what completed before the failure.
                if !aggregated.is_empty() {
                    finalize.result = Some(A2ATaskResult::new(aggregated.clone()));
                }
                finalize.error = Some(format!("step `{step_id}` failed: {message}"));
                let error = finalize.error.clone();
                finalize.touch();
                // 0.22.0 audit fix: CAS so a concurrent cancel cannot be
                // overwritten by the workflow's failure write.
                if cas_apply(store, task_id, &finalize).await {
                    publish_status(&event_bus, task_id, TaskStatus::Failed, error.as_deref());
                }
            }
        }
        None => {
            if finalize
                .task
                .status
                .can_transition_to(&TaskStatus::Completed)
            {
                finalize.task.status = TaskStatus::Completed;
                let task_result = A2ATaskResult::new(aggregated.clone());
                finalize.result = Some(task_result);
                finalize.error = None;
                finalize.touch();
                // 0.22.0 audit fix: CAS so a concurrent cancel cannot be
                // overwritten by the workflow's completion write.
                if cas_apply(store, task_id, &finalize).await {
                    publish_status(&event_bus, task_id, TaskStatus::Completed, None);
                    publish_artifact(&event_bus, task_id, A2AArtifact::text(task_id, aggregated));
                }
            }
        }
    }
}
