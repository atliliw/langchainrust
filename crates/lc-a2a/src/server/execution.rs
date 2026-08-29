use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::broadcast;

use lc_chains::base::BaseChain;

use crate::protocol::{
    A2AMessage, A2ATaskResult, TaskFilter, TaskPushNotification, TaskStatus, WorkflowStep,
};
use crate::store::TaskStore;

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

/// Execute a task in the background: `submitted -> working ->
/// completed/failed/input-required`, guarded by the task state machine so it
/// never clobbers a terminal status (e.g. a task cancelled while the chain
/// was still running).
pub(crate) async fn run_task(
    store: &Arc<dyn TaskStore>,
    chain: Arc<dyn BaseChain>,
    task_id: &str,
    history: Vec<A2AMessage>,
    event_bus: Option<Arc<broadcast::Sender<TaskPushNotification>>>,
) {
    // submitted / input-required -> working
    if let Ok(Some(mut stored)) = store.get(task_id).await {
        if stored.task.status.can_transition_to(&TaskStatus::Working) {
            stored.task.status = TaskStatus::Working;
            stored.touch();
            let _ = store.upsert(stored).await;
            publish_status(&event_bus, task_id, TaskStatus::Working, None);
        }
    }

    let input = build_chain_input_from_history(&history, chain.as_ref());
    match chain.invoke(input).await {
        Ok(result) => {
            let output = extract_output(&result);
            if let Ok(Some(mut stored)) = store.get(task_id).await {
                if stored.task.status.can_transition_to(&TaskStatus::Completed) {
                    stored.task.status = TaskStatus::Completed;
                    stored.result = Some(A2ATaskResult::new(output.clone()));
                    stored.error = None;
                    stored.touch();
                    let _ = store.upsert(stored).await;
                    publish_status(&event_bus, task_id, TaskStatus::Completed, None);
                    publish_artifact(&event_bus, task_id, A2ATaskResult::new(output));
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
                        let _ = store.upsert(stored).await;
                        publish_status(
                            &event_bus,
                            task_id,
                            TaskStatus::InputRequired,
                            Some(&prompt),
                        );
                    }
                }
            } else if let Ok(Some(mut stored)) = store.get(task_id).await {
                if stored.task.status.can_transition_to(&TaskStatus::Failed) {
                    stored.task.status = TaskStatus::Failed;
                    stored.error = Some(e.to_string());
                    stored.touch();
                    let _ = store.upsert(stored).await;
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

/// Execute a workflow in the background: run each step's chain in order, aggregate
/// per-step outputs, and finalize the backing task (`working -> completed/failed`),
/// guarded by the task state machine so a cancelled task is not clobbered.
///
/// The handler acknowledges the `working` task immediately and spawns this; the
/// client polls `tasks/get` for the outcome (0.20.0 S4 G2).
pub(crate) async fn run_workflow(
    store: &Arc<dyn TaskStore>,
    steps: Vec<WorkflowStep>,
    chains: Vec<Arc<dyn BaseChain>>,
    task_id: &str,
    event_bus: Option<Arc<broadcast::Sender<TaskPushNotification>>>,
) {
    let mut results = serde_json::Map::new();
    let mut failure: Option<(String, String)> = None; // (step_id, message)
    for (step, chain) in steps.iter().zip(chains.iter()) {
        let input = build_chain_input(&step.message.content, chain.as_ref());
        match chain.invoke(input).await {
            Ok(result) => {
                let output = extract_output(&result);
                results.insert(step.id.clone(), Value::String(output));
            }
            Err(e) => {
                failure = Some((step.id.clone(), e.to_string()));
                break;
            }
        }
    }

    let aggregated = results
        .values()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let Ok(Some(mut finalize)) = store.get(task_id).await else {
        return;
    };
    match failure {
        Some((step_id, message)) => {
            if finalize.task.status.can_transition_to(&TaskStatus::Failed) {
                finalize.task.status = TaskStatus::Failed;
                finalize.error = Some(format!("step `{step_id}` failed: {message}"));
                let error = finalize.error.clone();
                finalize.touch();
                let _ = store.upsert(finalize).await;
                publish_status(&event_bus, task_id, TaskStatus::Failed, error.as_deref());
            }
        }
        None => {
            if finalize
                .task
                .status
                .can_transition_to(&TaskStatus::Completed)
            {
                finalize.task.status = TaskStatus::Completed;
                let task_result = A2ATaskResult::new(aggregated);
                finalize.result = Some(task_result.clone());
                finalize.error = None;
                finalize.touch();
                let _ = store.upsert(finalize).await;
                publish_status(&event_bus, task_id, TaskStatus::Completed, None);
                publish_artifact(&event_bus, task_id, task_result);
            }
        }
    }
}
