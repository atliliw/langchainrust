use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;

use lc_chains::base::BaseChain;

use crate::protocol::{A2AMessage, A2ATaskResult, TaskFilter, TaskPushNotification, TaskStatus};
use crate::store::TaskStore;

use super::handlers::{publish_artifact, publish_status};
use super::message::{build_chain_input_from_history, extract_output, is_input_required};

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
