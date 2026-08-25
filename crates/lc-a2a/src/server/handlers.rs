use std::sync::Arc;

use serde_json::json;
use tokio::sync::broadcast;

use crate::protocol::{A2AErrorData, A2AResponse, A2ATaskResult, TaskPushNotification, TaskStatus};
use crate::store::StoredTask;

/// Build a `{task, result?, error?}` response for `tasks/get`.
pub(crate) fn task_details_response(id: u64, stored: &StoredTask) -> A2AResponse {
    let mut result = json!({ "task": stored.task });
    if let Some(ref task_result) = stored.result {
        result["result"] = json!(task_result);
    }
    if let Some(ref error) = stored.error {
        result["error"] = json!(error);
    }
    A2AResponse::ok(id, result)
}

/// `-32001` task-not-found error.
pub(crate) fn task_not_found(id: u64, task_id: &str) -> A2AResponse {
    A2AResponse::from_error_data(
        id,
        A2AErrorData::new(-32001, format!("Task not found: {}", task_id)),
    )
}

/// `-32003` ownership violation error.
pub(crate) fn forbidden(id: u64, message: impl Into<String>) -> A2AResponse {
    A2AResponse::from_error_data(id, A2AErrorData::new(-32003, message))
}

/// Publish a status-update event on the optional event bus (P2-1).
pub(crate) fn publish_status(
    bus: &Option<Arc<broadcast::Sender<TaskPushNotification>>>,
    task_id: &str,
    status: TaskStatus,
    error: Option<&str>,
) {
    if let Some(sender) = bus {
        let event = match error {
            Some(e) => TaskPushNotification::status_with_error(task_id, status, e),
            None => TaskPushNotification::status(task_id, status),
        };
        let _ = sender.send(event);
    }
}

/// Publish an artifact-update event on the optional event bus (P2-1).
pub(crate) fn publish_artifact(
    bus: &Option<Arc<broadcast::Sender<TaskPushNotification>>>,
    task_id: &str,
    artifact: A2ATaskResult,
) {
    if let Some(sender) = bus {
        let _ = sender.send(TaskPushNotification::artifact(task_id, artifact));
    }
}
