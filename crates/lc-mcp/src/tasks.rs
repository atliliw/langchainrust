//! MCP Tasks extension — durable handles for long-running operations (T3).
//!
//! The spec's Tasks extension lets a server hand a client a durable task handle
//! instead of a one-shot answer, so genuinely long work (a batch job, an agent
//! run) can be polled with `tasks/get` and cancelled with `tasks/cancel` rather
//! than blocking a request for its whole lifetime. A tool call may echo a task
//! object back; the client then polls that handle for completion.
//!
//! This module supplies the wire types plus a [`TaskProvider`] a server
//! registers. Consistent with every other MCP surface here, an unregistered
//! server still answers `tasks/*` with `method_not_found` (an honest boundary,
//! not a silent pretend).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::MCPError;

/// Lifecycle state of a server-side task (spec field `status.state`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// The task is waiting to run.
    #[serde(rename = "queued")]
    Queued,
    /// The task is executing.
    #[serde(rename = "running")]
    Running,
    /// The task finished successfully.
    #[serde(rename = "completed")]
    Completed,
    /// The task failed; a `reason` may accompany it.
    #[serde(rename = "failed")]
    Failed,
    /// The task was cancelled before completion.
    #[serde(rename = "cancelled")]
    Cancelled,
}

/// The `status` object field of a task: a lifecycle state plus `extensions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    /// Lifecycle state of the task.
    pub state: TaskState,
    /// Optional provider-specific metadata (e.g. a failure `reason`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Extensions bag carried verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
}

/// Tells a client how to continue fetching a task (`shipping_label`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShippingLabel {
    /// The method to call to continue (e.g. `tasks/get`).
    pub method: String,
    /// The protocol version the shipping label is valid for.
    pub protocol_version: String,
}

/// `tasks/get` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTaskParams {
    /// The opaque task id issued by the server.
    pub id: String,
}

/// The full task object returned by `tasks/get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponse {
    /// Opaque task id.
    pub id: String,
    /// Human-readable title of the task.
    pub title: String,
    /// Task lifecycle status.
    pub status: TaskStatus,
    /// Discriminator; always `"task"` on the wire.
    #[serde(rename = "type", default = "default_task_kind")]
    pub kind: String,
    /// How the client continues polling — `tasks/get` at the server's version.
    pub shipping_label: ShippingLabel,
    /// When the task's status last changed (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
    /// Whether a mid-flight/interrupted task can be resumed.
    #[serde(default)]
    pub restorable: bool,
    /// `auto` = server picks the executor; `interactive` = human assist needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_choice: Option<String>,
    /// When the task is scheduled to run, if deferred (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduled_for_time: Option<String>,
    /// Extensions bag carried verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
}

fn default_task_kind() -> String {
    "task".to_string()
}

/// `tasks/cancel` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelTaskParams {
    /// The opaque task id to cancel.
    pub id: String,
}

/// Task provider: once a server registers, `tasks/get` / `tasks/cancel` have a
/// data source. An unregistered server still answers `tasks/*` with
/// `method_not_found` (an honest boundary).
#[async_trait]
pub trait TaskProvider: Send + Sync {
    /// Polls a task by id, returning its current status and shipping label.
    /// Returns an error when the id is unknown.
    async fn get_task(&self, id: &str) -> Result<TaskResponse, MCPError>;

    /// Cancels a task by id. Returns `true` when the task was running/queued
    /// and is now cancelled; `false` when it is already finished.
    ///
    /// Providers that do not support cancellation can leave this default (an
    /// explicit unsupported error) rather than pretending to cancel.
    async fn cancel_task(&self, id: &str) -> Result<bool, MCPError> {
        let _ = id;
        Err(MCPError::method_not_found())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_state_wire_values() {
        assert_eq!(
            serde_json::to_value(TaskState::Queued).unwrap(),
            Value::String("queued".into())
        );
        assert_eq!(
            serde_json::to_value(TaskState::Cancelled).unwrap(),
            Value::String("cancelled".into())
        );
        let running: TaskState = serde_json::from_value("running".into()).unwrap();
        assert_eq!(running, TaskState::Running);
    }

    #[test]
    fn task_response_roundtrip() {
        let resp = TaskResponse {
            id: "t-1".to_string(),
            title: "summarize log".to_string(),
            status: TaskStatus {
                state: TaskState::Running,
                reason: None,
                extensions: None,
            },
            kind: "task".to_string(),
            shipping_label: ShippingLabel {
                method: "tasks/get".to_string(),
                protocol_version: "2026-07-28".to_string(),
            },
            last_updated: None,
            restorable: false,
            runner_choice: Some("auto".to_string()),
            scheduled_for_time: None,
            extensions: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "task");
        assert_eq!(json["status"]["state"], "running");
        assert_eq!(json["shipping_label"]["method"], "tasks/get");
        let back: TaskResponse = serde_json::from_value(json).unwrap();
        assert_eq!(back.id, "t-1");
    }

    #[test]
    fn failed_task_carries_reason() {
        let status = TaskStatus {
            state: TaskState::Failed,
            reason: Some("timeout".to_string()),
            extensions: None,
        };
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["state"], "failed");
        assert_eq!(json["reason"], "timeout");
    }
}