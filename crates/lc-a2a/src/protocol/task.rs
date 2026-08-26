use serde::{Deserialize, Serialize};

use super::message::A2AMessage;

/// Task lifecycle status.
///
/// Serialized using the wire names from the A2A v0.3 specification
/// (`input-required`, `auth-required`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task has been submitted but not yet started.
    #[serde(rename = "submitted")]
    Submitted,
    /// Task is currently being processed.
    #[serde(rename = "working")]
    Working,
    /// Task requires additional input from the user.
    #[serde(rename = "input-required")]
    InputRequired,
    /// Task completed successfully.
    #[serde(rename = "completed")]
    Completed,
    /// Task failed.
    #[serde(rename = "failed")]
    Failed,
    /// Task was cancelled.
    #[serde(rename = "cancelled")]
    Cancelled,
    /// Task was rejected by the agent (e.g. refused work).
    #[serde(rename = "rejected")]
    Rejected,
    /// Task requires authentication before it can proceed.
    #[serde(rename = "auth-required")]
    AuthRequired,
    /// Task expired before reaching a terminal state.
    #[serde(rename = "expired")]
    Expired,
}

impl TaskStatus {
    /// Whether a task in this status may legally transition to `target`.
    ///
    /// This is the A2A task lifecycle state machine:
    ///
    /// ```text
    ///               ┌──────────────┐
    ///               ▼              │
    /// auth-required ─┐            working ──→ completed
    ///                │             │  ├──→ failed
    /// submitted ─────┼─→ working ──┘  ├──→ input-required ──→ working
    ///    │           │                └──→ cancelled
    ///    ├──→ rejected
    ///    ├──→ cancelled
    ///    └──→ expired
    /// ```
    ///
    /// Terminal states (`completed`, `failed`, `cancelled`, `rejected`,
    /// `expired`) have no outgoing transitions.
    pub fn can_transition_to(&self, target: &TaskStatus) -> bool {
        use TaskStatus::*;
        matches!(
            (self, target),
            (Submitted, Working)
                | (Submitted, Rejected)
                | (Submitted, Cancelled)
                | (Submitted, Expired)
                | (Working, Completed)
                | (Working, Failed)
                | (Working, InputRequired)
                | (Working, Cancelled)
                | (Working, Expired)
                | (InputRequired, Working)
                | (InputRequired, Cancelled)
                | (InputRequired, Expired)
                | (AuthRequired, Submitted)
                | (AuthRequired, Expired)
        )
    }

    /// Whether this status is terminal (no further transitions allowed).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskStatus::Completed
                | TaskStatus::Failed
                | TaskStatus::Cancelled
                | TaskStatus::Rejected
                | TaskStatus::Expired
        )
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TaskStatus::Submitted => "submitted",
            TaskStatus::Working => "working",
            TaskStatus::InputRequired => "input-required",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
            TaskStatus::Rejected => "rejected",
            TaskStatus::AuthRequired => "auth-required",
            TaskStatus::Expired => "expired",
        };
        f.write_str(s)
    }
}

/// A unit of work in the A2A protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2ATask {
    /// Unique task identifier.
    pub id: String,
    /// The message that initiated this task.
    ///
    /// Kept for single-message compatibility; the full history lives in
    /// [`messages`](A2ATask::messages) and always starts with this message.
    pub message: A2AMessage,
    /// Current status of the task.
    pub status: TaskStatus,
    /// Identifier of the caller/organization that created this task (P1-4).
    ///
    /// Used for ownership authorization: `tasks/get`/`tasks/cancel` from a
    /// different caller are rejected with a `403`-style error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Full message history for multi-turn dialogue (P2-2).
    ///
    /// The first element is the initiating message (equal to [`message`](Self::message));
    /// subsequent turns are appended by `tasks/send` with a `taskId`. The
    /// chain is invoked over this whole history for continued tasks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<A2AMessage>,
}

impl A2ATask {
    /// Create a new task with `Submitted` status.
    pub fn new(id: impl Into<String>, message: A2AMessage) -> Self {
        let id = id.into();
        Self {
            messages: vec![message.clone()],
            id,
            message,
            status: TaskStatus::Submitted,
            owner: None,
        }
    }

    /// Set the task status.
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
    }

    /// Set the task owner (P1-4).
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Append a message to the multi-turn history (P2-2).
    pub fn push_message(&mut self, msg: A2AMessage) {
        self.messages.push(msg);
    }

    /// The full message history for this task (P2-2).
    ///
    /// Guaranteed to be non-empty and to start with the initiating message.
    /// Borrows the in-memory history when populated; falls back to a
    /// single-element owned history for tasks deserialized from an
    /// old single-message wire payload.
    pub fn message_history(&self) -> std::borrow::Cow<'_, [A2AMessage]> {
        if self.messages.is_empty() {
            std::borrow::Cow::Owned(vec![self.message.clone()])
        } else {
            std::borrow::Cow::Borrowed(&self.messages)
        }
    }
}
