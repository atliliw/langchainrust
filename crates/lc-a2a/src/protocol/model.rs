use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::message::A2AMessage;
use super::task::{A2ATask, TaskStatus};

pub(crate) fn default_protocol_version() -> String {
    "0.3.0".to_string()
}

pub(crate) fn default_input_modes() -> Vec<String> {
    vec!["text".to_string()]
}

pub(crate) fn default_output_modes() -> Vec<String> {
    vec!["text".to_string()]
}

/// Result of a completed A2A task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2ATaskResult {
    /// Output text from the task.
    pub output: String,
}

impl A2ATaskResult {
    /// Create a new task result.
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
        }
    }
}

/// Detailed view of a task including its result and any error message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2ATaskDetails {
    /// The task itself.
    pub task: A2ATask,
    /// Result of the task (present when the task completed).
    pub result: Option<A2ATaskResult>,
    /// Error message (present when the task failed).
    pub error: Option<String>,
}

/// A multi-step orchestration submitted via `tasks/runWorkflow` (P2-8).
///
/// Steps execute in order on the server; each step is an instruction (a
/// message) that can be routed to a different skill/chain. Results are
/// aggregated per step so the caller sees which output came from where.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AWorkflow {
    /// Caller-supplied workflow id, reused as the backing task id when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,
    /// Optional human-readable name (advisory only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Ordered steps to execute.
    pub steps: Vec<WorkflowStep>,
}

impl A2AWorkflow {
    /// Create a workflow from an ordered list of steps.
    pub fn new(steps: Vec<WorkflowStep>) -> Self {
        Self {
            workflow_id: None,
            name: None,
            steps,
        }
    }

    /// Attach a caller-supplied workflow id (reused as the backing task id).
    pub fn with_workflow_id(mut self, id: impl Into<String>) -> Self {
        self.workflow_id = Some(id.into());
        self
    }

    /// Attach an advisory name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// A single unit of work within an [`A2AWorkflow`] (P2-8).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Unique step identifier within its workflow; names the step's result.
    pub id: String,
    /// The instruction executed for this step.
    pub message: A2AMessage,
    /// Optional skill to route this step to a specialized chain (P2-4).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill_id: Option<String>,
}

impl WorkflowStep {
    /// Create a step that runs `content` through the default chain.
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            message: A2AMessage::user(content),
            skill_id: None,
        }
    }

    /// Create a step routed to a specific skill (P2-4).
    pub fn with_skill(
        id: impl Into<String>,
        content: impl Into<String>,
        skill_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            message: A2AMessage::user(content),
            skill_id: Some(skill_id.into()),
        }
    }
}

/// A2A JSON-RPC style request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2ARequest {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Request identifier.
    pub id: u64,
    /// Method name (e.g. "tasks/send", "tasks/get").
    pub method: String,
    /// Method parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    /// Optional metadata (P1-5 / P2-8).
    ///
    /// Carries the W3C `trace_id`, the caller/`owner` identity, and other
    /// cross-cutting data without polluting the method-specific params.
    /// Standard well-known keys (read via the accessor helpers):
    /// - `trace_id` — W3C-style trace id for distributed tracing (P1-5);
    /// - `owner` — caller identity used for task ownership authorization (P1-4);
    /// - `message_id` — idempotency key for `tasks/send` (P1-6).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Standard well-known metadata keys.
pub mod metadata_keys {
    /// W3C-style trace id (P1-5 / P2-8).
    pub const TRACE_ID: &str = "trace_id";
    /// Caller / organization identity (P1-4).
    pub const OWNER: &str = "owner";
    /// Idempotency key for `tasks/send` (P1-6).
    pub const MESSAGE_ID: &str = "message_id";
}

impl A2ARequest {
    /// Create a new request.
    pub fn new(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
            metadata: None,
        }
    }

    /// Create a `tasks/send` request.
    pub fn send_task(id: u64, message: &A2AMessage) -> Self {
        let params = serde_json::to_value(message)
            .ok()
            .map(|v| serde_json::json!({ "message": v }));
        Self::new(id, "tasks/send", params)
    }

    /// Create a `tasks/send` request with an idempotency key (P1-6).
    ///
    /// Re-sending the same `message_id` makes the server return the already
    /// created task instead of running the chain twice.
    pub fn send_task_with_message_id(id: u64, message: &A2AMessage, message_id: &str) -> Self {
        Self::send_task(id, message).with_message_id(message_id)
    }

    /// Create a `tasks/send` request from a [`MessageEnvelope`], propagating
    /// its owner and trace context into request metadata (P2-8).
    ///
    /// Lets a transport-neutral envelope be handed to the JSON-RPC HTTP layer
    /// without losing the cross-cutting metadata it carries.
    pub fn send_envelope(id: u64, envelope: &MessageEnvelope) -> Self {
        let mut req = Self::send_task(id, &envelope.message);
        if let Some(owner) = &envelope.owner {
            req = req.with_owner(owner);
        }
        if let Some(trace) = &envelope.trace {
            req = req.with_trace_id(trace.trace_id.as_str());
        }
        req
    }

    /// Create a `tasks/send` request that continues an existing task (P2-2/P2-3).
    ///
    /// Used to resume an `input-required` task or append a new turn to a
    /// multi-turn conversation: the server appends `message` to the task's
    /// history and (re)starts processing.
    pub fn continue_task(id: u64, task_id: &str, message: &A2AMessage) -> Self {
        let params = serde_json::to_value(message)
            .ok()
            .map(|v| serde_json::json!({ "taskId": task_id, "message": v }));
        Self::new(id, "tasks/send", params)
    }

    /// Create a `tasks/get` request.
    pub fn get_task(id: u64, task_id: &str) -> Self {
        Self::new(
            id,
            "tasks/get",
            Some(serde_json::json!({ "taskId": task_id })),
        )
    }

    /// Create a `tasks/cancel` request.
    pub fn cancel_task(id: u64, task_id: &str) -> Self {
        Self::new(
            id,
            "tasks/cancel",
            Some(serde_json::json!({ "taskId": task_id })),
        )
    }

    /// Create a `tasks/runWorkflow` request (P2-8).
    ///
    /// The server executes the workflow's steps in order and returns the
    /// per-step results in the response.
    pub fn run_workflow(id: u64, workflow: &A2AWorkflow) -> Self {
        let params = serde_json::to_value(workflow)
            .ok()
            .map(|v| serde_json::json!({ "workflow": v }));
        Self::new(id, "tasks/runWorkflow", params)
    }

    /// Set the request metadata payload (P1-5).
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the W3C-style trace id in metadata (P1-5 / P2-8).
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        let meta = self.metadata.get_or_insert_with(|| serde_json::json!({}));
        meta[metadata_keys::TRACE_ID] = serde_json::Value::String(trace_id.into());
        self
    }

    /// Set the caller/owner identity in metadata (P1-4).
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        let meta = self.metadata.get_or_insert_with(|| serde_json::json!({}));
        meta[metadata_keys::OWNER] = serde_json::Value::String(owner.into());
        self
    }

    /// Set the idempotency key for `tasks/send` (P1-6).
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Self {
        let meta = self.metadata.get_or_insert_with(|| serde_json::json!({}));
        meta[metadata_keys::MESSAGE_ID] = serde_json::Value::String(message_id.into());
        self
    }

    /// W3C-style trace id carried in metadata (P1-5).
    pub fn trace_id(&self) -> Option<&str> {
        self.metadata
            .as_ref()
            .and_then(|m| m.get(metadata_keys::TRACE_ID))
            .and_then(serde_json::Value::as_str)
    }

    /// Caller/owner identity carried in metadata (P1-4).
    pub fn owner(&self) -> Option<&str> {
        self.metadata
            .as_ref()
            .and_then(|m| m.get(metadata_keys::OWNER))
            .and_then(serde_json::Value::as_str)
    }

    /// Idempotency key for `tasks/send` (P1-6).
    ///
    /// Checks the metadata `message_id` first, then falls back to a
    /// top-level `params.messageId` (the A2A wire convention).
    pub fn message_id(&self) -> Option<&str> {
        if let Some(id) = self
            .metadata
            .as_ref()
            .and_then(|m| m.get(metadata_keys::MESSAGE_ID))
            .and_then(serde_json::Value::as_str)
        {
            return Some(id);
        }
        self.params
            .as_ref()
            .and_then(|p| p.get("messageId"))
            .and_then(serde_json::Value::as_str)
    }

    /// The `taskId` param for continuation requests (P2-2/P2-3).
    pub fn task_id(&self) -> Option<&str> {
        self.params
            .as_ref()
            .and_then(|p| p.get("taskId"))
            .and_then(serde_json::Value::as_str)
    }
}

/// A2A JSON-RPC style response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AResponse {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Request identifier this response corresponds to.
    pub id: u64,
    /// Result payload (present on success).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error payload (present on failure).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<A2AErrorData>,
}

impl A2AResponse {
    /// Create a success response.
    pub fn ok(id: u64, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(id: u64, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(A2AErrorData {
                code,
                message: message.into(),
            }),
        }
    }

    /// Create an error response from error data.
    pub fn from_error_data(id: u64, error: A2AErrorData) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Whether this response represents an error.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Extract the result value, or return the error data.
    pub fn into_result(self) -> Result<Value, A2AErrorData> {
        if let Some(err) = self.error {
            return Err(err);
        }
        Ok(self.result.unwrap_or(Value::Null))
    }
}

/// Error payload within an A2A JSON-RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AErrorData {
    /// Error code.
    pub code: i32,
    /// Human-readable error message.
    pub message: String,
}

impl A2AErrorData {
    /// Create new error data.
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Standard error: method not found.
    pub fn method_not_found() -> Self {
        Self::new(-32601, "Method not found")
    }

    /// Standard error: invalid params.
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::new(-32602, msg)
    }

    /// Standard error: internal error.
    pub fn internal_error(msg: impl Into<String>) -> Self {
        Self::new(-32603, msg)
    }
}

impl std::fmt::Display for A2AErrorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A2A Error [{}]: {}", self.code, self.message)
    }
}

impl std::error::Error for A2AErrorData {}

/// A push notification emitted by a streaming A2A server (P2-1).
///
/// Sent over an SSE connection as `data: <json>` lines, discriminated by the
/// `kind` field. Mirrors the A2A `TaskStatusUpdateEvent` /
/// `TaskArtifactUpdateEvent` pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TaskPushNotification {
    /// A task status transition (e.g. `submitted` → `working`, or a terminal
    /// state). Carries an optional error message when the status is `failed`.
    #[serde(rename_all = "camelCase")]
    StatusUpdate {
        /// Task id.
        id: String,
        /// New status.
        status: TaskStatus,
        /// Optional error message (present when status is `failed`).
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// A chunk of partial output while the task is still `working`.
    #[serde(rename_all = "camelCase")]
    ArtifactUpdate {
        /// Task id.
        id: String,
        /// Partial (or final) output.
        artifact: A2ATaskResult,
    },
}

impl TaskPushNotification {
    /// Create a status-update notification.
    pub fn status(id: impl Into<String>, status: TaskStatus) -> Self {
        TaskPushNotification::StatusUpdate {
            id: id.into(),
            status,
            error: None,
        }
    }

    /// Create a status-update notification carrying an error message.
    pub fn status_with_error(
        id: impl Into<String>,
        status: TaskStatus,
        error: impl Into<String>,
    ) -> Self {
        TaskPushNotification::StatusUpdate {
            id: id.into(),
            status,
            error: Some(error.into()),
        }
    }

    /// Create an artifact-update notification.
    pub fn artifact(id: impl Into<String>, artifact: A2ATaskResult) -> Self {
        TaskPushNotification::ArtifactUpdate {
            id: id.into(),
            artifact,
        }
    }

    /// The task id this notification refers to.
    pub fn id(&self) -> &str {
        match self {
            TaskPushNotification::StatusUpdate { id, .. }
            | TaskPushNotification::ArtifactUpdate { id, .. } => id,
        }
    }

    /// The current status, if this is a status-update notification.
    pub fn status_value(&self) -> Option<TaskStatus> {
        match self {
            TaskPushNotification::StatusUpdate { status, .. } => Some(*status),
            TaskPushNotification::ArtifactUpdate { .. } => None,
        }
    }
}

/// W3C Trace Context (`traceparent` header) (P2-8).
///
/// Format: `version-trace_id-parent_id-flags`, e.g.
/// `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`.
/// `trace_id` is 32 lowercase hex chars, `parent_id` 16, flags 2 (bit 0 =
/// sampled).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceContext {
    /// Version byte (currently `0`, serialized as `"00"`).
    pub version: u8,
    /// 32-char lowercase hex trace id.
    pub trace_id: String,
    /// 16-char lowercase hex parent span id.
    pub parent_id: String,
    /// 2-char hex flags byte (bit 0 = sampled).
    pub flags: u8,
}

impl TraceContext {
    /// Create a new (unsampled) trace context with version `00`.
    pub fn new(trace_id: impl Into<String>, parent_id: impl Into<String>) -> Self {
        Self {
            version: 0,
            trace_id: trace_id.into(),
            parent_id: parent_id.into(),
            flags: 0,
        }
    }

    /// Mark the trace as sampled (sets flags bit 0).
    pub fn sampled(mut self) -> Self {
        self.flags |= 0b0000_0001;
        self
    }

    /// Whether this trace has been marked as sampled.
    pub fn is_sampled(&self) -> bool {
        self.flags & 0b0000_0001 != 0
    }

    /// Parse a `traceparent` header value.
    pub fn parse(s: &str) -> Option<Self> {
        let mut parts = s.trim().split('-');
        let version = parts.next()?;
        let trace_id = parts.next()?;
        let parent_id = parts.next()?;
        let flags = parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        let version = u8::from_str_radix(version, 16).ok()?;
        if trace_id.len() != 32 || !trace_id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        if parent_id.len() != 16 || !parent_id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let flags = u8::from_str_radix(flags, 16).ok()?;
        Some(Self {
            version,
            trace_id: trace_id.to_string(),
            parent_id: parent_id.to_string(),
            flags,
        })
    }

    /// Serialize to a `traceparent` header value.
    pub fn to_traceparent(&self) -> String {
        format!(
            "{:02x}-{}-{}-{:02x}",
            self.version, self.trace_id, self.parent_id, self.flags
        )
    }
}

/// Transport-neutral message envelope shared across HTTP and gRPC (P2-8).
///
/// A2A messages carry identical semantics over JSON-RPC/HTTP and (future)
/// gRPC; this envelope is the common representation so a message produced on
/// one transport can be handed to the other without reshaping. It bundles the
/// message with the cross-cutting metadata that would otherwise live in an
/// HTTP header or a gRPC field: W3C trace context, caller identity, and
/// arbitrary application headers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Protocol version this envelope conforms to.
    pub protocol_version: String,
    /// The message payload.
    pub message: A2AMessage,
    /// W3C trace context carried on the message (P2-8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceContext>,
    /// Caller / organization identity (P1-4).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Arbitrary application-defined headers.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
}

impl MessageEnvelope {
    /// Wrap a message in a fresh envelope for the current protocol version.
    pub fn new(message: A2AMessage) -> Self {
        Self {
            protocol_version: "0.3.0".to_string(),
            message,
            trace: None,
            owner: None,
            headers: HashMap::new(),
        }
    }

    /// Attach the W3C trace context (P2-8).
    pub fn with_trace(mut self, trace: TraceContext) -> Self {
        self.trace = Some(trace);
        self
    }

    /// Attach the caller / organization identity (P1-4).
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Attach an application-defined header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Unwrap into the bare message.
    pub fn into_message(self) -> A2AMessage {
        self.message
    }
}

/// Filter for listing tasks from a task store (P1-1).
///
/// Used by `A2AServer::handle_tasks_list` and the store's `list` method to
/// scope results by owner and/or status.
#[derive(Debug, Clone, Default)]
pub struct TaskFilter {
    /// Only include tasks owned by this caller (None = no owner filter).
    pub owner: Option<String>,
    /// Only include tasks in these statuses (None = all statuses).
    pub statuses: Option<Vec<TaskStatus>>,
}

impl TaskFilter {
    /// Create an empty filter (matches all tasks).
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict to tasks owned by `owner`.
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Restrict to tasks in the given statuses.
    pub fn with_statuses(mut self, statuses: Vec<TaskStatus>) -> Self {
        self.statuses = Some(statuses);
        self
    }

    /// Whether a task matches this filter.
    pub fn matches(&self, task: &A2ATask) -> bool {
        if let Some(owner) = &self.owner {
            if task.owner.as_deref() != Some(owner.as_str()) {
                return false;
            }
        }
        if let Some(statuses) = &self.statuses {
            if !statuses.contains(&task.status) {
                return false;
            }
        }
        true
    }
}
