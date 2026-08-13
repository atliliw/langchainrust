//! A2A (Agent-to-Agent) protocol types.
//!
//! Defines the core data types for the A2A protocol, which enables
//! inter-agent communication over JSON-RPC style messaging.
//!
//! # Core Types
//!
//! - **AgentCard**: Metadata describing an agent's identity and capabilities.
//! - **A2ATask**: A unit of work sent between agents.
//! - **A2AMessage**: A message within a task (role + content).
//! - **TaskStatus**: Lifecycle states for a task.
//! - **A2ARequest / A2AResponse**: JSON-RPC style request/response envelope.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A skill that an agent can perform.
///
/// Aligned with the structured skill objects required by the A2A v0.3
/// Agent Card specification (a flat string list is not sufficient).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    /// Stable identifier for the skill.
    pub id: String,
    /// Human-readable skill name.
    pub name: String,
    /// Description of what the skill does.
    pub description: String,
}

impl AgentSkill {
    /// Create a new skill.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Agent metadata card, served at `/.well-known/agent-card.json`.
///
/// Describes an agent's identity, endpoint, and capabilities so that
/// other agents can discover and interact with it. Aligned with the
/// A2A v0.3 Agent Card specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    /// Human-readable agent name.
    pub name: String,
    /// Description of what the agent does.
    pub description: String,
    /// Base URL where the agent accepts A2A requests.
    pub url: String,
    /// Structured list of skills the agent can perform.
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
    /// A2A protocol version supported by this agent.
    #[serde(default = "default_protocol_version", rename = "protocolVersion")]
    pub protocol_version: String,
    /// Security schemes the agent supports (e.g. `{"bearerAuth": {...}}`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "securitySchemes")]
    pub security_schemes: Option<Value>,
    /// Interfaces the agent exposes (e.g. `{"sse": true}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interfaces: Option<Value>,
    /// Provider/organization name (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Documentation URL (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
    /// Authentication schemes supported (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication: Option<Vec<String>>,
    /// Default input modes (e.g. ["text", "image"]).
    #[serde(default = "default_input_modes")]
    pub default_input_modes: Vec<String>,
    /// Default output modes (e.g. ["text"]).
    #[serde(default = "default_output_modes")]
    pub default_output_modes: Vec<String>,
    /// Digital signature over the canonical card content (P1-3 / P2-5).
    ///
    /// When present, clients SHOULD verify it against the agent's public key
    /// before trusting the card (see `lc_a2a::security`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Data classification this agent deals with (e.g. "public", "internal",
    /// "confidential"). Used for data-boundary / federation policy (P2-7/P2-8).
    #[serde(skip_serializing_if = "Option::is_none", rename = "dataClass")]
    pub data_class: Option<String>,
    /// Jurisdiction(s) this agent operates under (e.g. "US", "EU"). Used for
    /// compliance-aware routing in federations (P2-8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jurisdiction: Option<String>,
    /// Optional protocol capabilities the agent can negotiate (P2-8).
    ///
    /// Backward-compatible extension points, e.g. `"tasks/runWorkflow"`,
    /// `"streaming-sse"`, `"input-required-resume"`. Unknown entries are
    /// ignored by clients that do not understand them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
}

fn default_protocol_version() -> String {
    "0.3.0".to_string()
}

fn default_input_modes() -> Vec<String> {
    vec!["text".to_string()]
}

fn default_output_modes() -> Vec<String> {
    vec!["text".to_string()]
}

impl AgentCard {
    /// Create a new agent card.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            url: url.into(),
            skills: Vec::new(),
            protocol_version: default_protocol_version(),
            security_schemes: None,
            interfaces: None,
            provider: None,
            documentation_url: None,
            authentication: None,
            default_input_modes: default_input_modes(),
            default_output_modes: default_output_modes(),
            signature: None,
            data_class: None,
            jurisdiction: None,
            capabilities: Vec::new(),
        }
    }

    /// Add a skill.
    pub fn with_skill(mut self, skill: AgentSkill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Set the A2A protocol version this agent supports.
    pub fn with_protocol_version(mut self, version: impl Into<String>) -> Self {
        self.protocol_version = version.into();
        self
    }

    /// Set the security schemes advertised on the card.
    pub fn with_security_schemes(mut self, schemes: Value) -> Self {
        self.security_schemes = Some(schemes);
        self
    }

    /// Set the interfaces advertised on the card.
    pub fn with_interfaces(mut self, interfaces: Value) -> Self {
        self.interfaces = Some(interfaces);
        self
    }

    /// Set the provider/organization name.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set the documentation URL.
    pub fn with_documentation_url(mut self, url: impl Into<String>) -> Self {
        self.documentation_url = Some(url.into());
        self
    }

    /// Set the authentication schemes.
    pub fn with_authentication(mut self, schemes: Vec<String>) -> Self {
        self.authentication = Some(schemes);
        self
    }

    /// Set the digital signature over the card content (P1-3).
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }

    /// Set the data classification of this agent (P2-8).
    pub fn with_data_class(mut self, class: impl Into<String>) -> Self {
        self.data_class = Some(class.into());
        self
    }

    /// Set the jurisdiction(s) this agent operates under (P2-8).
    pub fn with_jurisdiction(mut self, jurisdiction: impl Into<String>) -> Self {
        self.jurisdiction = Some(jurisdiction.into());
        self
    }

    /// Advertise an optional protocol capability (P2-8).
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }
}

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

/// A message within an A2A task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AMessage {
    /// Role of the message sender (e.g. "user", "agent").
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

impl A2AMessage {
    /// Create a new message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Create an agent message.
    pub fn agent(content: impl Into<String>) -> Self {
        Self::new("agent", content)
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
    /// The first element is the initiating message (equal to [`message`]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_card_new() {
        let card = AgentCard::new("test-agent", "A test agent", "http://localhost:8080");
        assert_eq!(card.name, "test-agent");
        assert_eq!(card.description, "A test agent");
        assert_eq!(card.url, "http://localhost:8080");
        assert!(card.skills.is_empty());
        assert_eq!(card.protocol_version, "0.3.0");
    }

    #[test]
    fn agent_card_with_skills() {
        let card = AgentCard::new("agent", "desc", "http://localhost")
            .with_skill(AgentSkill::new("s1", "text-generation", "Generates text"))
            .with_skill(AgentSkill::new("s2", "tool-use", "Uses tools"));
        assert_eq!(card.skills.len(), 2);
        assert_eq!(card.skills[0].id, "s1");
        assert_eq!(card.skills[0].name, "text-generation");
        assert_eq!(card.skills[1].description, "Uses tools");
    }

    #[test]
    fn agent_card_serialization() {
        let card = AgentCard::new("agent", "desc", "http://localhost")
            .with_skill(AgentSkill::new("s1", "text-generation", "Generates text"))
            .with_security_schemes(serde_json::json!({ "bearerAuth": { "scheme": "bearer" } }))
            .with_interfaces(serde_json::json!({ "sse": false }));
        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("\"name\":\"agent\""));
        assert!(json.contains("\"skills\""));
        assert!(json.contains("\"text-generation\""));
        assert!(json.contains("\"protocolVersion\":\"0.3.0\""));
        assert!(json.contains("\"securitySchemes\""));
        assert!(json.contains("\"bearerAuth\""));
        assert!(json.contains("\"interfaces\""));
    }

    #[test]
    fn agent_card_deserialization() {
        let json = r#"{"name":"agent","description":"desc","url":"http://localhost","skills":[{"id":"s1","name":"text-generation","description":"Generates text"}],"protocolVersion":"0.3.0"}"#;
        let card: AgentCard = serde_json::from_str(json).unwrap();
        assert_eq!(card.name, "agent");
        assert_eq!(card.protocol_version, "0.3.0");
        assert_eq!(card.skills.len(), 1);
        assert_eq!(card.skills[0].id, "s1");
    }

    #[test]
    fn task_status_serialization() {
        let statuses = vec![
            TaskStatus::Submitted,
            TaskStatus::Working,
            TaskStatus::InputRequired,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
            TaskStatus::Rejected,
            TaskStatus::AuthRequired,
            TaskStatus::Expired,
        ];
        let json = serde_json::to_string(&statuses).unwrap();
        assert!(json.contains("\"submitted\""));
        assert!(json.contains("\"working\""));
        assert!(json.contains("\"input-required\""));
        assert!(json.contains("\"completed\""));
        assert!(json.contains("\"failed\""));
        assert!(json.contains("\"cancelled\""));
        assert!(json.contains("\"rejected\""));
        assert!(json.contains("\"auth-required\""));
        assert!(json.contains("\"expired\""));
    }

    #[test]
    fn task_status_display() {
        assert_eq!(TaskStatus::Submitted.to_string(), "submitted");
        assert_eq!(TaskStatus::Working.to_string(), "working");
        assert_eq!(TaskStatus::InputRequired.to_string(), "input-required");
        assert_eq!(TaskStatus::Completed.to_string(), "completed");
        assert_eq!(TaskStatus::Failed.to_string(), "failed");
        assert_eq!(TaskStatus::Cancelled.to_string(), "cancelled");
        assert_eq!(TaskStatus::Rejected.to_string(), "rejected");
        assert_eq!(TaskStatus::AuthRequired.to_string(), "auth-required");
        assert_eq!(TaskStatus::Expired.to_string(), "expired");
    }

    #[test]
    fn task_status_legal_transitions() {
        use TaskStatus::*;
        // Plan-mandated transitions.
        assert!(Submitted.can_transition_to(&Working));
        assert!(Submitted.can_transition_to(&Rejected));
        assert!(Working.can_transition_to(&Completed));
        assert!(Working.can_transition_to(&Failed));
        assert!(Working.can_transition_to(&InputRequired));
        assert!(Working.can_transition_to(&Cancelled));
        assert!(InputRequired.can_transition_to(&Working));
        assert!(AuthRequired.can_transition_to(&Submitted));
        // Practical additions: cancel/expire from non-terminal states.
        assert!(Submitted.can_transition_to(&Cancelled));
        assert!(InputRequired.can_transition_to(&Cancelled));
        assert!(Submitted.can_transition_to(&Expired));
        assert!(Working.can_transition_to(&Expired));
    }

    #[test]
    fn task_status_illegal_transitions() {
        use TaskStatus::*;
        // No backward / terminal / skipped transitions.
        assert!(!Working.can_transition_to(&Submitted));
        assert!(!Completed.can_transition_to(&Working));
        assert!(!Failed.can_transition_to(&Working));
        assert!(!Cancelled.can_transition_to(&Working));
        assert!(!Rejected.can_transition_to(&Working));
        assert!(!Expired.can_transition_to(&Working));
        assert!(!Submitted.can_transition_to(&Completed));
        assert!(!Working.can_transition_to(&Rejected));
    }

    #[test]
    fn task_status_terminal() {
        use TaskStatus::*;
        for s in [Completed, Failed, Cancelled, Rejected, Expired] {
            assert!(s.is_terminal(), "{s:?} should be terminal");
        }
        for s in [Submitted, Working, InputRequired, AuthRequired] {
            assert!(!s.is_terminal(), "{s:?} should not be terminal");
        }
    }

    #[test]
    fn a2a_task_details() {
        let details = A2ATaskDetails {
            task: A2ATask::new("t1", A2AMessage::user("hi")).with_status(TaskStatus::Completed),
            result: Some(A2ATaskResult::new("done")),
            error: None,
        };
        assert_eq!(details.task.status, TaskStatus::Completed);
        assert_eq!(details.result.unwrap().output, "done");
    }

    #[test]
    fn a2a_message_user() {
        let msg = A2AMessage::user("hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn a2a_message_agent() {
        let msg = A2AMessage::agent("response");
        assert_eq!(msg.role, "agent");
        assert_eq!(msg.content, "response");
    }

    #[test]
    fn a2a_task_new() {
        let task = A2ATask::new("task-1", A2AMessage::user("hello"));
        assert_eq!(task.id, "task-1");
        assert_eq!(task.status, TaskStatus::Submitted);
        assert_eq!(task.message.content, "hello");
    }

    #[test]
    fn a2a_task_with_status() {
        let task =
            A2ATask::new("task-1", A2AMessage::user("hello")).with_status(TaskStatus::Completed);
        assert_eq!(task.status, TaskStatus::Completed);
    }

    #[test]
    fn a2a_task_result() {
        let result = A2ATaskResult::new("output text");
        assert_eq!(result.output, "output text");
    }

    #[test]
    fn a2a_request_new() {
        let req = A2ARequest::new(1, "tasks/send", None);
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.id, 1);
        assert_eq!(req.method, "tasks/send");
        assert!(req.params.is_none());
    }

    #[test]
    fn a2a_request_send_task() {
        let msg = A2AMessage::user("hello");
        let req = A2ARequest::send_task(1, &msg);
        assert_eq!(req.method, "tasks/send");
        assert!(req.params.is_some());
        let params = req.params.unwrap();
        assert!(params.get("message").is_some());
    }

    #[test]
    fn a2a_request_get_task() {
        let req = A2ARequest::get_task(2, "task-123");
        assert_eq!(req.method, "tasks/get");
        let params = req.params.unwrap();
        assert_eq!(params["taskId"], "task-123");
    }

    #[test]
    fn a2a_request_cancel_task() {
        let req = A2ARequest::cancel_task(3, "task-456");
        assert_eq!(req.method, "tasks/cancel");
        let params = req.params.unwrap();
        assert_eq!(params["taskId"], "task-456");
    }

    #[test]
    fn a2a_request_serialization_skips_none_params() {
        let req = A2ARequest::new(1, "tasks/send", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("params"));
    }

    #[test]
    fn a2a_response_ok() {
        let resp = A2AResponse::ok(1, serde_json::json!({"status": "completed"}));
        assert!(!resp.is_error());
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn a2a_response_error() {
        let resp = A2AResponse::error(1, -32601, "Method not found");
        assert!(resp.is_error());
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
    }

    #[test]
    fn a2a_response_into_result_ok() {
        let resp = A2AResponse::ok(1, serde_json::json!({"output": "done"}));
        let result = resp.into_result();
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["output"], "done");
    }

    #[test]
    fn a2a_response_into_result_err() {
        let resp = A2AResponse::error(1, -32601, "Method not found");
        let result = resp.into_result();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, -32601);
    }

    #[test]
    fn a2a_response_serialization() {
        let resp = A2AResponse::ok(1, serde_json::json!({"status": "completed"}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn a2a_error_data_display() {
        let err = A2AErrorData::new(-1, "boom");
        assert_eq!(format!("{}", err), "A2A Error [-1]: boom");
    }

    #[test]
    fn a2a_error_data_standard_errors() {
        let err = A2AErrorData::method_not_found();
        assert_eq!(err.code, -32601);

        let err = A2AErrorData::invalid_params("bad input");
        assert_eq!(err.code, -32602);
        assert!(err.message.contains("bad input"));

        let err = A2AErrorData::internal_error("oops");
        assert_eq!(err.code, -32603);
    }

    #[test]
    fn roundtrip_request_json() {
        let req = A2ARequest::send_task(42, &A2AMessage::user("test"));
        let json = serde_json::to_string(&req).unwrap();
        let parsed: A2ARequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 42);
        assert_eq!(parsed.method, "tasks/send");
    }

    #[test]
    fn roundtrip_response_json() {
        let resp = A2AResponse::ok(7, serde_json::json!({"task": {"id": "t1"}}));
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: A2AResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 7);
        assert!(!parsed.is_error());
    }

    // ---- P1-3 / P2-8: AgentCard extensions ----

    #[test]
    fn agent_card_new_extension_fields_default() {
        let card = AgentCard::new("a", "d", "http://localhost");
        assert!(card.signature.is_none());
        assert!(card.data_class.is_none());
        assert!(card.jurisdiction.is_none());
        assert!(card.capabilities.is_empty());
    }

    #[test]
    fn agent_card_with_signature_and_extensions() {
        let card = AgentCard::new("a", "d", "http://localhost")
            .with_signature("sig-v1")
            .with_data_class("confidential")
            .with_jurisdiction("EU")
            .with_capability("streaming-sse")
            .with_capability("input-required-resume");
        assert_eq!(card.signature.as_deref(), Some("sig-v1"));
        assert_eq!(card.data_class.as_deref(), Some("confidential"));
        assert_eq!(card.jurisdiction.as_deref(), Some("EU"));
        assert_eq!(card.capabilities.len(), 2);

        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("\"signature\":\"sig-v1\""));
        assert!(json.contains("\"dataClass\":\"confidential\""));
        assert!(json.contains("\"jurisdiction\":\"EU\""));
        assert!(json.contains("\"capabilities\""));
    }

    #[test]
    fn agent_card_deserialize_without_extensions() {
        // Backward compatibility: a card without the new fields must still parse.
        let json =
            r#"{"name":"a","description":"d","url":"http://localhost","protocolVersion":"0.3.0"}"#;
        let card: AgentCard = serde_json::from_str(json).unwrap();
        assert!(card.signature.is_none());
        assert!(card.capabilities.is_empty());
    }

    // ---- P1-4 / P2-2: A2ATask extensions ----

    #[test]
    fn a2a_task_new_populates_history() {
        let task = A2ATask::new("t1", A2AMessage::user("hello"));
        assert_eq!(task.message_history().len(), 1);
        assert_eq!(task.message_history()[0].content, "hello");
        assert!(task.owner.is_none());
    }

    #[test]
    fn a2a_task_push_message_appends_history() {
        let mut task = A2ATask::new("t1", A2AMessage::user("hello"));
        task.push_message(A2AMessage::agent("hi there"));
        task.push_message(A2AMessage::user("continue"));
        let history = task.message_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[1].role, "agent");
        assert_eq!(history[2].content, "continue");
    }

    #[test]
    fn a2a_task_with_owner() {
        let task = A2ATask::new("t1", A2AMessage::user("hi")).with_owner("org-a");
        assert_eq!(task.owner.as_deref(), Some("org-a"));
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("\"owner\":\"org-a\""));
    }

    #[test]
    fn a2a_task_deserialize_without_owner_messages() {
        // Old single-message wire payload: history falls back to `message`.
        let json = r#"{"id":"t1","message":{"role":"user","content":"hi"},"status":"submitted"}"#;
        let task: A2ATask = serde_json::from_str(json).unwrap();
        assert!(task.owner.is_none());
        assert!(task.messages.is_empty());
        assert_eq!(task.message_history().len(), 1);
        assert_eq!(task.message_history()[0].content, "hi");
    }

    // ---- P1-5 / P1-6: A2ARequest metadata & idempotency ----

    #[test]
    fn a2a_request_with_trace_id() {
        let req = A2ARequest::send_task(1, &A2AMessage::user("hi")).with_trace_id("abc123");
        assert_eq!(req.trace_id(), Some("abc123"));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"trace_id\":\"abc123\""));
    }

    #[test]
    fn a2a_request_with_owner() {
        let req = A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("org-a");
        assert_eq!(req.owner(), Some("org-a"));
    }

    #[test]
    fn a2a_request_message_id_from_metadata() {
        let req = A2ARequest::send_task(1, &A2AMessage::user("hi")).with_message_id("msg-1");
        assert_eq!(req.message_id(), Some("msg-1"));
    }

    #[test]
    fn a2a_request_message_id_from_params_fallback() {
        // A2A wire convention: messageId at top level of params.
        let mut req = A2ARequest::send_task(1, &A2AMessage::user("hi"));
        req.params = Some(serde_json::json!({ "messageId": "wire-1" }));
        assert_eq!(req.message_id(), Some("wire-1"));
    }

    #[test]
    fn a2a_request_continue_task() {
        let req = A2ARequest::continue_task(5, "task-9", &A2AMessage::user("more"));
        assert_eq!(req.method, "tasks/send");
        assert_eq!(req.task_id(), Some("task-9"));
        let params = req.params.unwrap();
        assert!(params.get("message").is_some());
    }

    #[test]
    fn a2a_request_metadata_skipped_when_none() {
        let req = A2ARequest::send_task(1, &A2AMessage::user("hi"));
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("metadata"));
    }

    // ---- P2-1: TaskPushNotification ----

    #[test]
    fn push_notification_status_serialization() {
        let n = TaskPushNotification::status("t1", TaskStatus::Working);
        let json = serde_json::to_string(&n).unwrap();
        assert!(json.contains("\"kind\":\"status-update\""));
        assert!(json.contains("\"status\":\"working\""));
        assert_eq!(n.id(), "t1");
        assert_eq!(n.status_value(), Some(TaskStatus::Working));
    }

    #[test]
    fn push_notification_artifact_serialization() {
        let n = TaskPushNotification::artifact("t1", A2ATaskResult::new("partial"));
        let json = serde_json::to_string(&n).unwrap();
        assert!(json.contains("\"kind\":\"artifact-update\""));
        assert!(json.contains("\"output\":\"partial\""));
        assert_eq!(n.status_value(), None);
    }

    #[test]
    fn push_notification_roundtrip() {
        for original in [
            TaskPushNotification::status_with_error("t1", TaskStatus::Failed, "boom"),
            TaskPushNotification::artifact("t2", A2ATaskResult::new("chunk")),
            TaskPushNotification::status("t3", TaskStatus::Completed),
        ] {
            let json = serde_json::to_string(&original).unwrap();
            let parsed: TaskPushNotification = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.id(), original.id());
        }
    }

    // ---- P2-8: TraceContext ----

    #[test]
    fn trace_context_roundtrip() {
        let tc = TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7");
        let tp = tc.to_traceparent();
        assert_eq!(
            tp,
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        );
        let parsed = TraceContext::parse(&tp).unwrap();
        assert_eq!(parsed, tc);
    }

    #[test]
    fn trace_context_sampled() {
        let tc =
            TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").sampled();
        assert!(tc.is_sampled());
        assert!(tc.to_traceparent().ends_with("-01"));
    }

    #[test]
    fn trace_context_parse_invalid() {
        assert!(TraceContext::parse("").is_none());
        // Wrong trace id length.
        assert!(TraceContext::parse("00-1234-00f067aa0ba902b7-00").is_none());
        // Non-hex chars.
        assert!(
            TraceContext::parse("00-zz92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00").is_none()
        );
        // Too many fields.
        assert!(TraceContext::parse(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00-extra"
        )
        .is_none());
    }

    // ---- P1-1: TaskFilter ----

    #[test]
    fn task_filter_empty_matches_all() {
        let f = TaskFilter::new();
        assert!(f.matches(&A2ATask::new("t1", A2AMessage::user("hi"))));
    }

    #[test]
    fn task_filter_by_owner() {
        let f = TaskFilter::new().with_owner("org-a");
        let owned = A2ATask::new("t1", A2AMessage::user("hi")).with_owner("org-a");
        let foreign = A2ATask::new("t2", A2AMessage::user("hi")).with_owner("org-b");
        assert!(f.matches(&owned));
        assert!(!f.matches(&foreign));
        // Tasks with no owner never match an owner filter.
        assert!(!f.matches(&A2ATask::new("t3", A2AMessage::user("hi"))));
    }

    #[test]
    fn task_filter_by_status() {
        let f = TaskFilter::new().with_statuses(vec![TaskStatus::Working]);
        assert!(
            f.matches(&A2ATask::new("t1", A2AMessage::user("hi")).with_status(TaskStatus::Working))
        );
        assert!(!f.matches(
            &A2ATask::new("t2", A2AMessage::user("hi")).with_status(TaskStatus::Completed)
        ));
    }

    // ---- P2-8: MessageEnvelope (unified message model) ----

    #[test]
    fn message_envelope_roundtrips_through_json() {
        let trace =
            TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").sampled();
        let envelope = MessageEnvelope::new(A2AMessage::user("hello"))
            .with_trace(trace)
            .with_owner("alice")
            .with_header("x-region", "cn-east");

        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: MessageEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.message.role, "user");
        assert_eq!(decoded.message.content, "hello");
        assert_eq!(decoded.owner.as_deref(), Some("alice"));
        assert_eq!(
            decoded.headers.get("x-region").map(String::as_str),
            Some("cn-east")
        );
        let t = decoded.trace.expect("trace present");
        assert_eq!(t.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert!(t.is_sampled());
        assert_eq!(
            t.to_traceparent(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        );
    }

    #[test]
    fn message_envelope_minimal_roundtrip() {
        let envelope = MessageEnvelope::new(A2AMessage::agent("hi"));
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: MessageEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.message.role, "agent");
        assert!(decoded.owner.is_none());
        assert!(decoded.trace.is_none());
        assert!(decoded.headers.is_empty());
        assert_eq!(decoded.into_message().content, "hi");
    }

    #[test]
    fn send_envelope_propagates_owner_and_trace() {
        let trace = TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7");
        let envelope = MessageEnvelope::new(A2AMessage::user("hi"))
            .with_trace(trace)
            .with_owner("alice");

        let req = A2ARequest::send_envelope(7, &envelope);
        assert_eq!(req.method, "tasks/send");
        assert_eq!(req.owner(), Some("alice"));
        assert_eq!(req.trace_id(), Some("4bf92f3577b34da6a3ce929d0e0e4736"));
    }
}
