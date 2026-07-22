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

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Agent metadata card, served at `/.well-known/agent.json`.
///
/// Describes an agent's identity, endpoint, and capabilities so that
/// other agents can discover and interact with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    /// Human-readable agent name.
    pub name: String,
    /// Description of what the agent does.
    pub description: String,
    /// Base URL where the agent accepts A2A requests.
    pub url: String,
    /// List of capability identifiers (e.g. "text-generation", "tool-use").
    pub capabilities: Vec<String>,
    /// Protocol version string.
    #[serde(default = "default_version")]
    pub version: String,
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
}

fn default_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
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
            capabilities: Vec::new(),
            version: default_version(),
            provider: None,
            documentation_url: None,
            authentication: None,
            default_input_modes: default_input_modes(),
            default_output_modes: default_output_modes(),
        }
    }

    /// Add a capability.
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Set the version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
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
}

/// Task lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// Task has been submitted but not yet started.
    Submitted,
    /// Task is currently being processed.
    Working,
    /// Task requires additional input from the user.
    InputRequired,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task was cancelled.
    Cancelled,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Submitted => write!(f, "submitted"),
            TaskStatus::Working => write!(f, "working"),
            TaskStatus::InputRequired => write!(f, "input_required"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
            TaskStatus::Cancelled => write!(f, "cancelled"),
        }
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
    pub message: A2AMessage,
    /// Current status of the task.
    pub status: TaskStatus,
}

impl A2ATask {
    /// Create a new task with `Submitted` status.
    pub fn new(id: impl Into<String>, message: A2AMessage) -> Self {
        Self {
            id: id.into(),
            message,
            status: TaskStatus::Submitted,
        }
    }

    /// Set the task status.
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
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
}

impl A2ARequest {
    /// Create a new request.
    pub fn new(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }

    /// Create a `tasks/send` request.
    pub fn send_task(id: u64, message: &A2AMessage) -> Self {
        let params = serde_json::to_value(message)
            .ok()
            .map(|v| serde_json::json!({ "message": v }));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_card_new() {
        let card = AgentCard::new("test-agent", "A test agent", "http://localhost:8080");
        assert_eq!(card.name, "test-agent");
        assert_eq!(card.description, "A test agent");
        assert_eq!(card.url, "http://localhost:8080");
        assert!(card.capabilities.is_empty());
    }

    #[test]
    fn agent_card_with_capabilities() {
        let card = AgentCard::new("agent", "desc", "http://localhost")
            .with_capability("text-generation")
            .with_capability("tool-use");
        assert_eq!(card.capabilities.len(), 2);
        assert_eq!(card.capabilities[0], "text-generation");
        assert_eq!(card.capabilities[1], "tool-use");
    }

    #[test]
    fn agent_card_serialization() {
        let card = AgentCard::new("agent", "desc", "http://localhost")
            .with_capability("text-generation");
        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("\"name\":\"agent\""));
        assert!(json.contains("\"capabilities\""));
        assert!(json.contains("\"text-generation\""));
    }

    #[test]
    fn agent_card_deserialization() {
        let json = r#"{"name":"agent","description":"desc","url":"http://localhost","capabilities":[],"version":"0.1.0"}"#;
        let card: AgentCard = serde_json::from_str(json).unwrap();
        assert_eq!(card.name, "agent");
        assert_eq!(card.version, "0.1.0");
    }

    #[test]
    fn task_status_serialization() {
        let statuses = vec![
            TaskStatus::Submitted,
            TaskStatus::Working,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ];
        let json = serde_json::to_string(&statuses).unwrap();
        assert!(json.contains("\"submitted\""));
        assert!(json.contains("\"working\""));
        assert!(json.contains("\"completed\""));
        assert!(json.contains("\"failed\""));
        assert!(json.contains("\"cancelled\""));
    }

    #[test]
    fn task_status_display() {
        assert_eq!(TaskStatus::Submitted.to_string(), "submitted");
        assert_eq!(TaskStatus::Working.to_string(), "working");
        assert_eq!(TaskStatus::Completed.to_string(), "completed");
        assert_eq!(TaskStatus::Failed.to_string(), "failed");
        assert_eq!(TaskStatus::Cancelled.to_string(), "cancelled");
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
        let task = A2ATask::new("task-1", A2AMessage::user("hello"))
            .with_status(TaskStatus::Completed);
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
}
