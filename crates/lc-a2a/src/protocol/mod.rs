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

mod card;
mod message;
mod model;
mod task;

pub use card::{AgentCard, AgentSkill};
pub use message::A2AMessage;
pub use model::{
    metadata_keys, A2AErrorData, A2ARequest, A2AResponse, A2ATaskDetails, A2ATaskResult,
    A2AWorkflow, MessageEnvelope, TaskFilter, TaskPushNotification, TraceContext, WorkflowStep,
};
pub use task::{A2ATask, TaskStatus};

#[cfg(test)]
mod tests;
