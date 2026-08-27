//! Handoff type definitions

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// A Handoff handover directive
#[derive(Debug)]
pub struct Handoff {
    /// Name of the target Agent
    pub target_agent: String,
    /// Task description to hand over
    pub task: String,
    /// Handoff context
    pub context: Option<HandoffContext>,
}

/// Handoff context - carries information to the target Agent
#[derive(Debug)]
pub struct HandoffContext {
    /// Original request content
    pub original_request: String,
    /// Current execution result
    pub current_result: Option<String>,
    /// Current conversation summary (P2-4): carries the upstream conversation
    /// summary to the target Agent on handoff, rather than transferring control
    /// raw — the target Agent can continue the topic instead of starting over.
    pub conversation_summary: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}

impl HandoffContext {
    /// Creates a handoff context, recording the original request.
    pub fn new(original_request: impl Into<String>) -> Self {
        Self {
            original_request: original_request.into(),
            current_result: None,
            conversation_summary: None,
            metadata: HashMap::new(),
        }
    }

    /// Carries the current conversation summary.
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.conversation_summary = Some(summary.into());
        self
    }

    /// Carries the current execution result.
    pub fn with_result(mut self, result: impl Into<String>) -> Self {
        self.current_result = Some(result.into());
        self
    }
}

/// Handoff result
pub struct HandoffResult {
    /// Name of the target Agent
    pub agent_name: String,
    /// Handoff execution result
    pub result: String,
    /// Next handoff directive (optional)
    pub next_handoff: Option<Box<Handoff>>,
}

/// Handoff history record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffRecord {
    /// Source Agent name
    pub from_agent: String,
    /// Target Agent name
    pub to_agent: String,
    /// Task description handed over
    pub task: String,
    /// Handoff result
    pub result: String,
    /// Handoff timestamp
    pub timestamp: String,
}

/// Handoff error
#[derive(Debug)]
#[non_exhaustive]
pub enum HandoffError {
    /// The target Agent does not exist
    AgentNotFound(String),
    /// Agent execution error
    ExecutionError(String),
    /// Handoff cycle detected: A hands off to B, B hands back to A, infinite loop (P1-7).
    HandoffCycleDetected(String),
    /// Handoff depth exceeded the limit (P1-7).
    MaxHandoffDepthExceeded(usize),
}

impl std::fmt::Display for HandoffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandoffError::AgentNotFound(name) => {
                write!(f, "Agent does not exist: {}", name)
            }
            HandoffError::ExecutionError(msg) => write!(f, "Agent execution error: {}", msg),
            HandoffError::HandoffCycleDetected(name) => {
                write!(f, "handoff cycle detected: {} already in the handoff chain, cyclic handoff rejected", name)
            }
            HandoffError::MaxHandoffDepthExceeded(depth) => {
                write!(f, "handoff depth exceeded the limit: {}", depth)
            }
        }
    }
}

impl std::error::Error for HandoffError {}
