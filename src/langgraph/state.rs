// src/langgraph/state.rs
//! State management for LangGraph
//!
//! This module provides the state abstraction for graph execution.
//! States are data structures that flow through nodes in the graph.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// State Schema trait
///
/// All states must implement this trait to be used in a StateGraph.
/// States should be serializable for checkpointing and debugging.
pub trait StateSchema: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// Create initial state from input
    fn from_input(input: Self) -> Self {
        input
    }

    /// Get state as JSON for debugging/checkpointing
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// State update representation
///
/// Nodes return StateUpdate which contains partial updates to the state.
/// The reducer pattern determines how updates are merged into the full state.
#[derive(Debug, Clone, Serialize)]
pub struct StateUpdate<S: StateSchema> {
    /// Full or partial state update
    pub update: Option<S>,

    /// Additional metadata (for debugging/tracing)
    pub metadata: HashMap<String, serde_json::Value>,
}

impl<S: StateSchema> StateUpdate<S> {
    /// Create a full state update
    pub fn full(state: S) -> Self {
        Self {
            update: Some(state),
            metadata: HashMap::new(),
        }
    }

    /// Create update with metadata
    pub fn with_metadata(state: S, metadata: HashMap<String, serde_json::Value>) -> Self {
        Self {
            update: Some(state),
            metadata,
        }
    }

    /// Create a no-change update (for nodes that don't modify state)
    pub fn unchanged() -> Self {
        Self {
            update: None,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata entry
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }
}

/// Reducer trait for merging state updates
///
/// Reducers define how state updates are merged into the current state.
/// This enables patterns like `add_messages` which appends rather than replaces.
pub trait Reducer<S: StateSchema>: Send + Sync {
    /// Reduce current state with an update
    fn reduce(&self, current: &S, update: &S) -> S;
}

/// Default reducer that replaces state entirely
pub struct ReplaceReducer;

impl<S: StateSchema> Reducer<S> for ReplaceReducer {
    fn reduce(&self, _current: &S, update: &S) -> S {
        update.clone()
    }
}

/// Append reducer for vector fields (like add_messages pattern)
///
/// This reducer appends new items to vector fields in the state.
/// Useful for message history, steps history, etc.
pub struct AppendReducer<S: StateSchema, T: Clone + Send + Sync> {
    pub field_accessor: fn(&S) -> &[T],
    pub field_mutator: fn(&mut S, Vec<T>),
}

// Note: AppendReducer requires specific implementation per state type
// Users should implement custom reducers for their state types

/// Common state with messages (agent-style)
///
/// This provides a pre-built state schema for agent-style graphs
/// that track messages through the execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Input query
    pub input: String,

    /// Chat messages history
    pub messages: Vec<MessageEntry>,

    /// Intermediate steps
    pub steps: Vec<StepEntry>,

    /// Output result
    pub output: Option<String>,
}

impl StateSchema for AgentState {}

impl AgentState {
    /// Create new agent state with input
    pub fn new(input: String) -> Self {
        let msg = MessageEntry::human(input.clone());
        Self {
            input,
            messages: vec![msg],
            steps: vec![],
            output: None,
        }
    }

    /// Add a message to history
    pub fn add_message(&mut self, message: MessageEntry) {
        self.messages.push(message);
    }

    /// Add a step to history
    pub fn add_step(&mut self, step: StepEntry) {
        self.steps.push(step);
    }

    /// Set output
    pub fn set_output(&mut self, output: String) {
        self.output = Some(output);
    }
}

/// Message entry for agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEntry {
    pub role: MessageRole,
    pub content: String,
}

impl MessageEntry {
    pub fn human(content: String) -> Self {
        Self {
            role: MessageRole::Human,
            content,
        }
    }

    pub fn ai(content: String) -> Self {
        Self {
            role: MessageRole::AI,
            content,
        }
    }

    pub fn system(content: String) -> Self {
        Self {
            role: MessageRole::System,
            content,
        }
    }

    pub fn tool(content: String) -> Self {
        Self {
            role: MessageRole::Tool,
            content,
        }
    }
}

/// Message role types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageRole {
    System,
    Human,
    AI,
    Tool,
}

/// Step entry for intermediate execution steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepEntry {
    pub action: String,
    pub observation: String,
}

impl StepEntry {
    pub fn new(action: String, observation: String) -> Self {
        Self {
            action,
            observation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_creation() {
        let state = AgentState::new("What is Rust?".to_string());
        assert_eq!(state.input, "What is Rust?");
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.output, None);
    }

    #[test]
    fn test_state_update() {
        let state = AgentState::new("test".to_string());
        let update = StateUpdate::full(state.clone());
        assert!(update.update.is_some());
    }

    #[test]
    fn test_message_entry() {
        let human_msg = MessageEntry::human("Hello".to_string());
        assert_eq!(human_msg.role, MessageRole::Human);

        let ai_msg = MessageEntry::ai("Hi there".to_string());
        assert_eq!(ai_msg.role, MessageRole::AI);
    }
}
