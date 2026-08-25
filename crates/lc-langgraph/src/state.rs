// crates/lc-langgraph/src/state.rs
//! State management for LangGraph
//!
//! This module provides the state abstraction for graph execution.
//! States are data structures that flow through nodes in the graph.

use crate::errors::{GraphError, GraphResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// State Schema trait
pub trait StateSchema:
    Clone + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de> + Debug
{
    /// Create initial state from input
    fn from_input(input: Self) -> Self {
        input
    }

    /// Get state as JSON for debugging/checkpointing.
    ///
    /// Returns a `Result` instead of silently producing `Value::Null` (Q2):
    /// a serialization failure is data corruption in the state machine and
    /// must surface at the call site, not be swallowed.
    fn to_json(&self) -> GraphResult<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| {
            GraphError::StateError(format!("Failed to serialize state to JSON: {}", e))
        })
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
    pub fn add_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
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
    /// Function that reads the vector field from a state.
    pub field_accessor: fn(&S) -> &[T],
    /// Function that writes the merged vector back into a state.
    pub field_mutator: fn(&mut S, Vec<T>),
}

impl<S: StateSchema, T: Clone + Send + Sync> Reducer<S> for AppendReducer<S, T> {
    fn reduce(&self, current: &S, update: &S) -> S {
        let current_items = (self.field_accessor)(current);
        let update_items = (self.field_accessor)(update);

        let mut merged: Vec<T> = current_items.to_vec();
        merged.extend(update_items.iter().cloned());

        let mut result = current.clone();
        (self.field_mutator)(&mut result, merged);
        result
    }
}

/// AgentState Messages Reducer - Appends messages instead of replacing
pub struct AppendMessagesReducer;

impl Reducer<AgentState> for AppendMessagesReducer {
    fn reduce(&self, current: &AgentState, update: &AgentState) -> AgentState {
        let mut result = update.clone();
        result.messages = current.messages.clone();
        result.messages.extend(update.messages.iter().cloned());
        result
    }
}

/// AgentState Steps Reducer - Appends steps instead of replacing
pub struct AppendStepsReducer;

impl Reducer<AgentState> for AppendStepsReducer {
    fn reduce(&self, current: &AgentState, update: &AgentState) -> AgentState {
        let mut result = update.clone();
        result.steps = current.steps.clone();
        result.steps.extend(update.steps.iter().cloned());
        result
    }
}

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
    pub fn new(input: impl Into<String>) -> Self {
        let input = input.into();
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
    pub fn set_output(&mut self, output: impl Into<String>) {
        self.output = Some(output.into());
    }
}

/// Message entry for agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEntry {
    /// Role of the message sender.
    pub role: MessageRole,
    /// Text content of the message.
    pub content: String,
}

impl MessageEntry {
    /// Create a human message entry.
    pub fn human(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Human,
            content: content.into(),
        }
    }

    /// Create an AI message entry.
    pub fn ai(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::AI,
            content: content.into(),
        }
    }

    /// Create a system message entry.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    /// Create a tool message entry.
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
        }
    }
}

/// Message role types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageRole {
    /// System message.
    System,
    /// Human message.
    Human,
    /// AI assistant message.
    AI,
    /// Tool result message.
    Tool,
}

/// Step entry for intermediate execution steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepEntry {
    /// Action that was taken.
    pub action: String,
    /// Observation or result of the action.
    pub observation: String,
}

impl StepEntry {
    /// Create a new step entry.
    pub fn new(action: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            observation: observation.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_messages_reducer() {
        let mut current = AgentState::new("Hello".to_string());
        current.add_message(MessageEntry::ai("Response 1".to_string()));

        let mut update = AgentState::new("Hello".to_string());
        update.add_message(MessageEntry::ai("Response 2".to_string()));
        update.set_output("Done".to_string());

        let reducer = AppendMessagesReducer;
        let result = reducer.reduce(&current, &update);

        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.output, Some("Done".to_string()));
    }

    #[test]
    fn test_append_steps_reducer() {
        let mut current = AgentState::new("Test".to_string());
        current.add_step(StepEntry::new(
            "Action 1".to_string(),
            "Result 1".to_string(),
        ));

        let mut update = AgentState::new("Test".to_string());
        update.add_step(StepEntry::new(
            "Action 2".to_string(),
            "Result 2".to_string(),
        ));

        let reducer = AppendStepsReducer;
        let result = reducer.reduce(&current, &update);

        assert_eq!(result.steps.len(), 2);
    }
}
