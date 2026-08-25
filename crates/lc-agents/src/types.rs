// lc-agents/src/types.rs
//! Agent related type definitions

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent action
///
/// Represents an action that the Agent decides to execute (usually a tool call).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    /// Tool name
    pub tool: String,

    /// Tool input (string or JSON object)
    pub tool_input: ToolInput,

    /// Log information (contains the full LLM output)
    pub log: String,
}

/// Tool input type
///
/// Uses internally tagged serialization to avoid `untagged` ambiguity:
/// - String inputs are tagged with `"type": "string"`
/// - Object inputs are tagged with `"type": "object"`
///
/// A `TryFrom<serde_json::Value>` implementation validates incoming untagged
/// JSON and dispatches to the correct variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolInput {
    /// String input
    String {
        /// The string value.
        value: String,
    },

    /// JSON object input
    Object {
        /// The JSON object value.
        value: serde_json::Value,
    },
}

impl Default for ToolInput {
    fn default() -> Self {
        ToolInput::String {
            value: String::new(),
        }
    }
}

impl std::fmt::Display for ToolInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolInput::String { value } => write!(f, "{}", value),
            ToolInput::Object { value } => write!(
                f,
                "{}",
                serde_json::to_string(value).unwrap_or_else(|_| "unknown".to_string())
            ),
        }
    }
}

/// Validates untagged JSON into a `ToolInput`.
///
/// This handles the case where external systems send JSON without the
/// `type` tag. A JSON string becomes `ToolInput::String`, a JSON object
/// becomes `ToolInput::Object`, and anything else is an error.
impl TryFrom<serde_json::Value> for ToolInput {
    type Error = String;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match value {
            serde_json::Value::String(s) => Ok(ToolInput::String { value: s }),
            serde_json::Value::Object(_) => Ok(ToolInput::Object { value }),
            other => Err(format!(
                "ToolInput must be a string or object, got: {}",
                other
            )),
        }
    }
}

/// Agent finish state
///
/// Represents that the Agent has reached a final answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFinish {
    /// Return values (key-value pairs)
    pub return_values: HashMap<String, serde_json::Value>,

    /// Log information (contains the full LLM output)
    pub log: String,
}

impl AgentFinish {
    /// Create a new AgentFinish
    pub fn new(output: impl Into<String>, log: impl Into<String>) -> Self {
        let mut return_values = HashMap::new();
        return_values.insert(
            "output".to_string(),
            serde_json::Value::String(output.into()),
        );
        Self {
            return_values,
            log: log.into(),
        }
    }

    /// Get the output value
    pub fn output(&self) -> Option<&str> {
        self.return_values.get("output").and_then(|v| v.as_str())
    }
}

/// Agent execution step
///
/// Represents an executed action and its observation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    /// The executed action
    pub action: AgentAction,

    /// Observation result (tool output)
    pub observation: String,
}

impl AgentStep {
    /// Create a new AgentStep
    pub fn new(action: AgentAction, observation: impl Into<String>) -> Self {
        Self {
            action,
            observation: observation.into(),
        }
    }
}

/// Agent output
///
/// The plan method of the Agent may return an action or a final answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentOutput {
    /// Execute a single action
    Action(AgentAction),

    /// Execute multiple actions in parallel
    Actions(Vec<AgentAction>),

    /// Finish (return final answer)
    Finish(AgentFinish),
}

impl AgentOutput {
    /// Whether this is a final answer
    pub fn is_finish(&self) -> bool {
        matches!(self, AgentOutput::Finish(_))
    }

    /// Whether this is an action (single or multiple)
    pub fn is_action(&self) -> bool {
        matches!(self, AgentOutput::Action(_) | AgentOutput::Actions(_))
    }

    /// Get a single action (if any)
    pub fn action(&self) -> Option<&AgentAction> {
        match self {
            AgentOutput::Action(action) => Some(action),
            _ => None,
        }
    }

    /// Get all actions (single or multiple)
    pub fn actions(&self) -> Vec<&AgentAction> {
        match self {
            AgentOutput::Action(action) => vec![action],
            AgentOutput::Actions(actions) => actions.iter().collect(),
            _ => vec![],
        }
    }

    /// Get the finish state (if any)
    pub fn finish(&self) -> Option<&AgentFinish> {
        match self {
            AgentOutput::Finish(finish) => Some(finish),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_action(tool: &str, input: &str) -> AgentAction {
        AgentAction {
            tool: tool.to_string(),
            tool_input: ToolInput::String {
                value: input.to_string(),
            },
            log: "test".to_string(),
        }
    }

    #[test]
    fn test_agent_output_single_action() {
        let action = create_action("calculator", "1+2");
        let output = AgentOutput::Action(action);

        assert!(output.is_action());
        assert!(!output.is_finish());
        assert_eq!(output.actions().len(), 1);
    }

    #[test]
    fn test_agent_output_multiple_actions() {
        let actions = vec![
            create_action("calculator", "1+2"),
            create_action("datetime", "now"),
        ];
        let output = AgentOutput::Actions(actions);

        assert!(output.is_action());
        assert!(!output.is_finish());
        assert_eq!(output.actions().len(), 2);
        assert!(output.action().is_none());
    }

    #[test]
    fn test_agent_output_finish() {
        let finish = AgentFinish::new("answer".to_string(), "log".to_string());
        let output = AgentOutput::Finish(finish);

        assert!(!output.is_action());
        assert!(output.is_finish());
        assert_eq!(output.actions().len(), 0);
        assert!(output.finish().is_some());
    }

    #[test]
    fn test_agent_finish_output() {
        let finish = AgentFinish::new("the answer is 42".to_string(), String::new());
        assert_eq!(finish.output(), Some("the answer is 42"));
    }
}
