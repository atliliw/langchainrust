// lc-shared/src/tool_types.rs
//! Tool call types shared across crates.
//!
//! These types are needed by both `lc-schema` (Message uses ToolCall)
//! and `lc-core` (tool definitions), so they live here to break the
//! circular dependency between schema and core.

use crate::json_repair::{parse_tolerant_json, JsonRepairError};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Tool call from LLM response
///
/// When an LLM decides to call a tool, it returns a ToolCall structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool call ID (used to reference the call result)
    pub id: String,

    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function call details
    pub function: FunctionCall,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    /// Get the function name
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// Get the arguments as string
    pub fn arguments(&self) -> &str {
        &self.function.arguments
    }

    /// Parse arguments as JSON.
    ///
    /// Arguments come from LLM output, so a tolerant parser is used — code
    /// fences, trailing commas, unescaped quotes and trailing garbage are
    /// repaired before deserialization (see [`crate::json_repair`]).
    pub fn parse_arguments<T: DeserializeOwned>(&self) -> Result<T, JsonRepairError> {
        parse_tolerant_json(&self.function.arguments)
    }
}

/// Function call inside a ToolCall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name
    pub name: String,

    /// Arguments as JSON string
    pub arguments: String,
}

/// Tool call result to send back to LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// Tool call ID (must match the ToolCall.id)
    pub tool_call_id: String,

    /// Role (always "tool")
    pub role: String,

    /// Tool output content
    pub content: String,
}

impl ToolCallResult {
    /// Create a new tool result
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            role: "tool".to_string(),
            content: content.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_tool_call() {
        let call = ToolCall::new(
            "call_123",
            "calculator",
            json!({"expression": "2 + 3"}).to_string(),
        );

        assert_eq!(call.id, "call_123");
        assert_eq!(call.name(), "calculator");

        let args: HashMap<String, String> = call.parse_arguments().unwrap();
        assert_eq!(args.get("expression").unwrap(), "2 + 3");
    }

    #[test]
    fn test_parse_arguments_tolerates_messy_llm_json() {
        // LLM-generated arguments with trailing comma + trailing garbage
        let call = ToolCall::new(
            "call_456",
            "weather",
            r#"{"city": "beijing", "unit": "celsius",} plus extra text"#,
        );

        let args: HashMap<String, String> = call.parse_arguments().unwrap();
        assert_eq!(args.get("city").unwrap(), "beijing");
        assert_eq!(args.get("unit").unwrap(), "celsius");
    }

    #[test]
    fn test_tool_call_result() {
        let result = ToolCallResult::new("call_123", "5");

        assert_eq!(result.tool_call_id, "call_123");
        assert_eq!(result.role, "tool");
        assert_eq!(result.content, "5");
    }
}
