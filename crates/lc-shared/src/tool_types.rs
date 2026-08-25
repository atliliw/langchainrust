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
    /// Create a new tool call.
    ///
    /// The three positional arguments are easy to swap (id/name/arguments);
    /// prefer [`ToolCall::builder`] for call sites that construct calls from
    /// untrusted or variable input.
    #[deprecated(note = "use ToolCall::builder(id).name(..).arguments(..).build() instead")]
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

    /// Create a builder for a [`ToolCall`].
    pub fn builder(id: impl Into<String>) -> ToolCallBuilder {
        ToolCallBuilder::new(id)
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

/// Builder for constructing a [`ToolCall`] field by field.
///
/// Replaces the error-prone 3-positional-argument [`ToolCall::new`].
///
/// ```
/// use lc_shared::ToolCall;
///
/// let call = ToolCall::builder("call_1")
///     .name("get_weather")
///     .arguments(r#"{"city":"beijing"}"#)
///     .build();
///
/// assert_eq!(call.id, "call_1");
/// assert_eq!(call.name(), "get_weather");
/// ```
#[derive(Debug, Clone)]
pub struct ToolCallBuilder {
    id: String,
    tool_type: String,
    function: FunctionCall,
}

impl ToolCallBuilder {
    /// Start building a tool call with its id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: String::new(),
                arguments: String::new(),
            },
        }
    }

    /// Set the function name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.function.name = name.into();
        self
    }

    /// Set the JSON-encoded function arguments.
    pub fn arguments(mut self, arguments: impl Into<String>) -> Self {
        self.function.arguments = arguments.into();
        self
    }

    /// Consume the builder and produce the [`ToolCall`].
    pub fn build(self) -> ToolCall {
        ToolCall {
            id: self.id,
            tool_type: self.tool_type,
            function: self.function,
        }
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
        let call = ToolCall::builder("call_123")
            .name("calculator")
            .arguments(json!({"expression": "2 + 3"}).to_string())
            .build();

        assert_eq!(call.id, "call_123");
        assert_eq!(call.name(), "calculator");

        let args: HashMap<String, String> = call.parse_arguments().unwrap();
        assert_eq!(args.get("expression").unwrap(), "2 + 3");
    }

    #[test]
    fn test_parse_arguments_tolerates_messy_llm_json() {
        // LLM-generated arguments with trailing comma + trailing garbage
        let call = ToolCall::builder("call_456")
            .name("weather")
            .arguments(r#"{"city": "beijing", "unit": "celsius",} plus extra text"#)
            .build();

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
