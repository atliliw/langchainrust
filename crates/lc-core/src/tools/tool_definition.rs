// src/core/tools/tool_definition.rs
//! Tool definition for function calling

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};

/// Tool definition for LLM function calling
///
/// This structure defines a tool that can be bound to an LLM
/// and invoked during generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool type (always "function" for now)
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function definition
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: Some(description.into()),
                parameters: None,
                strict: None,
            },
        }
    }

    /// Create with JSON Schema parameters
    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.function.parameters = Some(parameters);
        self
    }

    /// Create from a type that implements JsonSchema
    pub fn from_type<T: JsonSchema>(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let schema = schema_for!(T);
        let parameters = serde_json::to_value(schema).unwrap_or(serde_json::Value::Null);
        Self::new(name, description).with_parameters(parameters)
    }

    /// Enable strict mode (OpenAI specific)
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.function.strict = Some(strict);
        self
    }
}

/// Function definition inside a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,

    /// Function description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Parameters JSON Schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,

    /// Strict mode (OpenAI specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl FunctionDefinition {
    /// Creates a new function definition with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
            strict: None,
        }
    }

    /// Sets the function description (builder style).
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the parameters JSON schema (builder style).
    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = Some(parameters);
        self
    }
}

// Re-export shared tool call types from lc-shared
pub use lc_shared::tools::{FunctionCall, ToolCall, ToolCallBuilder, ToolCallResult};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition::new("calculator", "Calculate mathematical expressions")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate"
                    }
                },
                "required": ["expression"]
            }));

        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "calculator");
        assert!(tool.function.parameters.is_some());
    }
}

#[cfg(test)]
mod tool_call_tests {
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
    fn test_tool_call_result() {
        let result = ToolCallResult::new("call_123", "5");

        assert_eq!(result.tool_call_id, "call_123");
        assert_eq!(result.role, "tool");
        assert_eq!(result.content, "5");
    }
}
