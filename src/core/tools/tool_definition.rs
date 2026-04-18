// src/core/tools/tool_definition.rs
//! Tool definition for function calling

use schemars::{schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

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
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
            strict: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = Some(parameters);
        self
    }
}

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

    /// Parse arguments as JSON
    pub fn parse_arguments<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.function.arguments)
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
    fn test_tool_call_result() {
        let result = ToolCallResult::new("call_123", "5");

        assert_eq!(result.tool_call_id, "call_123");
        assert_eq!(result.role, "tool");
        assert_eq!(result.content, "5");
    }
}
