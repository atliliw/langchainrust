#![allow(dead_code)]
// tests/unit/tool_calling.rs
//! Unit tests for tool calling functionality

use langchainrust::{
    FunctionDefinition, Message, MessageType, ToolCall, ToolCallResult,
    ToolDefinition,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[test]
fn test_tool_definition_new() {
    let tool = ToolDefinition::new("calculator", "Calculate math expressions");

    assert_eq!(tool.tool_type, "function");
    assert_eq!(tool.function.name, "calculator");
    assert_eq!(
        tool.function.description,
        Some("Calculate math expressions".to_string())
    );
    assert!(tool.function.parameters.is_none());
}

#[test]
fn test_tool_definition_with_parameters() {
    let tool = ToolDefinition::new("weather", "Get weather").with_parameters(json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"]
    }));

    assert!(tool.function.parameters.is_some());
    let params = tool.function.parameters.unwrap();
    assert_eq!(params["type"], "object");
}

#[test]
fn test_tool_definition_from_type() {
    #[derive(JsonSchema, Deserialize)]
    struct WeatherInput {
        city: String,
        unit: Option<String>,
    }

    let tool = ToolDefinition::from_type::<WeatherInput>("weather", "Get weather for a city");

    assert_eq!(tool.function.name, "weather");
    assert!(tool.function.parameters.is_some());
}

#[test]
fn test_tool_definition_strict_mode() {
    let tool = ToolDefinition::new("test", "Test tool").with_strict(true);

    assert_eq!(tool.function.strict, Some(true));
}

#[test]
fn test_tool_call_new() {
    let call = ToolCall::new("call_123", "calculator", json!({"expr": "2+3"}).to_string());

    assert_eq!(call.id, "call_123");
    assert_eq!(call.tool_type, "function");
    assert_eq!(call.function.name, "calculator");
    assert_eq!(call.name(), "calculator");
    assert_eq!(call.arguments(), "{\"expr\":\"2+3\"}");
}

#[test]
fn test_tool_call_parse_arguments() {
    let call = ToolCall::new(
        "call_abc",
        "weather",
        json!({"city": "Beijing", "unit": "celsius"}).to_string(),
    );

    #[derive(Deserialize)]
    struct WeatherArgs {
        city: String,
        unit: Option<String>,
    }

    let args: WeatherArgs = call.parse_arguments().unwrap();
    assert_eq!(args.city, "Beijing");
    assert_eq!(args.unit, Some("celsius".to_string()));
}

#[test]
fn test_tool_call_result() {
    let result = ToolCallResult::new("call_xyz", "Result: 42");

    assert_eq!(result.tool_call_id, "call_xyz");
    assert_eq!(result.role, "tool");
    assert_eq!(result.content, "Result: 42");
}

#[test]
fn test_function_definition() {
    let func = FunctionDefinition::new("search")
        .with_description("Search the web")
        .with_parameters(json!({"query": "string"}));

    assert_eq!(func.name, "search");
    assert_eq!(func.description, Some("Search the web".to_string()));
    assert!(func.parameters.is_some());
}

#[test]
fn test_message_ai_with_tool_calls() {
    let call = ToolCall::new("call_1", "tool", "{}");
    let msg = Message::ai_with_tool_calls("Thinking...", vec![call.clone()]);

    assert_eq!(msg.content, "Thinking...");
    assert_eq!(msg.message_type, MessageType::AI);
    assert!(msg.has_tool_calls());

    let calls = msg.get_tool_calls().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_1");
}

#[test]
fn test_message_without_tool_calls() {
    let msg = Message::ai("Hello");

    assert!(!msg.has_tool_calls());
    assert!(msg.get_tool_calls().is_none());
}

#[test]
fn test_message_tool() {
    let msg = Message::tool("call_123", "Tool output");

    assert_eq!(msg.content, "Tool output");
    match msg.message_type {
        MessageType::Tool { tool_call_id } => {
            assert_eq!(tool_call_id, "call_123");
        }
        _ => panic!("Expected Tool message type"),
    }
}

#[test]
fn test_tool_definition_serialization() {
    let tool = ToolDefinition::new("test", "Test").with_parameters(json!({"type": "object"}));

    let json = serde_json::to_string(&tool).unwrap();
    assert!(json.contains("function"));
    assert!(json.contains("test"));

    let parsed: ToolDefinition = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.function.name, "test");
}

#[test]
fn test_tool_call_serialization() {
    let call = ToolCall::new("id_1", "name", "{}");

    let json = serde_json::to_string(&call).unwrap();
    assert!(json.contains("id_1"));
    assert!(json.contains("name"));

    let parsed: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "id_1");
}

#[test]
fn test_multiple_tool_calls_in_message() {
    let calls = vec![
        ToolCall::new("call_1", "tool_a", "{}"),
        ToolCall::new("call_2", "tool_b", "{}"),
    ];

    let msg = Message::ai_with_tool_calls("Using tools", calls);

    assert!(msg.has_tool_calls());
    let calls = msg.get_tool_calls().unwrap();
    assert_eq!(calls.len(), 2);
}

#[derive(JsonSchema, Deserialize)]
struct TestSchema {
    name: String,
    count: i32,
    items: Vec<String>,
}

#[test]
fn test_schema_generation() {
    let tool = ToolDefinition::from_type::<TestSchema>("test", "Test schema");

    assert!(tool.function.parameters.is_some());
    let params = tool.function.parameters.unwrap();

    assert_eq!(params["type"], "object");
    assert!(params["properties"]["name"].is_object());
    assert!(params["properties"]["count"].is_object());
    assert!(params["properties"]["items"].is_object());
}
