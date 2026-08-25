// lc-providers/src/providers/gemini/tests.rs

use super::*;
use lc_core::tools::ToolDefinition;
use serde_json::json;

#[test]
fn test_bind_tools_creates_new_chat_with_tools() {
    let config = GeminiConfig::new("test-key");
    let chat = GeminiChat::new(config);
    let tools = vec![ToolDefinition::new("calculator", "Do math")
        .with_parameters(json!({"type": "object", "properties": {"expr": {"type": "string"}}}))];

    let bound = chat.bind_tools(tools.clone());
    assert!(bound.config.tools.is_some());
    assert_eq!(bound.config.tools.as_ref().unwrap().len(), 1);
    assert_eq!(
        bound.config.tools.as_ref().unwrap()[0].function.name,
        "calculator"
    );
    // Original chat should not have tools
    assert!(chat.config.tools.is_none());
}

#[test]
fn test_with_tool_choice_sets_config() {
    let config = GeminiConfig::new("test-key");
    let chat = GeminiChat::new(config);
    let chat = chat.with_tool_choice("auto");
    assert_eq!(chat.config.tool_choice.as_deref(), Some("auto"));
}

#[test]
fn test_build_request_includes_tools() {
    let config = GeminiConfig::new("test-key");
    let tools = vec![ToolDefinition::new("get_weather", "Get weather")
        .with_parameters(json!({"type": "object", "properties": {"city": {"type": "string"}}}))];
    let chat = GeminiChat::new(config).bind_tools(tools);

    let request = chat.build_request(vec![]);
    assert!(request.tools.is_some());
    let tool_decls = &request.tools.as_ref().unwrap()[0].function_declarations;
    assert_eq!(tool_decls.len(), 1);
    assert_eq!(tool_decls[0].name, "get_weather");
    assert!(tool_decls[0].parameters.is_some());
}

#[test]
fn test_build_request_tool_choice_auto() {
    let config = GeminiConfig::new("test-key");
    let chat = GeminiChat::new(config).with_tool_choice("auto");
    let request = chat.build_request(vec![]);
    assert!(request.tool_config.is_some());
    assert_eq!(
        request
            .tool_config
            .as_ref()
            .unwrap()
            .function_calling_config
            .mode,
        "AUTO"
    );
}

#[test]
fn test_build_request_tool_choice_none() {
    let config = GeminiConfig::new("test-key");
    let chat = GeminiChat::new(config).with_tool_choice("none");
    let request = chat.build_request(vec![]);
    assert_eq!(
        request
            .tool_config
            .as_ref()
            .unwrap()
            .function_calling_config
            .mode,
        "NONE"
    );
}

#[test]
fn test_with_structured_output_binds_tool() {
    let config = GeminiConfig::new("test-key");
    let chat = GeminiChat::new(config);
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct TestOutput {
        answer: String,
    }
    let _method: GeminiStructuredOutputMethod<TestOutput> = chat.with_structured_output();
    // Just verify it compiles and the method is callable
}
