// src/language_models/openai/responses/tests.rs
//! Tests for the Responses API model.

use lc_core::language_models::BaseLanguageModel;
use lc_schema::Message;
use serde_json::json;

use super::model::ResponsesModel;
use super::types::*;

#[test]
fn test_builtin_tool_to_api_value() {
    let web_search = BuiltinTool::WebSearch;
    assert_eq!(web_search.to_api_value(), json!({"type": "web_search"}));

    let file_search = BuiltinTool::FileSearch {
        vector_store_ids: vec!["vs_123".to_string()],
    };
    assert_eq!(
        file_search.to_api_value(),
        json!({"type": "file_search", "vector_store_ids": ["vs_123"]})
    );

    let code_interp = BuiltinTool::CodeInterpreter;
    assert_eq!(
        code_interp.to_api_value(),
        json!({"type": "code_interpreter"})
    );

    let computer = BuiltinTool::ComputerUse {
        display_width: Some(1024),
        display_height: Some(768),
    };
    let cv = computer.to_api_value();
    assert_eq!(cv["type"], "computer_use");
    assert_eq!(cv["display_width"], 1024);
    assert_eq!(cv["display_height"], 768);

    let computer_no_dims = BuiltinTool::ComputerUse {
        display_width: None,
        display_height: None,
    };
    let cv2 = computer_no_dims.to_api_value();
    assert_eq!(cv2["type"], "computer_use");
    assert!(cv2.get("display_width").is_none());
    assert!(cv2.get("display_height").is_none());
}

#[test]
fn test_config_builder() {
    let config = ResponsesConfig::new("sk-test")
        .with_model("gpt-4o")
        .with_base_url("https://api.openai.com/v1")
        .with_temperature(0.7)
        .with_max_tokens(1024)
        .with_builtin_tool(BuiltinTool::WebSearch)
        .with_builtin_tool(BuiltinTool::CodeInterpreter);

    assert_eq!(config.api_key, "sk-test");
    assert_eq!(config.model, "gpt-4o");
    assert_eq!(config.base_url, "https://api.openai.com/v1");
    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_tokens, Some(1024));
    assert_eq!(config.builtin_tools.len(), 2);
}

#[test]
fn test_message_to_input_system() {
    let msg = Message::system("You are helpful.");
    let val = ResponsesModel::message_to_input(&msg);
    assert_eq!(val["role"], "system");
    assert_eq!(val["content"], "You are helpful.");
}

#[test]
fn test_message_to_input_human() {
    let msg = Message::human("Hello!");
    let val = ResponsesModel::message_to_input(&msg);
    assert_eq!(val["role"], "user");
    assert_eq!(val["content"], "Hello!");
}

#[test]
fn test_message_to_input_ai() {
    let msg = Message::ai("Hi there!");
    let val = ResponsesModel::message_to_input(&msg);
    assert_eq!(val["role"], "assistant");
    assert_eq!(val["content"], "Hi there!");
}

#[test]
fn test_message_to_input_tool() {
    let msg = Message::tool("call_123", "result data");
    let val = ResponsesModel::message_to_input(&msg);
    assert_eq!(val["type"], "function_call_output");
    assert_eq!(val["call_id"], "call_123");
    assert_eq!(val["output"], "result data");
}

#[test]
fn test_build_request_body_basic() {
    let config = ResponsesConfig::new("sk-test").with_model("gpt-4o");
    let model = ResponsesModel::new(config);
    let messages = vec![
        Message::system("Be helpful."),
        Message::human("What is 2+2?"),
    ];
    let body = model.build_request_body(messages, false);

    assert_eq!(body["model"], "gpt-4o");
    assert_eq!(body["stream"], false);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[1]["role"], "user");
}

#[test]
fn test_build_request_body_with_tools() {
    let config = ResponsesConfig::new("sk-test")
        .with_model("gpt-4o")
        .with_builtin_tool(BuiltinTool::WebSearch)
        .with_builtin_tool(BuiltinTool::FileSearch {
            vector_store_ids: vec!["vs_abc".to_string()],
        });
    let model = ResponsesModel::new(config);
    let messages = vec![Message::human("Search the web")];
    let body = model.build_request_body(messages, false);

    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["type"], "web_search");
    assert_eq!(tools[1]["type"], "file_search");
}

#[test]
fn test_build_request_body_with_options() {
    let config = ResponsesConfig::new("sk-test")
        .with_model("gpt-4o")
        .with_temperature(0.5)
        .with_max_tokens(2048);
    let model = ResponsesModel::new(config);
    let messages = vec![Message::human("Hi")];
    let body = model.build_request_body(messages, true);

    assert_eq!(body["temperature"], 0.5);
    assert_eq!(body["max_output_tokens"], 2048);
    assert_eq!(body["stream"], true);
}

#[test]
fn test_parse_response_text_only() {
    let api_response = ResponsesApiResponse {
        id: "resp_001".to_string(),
        object: Some("response".to_string()),
        model: Some("gpt-4o".to_string()),
        output: vec![ResponsesOutputItem::Message(ResponsesMessage {
            id: Some("msg_001".to_string()),
            role: Some("assistant".to_string()),
            content: vec![ResponsesContentPart::OutputText(ResponsesOutputText {
                text: "Hello, world!".to_string(),
                annotations: vec![],
            })],
            status: Some("completed".to_string()),
        })],
        usage: Some(ResponsesUsage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
        }),
    };

    let result = ResponsesModel::parse_response(api_response).unwrap();
    assert_eq!(result.content, "Hello, world!");
    assert_eq!(result.model, "gpt-4o");
    assert!(result.tool_calls.is_none());
    let usage = result.token_usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[test]
fn test_parse_response_with_web_search() {
    let api_response = ResponsesApiResponse {
        id: "resp_002".to_string(),
        object: Some("response".to_string()),
        model: Some("gpt-4o".to_string()),
        output: vec![
            ResponsesOutputItem::WebSearchCall(ResponsesWebSearchCall {
                id: "ws_001".to_string(),
                status: "completed".to_string(),
                query: Some("Rust programming".to_string()),
            }),
            ResponsesOutputItem::Message(ResponsesMessage {
                id: Some("msg_002".to_string()),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentPart::OutputText(ResponsesOutputText {
                    text: "Rust is a systems language.".to_string(),
                    annotations: vec![],
                })],
                status: Some("completed".to_string()),
            }),
        ],
        usage: None,
    };

    let result = ResponsesModel::parse_response(api_response).unwrap();
    assert_eq!(result.content, "Rust is a systems language.");
    let tc = result.tool_calls.unwrap();
    assert_eq!(tc.len(), 1);
    assert_eq!(tc[0].id, "ws_001");
    assert_eq!(tc[0].function.name, "web_search");
}

#[test]
fn test_parse_response_with_code_interpreter() {
    let api_response = ResponsesApiResponse {
        id: "resp_003".to_string(),
        object: Some("response".to_string()),
        model: Some("gpt-4o".to_string()),
        output: vec![
            ResponsesOutputItem::CodeInterpreterCall(ResponsesCodeInterpreterCall {
                id: "ci_001".to_string(),
                code: Some("print(2+2)".to_string()),
                results: Some(vec![json!({"type": "text", "text": "4"})]),
                status: Some("completed".to_string()),
            }),
            ResponsesOutputItem::Message(ResponsesMessage {
                id: Some("msg_003".to_string()),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentPart::OutputText(ResponsesOutputText {
                    text: "The answer is 4.".to_string(),
                    annotations: vec![],
                })],
                status: Some("completed".to_string()),
            }),
        ],
        usage: None,
    };

    let result = ResponsesModel::parse_response(api_response).unwrap();
    assert_eq!(result.content, "The answer is 4.");
    let tc = result.tool_calls.unwrap();
    assert_eq!(tc[0].function.name, "code_interpreter");
}

#[test]
fn test_parse_response_refusal() {
    let api_response = ResponsesApiResponse {
        id: "resp_004".to_string(),
        object: Some("response".to_string()),
        model: Some("gpt-4o".to_string()),
        output: vec![ResponsesOutputItem::Message(ResponsesMessage {
            id: Some("msg_004".to_string()),
            role: Some("assistant".to_string()),
            content: vec![ResponsesContentPart::Refusal(ResponsesRefusal {
                refusal: "I cannot do that.".to_string(),
            })],
            status: Some("completed".to_string()),
        })],
        usage: None,
    };

    let result = ResponsesModel::parse_response(api_response).unwrap();
    assert!(result.content.contains("[Refusal: I cannot do that.]"));
}

#[test]
fn test_parse_response_multiple_content_parts() {
    let api_response = ResponsesApiResponse {
        id: "resp_005".to_string(),
        object: Some("response".to_string()),
        model: Some("gpt-4o".to_string()),
        output: vec![ResponsesOutputItem::Message(ResponsesMessage {
            id: Some("msg_005".to_string()),
            role: Some("assistant".to_string()),
            content: vec![
                ResponsesContentPart::OutputText(ResponsesOutputText {
                    text: "Part 1".to_string(),
                    annotations: vec![],
                }),
                ResponsesContentPart::OutputText(ResponsesOutputText {
                    text: "Part 2".to_string(),
                    annotations: vec![],
                }),
            ],
            status: Some("completed".to_string()),
        })],
        usage: None,
    };

    let result = ResponsesModel::parse_response(api_response).unwrap();
    assert_eq!(result.content, "Part 1\nPart 2");
}

#[test]
fn test_error_display() {
    let http_err = ResponsesError::Http("connection refused".to_string());
    assert_eq!(format!("{}", http_err), "HTTP error: connection refused");

    let api_err = ResponsesError::Api("rate limited".to_string());
    assert_eq!(format!("{}", api_err), "API error: rate limited");

    let parse_err = ResponsesError::Parse("invalid json".to_string());
    assert_eq!(format!("{}", parse_err), "Parse error: invalid json");
}

#[test]
fn test_stream_event_deserialize_text_delta() {
    let data = r#"{"type":"response.output_text.delta","delta":"Hello"}"#;
    let event: ResponsesStreamEvent = serde_json::from_str(data).unwrap();
    match event {
        ResponsesStreamEvent::OutputTextDelta(delta) => {
            assert_eq!(delta.delta, "Hello");
        }
        _ => panic!("Expected OutputTextDelta"),
    }
}

#[test]
fn test_stream_event_deserialize_completed() {
    let data = r#"{
            "type": "response.completed",
            "response": {
                "id": "resp_001",
                "model": "gpt-4o",
                "output": [],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
            }
        }"#;
    let event: ResponsesStreamEvent = serde_json::from_str(data).unwrap();
    match event {
        ResponsesStreamEvent::Completed(completed) => {
            assert_eq!(completed.response.id, "resp_001");
        }
        _ => panic!("Expected Completed"),
    }
}

#[test]
fn test_with_builtin_tool_returns_new_instance() {
    let config = ResponsesConfig::new("sk-test");
    let model = ResponsesModel::new(config);
    assert_eq!(model.config.builtin_tools.len(), 0);

    let model2 = model.with_builtin_tool(BuiltinTool::WebSearch);
    assert_eq!(model2.config.builtin_tools.len(), 1);

    let model3 = model2.with_builtin_tool(BuiltinTool::CodeInterpreter);
    assert_eq!(model3.config.builtin_tools.len(), 2);
}

#[test]
fn test_base_language_model_trait() {
    let config = ResponsesConfig::new("sk-test")
        .with_model("gpt-4o")
        .with_temperature(0.7)
        .with_max_tokens(512);
    let model = ResponsesModel::new(config);

    assert_eq!(model.model_name(), "gpt-4o");
    assert_eq!(model.temperature(), Some(0.7));
    assert_eq!(model.max_tokens(), Some(512));
    assert!(model.get_num_tokens("hello world") > 0);
}

#[test]
fn test_with_temperature_and_max_tokens() {
    let config = ResponsesConfig::new("sk-test");
    let model = ResponsesModel::new(config);
    assert_eq!(model.temperature(), None);
    assert_eq!(model.max_tokens(), None);

    let model2 = model.with_temperature(0.3);
    assert_eq!(model2.temperature(), Some(0.3));

    let model3 = model2.with_max_tokens(1024);
    assert_eq!(model3.max_tokens(), Some(1024));
}
