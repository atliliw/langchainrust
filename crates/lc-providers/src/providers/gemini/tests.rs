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

// B7: unified multimodal request-body mapping (inlineData / fileData parts).
mod b7_multimodal {
    use super::*;
    use lc_schema::{AudioContent, FileContent, ImageContent, Message, VideoContent};

    fn chat() -> GeminiChat {
        GeminiChat::new(GeminiConfig::new("test-key"))
    }

    #[test]
    fn data_uri_media_becomes_inline_data_parts() {
        let msg = Message::human("素材")
            .with_image(ImageContent::from_url("data:image/png;base64,aW1n"))
            .with_audio(AudioContent::from_base64_with_mime("YXVk", "audio/wav"))
            .with_video(VideoContent::from_base64("dmlk"))
            .with_file(FileContent::from_base64("ZG9j", "application/pdf"));

        let request = chat().build_request(vec![msg]);
        let value = serde_json::to_value(&request).unwrap();
        let parts = value["contents"][0]["parts"].as_array().unwrap();

        let expected = json!([
            {"text": "素材"},
            {"inline_data": {"mime_type": "image/png", "data": "aW1n"}},
            {"inline_data": {"mime_type": "audio/wav", "data": "YXVk"}},
            {"inline_data": {"mime_type": "video/mp4", "data": "dmlk"}},
            {"inline_data": {"mime_type": "application/pdf", "data": "ZG9j"}},
        ]);
        assert_eq!(json!(parts), expected);
    }

    #[test]
    fn gs_uri_becomes_file_data_with_extension_mime() {
        // The sync mapper accepts gs:// directly; the async resolver is what
        // confines gs:// to Gemini policy and inlines every other scheme.
        let msg = Message::human("图").with_image(ImageContent::from_url("gs://bucket/a.png"));

        let request = chat().build_request(vec![msg]);
        let value = serde_json::to_value(&request).unwrap();
        let parts = value["contents"][0]["parts"].as_array().unwrap();

        assert_eq!(
            json!(&parts[1]),
            json!({
                "file_data": {
                    "file_uri": "gs://bucket/a.png",
                    "mime_type": "image/png"
                }
            })
        );
    }

    #[test]
    fn plain_text_message_is_unchanged() {
        let request = chat().build_request(vec![Message::human("hello")]);
        let value = serde_json::to_value(&request).unwrap();
        let parts = value["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(json!(parts), json!([{"text": "hello"}]));
    }

    #[test]
    fn hosted_url_is_skipped_by_sync_mapper() {
        // Without async resolution a bare http(s) URL cannot map to a Gemini
        // part; it is skipped rather than sent in a shape the API rejects.
        let msg =
            Message::human("x").with_image(ImageContent::from_url("https://example.com/a.png"));
        let request = chat().build_request(vec![msg]);
        let value = serde_json::to_value(&request).unwrap();
        let parts = value["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "x");
    }
}
