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

// 0.25.0 B2: the wire JSON sent to the Generative Language API is camelCase.
#[test]
fn test_request_serializes_with_camel_case_wire_keys() {
    let config = GeminiConfig::new("test-key")
        .with_temperature(0.5)
        .with_max_output_tokens(128)
        .with_base_url(GEMINI_BASE_URL);
    let tools = vec![ToolDefinition::new("get_weather", "Get weather")
        .with_parameters(json!({"type": "object", "properties": {"city": {"type": "string"}}}))];
    let chat = GeminiChat::new(config)
        .bind_tools(tools)
        .with_tool_choice("auto");
    let request = chat.build_request(vec![Message::system("be terse"), Message::human("hi")]);
    let value = serde_json::to_value(&request).unwrap();

    assert!(value.get("systemInstruction").is_some(), "got: {value}");
    assert!(value.get("generationConfig").is_some());
    assert_eq!(value["generationConfig"]["maxOutputTokens"], json!(128));
    assert_eq!(
        value["tools"][0]["functionDeclarations"][0]["name"],
        json!("get_weather")
    );
    assert_eq!(
        value["toolConfig"]["functionCallingConfig"]["mode"],
        json!("AUTO")
    );
    // Snake_case wire keys would be silently ignored by the real API.
    assert!(value.get("system_instruction").is_none());
    assert!(value.get("generation_config").is_none());
    assert!(value.get("tool_config").is_none());
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

        // 0.25.0: the real Generative Language API wire JSON is camelCase.
        let expected = json!([
            {"text": "素材"},
            {"inlineData": {"mimeType": "image/png", "data": "aW1n"}},
            {"inlineData": {"mimeType": "audio/wav", "data": "YXVk"}},
            {"inlineData": {"mimeType": "video/mp4", "data": "dmlk"}},
            {"inlineData": {"mimeType": "application/pdf", "data": "ZG9j"}},
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
                "fileData": {
                    "fileUri": "gs://bucket/a.png",
                    "mimeType": "image/png"
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

// 0.25.0 B2: streaming must request the documented `alt=sse` framing.
mod b2_streaming_contract {
    use super::*;
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn stream_uses_alt_sse_and_parses_chunks() {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let request_line = Arc::new(Mutex::new(String::new()));
        let captured = request_line.clone();
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut header = Vec::new();
                let mut byte = [0u8; 1];
                while header.len() < 64 * 1024 {
                    if socket.read_exact(&mut byte).await.is_err() {
                        return;
                    }
                    header.push(byte[0]);
                    if header.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let head_lower = String::from_utf8_lossy(&header).to_lowercase();
                let content_length: usize = head_lower
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                let mut body = vec![0u8; content_length];
                if content_length > 0 {
                    let _ = socket.read_exact(&mut body).await;
                }
                {
                    let line = head_lower.lines().next().unwrap_or("").to_string();
                    let mut guard = captured.lock().unwrap_or_else(|e| e.into_inner());
                    *guard = line;
                }
                let sse = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\r\n\r\n\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}]}}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":3,\"totalTokenCount\":5}}\r\n\r\n";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse}"
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });

        let chat =
            GeminiChat::new(GeminiConfig::new("test-key").with_base_url(format!("http://{addr}")));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut text = String::new();
        let mut usage = None;
        while let Some(item) = stream.next().await {
            let chunk = item.expect("stream ok");
            text.push_str(&chunk.text);
            if chunk.token_usage.is_some() {
                usage = chunk.token_usage;
            }
        }
        assert_eq!(text, "Hello world");
        assert_eq!(usage.expect("usage chunk").total_tokens, 5);

        let line = request_line
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        assert!(
            line.contains(":streamgeneratecontent?alt=sse"),
            "streaming request must use the documented alt=sse framing, got: {line}"
        );
        assert!(!line.contains("event-stream"));
    }
}
