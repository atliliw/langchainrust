// lc-providers/src/openai/chat/tests.rs

use super::*;

mod tests_env {
    use super::*;

    use std::env;

    fn save_and_set(key: &str, value: &str) -> Option<String> {
        let old = env::var(key).ok();
        env::set_var(key, value);
        old
    }

    fn restore(key: &str, old: Option<String>) {
        match old {
            Some(v) => env::set_var(key, v),
            None => env::remove_var(key),
        }
    }

    #[test]
    fn test_from_env_result_ok_when_key_set() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("OPENAI_API_KEY", "test-key-123");
        assert!(OpenAIChat::from_env_result().is_ok());
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");
        assert!(OpenAIChat::from_env_result().is_err());
        restore("OPENAI_API_KEY", old);
    }
}

mod tests_q3_q4 {
    use super::*;

    fn message(content: Option<&str>, reasoning: Option<&str>) -> OpenAIMessage {
        OpenAIMessage {
            role: "assistant".to_string(),
            content: content.map(|s| s.to_string()),
            reasoning_content: reasoning.map(|s| s.to_string()),
            tool_calls: None,
            refusal: None,
        }
    }

    #[test]
    fn test_llm_result_keeps_content_when_non_empty() {
        let msg = message(Some("Hello"), Some("hidden chain-of-thought"));
        let result = OpenAIChat::llm_result_from_message(
            &msg,
            "gpt-test".to_string(),
            Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                reasoning_tokens: None,
                completion_tokens_details: None,
            }),
        );

        assert_eq!(result.content, "Hello");
        assert_eq!(
            result.thinking_content.as_deref(),
            Some("hidden chain-of-thought")
        );
        assert_eq!(result.model, "gpt-test");
        let usage = result.token_usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_llm_result_reasoning_does_not_leak_into_content() {
        // Q3: reasoning-only responses keep `content` empty — no fallback.
        let msg = message(Some(""), Some("reasoning only"));
        let result = OpenAIChat::llm_result_from_message(&msg, "gpt-test".to_string(), None);

        assert_eq!(result.content, "");
        assert_eq!(result.thinking_content.as_deref(), Some("reasoning only"));
    }

    #[test]
    fn test_llm_result_empty_content_no_thinking() {
        let msg = message(None, Some(""));
        let result = OpenAIChat::llm_result_from_message(&msg, "gpt-test".to_string(), None);

        assert_eq!(result.content, "");
        assert!(result.thinking_content.is_none());
    }

    #[tokio::test]
    async fn test_aggregate_stream_concatenates_tokens_in_order() {
        // Q4: the aggregation helper produces the full content in order.
        let stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::new("Hello")),
                Ok(StreamChunk::new(", ")),
                Ok(StreamChunk::new("world")),
            ]));

        let (content, thinking, token_usage, tool_calls) =
            OpenAIChat::aggregate_stream(stream).await.unwrap();
        assert_eq!(content, "Hello, world");
        assert!(thinking.is_none());
        // 0.22.0 audit fix (Medium): a text-only stream carries no terminal
        // usage / tool calls.
        assert!(token_usage.is_none());
        assert!(tool_calls.is_none());
    }

    #[tokio::test]
    async fn test_aggregate_stream_carries_terminal_usage_and_tool_calls() {
        // 0.22.0 audit fix (Medium): the `config.streaming=true` aggregate path
        // must not drop the terminal usage chunk / accumulated tool calls.
        let usage_chunk = StreamChunk {
            thinking_content: None,
            text: String::new(),
            token_usage: Some(TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 5,
                total_tokens: 8,
            }),
            tool_calls: Some(vec![lc_core::tools::ToolCall::builder("call_1")
                .name("get_weather")
                .arguments(r#"{"city":"beijing"}"#)
                .build()]),
        };
        let stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk {
                    text: String::new(),
                    thinking_content: Some("think ".to_string()),
                    token_usage: None,
                    tool_calls: None,
                }),
                Ok(StreamChunk::new("Hello")),
                Ok(StreamChunk {
                    text: String::new(),
                    thinking_content: Some("twice".to_string()),
                    token_usage: None,
                    tool_calls: None,
                }),
                Ok(StreamChunk::new(" world")),
                Ok(usage_chunk),
            ]));

        let (content, thinking, token_usage, tool_calls) =
            OpenAIChat::aggregate_stream(stream).await.unwrap();
        assert_eq!(content, "Hello world");
        assert_eq!(thinking.as_deref(), Some("think twice"));
        let usage = token_usage.expect("usage carried through");
        assert_eq!(usage.total_tokens, 8);
        let calls = tool_calls.expect("tool_calls carried through");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name(), "get_weather");
    }

    #[tokio::test]
    async fn test_aggregate_stream_stops_on_error() {
        let stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::new("Hello")),
                Err(OpenAIError::Api("boom".to_string())),
                Ok(StreamChunk::new("never")),
            ]));

        let err = OpenAIChat::aggregate_stream(stream).await.unwrap_err();
        assert!(matches!(err, OpenAIError::Api(_)));
    }
}

/// 0.20.0 S3.2: the SSE streaming loop accumulates fragmented `delta.tool_calls`
/// and attaches the complete tool calls to the terminal chunk — the piece that
/// lets FunctionCalling's `plan_stream` stream tool-call steps natively.
mod tests_streaming_tool_calls {
    use super::*;
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Spawns a one-shot HTTP server that replies to POST /v1/chat/completions
    /// with the given OpenAI-style SSE body, returning the base URL and a
    /// handle to the captured request body.
    async fn spawn_sse_server(sse_body: &'static str) -> (String, Arc<Mutex<Vec<u8>>>) {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_task = captured.clone();
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                // Read the request header + body so reqwest's POST completes.
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
                let header_str = String::from_utf8_lossy(&header).to_lowercase();
                let content_length: usize = header_str
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                let mut body = vec![0u8; content_length];
                if content_length > 0 && socket.read_exact(&mut body).await.is_err() {
                    return;
                }
                *captured_task.lock().unwrap() = body;
                let response =
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{sse_body}");
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}"), captured)
    }

    #[tokio::test]
    async fn stream_chat_accumulates_fragmented_tool_calls() {
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"beij\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ing\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":8,\"total_tokens\":18}}\n\n\
data: [DONE]\n\n";
        let (base_url, _captured) = spawn_sse_server(sse_body).await;

        let chat =
            OpenAIChat::new(OpenAIConfig::new("test_key").with_base_url(format!("{base_url}/v1")));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("weather in beijing")])
            .await
            .unwrap();

        let mut terminal: Option<StreamChunk> = None;
        while let Some(item) = stream.next().await {
            let chunk = item.expect("chunk ok");
            if chunk.tool_calls.is_some() {
                terminal = Some(chunk);
            }
        }

        let final_chunk = terminal.expect("terminal chunk carries tool_calls");
        let calls = final_chunk.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name(), "get_weather");
        assert_eq!(
            calls[0].arguments(),
            r#"{"city":"beijing"}"#,
            "arguments concatenated across fragments"
        );
        let usage = final_chunk
            .token_usage
            .expect("usage on the same terminal chunk");
        assert_eq!(usage.total_tokens, 18);
    }

    #[tokio::test]
    async fn stream_chat_flushes_tool_calls_without_usage_chunk() {
        // Some compatible providers end the stream without a usage chunk; the
        // accumulated tool calls must still be flushed as a terminal chunk.
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"add\",\"arguments\":\"{\\\"a\\\":1}\"}}]},\"finish_reason\":null}]}\n\n\
data: [DONE]\n\n";
        let (base_url, _captured) = spawn_sse_server(sse_body).await;

        let chat =
            OpenAIChat::new(OpenAIConfig::new("test_key").with_base_url(format!("{base_url}/v1")));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("compute")])
            .await
            .unwrap();

        let mut terminal: Option<StreamChunk> = None;
        while let Some(item) = stream.next().await {
            let chunk = item.expect("chunk ok");
            if chunk.tool_calls.is_some() {
                terminal = Some(chunk);
            }
        }

        let final_chunk = terminal.expect("flushed tool-calls chunk");
        let calls = final_chunk.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name(), "add");
        assert_eq!(calls[0].arguments(), r#"{"a":1}"#);
    }

    /// A12: when the connection drops mid-stream — content chunks arrived but
    /// neither `[DONE]` nor a `finish_reason` chunk did — the consumer must see
    /// a terminal `StreamInterrupted` error instead of the partial text being
    /// mistaken for a complete answer.
    #[tokio::test]
    async fn stream_chat_truncated_without_terminal_emits_error() {
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n";
        let (base_url, _captured) = spawn_sse_server(sse_body).await;

        let chat =
            OpenAIChat::new(OpenAIConfig::new("test_key").with_base_url(format!("{base_url}/v1")));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut saw_partial = false;
        let mut terminal: Option<Result<StreamChunk, OpenAIError>> = None;
        while let Some(item) = stream.next().await {
            if item.is_ok() {
                saw_partial = true;
            }
            terminal = Some(item);
        }

        assert!(
            saw_partial,
            "partial chunks are still delivered before the error"
        );
        let err = terminal
            .expect("stream yields at least one item")
            .expect_err("truncated stream must end with an error, not a complete result");
        assert!(
            matches!(err, OpenAIError::StreamInterrupted(_)),
            "expected StreamInterrupted, got {err:?}"
        );
    }

    /// A12 regression guard: a stream that ends with a `finish_reason` chunk but
    /// no explicit `[DONE]` sentinel (common among OpenAI-compatible servers)
    /// is a normal completion and must NOT be flagged as interrupted.
    #[tokio::test]
    async fn stream_chat_finish_reason_without_done_completes_ok() {
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
        let (base_url, _captured) = spawn_sse_server(sse_body).await;

        let chat =
            OpenAIChat::new(OpenAIConfig::new("test_key").with_base_url(format!("{base_url}/v1")));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut chunks = 0usize;
        while let Some(item) = stream.next().await {
            item.expect("finish_reason chunk is a valid terminal");
            chunks += 1;
        }
        assert!(chunks >= 1, "content delivered and stream closed cleanly");
    }

    /// 0.25.0: `delta.reasoning_content` must reach the consumer as
    /// `StreamChunk::thinking_content`, and the streaming request must ask for
    /// the terminal usage chunk via `stream_options.include_usage`.
    #[tokio::test]
    async fn stream_chat_forwards_reasoning_and_requests_usage() {
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think first\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        let (base_url, captured) = spawn_sse_server(sse_body).await;

        let chat =
            OpenAIChat::new(OpenAIConfig::new("test_key").with_base_url(format!("{base_url}/v1")));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut thinking = Vec::new();
        let mut text = String::new();
        while let Some(item) = stream.next().await {
            let chunk = item.expect("chunk ok");
            if let Some(t) = chunk.thinking_content {
                thinking.push(t);
            }
            text.push_str(&chunk.text);
        }
        assert_eq!(thinking, vec!["think first".to_string()]);
        assert_eq!(text, "Hi");

        let request_body: serde_json::Value =
            serde_json::from_slice(&captured.lock().unwrap()).expect("request body json");
        assert_eq!(
            request_body["stream_options"]["include_usage"],
            serde_json::json!(true),
            "streaming requests must ask OpenAI to include usage"
        );
        assert_eq!(request_body["stream"], serde_json::json!(true));
    }
}

/// 0.21.0 S3.1: `response_format` plumbing — engine-side structured output.
mod tests_response_format {
    use super::*;
    use crate::openai::response_format::ResponseFormat;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct Person {
        /// The person's full name.
        name: String,
        /// The person's age in years.
        age: u32,
    }

    fn sample_messages() -> Vec<Message> {
        vec![Message::human("who are you")]
    }

    /// Default: no `response_format` key in the request body (unchanged behavior).
    #[test]
    fn build_request_body_has_no_response_format_by_default() {
        let chat = OpenAIChat::new(OpenAIConfig::new("k"));
        let body = chat.build_request_body(sample_messages(), false);
        assert!(body.get("response_format").is_none());
    }

    /// json_object mode is serialized with the `type` tag.
    #[test]
    fn build_request_body_includes_json_object_format() {
        let chat = OpenAIChat::new(OpenAIConfig::new("k"))
            .config
            .clone()
            .with_response_format(ResponseFormat::JsonObject);
        let chat = OpenAIChat::new(chat);
        let body = chat.build_request_body(sample_messages(), false);
        assert_eq!(body["response_format"]["type"], "json_object");
    }

    /// `with_json_schema_output` wires a strict json_schema response_format into
    /// the request body — and normalizes the generated schema for strict mode.
    #[test]
    fn with_json_schema_output_sets_strict_schema_format() {
        let chat = OpenAIChat::new(OpenAIConfig::new("k"));
        let method = chat.with_json_schema_output::<Person>();
        let body_chat = OpenAIChat::new(method.config.clone());
        let body = body_chat.build_request_body(sample_messages(), false);

        let format = &body["response_format"];
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["json_schema"]["name"], "output");
        assert_eq!(format["json_schema"]["strict"], true);

        let schema = &format["json_schema"]["schema"];
        assert_eq!(
            schema["additionalProperties"], false,
            "strict mode requires additionalProperties: false"
        );
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            required,
            vec!["age", "name"],
            "strict mode requires all properties"
        );
    }

    /// The tool-based `with_structured_output` path is unchanged (regression guard).
    #[test]
    fn with_structured_output_keeps_tool_based_path() {
        let chat = OpenAIChat::new(OpenAIConfig::new("k"));
        let method = chat.with_structured_output::<Person>();
        let body_chat = OpenAIChat::new(method.config.clone());
        let body = body_chat.build_request_body(sample_messages(), false);
        assert!(
            body.get("response_format").is_none(),
            "tool-based path must not set response_format"
        );
        assert_eq!(body["tools"][0]["function"]["strict"], true);
        assert_eq!(body["tool_choice"], "auto");
    }
}

// B5: optional auth + extra headers for generic OpenAI-compatible endpoints.
// 0.25.0: after the move to the unified HTTP layer these are behavioral
// loopback tests (the actual wire headers) instead of reqwest-builder probes.
mod tests_b5_headers {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One-shot stub that captures the request head (lowercased) and replies
    /// with a minimal valid chat completion.
    async fn spawn_json_server() -> (String, Arc<Mutex<Option<String>>>) {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let captured_task = captured.clone();
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
                let head = String::from_utf8_lossy(&header).to_string().to_lowercase();
                let content_length: usize = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                let mut body = vec![0u8; content_length];
                if content_length > 0 && socket.read_exact(&mut body).await.is_err() {
                    return;
                }
                *captured_task.lock().unwrap() = Some(head);
                // No Content-Length: reqwest reads until close (Connection: close).
                let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"id\":\"1\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":null}";
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}"), captured)
    }

    #[tokio::test]
    async fn keyless_config_omits_authorization_and_keeps_extras() {
        let (base_url, captured) = spawn_json_server().await;
        let config = OpenAIConfig {
            send_auth: false,
            base_url: format!("{base_url}/v1"),
            extra_headers: vec![("X-Tenant".to_string(), "acme".to_string())],
            ..Default::default()
        };
        let chat = OpenAIChat::new(config);
        chat.chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let head = captured.lock().unwrap().clone().expect("request captured");
        assert!(!head.contains("authorization:"), "head was: {head}");
        assert!(head.contains("x-tenant: acme"), "head was: {head}");
        // .json() bodies still carry the JSON content type.
        assert!(
            head.contains("content-type: application/json"),
            "head was: {head}"
        );
    }

    #[tokio::test]
    async fn default_config_still_sends_bearer() {
        let (base_url, captured) = spawn_json_server().await;
        let config = OpenAIConfig {
            api_key: "sk-secret".to_string(),
            base_url: format!("{base_url}/v1"),
            ..Default::default()
        };
        let chat = OpenAIChat::new(config);
        chat.chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let head = captured.lock().unwrap().clone().expect("request captured");
        assert!(
            head.contains("authorization: bearer sk-secret"),
            "head was: {head}"
        );
    }
}

// B7: unified multimodal request-body mapping (Chat Completions blocks).
mod tests_b7_multimodal {
    use super::*;
    use lc_schema::{AudioContent, FileContent, ImageContent, Message, VideoContent};

    #[test]
    fn multimodal_user_emits_text_image_audio_video_file_blocks() {
        let msg = Message::human("请看这些素材")
            .with_image(ImageContent::from_url("data:image/png;base64,aW1n"))
            .with_audio(AudioContent::from_base64_with_mime("YXVkaW8", "audio/wav"))
            .with_video(VideoContent::from_url("https://cdn.example.com/clip.mp4"))
            .with_file(FileContent::from_base64("ZG9j", "application/pdf").with_name("brief.pdf"));

        let value = OpenAIChat::message_to_openai_format(&msg);
        assert_eq!(value["role"], "user");
        let blocks = value["content"]
            .as_array()
            .expect("multimodal user content must be a blocks array");

        // Exact wire-shape snapshot.
        let expected = serde_json::json!([
            {"type": "text", "text": "请看这些素材"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
            {"type": "input_audio", "input_audio": {"data": "YXVkaW8", "format": "wav"}},
            {"type": "input_video", "input_video": {"url": "https://cdn.example.com/clip.mp4"}},
            {"type": "file", "file": {
                "file_data": "data:application/pdf;base64,ZG9j",
                "filename": "brief.pdf"
            }},
        ]);
        assert_eq!(serde_json::json!(blocks), expected);
    }

    #[test]
    fn plain_text_user_stays_a_string() {
        let value = OpenAIChat::message_to_openai_format(&Message::human("just text"));
        assert_eq!(value["role"], "user");
        assert_eq!(value["content"], serde_json::json!("just text"));
    }

    #[test]
    fn image_only_message_keeps_text_block_first() {
        let msg = Message::human("看图")
            .with_image(ImageContent::from_base64_with_mime("cG5n", "image/png"));
        let value = OpenAIChat::message_to_openai_format(&msg);
        let blocks = value["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image_url");
        assert_eq!(blocks[1]["image_url"]["url"], "data:image/png;base64,cG5n");
    }

    #[test]
    fn mp3_audio_uses_mp3_format_token() {
        let msg = Message::human("听")
            .with_audio(AudioContent::from_base64_with_mime("bXAz", "audio/mpeg"));
        let value = OpenAIChat::message_to_openai_format(&msg);
        let blocks = value["content"].as_array().unwrap();
        assert_eq!(blocks[1]["type"], "input_audio");
        assert_eq!(blocks[1]["input_audio"]["format"], "mp3");
    }
}
