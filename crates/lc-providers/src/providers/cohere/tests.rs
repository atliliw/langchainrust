// lc-providers/src/providers/cohere/tests.rs

use super::*;
use crate::ENV_TEST_LOCK;

fn save_and_set(key: &str, value: &str) -> Option<String> {
    let old = std::env::var(key).ok();
    std::env::set_var(key, value);
    old
}

fn restore(key: &str, old: Option<String>) {
    match old {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
}

#[test]
fn test_config_new() {
    let config = CohereConfig::new("test-key");
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.base_url, COHERE_BASE_URL);
    assert_eq!(config.model, "command-r-plus");
}

#[test]
fn test_config_builder() {
    let config = CohereConfig::new("key")
        .with_model("command-r")
        .with_base_url("https://custom.cohere.com/v2")
        .with_temperature(0.5)
        .with_max_tokens(1024)
        .with_preamble("You are a helpful assistant.");
    assert_eq!(config.model, "command-r");
    assert_eq!(config.base_url, "https://custom.cohere.com/v2");
    assert_eq!(config.temperature, Some(0.5));
    assert_eq!(config.max_tokens, Some(1024));
    assert_eq!(
        config.preamble,
        Some("You are a helpful assistant.".to_string())
    );
}

#[test]
fn test_config_from_env_result_ok() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old = save_and_set("COHERE_API_KEY", "env-key");
    let result = CohereConfig::from_env_result();
    assert!(result.is_ok());
    assert_eq!(result.unwrap().api_key, "env-key");
    restore("COHERE_API_KEY", old);
}

#[test]
fn test_config_from_env_result_err_when_missing() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old = std::env::var("COHERE_API_KEY").ok();
    std::env::remove_var("COHERE_API_KEY");
    let result = CohereConfig::from_env_result();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("COHERE_API_KEY"));
    restore("COHERE_API_KEY", old);
}

#[test]
fn test_chat_new() {
    let config = CohereConfig::new("test-key");
    let _chat = CohereChat::new(config);
}

#[test]
fn test_model_name() {
    let config = CohereConfig::new("key").with_model("command-r");
    let chat = CohereChat::new(config);
    assert_eq!(chat.model_name(), "command-r");
}

#[test]
fn test_build_request_body() {
    let config = CohereConfig::new("key").with_preamble("System prompt");
    let chat = CohereChat::new(config);
    let body = chat.build_request_body(vec![Message::human("hello")], false);
    assert_eq!(body["model"], "command-r-plus");
    assert!(body.get("messages").is_some());
    assert_eq!(body["preamble"], "System prompt");
}

#[test]
fn test_error_display() {
    let err = CohereError::Http("timeout".to_string());
    assert!(err.to_string().contains("HTTP error"));
    let err = CohereError::Api("rate limit".to_string());
    assert!(err.to_string().contains("API error"));
    let err = CohereError::Parse("bad json".to_string());
    assert!(err.to_string().contains("parse error"));
}

#[test]
fn test_message_to_cohere_format_human() {
    let msg = Message::human("Hello");
    let formatted = CohereChat::message_to_cohere_format(&msg);
    assert_eq!(formatted["role"], "user");
    assert_eq!(formatted["content"], "Hello");
}

#[test]
fn test_message_to_cohere_format_system() {
    let msg = Message::system("You are helpful");
    let formatted = CohereChat::message_to_cohere_format(&msg);
    assert_eq!(formatted["role"], "system");
}

// ---- 0.20.0 P4: Cohere v2 streaming is its own SSE format, not OpenAI ----

#[test]
fn test_parse_cohere_event_done_terminator() {
    // Cohere never sends [DONE] (it closes the connection after message-end),
    // but the parser tolerates it for OpenAI-compatible proxies.
    let result = parse_cohere_event("[DONE]").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_parse_cohere_event_malformed_data_errors() {
    // 0.20.0 P4: malformed payloads surface as errors (logged, not silent),
    // matching the OpenAI streaming path.
    assert!(parse_cohere_event("not json").is_err());
}

#[test]
fn test_cohere_event_to_chunk_content_delta() {
    let data =
        r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hi"}}}}"#;
    let ev = parse_cohere_event(data).unwrap().unwrap();
    let mut acc = CohereToolCallAccumulator::default();
    let chunk = cohere_event_to_chunk(&mut acc, &ev).expect("content-delta -> chunk");
    assert_eq!(chunk.text, "Hi");
    assert!(chunk.token_usage.is_none());
    assert!(chunk.tool_calls.is_none());
}

#[test]
fn test_cohere_event_to_chunk_ignores_framing_events() {
    for data in [
        r#"{"type":"message-start","message":{"role":"assistant","id":"m1"}}"#,
        r#"{"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}"#,
        r#"{"type":"content-end","index":0}"#,
        // tool-call-start seeds the accumulator but does not itself emit a chunk.
        r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}}"#,
    ] {
        let ev = parse_cohere_event(data).unwrap().unwrap();
        let mut acc = CohereToolCallAccumulator::default();
        assert!(
            cohere_event_to_chunk(&mut acc, &ev).is_none(),
            "framing/tool event must not emit a chunk: {data}"
        );
    }
}

#[test]
fn test_cohere_event_to_chunk_message_end_usage() {
    let data = r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}}"#;
    let ev = parse_cohere_event(data).unwrap().unwrap();
    let mut acc = CohereToolCallAccumulator::default();
    let chunk = cohere_event_to_chunk(&mut acc, &ev).expect("message-end -> chunk");
    assert!(chunk.text.is_empty());
    let usage = chunk.token_usage.expect("usage parsed");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(usage.total_tokens, 7);
}

#[test]
fn test_stream_parses_full_cohere_v2_stream() {
    // 0.20.0 P4 lock-in: a realistic Cohere v2 SSE stream (framing + text +
    // usage) is parsed into the expected concatenated text. Before the fix the
    // old OpenAI-format parser rejected every event, so the stream was empty.
    use crate::openai::sse::SSEParser;

    let mut parser = SSEParser::new();
    let raw = format!(
        "{}\n\n",
        [
            "event: message-start\ndata: {\"type\":\"message-start\",\"message\":{\"role\":\"assistant\",\"id\":\"m1\"}}",
            "event: content-start\ndata: {\"type\":\"content-start\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"\"}}}}",
            "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"Hello\"}}}}",
            "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\" world\"}}}}",
            "event: content-end\ndata: {\"type\":\"content-end\",\"index\":0}",
            "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":2}}}}",
        ]
        .join("\n\n")
    );

    let events = parser.parse(&raw);
    assert_eq!(events.len(), 6, "six SSE events");
    let mut acc = CohereToolCallAccumulator::default();
    let chunks: Vec<StreamChunk> = events
        .iter()
        .filter_map(|e| parse_cohere_event(&e.data).ok().flatten())
        .filter_map(|ev| cohere_event_to_chunk(&mut acc, &ev))
        .collect();

    let text: String = chunks.iter().map(|c| c.text.clone()).collect();
    assert_eq!(text, "Hello world");

    let usage = chunks
        .iter()
        .find_map(|c| c.token_usage.clone())
        .expect("usage from message-end");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(usage.total_tokens, 7);
}

// ---- 0.25.0 B2: tool-plan, fragmented tool calls, real arguments shape ----

#[test]
fn test_tool_plan_delta_maps_to_thinking_content() {
    let data = r#"{"type":"tool-plan-delta","delta":{"message":{"tool_plan":"I should check"}}}"#;
    let ev = parse_cohere_event(data).unwrap().unwrap();
    let mut acc = CohereToolCallAccumulator::default();
    let chunk = cohere_event_to_chunk(&mut acc, &ev).expect("tool-plan-delta -> chunk");
    assert_eq!(chunk.thinking_content.as_deref(), Some("I should check"));
    assert!(chunk.text.is_empty());
    assert!(chunk.tool_calls.is_none());
}

#[test]
fn test_streaming_tool_calls_accumulate_by_index_and_flush_on_message_end() {
    let mut acc = CohereToolCallAccumulator::default();

    let start = parse_cohere_event(
        r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}}"#,
    )
    .unwrap()
    .unwrap();
    assert!(cohere_event_to_chunk(&mut acc, &start).is_none());

    let delta1 = parse_cohere_event(
        r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}}"#,
    )
    .unwrap()
    .unwrap();
    assert!(cohere_event_to_chunk(&mut acc, &delta1).is_none());

    let delta2 = parse_cohere_event(
        r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":[{"index":0,"function":{"arguments":" \"beijing\"}"}}]}}}"#,
    )
    .unwrap()
    .unwrap();
    assert!(cohere_event_to_chunk(&mut acc, &delta2).is_none());

    let call_end = parse_cohere_event(r#"{"type":"tool-call-end","index":0,"delta":{"index":0}}"#)
        .unwrap()
        .unwrap();
    assert!(cohere_event_to_chunk(&mut acc, &call_end).is_none());

    let message_end = parse_cohere_event(
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":3}}}}"#,
    )
    .unwrap()
    .unwrap();
    let chunk = cohere_event_to_chunk(&mut acc, &message_end).expect("message-end -> chunk");

    let calls = chunk
        .tool_calls
        .expect("accumulated calls flushed on message-end");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].function.arguments, "{\"city\": \"beijing\"}");
    assert_eq!(chunk.token_usage.expect("usage present").total_tokens, 13);
}

#[test]
fn test_accumulator_drops_unfinished_calls() {
    // A delta with arguments but neither start (id) nor name must not produce
    // an un-routable tool call.
    let mut acc = CohereToolCallAccumulator::default();
    let orphan = parse_cohere_event(
        r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}}"#,
    )
    .unwrap()
    .unwrap();
    assert!(cohere_event_to_chunk(&mut acc, &orphan).is_none());
    assert!(acc.build().is_none());
}

#[test]
fn test_non_stream_arguments_accept_json_object() {
    // Real v2 wire shape: non-streaming function.arguments is a JSON object.
    let raw = r#"{
        "id":"resp_1","finish_reason":"COMPLETE","model":"command-r-plus",
        "message":{"role":"assistant","content":[{"type":"text","text":""}],
        "tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":{"city":"beijing"}}}]},
        "usage":{"tokens":{"input_tokens":12,"output_tokens":6}}
    }"#;
    let parsed: CohereChatResponse = serde_json::from_str(raw).unwrap();
    let message = parsed.message.expect("message present");
    let call = &message.tool_calls[0];
    assert_eq!(
        normalize_arguments(call.function.arguments.clone()),
        r#"{"city":"beijing"}"#
    );
}

#[test]
fn test_non_stream_arguments_accept_string_too() {
    // Streaming histories echo arguments back as a pre-serialized string.
    let fc: CohereFunctionCall =
        serde_json::from_str(r#"{"name":"f","arguments":"{\"a\":1}"}"#).unwrap();
    assert_eq!(normalize_arguments(fc.arguments), r#"{"a":1}"#);
}

#[test]
fn test_content_part_without_text_deserializes() {
    // Tool-history parts can carry no top-level `text`; a required String made
    // every such response fail.
    let msg: CohereMessage =
        serde_json::from_str(r#"{"role":"assistant","content":[{"type":"tool"}]}"#).unwrap();
    assert!(msg.content[0].text.is_none());
}

/// 0.25.0 B2: real wire contracts for Cohere v2 — Bearer auth, tools/
/// tool_choice request shapes, object-form non-stream `arguments`, and the
/// streaming tool-plan / fragmented tool-call / message-end sequence.
mod tests_b2_contracts {
    use super::*;
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One-shot loopback server: capture the full raw request, reply with the
    /// fixed body and close. The reply carries no Content-Length — the unified
    /// HTTP layer reads until close.
    async fn spawn_server(
        response_body: &'static str,
        content_type: &'static str,
    ) -> (String, Arc<Mutex<Vec<u8>>>) {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
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
                if content_length > 0 && socket.read_exact(&mut body).await.is_err() {
                    return;
                }
                {
                    // Drop the guard before any await: std MutexGuard is !Send.
                    let mut raw = captured_clone.lock().unwrap_or_else(|e| e.into_inner());
                    raw.extend_from_slice(&header);
                    raw.extend_from_slice(&body);
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n{response_body}"
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}"), captured)
    }

    fn head_and_body(raw: &[u8]) -> (String, serde_json::Value) {
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("request head terminated");
        let head = String::from_utf8_lossy(&raw[..split]).to_lowercase();
        let body: serde_json::Value =
            serde_json::from_slice(&raw[split + 4..]).expect("request body is json");
        (head, body)
    }

    fn weather_tool() -> lc_core::tools::ToolDefinition {
        lc_core::tools::ToolDefinition::new("get_weather", "Weather lookup").with_parameters(
            serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }),
        )
    }

    #[tokio::test]
    async fn non_stream_sends_bearer_tools_and_parses_object_arguments() {
        let response = "{\"id\":\"resp_1\",\"finish_reason\":\"COMPLETE\",\
\"model\":\"command-r-plus\",\"message\":{\"role\":\"assistant\",\
\"content\":[{\"type\":\"text\",\"text\":\"\"}],\"tool_calls\":[{\
\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\
\"arguments\":{\"city\":\"SF\"}}}]},\
\"usage\":{\"tokens\":{\"input_tokens\":11,\"output_tokens\":7}}}";
        let (base_url, captured) = spawn_server(response, "application/json").await;

        let chat = CohereChat::new(CohereConfig::new("cohere-secret").with_base_url(base_url))
            .bind_tools(vec![weather_tool()])
            .with_tool_choice("AUTO");
        let result = chat
            .chat_internal(vec![Message::human("weather?")])
            .await
            .unwrap();

        let calls = result.tool_calls.expect("non-stream tool call parsed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"SF\"}");
        assert_eq!(result.token_usage.unwrap().total_tokens, 18);

        let (head, body) = head_and_body(&captured.lock().unwrap());
        assert!(
            head.lines()
                .any(|l| l == "authorization: bearer cohere-secret"),
            "Cohere v2 auth is Bearer, got:\n{head}"
        );
        assert_eq!(body["stream"], serde_json::json!(false));
        assert_eq!(body["tool_choice"], serde_json::json!("AUTO"));
        assert_eq!(body["tools"][0]["type"], serde_json::json!("function"));
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    }

    #[tokio::test]
    async fn stream_forwards_tool_plan_fragmented_tool_calls_and_usage() {
        let sse = "\
event: message-start\n\
data: {\"type\":\"message-start\",\"delta\":{\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\"}}}\n\n\
event: tool-plan-delta\n\
data: {\"type\":\"tool-plan-delta\",\"delta\":{\"message\":{\"tool_plan\":\"I should \"}}}\n\n\
event: tool-plan-delta\n\
data: {\"type\":\"tool-plan-delta\",\"delta\":{\"message\":{\"tool_plan\":\"check weather\"}}}\n\n\
event: content-delta\n\
data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"Beijing \"}}}}\n\n\
event: content-delta\n\
data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"is sunny\"}}}}\n\n\
event: tool-call-start\n\
data: {\"type\":\"tool-call-start\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}}\n\n\
event: tool-call-delta\n\
data: {\"type\":\"tool-call-delta\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}}\n\n\
event: tool-call-delta\n\
data: {\"type\":\"tool-call-delta\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" \\\"beijing\\\"}\"}}]}}}\n\n\
event: tool-call-end\n\
data: {\"type\":\"tool-call-end\",\"index\":0,\"delta\":{\"index\":0}}\n\n\
event: message-end\n\
data: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":11,\"output_tokens\":7}}}}\n\n";
        let (base_url, captured) = spawn_server(sse, "text/event-stream").await;

        let chat = CohereChat::new(CohereConfig::new("cohere-secret").with_base_url(base_url))
            .bind_tools(vec![weather_tool()]);
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("weather?")])
            .await
            .unwrap();

        let mut thinking = String::new();
        let mut text = String::new();
        let mut terminal_usage = None;
        let mut terminal_calls = None;
        while let Some(item) = stream.next().await {
            let chunk = item.expect("stream ok");
            if let Some(plan) = chunk.thinking_content {
                thinking.push_str(&plan);
            }
            text.push_str(&chunk.text);
            if chunk.token_usage.is_some() {
                terminal_usage = chunk.token_usage;
                terminal_calls = chunk.tool_calls;
            }
        }
        assert_eq!(thinking, "I should check weather");
        assert_eq!(text, "Beijing is sunny");
        assert_eq!(
            terminal_usage.expect("usage chunk forwarded").total_tokens,
            18
        );
        let calls = terminal_calls.expect("accumulated tool calls on message-end");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\": \"beijing\"}");

        let (_head, body) = head_and_body(&captured.lock().unwrap());
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    }

    /// M-8: a stream that closes without a terminal `message-end` is surfaced as
    /// `StreamInterrupted` rather than handed back as a partial but "complete" reply.
    #[tokio::test]
    async fn stream_truncation_without_message_end_is_interrupted() {
        let sse = "\
event: content-delta\n\
data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"Partial \"}}}}\n\n\
event: content-delta\n\
data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"output\"}}}}\n\n";
        let (base_url, _captured) = spawn_server(sse, "text/event-stream").await;

        let chat = CohereChat::new(CohereConfig::new("cohere-secret").with_base_url(base_url));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut text = String::new();
        let mut saw_interrupt = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => text.push_str(&chunk.text),
                Err(CohereError::StreamInterrupted(_)) => {
                    saw_interrupt = true;
                    break;
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert_eq!(text, "Partial output");
        assert!(
            saw_interrupt,
            "truncated stream must surface StreamInterrupted, got text {text:?}"
        );
    }
}
