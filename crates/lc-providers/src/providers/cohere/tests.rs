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
    let chunk = cohere_event_to_chunk(&ev).expect("content-delta -> chunk");
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
        // Streaming tool calls are framed by the API but not consumed here yet.
        r#"{"type":"tool-call-start","index":0,"delta":{"tool_call":{"name":"get_weather"}}}"#,
    ] {
        let ev = parse_cohere_event(data).unwrap().unwrap();
        assert!(
            cohere_event_to_chunk(&ev).is_none(),
            "framing/tool event must not emit a chunk: {data}"
        );
    }
}

#[test]
fn test_cohere_event_to_chunk_message_end_usage() {
    let data = r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}}"#;
    let ev = parse_cohere_event(data).unwrap().unwrap();
    let chunk = cohere_event_to_chunk(&ev).expect("message-end -> chunk");
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
    let chunks: Vec<StreamChunk> = events
        .iter()
        .filter_map(|e| parse_cohere_event(&e.data).ok().flatten())
        .filter_map(|ev| cohere_event_to_chunk(&ev))
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
