// src/language_models/providers/anthropic/tests.rs
//! Tests for the Anthropic module.

use super::chat::AnthropicChat;
use super::config::{AnthropicConfig, ThinkingConfig, ThinkingType};
use super::types::{AnthropicContent, AnthropicDelta};

#[test]
fn test_thinking_type_default() {
    assert_eq!(ThinkingType::default(), ThinkingType::Disabled);
}

#[test]
fn test_thinking_config_enabled() {
    let config = ThinkingConfig::enabled(10000);
    assert!(config.is_enabled());
    assert_eq!(config.budget_tokens, 10000);
}

#[test]
fn test_thinking_config_disabled() {
    let config = ThinkingConfig::disabled();
    assert!(!config.is_enabled());
}

#[test]
fn test_anthropic_config_with_thinking() {
    let config = AnthropicConfig::new("test-key").with_thinking(ThinkingConfig::enabled(5000));
    assert!(config.thinking.is_enabled());
    assert_eq!(config.thinking.budget_tokens, 5000);
}

#[test]
fn test_anthropic_config_default_no_thinking() {
    let config = AnthropicConfig::default();
    assert!(!config.thinking.is_enabled());
}

#[test]
fn test_anthropic_chat_with_thinking() {
    let config = AnthropicConfig::new("test-key");
    let chat = AnthropicChat::new(config).with_thinking(8000);
    assert!(chat.thinking_config().is_enabled());
    assert_eq!(chat.thinking_config().budget_tokens, 8000);
}

#[test]
fn test_build_request_body_without_thinking() {
    let config = AnthropicConfig::new("test-key");
    let chat = AnthropicChat::new(config);
    let body = chat.build_request_body(vec![], false);
    assert!(body.get("thinking").is_none());
}

#[test]
fn test_build_request_body_with_thinking() {
    let config = AnthropicConfig::new("test-key").with_thinking(ThinkingConfig::enabled(10000));
    let chat = AnthropicChat::new(config);
    let body = chat.build_request_body(vec![], false);

    let thinking = body.get("thinking").expect("thinking should be present");
    assert_eq!(thinking["type"], "enabled");
    assert_eq!(thinking["budget_tokens"], 10000);
}

#[test]
fn test_anthropic_content_deserialize_thinking() {
    let json = r#"{"type": "thinking", "thinking": "Let me analyze..."}"#;
    let content: AnthropicContent = serde_json::from_str(json).unwrap();
    assert_eq!(content.content_type, "thinking");
    assert_eq!(content.thinking, "Let me analyze...");
}

#[test]
fn test_anthropic_content_deserialize_text() {
    let json = r#"{"type": "text", "text": "The answer is 42."}"#;
    let content: AnthropicContent = serde_json::from_str(json).unwrap();
    assert_eq!(content.content_type, "text");
    assert_eq!(content.text, "The answer is 42.");
}

#[test]
fn test_anthropic_delta_deserialize_thinking_delta() {
    let json = r#"{"type": "thinking_delta", "thinking": "Hmm..."}"#;
    let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
    assert_eq!(delta.type_field, "thinking_delta");
    assert_eq!(delta.thinking, "Hmm...");
}

#[test]
fn test_anthropic_delta_deserialize_text_delta() {
    let json = r#"{"type": "text_delta", "text": "Hello"}"#;
    let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
    assert_eq!(delta.type_field, "text_delta");
    assert_eq!(delta.text, "Hello");
}

/// A12: a connection that closes after text deltas but before the terminal
/// `message_stop` event delivered a truncated answer — the stream must end with
/// `StreamInterrupted`, not a silent clean completion.
mod tests_a12_stream_truncation {
    use super::AnthropicChat;
    use crate::providers::anthropic::config::AnthropicConfig;
    use crate::providers::anthropic::error::AnthropicError;
    use crate::providers::anthropic::types::AnthropicStreamToken;
    use futures_util::StreamExt;
    use lc_schema::Message;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One-shot server replying to POST {base}/messages with a fixed SSE body.
    async fn spawn_sse_server(sse_body: &'static str) -> String {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
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
                let response =
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{sse_body}");
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn stream_truncated_before_message_stop_emits_error() {
        // Text deltas arrive, but content_block_stop / message_delta /
        // message_stop never do — the server drops the connection.
        let sse_body = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n";
        let base_url = spawn_sse_server(sse_body).await;

        let chat = AnthropicChat::new(
            AnthropicConfig::new("test_key").with_base_url(format!("{base_url}/v1")),
        );
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut text = String::new();
        let mut terminal: Option<Result<AnthropicStreamToken, AnthropicError>> = None;
        while let Some(item) = stream.next().await {
            if let Ok(AnthropicStreamToken::Text(t)) = &item {
                text.push_str(t);
            }
            terminal = Some(item);
        }

        assert_eq!(text, "Hello", "partial text is still delivered");
        let err = terminal
            .expect("stream yields items")
            .expect_err("truncated stream must end with StreamInterrupted");
        assert!(
            matches!(err, AnthropicError::StreamInterrupted(_)),
            "expected StreamInterrupted, got {err:?}"
        );
    }

    #[tokio::test]
    async fn stream_ending_with_message_stop_completes_ok() {
        // A full, well-formed Anthropic SSE sequence must not regress.
        let sse_body = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";
        let base_url = spawn_sse_server(sse_body).await;

        let chat = AnthropicChat::new(
            AnthropicConfig::new("test_key").with_base_url(format!("{base_url}/v1")),
        );
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut text = String::new();
        let mut saw_usage = false;
        while let Some(item) = stream.next().await {
            match item.expect("well-formed stream has no errors") {
                AnthropicStreamToken::Text(t) => text.push_str(&t),
                AnthropicStreamToken::Usage(_) => saw_usage = true,
                _ => {}
            }
        }
        assert_eq!(text, "Hi");
        assert!(saw_usage, "message_delta usage token still delivered");
    }
}
