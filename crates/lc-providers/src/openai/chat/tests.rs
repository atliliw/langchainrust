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

        let content = OpenAIChat::aggregate_stream(stream).await.unwrap();
        assert_eq!(content, "Hello, world");
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Spawns a one-shot HTTP server that replies to POST /v1/chat/completions
    /// with the given OpenAI-style SSE body, returning the base URL.
    async fn spawn_sse_server(sse_body: &'static str) -> String {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
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
                let response =
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{sse_body}");
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn stream_chat_accumulates_fragmented_tool_calls() {
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"beij\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ing\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":8,\"total_tokens\":18}}\n\n\
data: [DONE]\n\n";
        let base_url = spawn_sse_server(sse_body).await;

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
        let base_url = spawn_sse_server(sse_body).await;

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
}
