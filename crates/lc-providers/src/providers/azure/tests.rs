// lc-providers/src/providers/azure/tests.rs

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
    let config = AzureOpenAIConfig::new(
        "https://myresource.openai.azure.com",
        "gpt-4-deployment",
        "test-key",
    );
    assert_eq!(config.endpoint, "https://myresource.openai.azure.com");
    assert_eq!(config.deployment_name, "gpt-4-deployment");
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.api_version, AZURE_DEFAULT_API_VERSION);
}

#[test]
fn test_config_builder() {
    let config = AzureOpenAIConfig::new("https://res.openai.azure.com", "deploy", "key")
        .with_api_version("2024-06-01")
        .with_model("gpt-4o")
        .with_temperature(0.5)
        .with_max_tokens(2048)
        .with_top_p(0.9);
    assert_eq!(config.api_version, "2024-06-01");
    assert_eq!(config.model, Some("gpt-4o".to_string()));
    assert_eq!(config.temperature, Some(0.5));
    assert_eq!(config.max_tokens, Some(2048));
    assert_eq!(config.top_p, Some(0.9));
}

#[test]
fn test_config_from_env_result_ok() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old_ep = save_and_set("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com");
    let old_dn = save_and_set("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deploy");
    let old_key = save_and_set("AZURE_OPENAI_API_KEY", "azure-key-123");
    let result = AzureOpenAIConfig::from_env_result();
    assert!(result.is_ok());
    let config = result.unwrap();
    assert_eq!(config.endpoint, "https://test.openai.azure.com");
    assert_eq!(config.deployment_name, "my-deploy");
    assert_eq!(config.api_key, "azure-key-123");
    restore("AZURE_OPENAI_ENDPOINT", old_ep);
    restore("AZURE_OPENAI_DEPLOYMENT_NAME", old_dn);
    restore("AZURE_OPENAI_API_KEY", old_key);
}

#[test]
fn test_config_from_env_result_err_when_missing() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old_ep = std::env::var("AZURE_OPENAI_ENDPOINT").ok();
    std::env::remove_var("AZURE_OPENAI_ENDPOINT");
    let result = AzureOpenAIConfig::from_env_result();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("AZURE_OPENAI_ENDPOINT"));
    restore("AZURE_OPENAI_ENDPOINT", old_ep);
}

#[test]
fn test_chat_url() {
    let config = AzureOpenAIConfig::new("https://myresource.openai.azure.com", "gpt4", "key");
    let url = config.chat_url();
    assert!(url.contains("myresource.openai.azure.com"));
    assert!(url.contains("/openai/deployments/gpt4/chat/completions"));
    assert!(url.contains("api-version="));
}

#[test]
fn test_chat_url_trailing_slash() {
    let config = AzureOpenAIConfig::new("https://myresource.openai.azure.com/", "gpt4", "key");
    let url = config.chat_url();
    // Should not have double slashes
    assert!(!url.contains("//openai"));
}

#[test]
fn test_effective_model() {
    let config_no_model = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    assert_eq!(config_no_model.effective_model(), "deploy");

    let config_with_model =
        AzureOpenAIConfig::new("https://ep", "deploy", "key").with_model("gpt-4o");
    assert_eq!(config_with_model.effective_model(), "gpt-4o");
}

#[test]
fn test_chat_new() {
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    let _chat = AzureOpenAIChat::new(config);
}

#[test]
fn test_model_name() {
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key").with_model("gpt-4o");
    let chat = AzureOpenAIChat::new(config);
    assert_eq!(chat.model_name(), "gpt-4o");
}

#[test]
fn test_build_request_body_no_model() {
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    let chat = AzureOpenAIChat::new(config);
    let body = chat.build_request_body(vec![Message::human("hello")], false);
    // Azure request body should NOT contain "model" field
    assert!(body.get("model").is_none());
    assert!(body.get("messages").is_some());
}

#[test]
fn test_build_request_body_tools_and_tool_choice() {
    // 0.25.0: Azure previously never sent `tools` at all.
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    let tool = lc_core::tools::ToolDefinition::new("get_weather", "Weather lookup")
        .with_parameters(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }));
    let chat = AzureOpenAIChat::new(config)
        .bind_tools(vec![tool])
        .with_tool_choice("auto");
    let body = chat.build_request_body(vec![Message::human("hello")], false);
    let tools = body.get("tools").expect("tools present in body");
    assert_eq!(tools[0]["function"]["name"], "get_weather");
    assert_eq!(body["tool_choice"], serde_json::json!("auto"));
    // Non-streaming bodies must not gain stream_options.
    assert!(body.get("stream_options").is_none());
}

#[test]
fn test_build_request_body_stream_requests_usage() {
    // 0.25.0: streaming requests ask for the terminal usage chunk.
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    let chat = AzureOpenAIChat::new(config);
    let body = chat.build_request_body(vec![Message::human("hello")], true);
    assert_eq!(
        body["stream_options"]["include_usage"],
        serde_json::json!(true)
    );
    assert_eq!(body["stream"], serde_json::json!(true));
}

#[test]
fn test_error_display() {
    let err = AzureOpenAIError::Http("timeout".to_string());
    assert!(err.to_string().contains("HTTP error"));
    let err = AzureOpenAIError::Api("rate limit".to_string());
    assert!(err.to_string().contains("API error"));
    let err = AzureOpenAIError::Parse("bad json".to_string());
    assert!(err.to_string().contains("parse error"));
    let err = AzureOpenAIError::StreamInterrupted("closed".to_string());
    assert!(err.to_string().contains("stream interrupted"));
}

/// A12: mid-stream truncation detection for the Azure SSE loop (which is a
/// structural duplicate of OpenAI's).
mod tests_a12_stream_truncation {
    use super::*;
    use futures_util::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One-shot server that answers the POST with a fixed SSE body and closes
    /// the connection (path is ignored, as Azure's deployment path is built by
    /// `chat_url()`). Captures the full raw request (head + body).
    async fn spawn_sse_server(
        sse_body: &'static str,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<u8>>>) {
        use std::sync::{Arc, Mutex};
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
                {
                    // Drop the guard before any await: std MutexGuard is !Send.
                    let mut raw = captured_clone.lock().unwrap_or_else(|e| e.into_inner());
                    raw.extend_from_slice(&header);
                    raw.extend_from_slice(&body);
                }
                let response =
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{sse_body}");
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}"), captured)
    }

    #[tokio::test]
    async fn stream_truncated_without_terminal_emits_error() {
        // Two content chunks, no finish_reason, no [DONE] — a dropped connection.
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n";
        let (base_url, _captured) = spawn_sse_server(sse_body).await;

        let config = AzureOpenAIConfig::new(base_url, "deploy", "test_key");
        let chat = AzureOpenAIChat::new(config);
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut saw_partial = false;
        let mut terminal: Option<Result<StreamChunk, AzureOpenAIError>> = None;
        while let Some(item) = stream.next().await {
            if item.is_ok() {
                saw_partial = true;
            }
            terminal = Some(item);
        }

        assert!(
            saw_partial,
            "partial chunks still delivered before the error"
        );
        let err = terminal
            .expect("stream yields items")
            .expect_err("truncated Azure stream must end with an error");
        assert!(
            matches!(err, AzureOpenAIError::StreamInterrupted(_)),
            "expected StreamInterrupted, got {err:?}"
        );
    }

    #[tokio::test]
    async fn stream_finish_reason_completes_ok() {
        // finish_reason without [DONE] is a normal completion for compat servers.
        let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
        let (base_url, _captured) = spawn_sse_server(sse_body).await;

        let chat = AzureOpenAIChat::new(AzureOpenAIConfig::new(base_url, "deploy", "test_key"));
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut chunks = 0usize;
        while let Some(item) = stream.next().await {
            item.expect("finish_reason is a valid terminal");
            chunks += 1;
        }
        assert!(chunks >= 1);
    }
}

/// 0.25.0 B2: real wire contracts for Azure — function calling on the
/// non-stream path, reasoning/tool-call/usage passthrough on the stream path,
/// `api-key` auth header, and `stream_options.include_usage` in the request.
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
    async fn non_stream_sends_api_key_and_tools_and_parses_tool_call() {
        let response =
            "{\"id\":\"x\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"gpt-4o\",\
\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":null,\
\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\
\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]},\
\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":11,\
\"completion_tokens\":7,\"total_tokens\":18}}";
        let (base_url, captured) = spawn_server(response, "application/json").await;

        let chat = AzureOpenAIChat::new(AzureOpenAIConfig::new(base_url, "deploy", "azure-secret"))
            .bind_tools(vec![weather_tool()]);
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
            head.lines().any(|l| l == "api-key: azure-secret"),
            "Azure auth uses the api-key header, got:\n{head}"
        );
        assert!(
            !head.contains("authorization:"),
            "Azure must not send a Bearer authorization header"
        );
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert!(body.get("stream_options").is_none());
    }

    #[tokio::test]
    async fn stream_forwards_reasoning_tool_calls_and_usage() {
        let sse = "\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"beijing\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}\n\n\
data: [DONE]\n\n";
        let (base_url, captured) = spawn_server(sse, "text/event-stream").await;

        let chat = AzureOpenAIChat::new(AzureOpenAIConfig::new(base_url, "deploy", "azure-secret"))
            .bind_tools(vec![weather_tool()]);
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("weather?")])
            .await
            .unwrap();

        let mut thinking = Vec::new();
        let mut terminal_usage = None;
        let mut terminal_calls = None;
        while let Some(item) = stream.next().await {
            let chunk = item.expect("stream ok");
            if let Some(t) = chunk.thinking_content {
                thinking.push(t);
            }
            if chunk.token_usage.is_some() {
                terminal_usage = chunk.token_usage;
                terminal_calls = chunk.tool_calls;
            }
        }
        assert_eq!(thinking, vec!["think".to_string()]);
        assert_eq!(
            terminal_usage.expect("usage chunk forwarded").total_tokens,
            18
        );
        let calls = terminal_calls.expect("accumulated tool calls on terminal chunk");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"beijing\"}");

        let (_head, body) = head_and_body(&captured.lock().unwrap());
        assert_eq!(
            body["stream_options"]["include_usage"],
            serde_json::json!(true),
            "Azure streaming request must ask for usage"
        );
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    }
}
