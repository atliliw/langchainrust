#![allow(unused_imports)]

use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::timeout;

use super::sse::parse_sse_line;
use super::*;
use crate::types::MCPConfig;

#[tokio::test]
async fn test_stdio_transport_new_invalid_command() {
    let config = MCPConfig::stdio("nonexistent_command_xyz_zzz", vec![]);
    let result = StdioTransport::new(&config).await;
    assert!(result.is_err());
}

#[test]
fn test_sse_transport_new_wrong_config() {
    let config = MCPConfig::stdio("npx", vec![]);
    let result = SseTransport::new(&config);
    assert!(result.is_err());
}

#[test]
fn test_sse_transport_new_ok() {
    let config = MCPConfig::sse("http://localhost:3001/sse");
    let transport = SseTransport::new(&config);
    assert!(transport.is_ok());
}

#[test]
fn test_backoff_delay_starts_small() {
    assert_eq!(backoff_delay(0), Duration::from_millis(500));
    assert_eq!(backoff_delay(1), Duration::from_millis(1000));
    assert_eq!(backoff_delay(2), Duration::from_millis(2000));
}

#[test]
fn test_backoff_delay_capped() {
    // attempt=6 → 0.5 * 2^6 = 32s → cap 30s
    assert_eq!(backoff_delay(6), Duration::from_millis(30_000));
    assert_eq!(backoff_delay(100), Duration::from_millis(30_000));
}

#[test]
fn test_parse_sse_line_event_name() {
    let mut current = String::new();
    let result = parse_sse_line("event: endpoint", &mut current);
    assert!(result.is_none());
    assert_eq!(current, "endpoint");
}

#[test]
fn test_parse_sse_line_data() {
    let mut current = "endpoint".to_string();
    let result = parse_sse_line("data: http://localhost:3001/message", &mut current);
    let (evt, data) = result.unwrap();
    assert_eq!(evt, "endpoint");
    assert_eq!(data, "http://localhost:3001/message");
}

#[test]
fn test_parse_sse_line_other_line_ignored() {
    let mut current = String::new();
    let result = parse_sse_line(": keep-alive comment", &mut current);
    assert!(result.is_none());
    assert!(current.is_empty());
}

#[test]
fn test_connection_lost_error() {
    let err = MCPError::connection_lost();
    assert!(err.is_connection_lost());
    let other = MCPError::new(-1, "boom");
    assert!(!other.is_connection_lost());
}

/// Waits for the SSE background read loop to establish the connection (the early-exit check in request
/// requires connected=true).
async fn wait_connected(transport: &SseTransport) {
    transport.ensure_reader();
    timeout(Duration::from_secs(5), async {
        while !transport.is_connected() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("SSE connection should establish within 5s");
}

#[tokio::test]
async fn test_sse_request_success() {
    // Normal flow: discover the endpoint → POST succeeds.
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::Quiet).await;
    let config = MCPConfig::sse(&server.sse_url);
    let transport = SseTransport::new(&config).unwrap();
    wait_connected(&transport).await;

    // Use an unknown method (the test server replies {"ok": true} to unrecognized methods)
    let req = MCPRequest::new(1, "ping", None);
    let resp = transport
        .request(req)
        .await
        .expect("request should succeed");
    assert!(!resp.is_error());
    assert_eq!(resp.result, Some(serde_json::json!({ "ok": true })));
}

#[tokio::test]
async fn test_sse_request_retries_after_post_failure() {
    // P1-1: the first POST returns 500 → clear the cache + reconnect re-discover + retry once → success.
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::FailFirstPost)
            .await;
    let config = MCPConfig::sse(&server.sse_url);
    let transport = SseTransport::new(&config).unwrap();
    wait_connected(&transport).await;

    // Use an unknown method (the test server replies {"ok": true} to unrecognized methods)
    let req = MCPRequest::new(1, "ping", None);
    let resp = transport.request(req).await.expect("retry should succeed");
    assert_eq!(resp.result, Some(serde_json::json!({ "ok": true })));
    // At least 2 POSTs happened (first failure + successful retry), proving the cache was really cleared and
    // retried
    assert!(
        server.post_count.load(Ordering::SeqCst) >= 2,
        "expected >=2 POSTs after failure+retry, got {}",
        server.post_count.load(Ordering::SeqCst)
    );
}

#[tokio::test]
async fn test_sse_request_accepts_202_and_reads_response_via_sse_push() {
    // F4: the server replies 202 Accepted to the POST and pushes the JSON-RPC response over SSE
    // `event: message` → request must correlate the push by `id` and return the result (interop with
    // direct-response servers). Under PushResponse mode the POST body is always empty, so the result can only
    // come from the SSE push — a successful return proves the push-correlation path works.
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::PushResponse)
            .await;
    let config = MCPConfig::sse(&server.sse_url);
    let transport = SseTransport::new(&config).unwrap();
    wait_connected(&transport).await;

    // Use an unknown method (the test server replies {"ok": true} to unrecognized methods), consistent with
    // the existing cases.
    let req = MCPRequest::new(1, "ping", None);
    let result = timeout(Duration::from_secs(10), transport.request(req)).await;
    let resp = result
        .expect("request must not hang forever (10s guard)")
        .expect("request should succeed via SSE-pushed response");
    assert!(!resp.is_error());
    assert_eq!(resp.result, Some(serde_json::json!({ "ok": true })));
}

#[tokio::test]
async fn test_sse_request_times_out_when_server_hangs() {
    // F2: the server "connected but swallows the POST without returning a body" → the request must return an
    // error carrying "timed out" within request_timeout, never hanging forever.
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::HangPost).await;
    let config = MCPConfig::sse(&server.sse_url);
    // The test shortens the timeout window to 300ms; otherwise it would really wait 30s.
    let transport = SseTransport::new(&config)
        .unwrap()
        .with_request_timeout(Duration::from_millis(300));
    wait_connected(&transport).await;

    let req = MCPRequest::new(1, "ping", None);
    // Outer 10s backstop: if the timeout mechanism fails, this fails with an error instead of hanging the test.
    let result = timeout(Duration::from_secs(10), transport.request(req)).await;
    let err = result
        .expect("request must not hang forever (10s guard)")
        .expect_err("request should time out when server never responds");
    assert!(
        err.to_string().contains("timed out"),
        "expected 'timed out' in error, got: {}",
        err
    );
}

/// P2-6: in-process transport + a real `MCPServer` wires the Client↔Server protocol chain end-to-end.
///
/// Goes through `MCPClient::with_transport` (handshake) → `list_tools` → `call_tool`, no child process /
/// network at all, verifying `tools/call` is really executed via `BaseTool::run`.
#[tokio::test]
async fn test_in_memory_transport_round_trip() {
    use crate::MCPClient;
    use lc_core::tools::ToolError;
    use lc_core::BaseTool;

    struct EchoTool;
    #[async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo back input"
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(input)
        }
    }

    let server =
        Arc::new(crate::MCPServer::new().with_tool(Arc::new(EchoTool) as Arc<dyn BaseTool>));
    let client = MCPClient::with_transport(Box::new(InMemoryTransport::new(server)))
        .await
        .unwrap();
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");

    let result = client
        .call_tool("echo", serde_json::json!({"msg": "hi"}))
        .await
        .unwrap();
    assert!(!result.is_error, "server tool should not error");
    assert_eq!(result.text(), r#"{"msg":"hi"}"#);
}
