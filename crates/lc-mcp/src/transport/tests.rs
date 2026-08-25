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
    // attempt=6 → 0.5 * 2^6 = 32s → 上限 30s
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

/// 等待 SSE 后台读循环建立连接(request 的早退检查要求 connected=true)。
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
    // 正常流程:发现 endpoint → POST 成功。
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::Quiet).await;
    let config = MCPConfig::sse(&server.sse_url);
    let transport = SseTransport::new(&config).unwrap();
    wait_connected(&transport).await;

    // 用未知方法(测试服务器对未识别方法回 {"ok": true})
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
    // P1-1:第一次 POST 返回 500 → 清空缓存 + 重连重发现 + 重试一次 → 成功。
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::FailFirstPost)
            .await;
    let config = MCPConfig::sse(&server.sse_url);
    let transport = SseTransport::new(&config).unwrap();
    wait_connected(&transport).await;

    // 用未知方法(测试服务器对未识别方法回 {"ok": true})
    let req = MCPRequest::new(1, "ping", None);
    let resp = transport.request(req).await.expect("retry should succeed");
    assert_eq!(resp.result, Some(serde_json::json!({ "ok": true })));
    // 至少发生了 2 次 POST(首次失败 + 重试成功),证明缓存确实被清掉重试了
    assert!(
        server.post_count.load(Ordering::SeqCst) >= 2,
        "expected >=2 POSTs after failure+retry, got {}",
        server.post_count.load(Ordering::SeqCst)
    );
}

#[tokio::test]
async fn test_sse_request_accepts_202_and_reads_response_via_sse_push() {
    // F4:服务器对 POST 回 202 Accepted、JSON-RPC 响应经 SSE `event: message`
    // 推送 → request 必须按 `id` 关联到推送并返回结果(与直接响应型服务器
    // 互操作)。PushResponse 模式下 POST body 恒为空,结果只能来自 SSE 推送,
    // 因此成功返回即证明推送关联路径走通。
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::PushResponse)
            .await;
    let config = MCPConfig::sse(&server.sse_url);
    let transport = SseTransport::new(&config).unwrap();
    wait_connected(&transport).await;

    // 用未知方法(测试服务器对未识别方法回 {"ok": true}),与既有用例一致。
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
    // F2:服务器"连上了但吞 POST 不回 body"→ 请求必须在 request_timeout
    // 内返回带 "timed out" 的错误,绝不永久挂起。
    let server =
        crate::test_support::start_fake_sse_server(crate::test_support::PostMode::HangPost).await;
    let config = MCPConfig::sse(&server.sse_url);
    // 测试缩短超时窗口到 300ms,否则真等 30s。
    let transport = SseTransport::new(&config)
        .unwrap()
        .with_request_timeout(Duration::from_millis(300));
    wait_connected(&transport).await;

    let req = MCPRequest::new(1, "ping", None);
    // 外层 10s 兜底:若超时机制失效,这里 fail 报错而不是把测试挂死。
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

/// P2-6: 进程内传输 + 真实 `MCPServer` 打通 Client↔Server 协议链路。
///
/// 走 `MCPClient::with_transport`(握手) → `list_tools` → `call_tool`,
/// 全程无子进程 / 网络,验证 `tools/call` 经 `BaseTool::run` 被真实执行。
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
