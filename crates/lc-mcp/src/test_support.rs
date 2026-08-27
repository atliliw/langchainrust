//! Test support: a fake MCP SSE server (compiled only for test builds).
//!
//! Shared by the `transport` / `client` test modules, covering:
//! - SSE long connection + `endpoint` event discovery;
//! - POST routing by JSON-RPC method (initialize / tools/list / tools/call);
//! - P1-1 retry after a failed first POST;
//! - P1-8 `tools/list_changed` push notification.

use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::Duration;

/// POST behavior mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostMode {
    /// All succeed; the SSE connection periodically pushes `notifications/tools/list_changed` (for P1-8).
    NotifyChanged,
    /// All succeed; the SSE connection only sends heartbeats, no change notifications.
    Quiet,
    /// The first POST returns 500 (for P1-1 cache-invalidation retry).
    FailFirstPost,
    /// Swallows the request after receiving POST, never writing a body (for the F2 request-timeout fallback test).
    HangPost,
    /// Replies `202 Accepted` to POST, with the JSON-RPC response pushed over SSE as `event: message`
    /// (for F4: interoperability with 202 + SSE-push servers).
    PushResponse,
    /// `tools/call` returns `is_error=true` content (for P1-6).
    ToolError,
    /// `tools/call` responds after a `Duration` delay (for per-tool timeout tests, no progress).
    SlowCall(Duration),
    /// As above, but the SSE long connection periodically pushes `notifications/progress`
    /// (for progress-reset-timer tests).
    ProgressSlowCall(Duration),
    /// Streaming tool output (P2-9): after the first `tools/call`, the SSE long connection pushes 3
    /// incremental chunks over `notifications/tool_partial` (seq 0/1/2, last chunk final).
    StreamingCall,
}

/// A fake MCP SSE server handle.
pub struct FakeSseServer {
    /// SSE entry URL (`GET` this address to open the long connection).
    pub sse_url: String,
    /// Total POST count (for P1-1 asserting "retry after clearing the cache").
    pub post_count: Arc<AtomicUsize>,
    /// Number of times `tools/list` was requested (for P1-8 asserting a re-fetch after cache invalidation).
    pub tools_list_count: Arc<AtomicUsize>,
}

/// Reads one HTTP request, returning (request line, body).
///
/// Simplified test implementation: first reads up to `\r\n\r\n`, then reads the full body by `Content-Length`.
async fn read_http_request(sock: &mut tokio::net::TcpStream) -> (String, String) {
    use tokio::io::AsyncReadExt;
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = sock.read(&mut tmp).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let header = String::from_utf8_lossy(&buf[..pos]).to_string();
            let content_length = header
                .lines()
                .find_map(|l| {
                    l.to_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            let body_start = pos + 4;
            let mut body: Vec<u8> = buf[body_start..].to_vec();
            while body.len() < content_length {
                let n = sock.read(&mut tmp).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            let first_line = header.lines().next().unwrap_or("").to_string();
            let body = String::from_utf8_lossy(&body[..body.len().min(content_length)]).to_string();
            return (first_line, body);
        }
    }
    (String::new(), String::new())
}

/// Starts a fake MCP SSE server.
///
/// - `GET /sse` → 200 + `text/event-stream`, sending the `endpoint` event first, then holding the long connection;
/// - `POST /message` → routes the response by JSON-RPC method.
pub async fn start_fake_sse_server(mode: PostMode) -> FakeSseServer {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let post_count = Arc::new(AtomicUsize::new(0));
    let tools_list_count = Arc::new(AtomicUsize::new(0));
    let call_seen = Arc::new(AtomicBool::new(false));

    // F4: the POST side hands JSON-RPC responses to the SSE long connections for pushing (broadcast with many
    // receivers, each GET connection subscribes to its own copy; responses are pushed as `event: message`).
    let (push_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let push_tx_clone = push_tx.clone();

    let post_count_clone = post_count.clone();
    let tools_list_count_clone = tools_list_count.clone();
    let call_seen_clone = call_seen.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => break,
            };
            let post_count = post_count_clone.clone();
            let tools_list_count = tools_list_count_clone.clone();
            let call_seen = call_seen_clone.clone();
            let push_tx = push_tx_clone.clone();
            tokio::spawn(async move {
                let (first_line, body) = read_http_request(&mut sock).await;
                if first_line.starts_with("GET ") {
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n",
                        )
                        .await;
                    let endpoint = format!("event: endpoint\ndata: http://{}/message\n\n", addr);
                    let _ = sock.write_all(endpoint.as_bytes()).await;
                    // Hold the long connection: periodically send heartbeat comments; NotifyChanged mode
                    // additionally pushes tools/list_changed notifications (P1-8); ProgressSlowCall mode
                    // periodically pushes progress (P2-4); StreamingCall mode pushes 3 tool_partial incremental
                    // chunks after the first tools/call (P2-9). Exits when the peer closes.
                    let heartbeat_ms = if matches!(
                        mode,
                        PostMode::ProgressSlowCall(_) | PostMode::StreamingCall
                    ) {
                        50
                    } else {
                        300
                    };
                    // F4: subscribe to the push channel before sending the endpoint event — the client only POSTs
                    // after discovering the endpoint, so the subscription necessarily happens before any push and
                    // no push is lost.
                    let mut push_rx = push_tx.subscribe();
                    let mut partial_seq = 0u64;
                    let mut heartbeat = tokio::time::interval(Duration::from_millis(heartbeat_ms));
                    loop {
                        tokio::select! {
                            _ = heartbeat.tick() => {
                                if sock.write_all(b": keep-alive\n\n").await.is_err() {
                                    break;
                                }
                                if mode == PostMode::NotifyChanged
                                    && sock
                                        .write_all(b"event: notifications/tools/list_changed\ndata: {}\n\n")
                                        .await
                                        .is_err()
                                {
                                    break;
                                }
                                if matches!(mode, PostMode::ProgressSlowCall(_))
                                    && sock
                                        .write_all(
                                            b"event: notifications/progress\ndata: {\"progress\":0.5}\n\n",
                                        )
                                        .await
                                        .is_err()
                                {
                                    break;
                                }
                                // P2-9: start streaming after the first tools/call, 3 chunks total
                                // (seq 0/1/2, last chunk final).
                                if mode == PostMode::StreamingCall
                                    && call_seen.load(Ordering::SeqCst)
                                    && partial_seq < 3
                                {
                                    let payload = serde_json::json!({
                                        "tool": "echo",
                                        "seq": partial_seq,
                                        "progress": (partial_seq as f32 + 1.0) / 3.0,
                                        "content": { "type": "text", "text": format!("chunk{}", partial_seq) },
                                        "final": partial_seq == 2,
                                    });
                                    let chunk =
                                        format!("event: notifications/tool_partial\ndata: {}\n\n", payload);
                                    if sock.write_all(chunk.as_bytes()).await.is_err() {
                                        break;
                                    }
                                    partial_seq += 1;
                                }
                            }
                            pushed = push_rx.recv() => {
                                // F4: JSON-RPC responses delivered from the POST side via broadcast → SSE push.
                                // `Err`s like Lagged / no sender are ignored, heartbeat continues.
                                if let Ok(data) = pushed {
                                    let chunk = format!("event: message\ndata: {}\n\n", data);
                                    if sock.write_all(chunk.as_bytes()).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } else if first_line.starts_with("POST ") {
                    let count = post_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if mode == PostMode::FailFirstPost && count == 1 {
                        let _ = sock
                            .write_all(
                                b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n",
                            )
                            .await;
                        return;
                    }
                    // F2: swallow the request without a body — hold the connection but never write a byte, so
                    // SseTransport's request_timeout triggers a timeout error.
                    if mode == PostMode::HangPost {
                        tokio::time::sleep(Duration::from_secs(3600)).await;
                        return;
                    }
                    let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                    let method = parsed.get("method").and_then(Value::as_str).unwrap_or("");
                    let id = parsed.get("id").cloned().unwrap_or(Value::Null);
                    let result: Value = match method {
                        "initialize" => serde_json::json!({
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": { "name": "fake-mcp", "version": "0.0.1" }
                        }),
                        "tools/list" => {
                            tools_list_count.fetch_add(1, Ordering::SeqCst);
                            serde_json::json!({ "tools": [
                                { "name": "echo", "description": "echo tool",
                                  "inputSchema": { "type": "object" } }
                            ]})
                        }
                        "tools/call" if mode == PostMode::ToolError => serde_json::json!({
                            "content": [{ "type": "text", "text": "server exploded" }],
                            "is_error": true
                        }),
                        // Slow-call mode (P2-4): sleep for delay first, then echo the tool name.
                        "tools/call"
                            if matches!(
                                mode,
                                PostMode::SlowCall(_) | PostMode::ProgressSlowCall(_)
                            ) =>
                        {
                            let delay = match mode {
                                PostMode::SlowCall(d) | PostMode::ProgressSlowCall(d) => d,
                                _ => unreachable!(),
                            };
                            tokio::time::sleep(delay).await;
                            let called = parsed
                                .get("params")
                                .and_then(|p| p.get("name"))
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            serde_json::json!({
                                "content": [{ "type": "text", "text": called }],
                                "is_error": false
                            })
                        }
                        // Normal mode: echo the received tool name, so assertions can confirm the caller stripped
                        // the namespace prefix and the call uses the Server-side original tool name (P2-2).
                        "tools/call" => {
                            // P2-9: in StreamingCall mode, the first tools/call triggers the SSE loop to start
                            // streaming (see the partial_seq loop in the GET branch).
                            if mode == PostMode::StreamingCall {
                                call_seen.store(true, Ordering::SeqCst);
                            }
                            let called = parsed
                                .get("params")
                                .and_then(|p| p.get("name"))
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            serde_json::json!({
                                "content": [{ "type": "text", "text": called }],
                                "is_error": false
                            })
                        }
                        _ => serde_json::json!({ "ok": true }),
                    };
                    let resp = serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result });
                    if mode == PostMode::PushResponse {
                        // F4: push the JSON-RPC response over the SSE long connection first, then reply
                        // 202 Accepted.
                        let _ = push_tx.send(resp.to_string());
                        let _ = sock
                            .write_all(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n")
                            .await;
                    } else {
                        let resp_body = resp.to_string();
                        let out = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            resp_body.len(),
                            resp_body
                        );
                        let _ = sock.write_all(out.as_bytes()).await;
                    }
                }
            });
        }
    });

    FakeSseServer {
        sse_url: format!("http://{}/sse", addr),
        post_count,
        tools_list_count,
    }
}
