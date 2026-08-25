//! 测试支撑:假 MCP SSE 服务器(仅测试构建编译)。
//!
//! 由 `transport` / `client` 的测试模块共享,覆盖:
//! - SSE 长连接 + `endpoint` 事件发现;
//! - POST 按 JSON-RPC 方法路由(initialize / tools/list / tools/call);
//! - P1-1 首次 POST 失败重试;
//! - P1-8 `tools/list_changed` 推送通知。

use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::Duration;

/// POST 行为模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostMode {
    /// 全部成功;SSE 周期推送 `notifications/tools/list_changed`(P1-8 用)。
    NotifyChanged,
    /// 全部成功;SSE 只发心跳,不发变更通知。
    Quiet,
    /// 第一次 POST 返回 500(P1-1 缓存失效重试用)。
    FailFirstPost,
    /// 收到 POST 后吞掉请求、不回 body(F2 请求超时兜底测试用)。
    HangPost,
    /// 对 POST 回 `202 Accepted`,JSON-RPC 响应经 SSE `event: message` 推送
    /// (F4:202 + SSE 推送型服务器的互操作用)。
    PushResponse,
    /// `tools/call` 返回 `is_error=true` 内容(P1-6 用)。
    ToolError,
    /// `tools/call` 延迟 `Duration` 后响应(per-tool 超时测试用,无 progress)。
    SlowCall(Duration),
    /// 同上,但 SSE 长连接周期推送 `notifications/progress`(进度重置计时器测试用)。
    ProgressSlowCall(Duration),
    /// 流式工具输出(P2-9):首次 `tools/call` 后,SSE 长连接沿
    /// `notifications/tool_partial` 推送 3 个增量片段(seq 0/1/2,末段 final)。
    StreamingCall,
}

/// 假 MCP SSE 服务器句柄。
pub struct FakeSseServer {
    /// SSE 入口 URL(`GET` 该地址建立长连接)。
    pub sse_url: String,
    /// 全部 POST 次数(P1-1 断言"清掉缓存后重试"用)。
    pub post_count: Arc<AtomicUsize>,
    /// `tools/list` 被请求次数(P1-8 断言缓存失效后重新拉取用)。
    pub tools_list_count: Arc<AtomicUsize>,
}

/// 读取一个 HTTP 请求,返回 (请求行, body)。
///
/// 测试用简化实现:先读到 `\r\n\r\n`,再按 `Content-Length` 读全 body。
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

/// 启动假 MCP SSE 服务器。
///
/// - `GET /sse` → 200 + `text/event-stream`,先发 `endpoint` 事件再保持长连接;
/// - `POST /message` → 按 JSON-RPC 方法路由响应。
pub async fn start_fake_sse_server(mode: PostMode) -> FakeSseServer {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let post_count = Arc::new(AtomicUsize::new(0));
    let tools_list_count = Arc::new(AtomicUsize::new(0));
    let call_seen = Arc::new(AtomicBool::new(false));

    // F4:POST 侧把 JSON-RPC 响应投给 SSE 长连接推送(broadcast 多接收者,
    // 每个 GET 连接各订阅一份;响应经 `event: message` 推送)。
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
                    // 保持长连接:周期发心跳注释;NotifyChanged 模式额外推送
                    // tools/list_changed 通知(P1-8);ProgressSlowCall 模式周期推送
                    // progress(P2-4);StreamingCall 模式在首次 tools/call 后
                    // 推送 3 个 tool_partial 增量片段(P2-9)。对端关闭后退出。
                    let heartbeat_ms = if matches!(
                        mode,
                        PostMode::ProgressSlowCall(_) | PostMode::StreamingCall
                    ) {
                        50
                    } else {
                        300
                    };
                    // F4:先订阅推送通道、再发 endpoint 事件——客户端 discover 到
                    // endpoint 后才发 POST,故订阅必然发生在推送之前,推送不丢失。
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
                                // P2-9:首次 tools/call 后开始推流,共 3 段(seq 0/1/2,末段 final)。
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
                                // F4:POST 侧经广播投递的 JSON-RPC 响应 → SSE 推送。
                                // Lagged / 无发送者等 `Err` 忽略,继续心跳即可。
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
                    // F2:吞掉请求不回 body——保持连接但永远不写任何字节,
                    // 让 SseTransport 的 request_timeout 触发超时错误。
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
                        // 慢调用模式(P2-4):先睡 delay,再回显工具名。
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
                        // 普通模式:回显收到的工具名,便于断言调用方剥掉了命名空间前缀,
                        // 走的是 Server 侧原始工具名(P2-2)。
                        "tools/call" => {
                            // P2-9:StreamingCall 模式下,首次 tools/call 触发
                            // SSE 循环开始推流(见 GET 分支的 partial_seq 循环)。
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
                        // F4:先经 SSE 长连接推送 JSON-RPC 响应,再回 202 Accepted。
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
