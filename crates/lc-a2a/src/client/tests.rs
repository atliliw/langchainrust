use super::*;
use serde_json::json;

/// Spawn a minimal HTTP/1.1 server that passes the request head (first
/// line) and body to `handler` and writes the returned string as the body.
async fn spawn_http_server(
    handler: impl Fn(&str, &str) -> String + Send + Sync + 'static,
) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handler = std::sync::Arc::new(handler);
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => break,
            };
            let handler = handler.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = Vec::new();
                let mut tmp = [0u8; 1024];
                loop {
                    let n = match socket.read(&mut tmp).await {
                        Ok(n) => n,
                        Err(_) => return,
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                let content_length: usize = head
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|v| v.trim().parse().ok())
                    })
                    .unwrap_or(0);
                while buf.len() < head_end + content_length {
                    let n = match socket.read(&mut tmp).await {
                        Ok(n) => n,
                        Err(_) => break,
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                let response_body = handler(&head, &body);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                let _ = socket.write_all(resp.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    format!("http://{}", addr)
}

/// Spawn a server that answers one GET request with an SSE event stream.
///
/// Each event is written as `event`/`data` lines followed by a blank line;
/// the connection is closed after all events are written.
async fn spawn_sse_server(events: &'static [&'static str]) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        if let Ok((mut socket, _)) = listener.accept().await {
            let mut tmp = [0u8; 4096];
            let _ = socket.read(&mut tmp).await; // consume the request
            let mut body =
                String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
            for ev in events {
                body.push_str(ev);
                body.push_str("\n\n");
            }
            let _ = socket.write_all(body.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });
    format!("http://{}", addr)
}

/// Read a request head up to and including the blank line, returning its text.
///
/// Consuming the whole head matters: closing a socket with unread request bytes
/// makes the OS send RST instead of FIN, and an RST makes the client discard
/// already-buffered response bytes — which would lose the events the test is
/// about to assert on.
async fn read_request_head<S>(socket: &mut S) -> Option<String>
where
    S: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        match socket.read(&mut tmp).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            return Some(String::from_utf8_lossy(&buf[..pos + 4]).to_string());
        }
    }
}

/// Per-connection behaviour for [`spawn_sse_scripted_server`] (0.27.0 T3 #95).
#[derive(Clone, Copy)]
enum SseConn {
    /// Deliver these events, then cut the connection mid-body: the response is
    /// chunked and closed without the terminating `0\r\n\r\n`, so the client
    /// sees a transport error rather than a clean close.
    Break(&'static [&'static str]),
    /// Deliver these events, then go silent while holding the socket open (the
    /// client's idle guard is what must notice).
    Silent(&'static [&'static str]),
    /// Deliver these events, then close cleanly.
    Clean(&'static [&'static str]),
}

/// Spawn a server that answers each accepted connection according to `script`,
/// returning its base URL and the request head of every connection (so tests can
/// assert on replayed auth/trace headers).
async fn spawn_sse_scripted_server(
    script: &'static [SseConn],
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let heads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = heads.clone();
    tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        for step in script {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            // `step` is a `&SseConn` (Copy); deref it so match binding gets the
            // value fields directly rather than double references to them.
            let step = *step;
            // Each connection is served on its own task: a `Silent` connection
            // stays open while the client reconnects, so the accept loop must
            // not be blocked behind it.
            let captured = captured.clone();
            tokio::spawn(async move {
                let Some(head) = read_request_head(&mut socket).await else {
                    return;
                };
                captured
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(head);

                let chunked = |events: &'static [&'static str]| {
                    let mut body = String::from(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                         Transfer-Encoding: chunked\r\n\r\n",
                    );
                    for ev in events {
                        let payload = format!("{ev}\n\n");
                        body.push_str(&format!("{:x}\r\n{}\r\n", payload.len(), payload));
                    }
                    body
                };
                match step {
                    SseConn::Break(events) => {
                        let _ = socket.write_all(chunked(events).as_bytes()).await;
                        let _ = socket.flush().await;
                        drop(socket); // abrupt: no `0\r\n\r\n` terminator
                    }
                    SseConn::Silent(events) => {
                        let _ = socket.write_all(chunked(events).as_bytes()).await;
                        let _ = socket.flush().await;
                        // Hold the connection open and stay silent: the client
                        // must give up on its own idle budget.
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        let _ = socket.shutdown().await;
                    }
                    SseConn::Clean(events) => {
                        let mut body = String::from(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n",
                        );
                        for ev in events {
                            body.push_str(ev);
                            body.push_str("\n\n");
                        }
                        let _ = socket.write_all(body.as_bytes()).await;
                        let _ = socket.shutdown().await;
                    }
                }
            });
        }
    });
    (format!("http://{}", addr), heads)
}

/// Spawn a server that answers JSON-RPC `POST /` requests and serves SSE
/// events on `GET /sse` (used by `send_task_streaming` tests).
async fn spawn_sse_rpc_server(events: &'static [&'static str]) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => break,
            };
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 1024];
                loop {
                    let n = match socket.read(&mut tmp).await {
                        Ok(n) => n,
                        Err(_) => return,
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                let head = String::from_utf8_lossy(&buf[..head_end]).to_string();

                if head.starts_with("GET") {
                    let mut body =
                        String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
                    for ev in events {
                        body.push_str(ev);
                        body.push_str("\n\n");
                    }
                    let _ = socket.write_all(body.as_bytes()).await;
                } else {
                    let content_length: usize = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse().ok())
                        })
                        .unwrap_or(0);
                    while buf.len() < head_end + content_length {
                        let n = match socket.read(&mut tmp).await {
                            Ok(n) => n,
                            Err(_) => break,
                        };
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                    }
                    let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                    let req: A2ARequest = serde_json::from_str(&body).unwrap();
                    let json = serde_json::to_string(&A2AResponse::ok(
                        req.id,
                        json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
                    ))
                    .unwrap();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        json.len(),
                        json
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                }
                let _ = socket.shutdown().await;
            });
        }
    });
    format!("http://{}", addr)
}

/// Spawn a server that delays its response by `delay` (for timeout tests).
async fn spawn_slow_server(delay: Duration) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        if let Ok((mut socket, _)) = listener.accept().await {
            let mut tmp = [0u8; 1024];
            let _ = socket.read(&mut tmp).await; // read request headers
            tokio::time::sleep(delay).await;
            let resp = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = socket.write_all(resp.as_bytes()).await;
        }
    });
    format!("http://{}", addr)
}

#[test]
fn a2a_error_from_reqwest_timeout() {
    // We can't easily create a reqwest::Error, so test the variant exists.
    let err = A2AError::Timeout("connection timed out".to_string());
    assert!(err.to_string().contains("Timeout"));
}

#[test]
fn a2a_error_from_reqwest_http() {
    let err = A2AError::Http("404 not found".to_string());
    assert!(err.to_string().contains("HTTP error"));
}

#[test]
fn a2a_error_from_error_data() {
    let data = A2AErrorData::method_not_found();
    let err: A2AError = data.into();
    match err {
        A2AError::Api { code, message } => {
            assert_eq!(code, -32601);
            assert!(message.contains("Method not found"));
        }
        _ => panic!("Expected Api variant"),
    }
}

#[test]
fn a2a_error_parse() {
    let err = A2AError::Parse("bad json".to_string());
    assert!(err.to_string().contains("Parse error"));
}

#[test]
fn client_new_trims_trailing_slash() {
    let client = A2AClient::new("http://localhost:8080/".to_string()).unwrap();
    assert_eq!(client.base_url, "http://localhost:8080");
}

#[test]
fn client_alloc_id_increments() {
    let client = A2AClient::new("http://localhost:8080".to_string()).unwrap();
    assert_eq!(client.alloc_id(), 1);
    assert_eq!(client.alloc_id(), 2);
    assert_eq!(client.alloc_id(), 3);
}

#[test]
fn client_with_custom_http() {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap();
    let client = A2AClient::with_http_client("http://localhost:8080".to_string(), http);
    assert_eq!(client.base_url, "http://localhost:8080");
}

#[test]
fn builder_rejects_insecure_url_when_enforcing_https() {
    let result = A2AClient::builder("http://localhost:8080")
        .enforce_https(true)
        .build();
    match result {
        Err(A2AError::Http(msg)) => assert!(msg.contains("HTTPS is required")),
        _ => panic!("expected an HTTPS enforcement error"),
    }
}

#[tokio::test]
async fn get_agent_card_uses_agent_card_json_path() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen_clone = seen.clone();
    let base = spawn_http_server(move |head, _body| {
        seen_clone
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(head.to_string());
        serde_json::to_string(&AgentCard::new("agent", "desc", "http://localhost")).unwrap()
    })
    .await;
    let client = A2AClient::new(base).unwrap();
    let card = client.get_agent_card().await.unwrap();
    assert_eq!(card.name, "agent");

    let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        lines
            .iter()
            .any(|l| l.contains("/.well-known/agent-card.json")),
        "expected request to use /.well-known/agent-card.json, got: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("/.well-known/agent.json")),
        "must not use the legacy path: {lines:?}"
    );
}

#[tokio::test]
async fn send_task_and_wait_polls_until_completed() {
    let base = spawn_http_server(|_head, body| {
        let req: A2ARequest = serde_json::from_str(body).unwrap();
        match req.method.as_str() {
            "tasks/send" => serde_json::to_string(&A2AResponse::ok(
                req.id,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap(),
            "tasks/get" => serde_json::to_string(&A2AResponse::ok(
                req.id,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "completed"}, "result": {"output": "done"}}),
            ))
            .unwrap(),
            _ => serde_json::to_string(&A2AResponse::error(req.id, -32601, "Method not found"))
                .unwrap(),
        }
    })
    .await;
    let client = A2AClient::new(base).unwrap();
    let result = client
        .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(10))
        .await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().output, "done");
}

#[tokio::test]
async fn send_task_and_wait_returns_error_on_failed() {
    let base = spawn_http_server(|_head, body| {
        let req: A2ARequest = serde_json::from_str(body).unwrap();
        match req.method.as_str() {
            "tasks/send" => serde_json::to_string(&A2AResponse::ok(
                req.id,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap(),
            "tasks/get" => serde_json::to_string(&A2AResponse::ok(
                req.id,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "failed"}, "error": "boom"}),
            ))
            .unwrap(),
            _ => serde_json::to_string(&A2AResponse::error(req.id, -32601, "Method not found"))
                .unwrap(),
        }
    })
    .await;
    let client = A2AClient::new(base).unwrap();
    let result = client
        .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(10))
        .await;
    match result {
        Err(A2AError::Api { code, message }) => {
            assert_eq!(code, -32000);
            assert!(message.contains("boom"));
        }
        other => panic!("expected Api error, got: {:?}", other),
    }
}

#[tokio::test]
async fn send_task_and_wait_times_out() {
    let base = spawn_http_server(|_head, _body| {
        // Always report `submitted`, so the poll never terminates.
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
        ))
        .unwrap()
    })
    .await;
    let client = A2AClient::new(base).unwrap();
    let result = client
        .send_task_and_wait(A2AMessage::user("hi"), Duration::from_millis(1500))
        .await;
    assert!(matches!(result, Err(A2AError::Timeout(_))));
}

#[tokio::test]
async fn client_sends_bearer_token() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen_clone = seen.clone();
    let base = spawn_http_server(move |head, _body| {
        seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(head.to_string());
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
        ))
        .unwrap()
    })
    .await;
    let client = A2AClient::builder(base)
        .bearer_token("s3cret")
        .build()
        .unwrap();
    let _ = client.send_task(A2AMessage::user("hi")).await;

    let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        lines.iter().any(|l| l
            .to_ascii_lowercase()
            .contains("authorization: bearer s3cret")),
        "expected Authorization: Bearer s3cret, got: {lines:?}"
    );
}

#[tokio::test]
async fn client_enforces_per_request_timeout() {
    let base = spawn_slow_server(Duration::from_secs(5)).await;
    let client = A2AClient::builder(base)
        .timeout(Duration::from_millis(300))
        .connect_timeout(Duration::from_millis(300))
        .build()
        .unwrap();
    let result = client.send_task(A2AMessage::user("hi")).await;
    assert!(matches!(result, Err(A2AError::Timeout(_))));
}

#[tokio::test]
async fn get_agent_card_invalid_url() {
    let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
    let result = client.get_agent_card().await;
    assert!(result.is_err());
    match result.unwrap_err() {
        A2AError::Http(_) | A2AError::Timeout(_) => {} // expected
        other => panic!("Expected Http or Timeout error, got: {:?}", other),
    }
}

#[tokio::test]
async fn send_task_invalid_url() {
    let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
    let result = client.send_task(A2AMessage::user("hello")).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn get_task_invalid_url() {
    let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
    let result = client.get_task("task-123").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn cancel_task_invalid_url() {
    let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
    let result = client.cancel_task("task-123").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn post_request_invalid_url() {
    let client = A2AClient::new("http://localhost:19999".to_string()).unwrap();
    let req = A2ARequest::new(1, "test", None);
    let result = client.post_request(req).await;
    assert!(result.is_err());
}

// ---- P1-3: card signature ----

#[test]
fn sign_and_verify_card_signature_roundtrip() {
    let mut card = AgentCard::new("agent", "desc", "http://localhost");
    sign_agent_card(&mut card, b"secret").unwrap();
    assert!(card.signature.is_some());
    assert!(verify_card_signature(&card, b"secret").is_ok());
}

#[test]
fn verify_card_signature_rejects_tampered_card() {
    let mut card = AgentCard::new("agent", "desc", "http://localhost");
    sign_agent_card(&mut card, b"secret").unwrap();
    card.name = "evil".to_string();
    assert!(matches!(
        verify_card_signature(&card, b"secret"),
        Err(A2AError::Signature(_))
    ));
}

#[test]
fn verify_card_signature_rejects_wrong_secret() {
    let mut card = AgentCard::new("agent", "desc", "http://localhost");
    sign_agent_card(&mut card, b"secret").unwrap();
    assert!(matches!(
        verify_card_signature(&card, b"other"),
        Err(A2AError::Signature(_))
    ));
}

#[test]
fn verify_card_signature_unsigned_is_ok() {
    let card = AgentCard::new("agent", "desc", "http://localhost");
    assert!(verify_card_signature(&card, b"secret").is_ok());
}

#[tokio::test]
async fn get_agent_card_verifies_signed_card() {
    let mut card = AgentCard::new("agent", "desc", "http://localhost");
    sign_agent_card(&mut card, b"secret").unwrap();
    let card_json = serde_json::to_string(&card).unwrap();
    let base = spawn_http_server(move |_head, _body| card_json.clone()).await;

    // Correct secret -> verified.
    let client = A2AClient::builder(base.clone())
        .card_verification_secret(b"secret")
        .build()
        .unwrap();
    let got = client.get_agent_card().await.unwrap();
    assert_eq!(got.name, "agent");

    // Wrong secret -> hard error.
    let client = A2AClient::builder(base)
        .card_verification_secret(b"wrong")
        .build()
        .unwrap();
    assert!(matches!(
        client.get_agent_card().await,
        Err(A2AError::Signature(_))
    ));
}

#[tokio::test]
async fn get_agent_card_requires_signature_without_secret() {
    let mut card = AgentCard::new("agent", "desc", "http://localhost");
    sign_agent_card(&mut card, b"secret").unwrap();
    let card_json = serde_json::to_string(&card).unwrap();
    let base = spawn_http_server(move |_head, _body| card_json.clone()).await;

    let client = A2AClient::builder(base)
        .require_card_signature(true)
        .build()
        .unwrap();
    assert!(matches!(
        client.get_agent_card().await,
        Err(A2AError::Signature(_))
    ));
}

#[tokio::test]
async fn get_agent_card_unsigned_passes_with_require_signature() {
    let base = spawn_http_server(|_head, _body| {
        serde_json::to_string(&AgentCard::new("agent", "desc", "http://localhost")).unwrap()
    })
    .await;
    let client = A2AClient::builder(base)
        .require_card_signature(true)
        .build()
        .unwrap();
    assert!(client.get_agent_card().await.is_ok());
}

// ---- P1-5: trace propagation ----

#[tokio::test]
async fn client_attaches_trace_id_to_requests() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen_clone = seen.clone();
    let base = spawn_http_server(move |_head, body| {
        seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(body.to_string());
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
        ))
        .unwrap()
    })
    .await;
    let client = A2AClient::builder(base)
        .trace_id("trace-123")
        .build()
        .unwrap();
    let _ = client.send_task(A2AMessage::user("hi")).await;

    let bodies = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert!(!bodies.is_empty());
    let parsed: A2ARequest = serde_json::from_str(&bodies[0]).unwrap();
    assert_eq!(parsed.trace_id(), Some("trace-123"));
}

// ---- P2-8: W3C traceparent header ----

#[tokio::test]
async fn client_sends_traceparent_header() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen_clone = seen.clone();
    let base = spawn_http_server(move |head, _body| {
        seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(head.to_string());
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
        ))
        .unwrap()
    })
    .await;

    let trace = TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").sampled();
    let client = A2AClient::builder(base)
        .with_traceparent(trace)
        .build()
        .unwrap();
    let _ = client.send_task(A2AMessage::user("hi")).await;

    let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert!(!lines.is_empty());
    let head = &lines[0];
    assert!(
        head.to_ascii_lowercase()
            .contains("traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        "expected W3C traceparent header, got: {head}"
    );
}

#[tokio::test]
async fn client_without_trace_context_sends_no_traceparent() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen_clone = seen.clone();
    let base = spawn_http_server(move |head, _body| {
        seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(head.to_string());
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
        ))
        .unwrap()
    })
    .await;

    let client = A2AClient::new(base).unwrap();
    let _ = client.send_task(A2AMessage::user("hi")).await;

    let lines = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        !lines[0].to_ascii_lowercase().contains("traceparent:"),
        "no traceparent expected, got: {:?}",
        lines[0]
    );
}

// ---- P1-6: idempotent send ----

#[tokio::test]
async fn client_sends_message_id() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen_clone = seen.clone();
    let base = spawn_http_server(move |_head, body| {
        seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(body.to_string());
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
        ))
        .unwrap()
    })
    .await;
    let client = A2AClient::new(base).unwrap();
    let _ = client
        .send_task_with_message_id(A2AMessage::user("hi"), "idem-1")
        .await;

    let bodies = seen.lock().unwrap_or_else(|e| e.into_inner());
    let parsed: A2ARequest = serde_json::from_str(&bodies[0]).unwrap();
    assert_eq!(parsed.message_id(), Some("idem-1"));
}

// ---- P2-3: input-required handling ----

#[tokio::test]
async fn send_task_and_wait_surfaces_input_required() {
    let base = spawn_http_server(|_head, body| {
        let req: A2ARequest = serde_json::from_str(body).unwrap();
        match req.method.as_str() {
            "tasks/send" => serde_json::to_string(&A2AResponse::ok(
                req.id,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
            ))
            .unwrap(),
            "tasks/get" => serde_json::to_string(&A2AResponse::ok(
                req.id,
                json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "input-required"}, "error": "please provide your name"}),
            ))
            .unwrap(),
            _ => serde_json::to_string(&A2AResponse::error(req.id, -32601, "Method not found"))
                .unwrap(),
        }
    })
    .await;
    let client = A2AClient::new(base).unwrap();
    let result = client
        .send_task_and_wait(A2AMessage::user("hi"), Duration::from_secs(10))
        .await;
    match result {
        Err(A2AError::InputRequired { task_id, prompt }) => {
            assert_eq!(task_id, "t1");
            assert!(prompt.contains("please provide your name"));
        }
        other => panic!("expected InputRequired error, got: {:?}", other),
    }
}

#[tokio::test]
async fn resume_task_carries_task_id_and_message() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen_clone = seen.clone();
    let base = spawn_http_server(move |_head, body| {
        seen_clone.lock().unwrap_or_else(|e| e.into_inner()).push(body.to_string());
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "working"}}),
        ))
        .unwrap()
    })
    .await;
    let client = A2AClient::new(base).unwrap();
    let _ = client
        .resume_task("t1", A2AMessage::user("my name is alice"))
        .await;

    let bodies = seen.lock().unwrap_or_else(|e| e.into_inner());
    let parsed: A2ARequest = serde_json::from_str(&bodies[0]).unwrap();
    assert_eq!(parsed.method, "tasks/send");
    let params = parsed.params.as_ref().unwrap();
    assert_eq!(params["taskId"], "t1");
    assert_eq!(params["message"]["content"], "my name is alice");
}

// ---- P2-1: SSE streaming ----

#[tokio::test]
async fn connect_sse_parses_task_notifications() {
    let base = spawn_sse_server(&[
        "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}",
        "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"completed\"}",
    ])
    .await;
    let client = A2AClient::new("http://localhost:1".to_string()).unwrap(); // URL unused by connect_sse
    let mut stream = client.connect_sse(&base).await.unwrap();

    let first = stream.next().await.expect("first event").unwrap();
    assert_eq!(first.id(), "t1");
    assert_eq!(first.status_value(), Some(TaskStatus::Working));

    let second = stream.next().await.expect("second event").unwrap();
    assert_eq!(second.status_value(), Some(TaskStatus::Completed));

    assert!(stream.next().await.is_none(), "stream should end");
}

#[tokio::test]
async fn send_task_streaming_sends_then_streams() {
    let base = spawn_sse_rpc_server(&[
        "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}",
        "event: artifact-update\ndata: {\"kind\":\"artifact-update\",\"id\":\"t1\",\"artifact\":{\"artifactId\":\"a1\",\"parts\":[{\"type\":\"text\",\"text\":\"hi\"}]}}",
    ])
    .await;
    let client = A2AClient::new(base.clone()).unwrap();
    let mut stream = client
        .send_task_streaming(&format!("{}/sse", base), A2AMessage::user("hi"))
        .await
        .unwrap();

    let first = stream.next().await.expect("first event").unwrap();
    assert_eq!(first.id(), "t1");
    assert_eq!(first.status_value(), Some(TaskStatus::Working));

    let second = stream.next().await.expect("second event").unwrap();
    assert!(
        second.status_value().is_none(),
        "artifact events carry no status"
    );
}

// ---- 0.27.0 T3 #95: opt-in SSE self-heal ----

const STATUS_WORKING: &str =
    "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}";
const STATUS_COMPLETED: &str =
    "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"completed\"}";

#[tokio::test]
async fn sse_reconnects_after_transport_break_when_enabled() {
    let (base, heads) = spawn_sse_scripted_server(&[
        SseConn::Break(&[STATUS_WORKING]),
        SseConn::Clean(&[STATUS_COMPLETED]),
    ])
    .await;
    let client = A2AClient::builder("http://localhost:1".to_string())
        .sse_reconnect_max_attempts(3)
        .build()
        .unwrap();
    let mut stream = client.connect_sse(&base).await.unwrap();

    let first = stream.next().await.expect("first event").unwrap();
    assert_eq!(first.status_value(), Some(TaskStatus::Working));

    // The break is healed on a fresh connection rather than surfaced.
    let second = stream.next().await.expect("reconnected event").unwrap();
    assert_eq!(second.status_value(), Some(TaskStatus::Completed));

    assert!(stream.next().await.is_none(), "clean close is terminal");
    assert_eq!(
        heads.lock().unwrap().len(),
        2,
        "one reconnect after the break"
    );
}

#[tokio::test]
async fn sse_transport_break_surfaces_when_self_heal_disabled() {
    let (base, heads) = spawn_sse_scripted_server(&[
        SseConn::Break(&[STATUS_WORKING]),
        SseConn::Clean(&[STATUS_COMPLETED]),
    ])
    .await;
    // Default: no self-heal, so the pre-0.27 behavior is preserved.
    let client = A2AClient::new("http://localhost:1".to_string()).unwrap();
    let mut stream = client.connect_sse(&base).await.unwrap();

    let first = stream.next().await.expect("first event").unwrap();
    assert_eq!(first.status_value(), Some(TaskStatus::Working));

    let second = stream.next().await.expect("break surfaces as an error");
    assert!(second.is_err(), "default path must not reconnect");
    assert_eq!(heads.lock().unwrap().len(), 1, "no reconnect when disabled");
}

#[tokio::test]
async fn sse_reconnects_after_idle_timeout() {
    let (base, heads) = spawn_sse_scripted_server(&[
        SseConn::Silent(&[STATUS_WORKING]),
        SseConn::Clean(&[STATUS_COMPLETED]),
    ])
    .await;
    let client = A2AClient::builder("http://localhost:1".to_string())
        // A short idle budget so the silent connection trips the guard quickly.
        .sse_idle_timeout(std::time::Duration::from_millis(150))
        .sse_reconnect_max_attempts(3)
        .build()
        .unwrap();
    let mut stream = client.connect_sse(&base).await.unwrap();

    let first = stream.next().await.expect("first event").unwrap();
    assert_eq!(first.status_value(), Some(TaskStatus::Working));

    // The server goes silent; the idle guard treats it as a transport break
    // and self-heals onto a fresh connection that completes the task.
    let second = stream
        .next()
        .await
        .expect("idle timeout is healed")
        .unwrap();
    assert_eq!(second.status_value(), Some(TaskStatus::Completed));

    assert!(stream.next().await.is_none(), "clean close is terminal");
    assert_eq!(
        heads.lock().unwrap().len(),
        2,
        "one reconnect after the idle timeout"
    );
}

#[tokio::test]
async fn sse_reconnect_budget_exhaustion_surfaces_original_error() {
    // Both connections break; a budget of 1 allows exactly one reconnect, after
    // which the original transport error is surfaced.
    let (base, heads) =
        spawn_sse_scripted_server(&[SseConn::Break(&[STATUS_WORKING]), SseConn::Break(&[])]).await;
    let client = A2AClient::builder("http://localhost:1".to_string())
        .sse_reconnect_max_attempts(1)
        .build()
        .unwrap();
    let mut stream = client.connect_sse(&base).await.unwrap();

    let first = stream.next().await.expect("first event").unwrap();
    assert_eq!(first.status_value(), Some(TaskStatus::Working));

    let second = stream
        .next()
        .await
        .expect("budget exhaustion surfaces an error");
    assert!(
        second.is_err(),
        "exhausted retry budget must surface the transport error"
    );
    assert_eq!(
        heads.lock().unwrap().len(),
        2,
        "original connection + one reconnect"
    );
}

#[tokio::test]
async fn sse_reconnect_replays_bearer_and_traceparent() {
    use crate::protocol::TraceContext;
    let (base, heads) = spawn_sse_scripted_server(&[
        SseConn::Break(&[STATUS_WORKING]),
        SseConn::Clean(&[STATUS_COMPLETED]),
    ])
    .await;
    let client = A2AClient::builder("http://localhost:1".to_string())
        .bearer_token("tok-1".to_string())
        .with_traceparent(TraceContext::new(
            "11111111111111111111111111111111",
            "2222222222222222",
        ))
        .sse_reconnect_max_attempts(3)
        .build()
        .unwrap();
    let mut stream = client.connect_sse(&base).await.unwrap();

    let _ = stream.next().await.expect("first event").unwrap();
    let _ = stream.next().await.expect("reconnected event").unwrap();

    let heads = heads.lock().unwrap();
    assert_eq!(heads.len(), 2, "original + reconnect");
    // Header names could arrive in any casing on the wire; compare lowercase.
    let reconnected = heads[1].to_lowercase();
    assert!(
        reconnected.contains("authorization: bearer tok-1"),
        "bearer token must be replayed on reconnect, got: {reconnected}"
    );
    assert!(
        reconnected
            .contains("traceparent: 00-11111111111111111111111111111111-2222222222222222-00"),
        "traceparent must be replayed on reconnect, got: {reconnected}"
    );
}

/// T3 #100 event replay: when a frame carried an SSE `id:`, the reconnect GET
/// sends `Last-Event-ID` set to the newest one, so a server with a replay
/// buffer can resume the delivery gap instead of dropping it.
#[tokio::test]
async fn sse_reconnect_sends_last_event_id_for_replay() {
    // Frame that carries an SSE `id:` (distinct from the JSON task id) so the
    // parser has an event id to replay.
    const WORKING_WITH_ID: &str = "id: 17\nevent: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}";
    const COMPLETED_NO_ID: &str =
        "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"completed\"}";
    let (base, heads) = spawn_sse_scripted_server(&[
        SseConn::Break(&[WORKING_WITH_ID]),
        SseConn::Clean(&[COMPLETED_NO_ID]),
    ])
    .await;

    let client = A2AClient::builder("http://localhost:1".to_string())
        .sse_reconnect_max_attempts(3)
        .build()
        .unwrap();
    let mut stream = client.connect_sse(&base).await.unwrap();

    let _ = stream.next().await.expect("first event").unwrap();
    let _ = stream.next().await.expect("reconnected event").unwrap();

    let heads = heads.lock().unwrap();
    assert_eq!(heads.len(), 2, "original + reconnect");
    let reconnected = heads[1].to_lowercase();
    assert!(
        reconnected.contains("last-event-id: 17"),
        "reconnect must ask the server to replay from the last event id, got: {reconnected}"
    );
}

// ---- 0.25.0 C3: trust registry + sandbox wired into client paths ----

#[tokio::test]
async fn get_agent_card_accepted_when_registered_and_signature_verifies() {
    use crate::security::{TrustConfig, TrustRegistry, TrustRole, TrustedAgent};

    let card_url = "http://registry-agent.example.com";
    let mut card = AgentCard::new("agent", "desc", card_url);
    sign_agent_card(&mut card, b"reg-secret").unwrap();
    let card_json = serde_json::to_string(&card).unwrap();
    let base = spawn_http_server(move |_head, _body| card_json.clone()).await;

    let registry = std::sync::Arc::new(TrustRegistry::new(TrustConfig::default()).with_agent(
        TrustedAgent::new(card_url, "agent", TrustRole::Leaf).with_key(b"reg-secret".to_vec()),
    ));
    let client = A2AClient::builder(base)
        .trust_registry(registry)
        .build()
        .unwrap();
    let got = client.get_agent_card().await.unwrap();
    assert_eq!(got.url, card_url);
}

#[tokio::test]
async fn get_agent_card_rejected_when_registry_does_not_know_agent() {
    use crate::security::{TrustConfig, TrustRegistry};

    let card_url = "http://stranger.example.com";
    let mut card = AgentCard::new("agent", "desc", card_url);
    sign_agent_card(&mut card, b"reg-secret").unwrap();
    let card_json = serde_json::to_string(&card).unwrap();
    let base = spawn_http_server(move |_head, _body| card_json.clone()).await;

    // Empty registry: the card is signed but no identity is attested.
    let registry = std::sync::Arc::new(TrustRegistry::new(TrustConfig::default()));
    let client = A2AClient::builder(base)
        .trust_registry(registry)
        .build()
        .unwrap();
    assert!(matches!(
        client.get_agent_card().await,
        Err(A2AError::Signature(_))
    ));
}

#[tokio::test]
async fn get_agent_card_rejected_when_signature_tampered() {
    use crate::security::{TrustConfig, TrustRegistry, TrustRole, TrustedAgent};

    let card_url = "http://registry-agent.example.com";
    let mut card = AgentCard::new("agent", "desc", card_url);
    sign_agent_card(&mut card, b"reg-secret").unwrap();
    // Tamper after signing; serialize the tampered card directly.
    card.name = "impostor".to_string();
    let card_json = serde_json::to_string(&card).unwrap();
    let base = spawn_http_server(move |_head, _body| card_json.clone()).await;

    let registry = std::sync::Arc::new(TrustRegistry::new(TrustConfig::default()).with_agent(
        TrustedAgent::new(card_url, "agent", TrustRole::Leaf).with_key(b"reg-secret".to_vec()),
    ));
    let client = A2AClient::builder(base)
        .trust_registry(registry)
        .build()
        .unwrap();
    assert!(matches!(
        client.get_agent_card().await,
        Err(A2AError::Signature(_))
    ));
}

#[test]
fn builder_sandbox_allows_listed_egress_host() {
    use crate::security::SandboxConfig;
    let sandbox = std::sync::Arc::new(SandboxConfig::new().allow_domain("localhost"));
    assert!(A2AClient::builder("http://localhost:8080")
        .sandbox(sandbox)
        .build()
        .is_ok());
}

#[test]
fn builder_sandbox_denies_unlisted_egress_host() {
    use crate::security::SandboxConfig;
    let sandbox = std::sync::Arc::new(SandboxConfig::new().allow_domain("example.com"));
    let result = A2AClient::builder("http://localhost:8080")
        .sandbox(sandbox)
        .build();
    assert!(
        matches!(result, Err(e) if e.to_string().contains("sandbox denied egress")),
        "egress to an unlisted host must be denied at build time"
    );
}

#[tokio::test]
async fn get_agent_card_falls_back_to_agent_json() {
    // B7 wire-compat: when the standard agent-card.json path is absent, the
    // client must retry the `/.well-known/agent.json` discovery alias.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let card_json =
        serde_json::to_string(&AgentCard::new("fallback", "desc", "http://localhost")).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => return,
            };
            let next_card_json = card_json.clone();
            tokio::spawn(async move {
                let card_json = next_card_json;
                let mut buf = [0u8; 4096];
                let mut req = Vec::new();
                let mut head_end = None;
                while head_end.is_none() {
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    req.extend_from_slice(&buf[..n]);
                    head_end = req.windows(4).position(|w| w == b"\r\n\r\n");
                }
                let Some(head_end) = head_end else { return };
                let path = String::from_utf8_lossy(&req[..head_end])
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .to_string();
                let (status, body) = if path.contains("agent-card.json") {
                    (404, r#"{"error":"not found"}"#.to_string())
                } else {
                    (200, card_json)
                };
                let resp = format!(
                    "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    if status == 404 { "Not Found" } else { "OK" },
                    body.len(),
                    body
                );
                let _ = socket.write_all(resp.as_bytes()).await;
                let _ = socket.flush().await;
            });
        }
    });

    let client = A2AClient::new(format!("http://{addr}")).unwrap();
    let card = client.get_agent_card().await.unwrap();
    assert_eq!(
        card.name, "fallback",
        "must recover the card from agent.json"
    );
}

// ---- P2-9: tasks/sendSubscribe (subscribe) ----

#[tokio::test]
async fn subscribe_returns_task_and_streams_parts_artifact() {
    // The RPC endpoint answers any POST with a submitted task ack; the SSE
    // endpoint streams a status increment then a parts-based artifact update.
    let base = spawn_sse_rpc_server(&[
        "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}",
        "event: artifact-update\ndata: {\"kind\":\"artifact-update\",\"id\":\"t1\",\"artifact\":{\"artifactId\":\"a1\",\"parts\":[{\"type\":\"text\",\"text\":\"hi\"}]}}",
    ])
    .await;
    let client = A2AClient::new(base.clone()).unwrap();

    let mut sub = client
        .subscribe(&format!("{}/sse", base), A2AMessage::user("hello"))
        .await
        .expect("subscription established");
    // The subscription confirmation carries the acknowledged task.
    assert_eq!(sub.task().id, "t1");
    assert_eq!(sub.task().status, TaskStatus::Submitted);

    // Incremental push events flow over the stream without polling.
    let first = sub.next().await.expect("first event").unwrap();
    assert_eq!(first.id(), "t1");
    assert_eq!(first.status_value(), Some(TaskStatus::Working));

    let second = sub.next().await.expect("second event").unwrap();
    assert_eq!(
        second.status_value(),
        None,
        "artifact updates carry no status"
    );
    if let TaskPushNotification::ArtifactUpdate { artifact, .. } = second {
        assert_eq!(artifact.parts().len(), 1);
        assert_eq!(artifact.text_content().as_deref(), Some("hi"));
    } else {
        panic!("expected an artifact update");
    }
}

// ---- T3: fleet routing wired into the send path ----

/// A stub agent that records every request body it receives and acknowledges
/// each `tasks/send` with a `submitted` task.
async fn spawn_fleet_agent(hits: std::sync::Arc<std::sync::Mutex<Vec<String>>>) -> String {
    use crate::protocol::AgentSkill;
    let base = spawn_http_server(move |_head, body| {
        hits.lock().unwrap_or_else(|e| e.into_inner()).push(body.to_string());
        serde_json::to_string(&A2AResponse::ok(
            0,
            json!({"task": {"id": "t1", "message": {"role": "user", "content": "hi"}, "status": "submitted"}}),
        ))
        .unwrap()
    })
    .await;
    let _ = AgentSkill::new("unused", "unused", "unused");
    base
}

#[tokio::test]
async fn fleet_routes_send_task_to_the_relevant_agent() {
    use crate::protocol::AgentSkill;

    let hits_retriever = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let hits_summarizer = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let retriever_url = spawn_fleet_agent(hits_retriever.clone()).await;
    let summarizer_url = spawn_fleet_agent(hits_summarizer.clone()).await;

    let mut fleet = AgentFleet::new();
    fleet.register(AgentCard::new("retriever", "doc search", retriever_url.clone()).with_skill(
        AgentSkill::new(
            "retrieve",
            "retrieve",
            "retrieve relevant documents from the corpus",
        ),
    ));
    fleet.register(AgentCard::new("summarizer", "text", summarizer_url.clone()).with_skill(
        AgentSkill::new(
            "summarize",
            "summarize",
            "condense long documents into a short summary",
        ),
    ));

    let client = A2AClient::builder("http://localhost:1")
        .fleet(std::sync::Arc::new(fleet))
        .build()
        .unwrap();

    // The routing policy picks the retriever for a retrieval query.
    let routed = client
        .send_task_routed("retrieve documents", A2AMessage::user("hi"))
        .await
        .unwrap();
    assert_eq!(routed.agent_url, retriever_url);
    assert_eq!(routed.task.id, "t1");
    assert_eq!(hits_retriever.lock().unwrap().len(), 1);
    assert_eq!(hits_summarizer.lock().unwrap().len(), 0);

    // A summarization query goes to the other agent.
    let routed = client
        .send_task_routed("condense documents summary", A2AMessage::user("hi"))
        .await
        .unwrap();
    assert_eq!(routed.agent_url, summarizer_url);
    assert_eq!(hits_retriever.lock().unwrap().len(), 1);
    assert_eq!(hits_summarizer.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn fleet_skips_a_tripped_agent_through_the_send_path() {
    use crate::protocol::AgentSkill;

    let hits_primary = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let hits_secondary = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let primary = spawn_fleet_agent(hits_primary.clone()).await;
    let secondary = spawn_fleet_agent(hits_secondary.clone()).await;

    // Threshold 1: one recorded failure trips the breaker open.
    let mut fleet = AgentFleet::new().with_breaker_config(crate::scale::CircuitBreakerConfig {
        failure_threshold: 1,
        open_duration: std::time::Duration::from_secs(30),
    });
    for (name, url) in [("a", &primary), ("b", &secondary)] {
        fleet.register(AgentCard::new(name, "doc", url.clone()).with_skill(AgentSkill::new(
            "retrieve",
            "retrieve",
            "retrieve relevant documents from the corpus",
        )));
    }
    let client = A2AClient::builder("http://localhost:1")
        .fleet(std::sync::Arc::new(fleet))
        .build()
        .unwrap();

    // First routed send goes to the primary (the higher-ranked candidate).
    let first = client
        .send_task_routed("retrieve documents", A2AMessage::user("hi"))
        .await
        .unwrap();
    assert_eq!(first.agent_url, primary);

    // Trip the primary; the next routed send must skip it for the secondary.
    let fleet = client.fleet().unwrap();
    fleet.record_failure(&primary);

    let second = client
        .send_task_routed("retrieve documents", A2AMessage::user("hi"))
        .await
        .unwrap();
    assert_eq!(
        second.agent_url, secondary,
        "a tripped agent must be skipped on a real send"
    );
    assert_eq!(hits_primary.lock().unwrap().len(), 1);
    assert_eq!(hits_secondary.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn fleet_requires_a_fleet_to_route() {
    let client = A2AClient::new("http://localhost:1".to_string()).unwrap();
    assert!(matches!(
        client.route_task("anything"),
        Err(A2AError::NoRoute(_))
    ));
    assert!(client.fleet().is_none());
}

#[tokio::test]
async fn fleet_delegation_enforces_the_hierarchy_and_hops() {
    use crate::protocol::AgentSkill;

    let hits = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let worker = spawn_fleet_agent(hits.clone()).await;

    let mut fleet = AgentFleet::new()
        .with_orchestrator("http://orch")
        .with_worker(worker.clone())
        .with_max_hops(2);
    fleet.register(AgentCard::new("worker", "doc", worker.clone()).with_skill(
        AgentSkill::new(
            "retrieve",
            "retrieve",
            "retrieve relevant documents from the corpus",
        ),
    ));
    let client = A2AClient::builder("http://localhost:1")
        .fleet(std::sync::Arc::new(fleet))
        .build()
        .unwrap();

    // An orchestrator may delegate within the hop budget.
    let routed = client
        .delegate_task("parent-task", "http://orch", 1, "retrieve documents", A2AMessage::user("hi"))
        .await
        .unwrap();
    assert_eq!(routed.agent_url, worker);

    // A worker may not delegate (hierarchy rule surfaces as NoRoute).
    let err = client
        .delegate_task("parent-task", &worker, 1, "retrieve documents", A2AMessage::user("hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, A2AError::NoRoute(msg) if msg.contains("may not delegate")),
        "worker delegation must be refused, got: {err:?}"
    );

    // Exceeding the hop budget is refused before any routing.
    let err = client
        .delegate_task("parent-task", "http://orch", 3, "retrieve documents", A2AMessage::user("hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, A2AError::NoRoute(msg) if msg.contains("hop limit")),
        "hop overflow must be refused, got: {err:?}"
    );
}
