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
        "event: artifact-update\ndata: {\"kind\":\"artifact-update\",\"id\":\"t1\",\"artifact\":{\"output\":\"hi\"}}",
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
