//! Test support: a fake stateless MCP server (compiled only for test builds).
//!
//! Shared by the client / adapter / timeout / gateway test modules, covering:
//! - POST routing by JSON-RPC method (`tools/list` / `tools/call` / `server/discover`);
//! - `Mcp-Method` header + self-contained `_meta` recording (routing-header contract);
//! - MRTR `input_required` flows (`Mrtr` / `MrtrLoop`), HTTP 401 (`Unauth`), slow calls.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Stateless fake server (2026-07-28 track, 0.22.0 S2) — for M1-M8 tests.
// ---------------------------------------------------------------------------

/// Behavior modes for [`start_fake_stateless_server`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatelessMode {
    /// Normal: tools/list, tools/call (echo), server/discover.
    Normal,
    /// The first tools/call (no requestState) answers input_required; the
    /// resent request (with requestState "state-1") succeeds and the echo
    /// carries "answered" + the state.
    Mrtr,
    /// Like `Mrtr` but every request (with or without state) answers
    /// input_required — used to trip the client's round-trip limit.
    MrtrLoop,
    /// Every request is answered with HTTP 401 (unauthorized).
    Unauth,
    /// Every request sleeps for the given duration before answering with the
    /// Normal behavior — used to exercise client-side timeouts.
    SlowCall(std::time::Duration),
}

/// A fake stateless MCP server handle.
pub struct FakeStatelessServer {
    /// Endpoint URL to POST JSON-RPC to.
    pub url: String,
    /// Total requests accepted.
    pub request_count: Arc<AtomicUsize>,
    /// `Mcp-Method` header values seen (routing-header contract).
    pub method_headers_seen: Arc<std::sync::Mutex<Vec<String>>>,
    /// Parsed `_meta` payloads seen (self-contained-request contract).
    pub metas_seen: Arc<std::sync::Mutex<Vec<crate::protocol::RequestMeta>>>,
    /// `requestState` values seen on resent requests.
    pub request_states_seen: Arc<std::sync::Mutex<Vec<String>>>,
}

/// Starts a fake stateless MCP server.
///
/// Accepts POST requests with a JSON-RPC body; every response is a plain HTTP
/// 200 JSON-RPC envelope (no SSE, no handshake). Headers `Mcp-Method` /
/// `_meta` are recorded for contract assertions.
pub async fn start_fake_stateless_server(mode: StatelessMode) -> FakeStatelessServer {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let request_count = Arc::new(AtomicUsize::new(0));
    let method_headers_seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let metas_seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let request_states_seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let rc = request_count.clone();
    let mh = method_headers_seen.clone();
    let ms = metas_seen.clone();
    let rs = request_states_seen.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => break,
            };
            let rc = rc.clone();
            let mh = mh.clone();
            let ms = ms.clone();
            let rs = rs.clone();
            tokio::spawn(async move {
                let (_first_line, mut headers, mut body) = read_http_request_full(&mut sock).await;
                // Keep-alive loops: handle further requests on the same conn.
                loop {
                    let _ = rc.fetch_add(1, Ordering::SeqCst);
                    let mut method_header = String::new();
                    for (k, v) in &headers {
                        if k.eq_ignore_ascii_case("mcp-method") {
                            method_header = v.clone();
                        }
                    }
                    if !method_header.is_empty() {
                        mh.lock().unwrap().push(method_header);
                    }

                    if mode == StatelessMode::Unauth {
                        let _ = sock
                            .write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                            .await;
                        return;
                    }

                    if let StatelessMode::SlowCall(delay) = mode {
                        tokio::time::sleep(delay).await;
                    }

                    let req: crate::protocol::MCPRequest = serde_json::from_str(&body)
                        .unwrap_or_else(|_| crate::protocol::MCPRequest::new(0, "", None));
                    if let Some(meta) = &req.meta {
                        ms.lock().unwrap().push(meta.clone());
                        if let Some(state) = &meta.request_state {
                            rs.lock().unwrap().push(state.clone());
                        }
                    }

                    let result: serde_json::Value = match req.method.as_str() {
                        "tools/list" => serde_json::json!({
                            "tools": [{
                                "name": "echo",
                                "description": "echo desc",
                                "inputSchema": {"type": "object"}
                            }]
                        }),
                        "tools/call" => {
                            let has_state = req
                                .meta
                                .as_ref()
                                .and_then(|m| m.request_state.as_deref())
                                .is_some();
                            let needs_input = match mode {
                                StatelessMode::Mrtr => !has_state,
                                StatelessMode::MrtrLoop => true,
                                _ => false,
                            };
                            if needs_input {
                                serde_json::json!({
                                    "input_required": {
                                        "requestState": "state-1",
                                        "questions": [
                                            {"id": "q1", "prompt": "confirm?"}
                                        ]
                                    }
                                })
                            } else {
                                let state = req
                                    .meta
                                    .as_ref()
                                    .and_then(|m| m.request_state.clone())
                                    .unwrap_or_default();
                                serde_json::json!({
                                    "content": [{
                                        "type": "text",
                                        "text": format!("echo answered state={state}")
                                    }],
                                    "isError": false
                                })
                            }
                        }
                        "server/discover" => serde_json::json!({
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake-stateless", "version": "0.0.1"}
                        }),
                        _ => serde_json::Value::Null,
                    };

                    let resp = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": req.id,
                        "result": result
                    });
                    let payload = resp.to_string();
                    let http = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        payload.len(),
                        payload
                    );
                    if sock.write_all(http.as_bytes()).await.is_err() {
                        return;
                    }
                    // Connection reuse: try to read the next request; a
                    // closed/reset socket ends the loop.
                    let (fl, h, b) = read_http_request_full(&mut sock).await;
                    if fl.is_empty() {
                        return;
                    }
                    headers = h;
                    body = b;
                }
            });
        }
    });

    FakeStatelessServer {
        url: format!("http://{addr}/mcp"),
        request_count,
        method_headers_seen,
        metas_seen,
        request_states_seen,
    }
}

/// Reads one HTTP request including headers: `(request_line, headers, body)`.
/// Empty `request_line` = connection closed.
async fn read_http_request_full(
    sock: &mut tokio::net::TcpStream,
) -> (String, Vec<(String, String)>, String) {
    use tokio::io::AsyncReadExt;
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    // Read until "\r\n\r\n" seen.
    loop {
        let header_end = find_subsequence(&buf, b"\r\n\r\n");
        if let Some(pos) = header_end {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let mut lines = head.lines();
            let first_line = lines.next().unwrap_or_default().to_string();
            let mut headers = Vec::new();
            for line in lines {
                if let Some((k, v)) = line.split_once(':') {
                    headers.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
            let content_length = headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .unwrap_or(0);
            let body_start = pos + 4;
            while buf.len() < body_start + content_length {
                let n = sock.read(&mut tmp).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = String::from_utf8_lossy(&buf[body_start..(body_start + content_length)])
                .to_string();
            return (first_line, headers, body);
        }
        let n = sock.read(&mut tmp).await.unwrap_or(0);
        if n == 0 {
            return (String::new(), Vec::new(), String::new());
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
