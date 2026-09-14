// lc-providers/tests/common/mod.rs
//! Offline cassette-playback harness (T4, v0.23.0).
//!
//! lc-providers' `tests/` historically had zero fixtures and all live tests were
//! `#[ignore]`d, so field-mapping bugs (usage capture, tool-call parsing,
//! finish_reason, streaming aggregation, error-status mapping) only surfaced
//! from user reports. This harness replays recorded responses through a tiny
//! loopback HTTP/1.1 server so provider behaviour is asserted **offline**,
//! without an API key or network.
//!
//! Each test points a provider's `base_url` at a fresh `127.0.0.1` port whose
//! routes map request paths to canned `MockResponse`s. The server is re-used
//! only by spawning per-test; it dies with the enclosing `#[tokio::test]`
//! runtime.
//!
//! T5 (v0.23.0), the request-body null assertion is lifted: [`spawn_mock_server_capture`]
//! additionally records the most recent request body so tests can assert what the
//! provider **sends** (cache-control TTL breakpoints, `tool_choice`, `response_format`)
//! — not just how it maps responses.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// A scripted HTTP response for one request path.
#[derive(Clone)]
pub struct MockResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: String,
}

impl MockResponse {
    /// A JSON response (e.g. a non-streaming completion).
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: body.into(),
        }
    }

    /// An SSE response (a streaming completion) — the body is the raw
    /// `data: {...}\n\n` event stream as the provider would receive it.
    pub fn sse_stream(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            body: body.into(),
        }
    }
}

/// Request path → canned response.
pub type Routes = Arc<HashMap<String, MockResponse>>;

/// The most recent request body the capture server observed (T5).
pub type CapturedBody = Arc<Mutex<Option<String>>>;

pub mod routes {
    use super::{HashMap, MockResponse, Routes};
    use std::sync::Arc;

    /// Builds route map from (path, response) pairs.
    pub fn build(pairs: Vec<(String, MockResponse)>) -> Routes {
        Arc::new(HashMap::from_iter(pairs))
    }
}

/// A bound mock server plus its request-body capture handle (T5).
pub struct MockServer {
    pub addr: std::net::SocketAddr,
    pub last_body: CapturedBody,
}

/// Binds a loopback listener WITHOUT body capture and spawns the responder.
/// Returns the bound address. Serves every request in `routes` until the
/// enclosing tokio runtime shuts down.
pub async fn spawn_mock_server(routes: Routes) -> std::net::SocketAddr {
    spawn_mock_server_capture(routes).await.addr
}

/// Binds a loopback listener that ALSO records the most recent request body.
/// Returns the bound address plus a handle exposing the captured body.
pub async fn spawn_mock_server_capture(routes: Routes) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("resolved local addr");
    let last_body: CapturedBody = Arc::new(Mutex::new(None));
    let sink = last_body.clone();
    tokio::spawn(async move { serve_loop(listener, routes, sink).await });
    MockServer { addr, last_body }
}

async fn serve_loop(listener: TcpListener, routes: Routes, sink: CapturedBody) {
    loop {
        let Ok((mut sock, _)) = listener.accept().await else {
            return;
        };
        let routes = routes.clone();
        let sink = sink.clone();
        tokio::spawn(async move {
            let _ = handle_connection(&mut sock, &routes, &sink).await;
        });
    }
}

/// Reads one HTTP/1.1 request, dispatches on its request path, records the
/// request body (T5), and writes the matching canned response.
async fn handle_connection(
    sock: &mut TcpStream,
    routes: &Routes,
    sink: &CapturedBody,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut scratch = [0u8; 1024];
    let mut path = "/".to_string();
    // Declared without initializer (deferred single-init, no `mut` needed): only
    // ever read inside the `header_end` block below, where
    // `find_map(...).unwrap_or(0)` assigns it on every path.
    let content_length;

    // Read until the request-header terminator so we can parse the request line.
    loop {
        let n = sock.read(&mut scratch).await?;
        if n == 0 {
            break; // client hung up before a full request
        }
        buf.extend_from_slice(&scratch[..n]);
        if let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..header_end]);
            path = head
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("/")
                .to_string();
            content_length = head
                .lines()
                .find_map(|l| {
                    let lower = l.to_ascii_lowercase();
                    lower
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            // Consume any body bytes already inside `buf` (POST headers and body
            // often arrive in the same TCP segment), then drain the rest.
            let mut have = buf.len() - (header_end + 4);
            while have < content_length {
                let n = sock.read(&mut scratch).await?;
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&scratch[..n]);
                have = buf.len() - (header_end + 4);
            }
            let body = String::from_utf8_lossy(&buf[header_end + 4..]).to_string();
            *sink.lock().unwrap_or_else(|e| e.into_inner()) = Some(body);
            break;
        }
    }

    let resp = routes
        .get(&path)
        .cloned()
        .unwrap_or_else(|| MockResponse::json(501, r#"{"error":"no fixture"}"#));
    let reason = match resp.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        status = resp.status,
        reason = reason,
        ct = resp.content_type,
        len = resp.body.len(),
    );
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(resp.body.as_bytes()).await?;
    sock.flush().await?;
    Ok(())
}
