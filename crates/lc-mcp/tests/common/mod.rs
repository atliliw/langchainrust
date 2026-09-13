//! Shared integration-test fixtures for lc-mcp.
//!
//! A spec-faithful fake of an official MCP **Streamable HTTP** server written
//! against raw HTTP/1.1 (no framework coupling — same role the official
//! TS/Python SDK servers play in CI). One connection per request; the same
//! listener also serves RFC 9728 / RFC 8414 metadata and a token endpoint for
//! the OAuth 2.1 interop tests.

#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// How the fixture delivers a JSON-RPC response to a POST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseMode {
    /// Direct `application/json` body.
    Json,
    /// `text/event-stream`: one server notification frame, then the response.
    Sse,
    /// `text/event-stream` headers, then immediate close (no frames).
    EmptySse,
}

/// Fixture behavior configuration.
#[derive(Debug, Clone)]
pub struct FixtureConfig {
    /// Response delivery mode.
    pub mode: ResponseMode,
    /// Whether initialize assigns a `Mcp-Session-Id`.
    pub issue_session: bool,
    /// Whether non-initialize POSTs must carry a live session id.
    pub enforce_session: bool,
    /// When set, every /mcp POST must carry `Bearer <value>`.
    pub require_bearer: Option<String>,
    /// Forces the initialize result's protocolVersion (negotiation tests).
    pub forced_version: Option<String>,
    /// Delay before answering /mcp POSTs (timeout tests).
    pub delay: Duration,
}

impl Default for FixtureConfig {
    fn default() -> Self {
        Self {
            mode: ResponseMode::Json,
            issue_session: true,
            enforce_session: true,
            require_bearer: None,
            forced_version: None,
            delay: Duration::ZERO,
        }
    }
}

impl FixtureConfig {
    /// Stateful JSON server (session issued and enforced).
    pub fn json() -> Self {
        Self::default()
    }
    /// Stateful SSE server.
    pub fn sse() -> Self {
        Self {
            mode: ResponseMode::Sse,
            ..Self::default()
        }
    }
    /// Fully stateless server (no session id, no enforcement).
    pub fn stateless() -> Self {
        Self {
            issue_session: false,
            enforce_session: false,
            ..Self::default()
        }
    }
}

/// Handle to the running fixture.
pub struct FakeStreamableServer {
    /// MCP endpoint (`http://127.0.0.1:port/mcp`).
    pub url: String,
    /// Protected-resource metadata URL.
    pub metadata_url: String,
    /// Authorization-server issuer URL (no trailing slash).
    pub as_issuer: String,
    /// Total POSTs to `/mcp`.
    pub posts: Arc<AtomicUsize>,
    /// `notifications/initialized` notifications received.
    pub initialized_notifications: Arc<AtomicUsize>,
    /// `Mcp-Session-Id` request header per POST (`None` when absent).
    pub session_header_seen: Arc<Mutex<Vec<Option<String>>>>,
    /// `Authorization` request header per POST.
    pub auth_headers_seen: Arc<Mutex<Vec<Option<String>>>>,
    /// Session ids ever issued, in order.
    pub sessions_issued: Arc<Mutex<Vec<String>>>,
    /// Token-endpoint form bodies received.
    pub token_requests: Arc<Mutex<Vec<String>>>,
    dead_sessions: Arc<Mutex<HashSet<String>>>,
}

impl FakeStreamableServer {
    /// Marks a previously issued session as invalid server-side (the next
    /// POST carrying it gets HTTP 404).
    pub fn kill_session(&self, id: &str) {
        self.dead_sessions.lock().unwrap().insert(id.to_string());
    }
}

/// Starts the fixture on an ephemeral port.
pub async fn start(config: FixtureConfig) -> FakeStreamableServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");

    let posts = Arc::new(AtomicUsize::new(0));
    let initialized_notifications = Arc::new(AtomicUsize::new(0));
    let session_header_seen = Arc::new(Mutex::new(Vec::new()));
    let auth_headers_seen = Arc::new(Mutex::new(Vec::new()));
    let sessions_issued = Arc::new(Mutex::new(Vec::new()));
    let token_requests = Arc::new(Mutex::new(Vec::new()));
    let dead_sessions = Arc::new(Mutex::new(HashSet::new()));
    let session_counter = Arc::new(AtomicUsize::new(0));

    let state = ServerState {
        config,
        posts: posts.clone(),
        initialized_notifications: initialized_notifications.clone(),
        session_header_seen: session_header_seen.clone(),
        auth_headers_seen: auth_headers_seen.clone(),
        sessions_issued: sessions_issued.clone(),
        token_requests: token_requests.clone(),
        dead_sessions: dead_sessions.clone(),
        session_counter,
    };

    tokio::spawn(async move {
        while let Ok((sock, _)) = listener.accept().await {
            let state = state.clone();
            tokio::spawn(async move {
                let _ = handle_connection(sock, state).await;
            });
        }
    });

    FakeStreamableServer {
        url: format!("{base}/mcp"),
        metadata_url: format!("{base}/.well-known/oauth-protected-resource"),
        as_issuer: format!("{base}/as"),
        posts,
        initialized_notifications,
        session_header_seen,
        auth_headers_seen,
        sessions_issued,
        token_requests,
        dead_sessions,
    }
}

#[derive(Clone)]
struct ServerState {
    config: FixtureConfig,
    posts: Arc<AtomicUsize>,
    initialized_notifications: Arc<AtomicUsize>,
    session_header_seen: Arc<Mutex<Vec<Option<String>>>>,
    auth_headers_seen: Arc<Mutex<Vec<Option<String>>>>,
    sessions_issued: Arc<Mutex<Vec<String>>>,
    token_requests: Arc<Mutex<Vec<String>>>,
    dead_sessions: Arc<Mutex<HashSet<String>>>,
    session_counter: Arc<AtomicUsize>,
}

async fn handle_connection(
    mut sock: tokio::net::TcpStream,
    state: ServerState,
) -> std::io::Result<()> {
    let (method, path, headers, body) = match read_request(&mut sock).await {
        Some(req) => req,
        None => return Ok(()),
    };

    // Metadata + token endpoints are always unauthenticated.
    if method == "GET" && path == "/.well-known/oauth-protected-resource" {
        let resource = format!("http://{}/mcp", sock_local_addr(&sock));
        let issuer = format!("http://{}/as", sock_local_addr(&sock));
        let payload = json!({
            "resource": resource,
            "authorization_servers": [issuer],
            "scopes_supported": ["mcp.read", "mcp.write"],
            "bearer_methods_supported": ["header"],
        });
        return write_json(&mut sock, 200, "OK", payload, &[]).await;
    }
    if method == "GET" && path == "/as/.well-known/oauth-authorization-server" {
        let host = sock_local_addr(&sock);
        let payload = json!({
            "issuer": format!("http://{host}/as"),
            "authorization_endpoint": format!("http://{host}/as/authorize"),
            "token_endpoint": format!("http://{host}/as/token"),
            "registration_endpoint": format!("http://{host}/as/register"),
            "grant_types_supported": ["authorization_code", "refresh_token", "client_credentials"],
            "code_challenge_methods_supported": ["S256"],
        });
        return write_json(&mut sock, 200, "OK", payload, &[]).await;
    }
    if method == "POST" && path == "/as/token" {
        state.token_requests.lock().unwrap().push(body.clone());
        let form = parse_form(&body);
        let grant = form.get("grant_type").map(String::as_str).unwrap_or("");
        assert!(form.contains_key("client_id"), "DCR-style body client_id");
        let access_token = match grant {
            "refresh_token" => "rotated-access",
            "authorization_code" => "code-access",
            "client_credentials" => "cc-access",
            _ => "access",
        };
        let payload = json!({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "rotated-refresh",
            "scope": "mcp.read",
        });
        return write_json(&mut sock, 200, "OK", payload, &[]).await;
    }

    if method != "POST" || path != "/mcp" {
        return write_status(&mut sock, 404, "Not Found").await;
    }

    state.posts.fetch_add(1, Ordering::SeqCst);
    let auth = header_value(&headers, "authorization");
    let session = header_value(&headers, "mcp-session-id");
    state.auth_headers_seen.lock().unwrap().push(auth.clone());
    state
        .session_header_seen
        .lock()
        .unwrap()
        .push(session.clone());

    // --- Auth gate (OAuth 2.1 resource server) ---------------------------
    if let Some(expected) = &state.config.require_bearer {
        if auth.as_deref() != Some(format!("Bearer {expected}").as_str()) {
            let host = sock_local_addr(&sock);
            let challenge = format!(
                "Bearer realm=\"echo-streamable\",resource_metadata=\"http://{host}/.well-known/oauth-protected-resource\",scope=\"mcp.read\""
            );
            return write_raw(
                &mut sock,
                401,
                "Unauthorized",
                b"",
                &[("WWW-Authenticate", challenge.as_str())],
            )
            .await;
        }
    }

    if state.config.delay != Duration::ZERO {
        tokio::time::sleep(state.config.delay).await;
    }

    let message: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => {
            let payload = json!({"jsonrpc":"2.0","id":Value::Null,"error":{"code":-32700,"message":"Parse error"}});
            return write_json(&mut sock, 400, "Bad Request", payload, &[]).await;
        }
    };

    let is_notification = message.get("id").is_none();
    let req_method = message.get("method").and_then(|m| m.as_str()).unwrap_or("");

    // --- Session semantics ------------------------------------------------
    let mut new_session: Option<String> = None;
    if req_method == "initialize" && state.config.issue_session {
        let n = state.session_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let id = format!("sess-{n}");
        state.sessions_issued.lock().unwrap().push(id.clone());
        new_session = Some(id);
    } else if state.config.enforce_session {
        let live = match &session {
            Some(id) => {
                !state.dead_sessions.lock().unwrap().contains(id)
                    && state
                        .sessions_issued
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|s| s == id)
            }
            None => false,
        };
        if !live {
            // Unknown/expired/missing Mcp-Session-Id → 404 per spec.
            return write_status(&mut sock, 404, "Not Found").await;
        }
    }

    if is_notification {
        if req_method == "notifications/initialized" {
            state
                .initialized_notifications
                .fetch_add(1, Ordering::SeqCst);
        }
        // Notifications get HTTP 202 Accepted (no body). The session id is
        // delivered by the initialize response, not the 202.
        return write_status(&mut sock, 202, "Accepted").await;
    }

    let id = message.get("id").cloned().unwrap_or(Value::Null);
    let params = message.get("params");

    let result_value: Result<Value, (i32, String)> = match req_method {
        "initialize" => {
            let requested = params
                .and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str())
                .unwrap_or("2024-11-05");
            let version = state.config.forced_version.as_deref().unwrap_or(requested);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo-streamable", "version": "0.0.1"},
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({
            "tools": [{
                "name": "echo",
                "description": "echoes its input",
                "inputSchema": {"type": "object"}
            }]
        })),
        "tools/call" => {
            let name = params
                .and_then(|p| p.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if name != "echo" {
                Err((-32601, format!("unknown tool: {name}")))
            } else {
                let msg = params
                    .and_then(|p| p.get("arguments"))
                    .and_then(|a| a.get("msg"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(json!({
                    "content": [{"type": "text", "text": format!("echo: {msg}")}],
                    "isError": false
                }))
            }
        }
        _ => Err((-32601, format!("unknown method: {req_method}"))),
    };

    let envelope = match result_value {
        Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
        Err((code, message)) => {
            json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
        }
    };

    let session_header: Vec<(&str, &str)> = new_session
        .as_ref()
        .map(|id| vec![("Mcp-Session-Id", id.as_str())])
        .unwrap_or_default();

    match state.config.mode {
        ResponseMode::Json => write_json(&mut sock, 200, "OK", envelope, &session_header).await,
        ResponseMode::EmptySse => write_sse(&mut sock, std::iter::empty(), &session_header).await,
        ResponseMode::Sse => {
            let notification = json!({
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": req_method}
            });
            write_sse(
                &mut sock,
                [notification, envelope].into_iter(),
                &session_header,
            )
            .await
        }
    }
}

fn sock_local_addr(sock: &tokio::net::TcpStream) -> String {
    sock.local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "127.0.0.1:0".to_string())
}

fn header_value(headers: &[(String, String)], name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

fn parse_form(body: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for pair in body.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            map.insert(k.to_string(), v.replace('+', " "));
        }
    }
    map
}

async fn write_json(
    sock: &mut tokio::net::TcpStream,
    status: u16,
    reason: &str,
    payload: Value,
    extra_headers: &[(&str, &str)],
) -> std::io::Result<()> {
    let body = serde_json::to_vec(&payload).unwrap();
    let mut headers = vec![("Content-Type", "application/json")];
    headers.extend_from_slice(extra_headers);
    write_raw(sock, status, reason, &body, &headers).await
}

async fn write_status(
    sock: &mut tokio::net::TcpStream,
    status: u16,
    reason: &str,
) -> std::io::Result<()> {
    write_raw(sock, status, reason, b"", &[]).await
}

async fn write_raw(
    sock: &mut tokio::net::TcpStream,
    status: u16,
    reason: &str,
    body: &[u8],
    extra_headers: &[(&str, &str)],
) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra_headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(body).await?;
    let _ = sock.shutdown().await;
    Ok(())
}

/// Writes an SSE response: `message` events for each JSON message, then closes.
async fn write_sse<I>(
    sock: &mut tokio::net::TcpStream,
    messages: I,
    extra_headers: &[(&str, &str)],
) -> std::io::Result<()>
where
    I: IntoIterator<Item = Value>,
{
    let mut head = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n",
    );
    for (k, v) in extra_headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("Connection: close\r\n\r\n");
    sock.write_all(head.as_bytes()).await?;
    for message in messages {
        let data = serde_json::to_string(&message).unwrap();
        let frame = format!("event: message\r\ndata: {data}\r\n\r\n");
        sock.write_all(frame.as_bytes()).await?;
    }
    let _ = sock.flush().await;
    let _ = sock.shutdown().await;
    Ok(())
}

async fn read_request(
    sock: &mut tokio::net::TcpStream,
) -> Option<(String, String, Vec<(String, String)>, String)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
            break pos;
        }
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("/").to_string();
    let path = raw_path.split('?').next().unwrap_or("/").to_string();
    let mut headers = Vec::new();
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_string();
            let v = v.trim().to_string();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = String::from_utf8_lossy(&buf[body_start..body_start + content_length]).to_string();
    Some((method, path, headers, body))
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
