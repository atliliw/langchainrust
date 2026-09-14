//! Official MCP **Streamable HTTP server** transport (B1, 0.22.4).
//!
//! Server counterpart of [`super::streamable_http::StreamableHttpTransport`]:
//! exposes an [`MCPServer`] over the 2025-03-26 Streamable HTTP protocol so
//! that official TS/Python SDK clients (and [`crate::StreamableMcpClient`])
//! can connect:
//!
//! - one POST endpoint; `GET`/`DELETE` get 405 (server-initiated SSE is not
//!   offered — an honest, spec-allowed boundary);
//! - `initialize` creates an opaque session id returned as `Mcp-Session-Id`;
//!   every later message must echo it (missing → 400, unknown → 404);
//! - notifications (messages without an `id`) answer HTTP 202;
//! - responses are content-negotiated: `application/json` by default, an SSE
//!   `event: message` frame when the client only accepts `text/event-stream`;
//! - `415`/`406` enforce the JSON-RPC content types; an optional
//!   [`crate::auth::TokenValidator`] rejects bad bearer tokens with 401 and a
//!   `WWW-Authenticate: Bearer` challenge the client OAuth layer can parse.
//!
//! One request is served per connection (`Connection: close`): official
//! clients issue a fresh POST per JSON-RPC message, so correctness and session
//! semantics stay simple without HTTP keep-alive bookkeeping.

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

use crate::protocol::{MCPRequest, MCPResponse, MCP_VERSION};
use crate::server::{
    read_http_request, HttpRequest, MCPServer, HTTP_MAX_CONCURRENT_CONNECTIONS,
    HTTP_READ_TIMEOUT_SECS,
};

/// Idle sessions older than this are swept away on the next `initialize`,
/// bounding the in-memory session table.
const SESSION_TTL: Duration = Duration::from_secs(60 * 60);

/// A live Streamable HTTP session.
struct Session {
    /// Set once the client sends `notifications/initialized`.
    initialized: bool,
    born: Instant,
}

/// Shared state behind the accept loop.
struct StreamableState {
    server: Arc<MCPServer>,
    sessions: Mutex<HashMap<String, Session>>,
}

/// Entry point used by [`MCPServer::serve_streamable_http`]; spawns the accept
/// loop and returns the endpoint URL.
pub(crate) fn serve(server: Arc<MCPServer>, listener: TcpListener) -> String {
    let addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_default();
    let state = Arc::new(StreamableState {
        server,
        sessions: Mutex::new(HashMap::new()),
    });
    let semaphore = Arc::new(Semaphore::new(HTTP_MAX_CONCURRENT_CONNECTIONS));

    tokio::spawn(async move {
        loop {
            let (sock, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => break,
            };
            let state = state.clone();
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => continue,
            };
            tokio::spawn(async move {
                let _permit = permit; // released when the connection ends
                let mut sock = sock;
                let read = tokio::time::timeout(
                    Duration::from_secs(HTTP_READ_TIMEOUT_SECS),
                    read_http_request(&mut sock),
                )
                .await;
                let request = match read {
                    Ok(Ok(req)) => req,
                    Ok(Err(())) => return, // closed / malformed transport
                    Err(_) => {
                        // Slow loris: reap without spending a response.
                        return;
                    }
                };
                let _ = handle_connection(&state, request, &mut sock).await;
            });
        }
    });

    format!("http://{addr}/mcp")
}

async fn handle_connection(
    state: &Arc<StreamableState>,
    request: HttpRequest,
    sock: &mut TcpStream,
) -> io::Result<()> {
    let mut request_line = request.first_line.splitn(3, ' ');
    let http_method = request_line.next().unwrap_or_default();
    if http_method != "POST" {
        return write_plain(sock, 405, "Method Not Allowed", &[("Allow", "POST")], b"").await;
    }

    let header = |name: &str| -> Option<&str> {
        request
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };

    // Authenticate before content negotiation (same validator as the
    // stateless track): unauthenticated callers must get the challenge, not a
    // 406 caused by their HTTP stack's default `Accept: */*`.
    if let Some(validator) = state.server.token_validator() {
        let token = header("authorization").and_then(|v| v.strip_prefix("Bearer "));
        let authorized = match token {
            Some(t) => validator.validate(t).await.is_ok(),
            None => false,
        };
        if !authorized {
            return write_plain(
                sock,
                401,
                "Unauthorized",
                &[("WWW-Authenticate", r#"Bearer realm="langchainrust-mcp""#)],
                b"",
            )
            .await;
        }
    }

    // Content negotiation and content type (2025-03-26 §5.1).
    let content_type = header("content-type").unwrap_or("");
    if !content_type_is_json(content_type) {
        return write_plain(
            sock,
            415,
            "Unsupported Media Type",
            &[("Accept", "application/json")],
            b"expected application/json",
        )
        .await;
    }
    let sse_only = match response_mode(header("accept")) {
        AcceptMode::Json => false,
        AcceptMode::Sse => true,
        AcceptMode::None => {
            return write_plain(
                sock,
                406,
                "Not Acceptable",
                &[("Accept", "application/json, text/event-stream")],
                b"",
            )
            .await;
        }
    };

    let body = match &request.body {
        Some(b) => b.as_str(),
        None => {
            return write_plain(sock, 400, "Bad Request", &[], b"missing request body").await;
        }
    };
    let message: Value = match serde_json::from_str(body) {
        Ok(Value::Object(map)) if map.contains_key("method") => Value::Object(map),
        _ => {
            // JSON-RPC parse error: 400 with an id-null error envelope.
            let envelope = json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": {"code": -32700, "message": "parse error"},
            });
            return write_envelope(sock, 400, false, None, &envelope).await;
        }
    };
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let given_session = header("mcp-session-id").map(str::to_string);

    // Notifications (no id) → 202 Accepted.
    if message.get("id").is_none() {
        let session_id = match require_session(state, given_session) {
            Ok(id) => id,
            Err(status) => return write_plain(sock, status.0, status.1, &[], b"").await,
        };
        if method == "notifications/initialized" {
            if let Some(session) = state
                .sessions
                .lock()
                .expect("session lock")
                .get_mut(&session_id)
            {
                session.initialized = true;
            }
        }
        state
            .server
            .handle_notification(&method, message.get("params").cloned())
            .await;
        return write_plain(
            sock,
            202,
            "Accepted",
            &[("Mcp-Session-Id", &session_id)],
            b"",
        )
        .await;
    }

    if method == "initialize" {
        if given_session.is_some() {
            return write_plain(
                sock,
                400,
                "Bad Request",
                &[],
                b"initialize must not carry Mcp-Session-Id",
            )
            .await;
        }
        let (id_value, typed_request) = match coerce_request(message) {
            Ok(pair) => pair,
            Err(()) => return write_plain(sock, 400, "Bad Request", &[], b"invalid request").await,
        };
        let response = state.server.handle_request(typed_request).await;
        if response.error.is_some() {
            let envelope = envelope_with_id(&id_value, &response);
            return write_envelope(sock, 400, sse_only, None, &envelope).await;
        }
        let session_id = insert_session(state);
        let envelope = envelope_with_id(&id_value, &response);
        return write_envelope(
            sock,
            200,
            sse_only,
            Some(SessionHeaders {
                session_id: &session_id,
                protocol_version: true,
            }),
            &envelope,
        )
        .await;
    }

    // Ordinary request: needs a live, initialized session.
    let session_id = match require_session(state, given_session) {
        Ok(id) => id,
        Err(status) => return write_plain(sock, status.0, status.1, &[], b"").await,
    };
    let initialized = {
        let sessions = state.sessions.lock().expect("session lock");
        sessions
            .get(&session_id)
            .map(|s| s.initialized)
            .unwrap_or(false)
        // guard dropped here — never across an await
    };
    if !initialized {
        return write_plain(sock, 400, "Bad Request", &[], b"session not initialized").await;
    }

    let (id_value, typed_request) = match coerce_request(message) {
        Ok(pair) => pair,
        Err(()) => return write_plain(sock, 400, "Bad Request", &[], b"invalid request").await,
    };
    let response = state.server.handle_request(typed_request).await;
    let envelope = envelope_with_id(&id_value, &response);
    write_envelope(
        sock,
        200,
        sse_only,
        Some(SessionHeaders {
            session_id: &session_id,
            protocol_version: false,
        }),
        &envelope,
    )
    .await
}

/// How the client wants responses encoded, derived from `Accept`.
enum AcceptMode {
    Json,
    Sse,
    /// Neither supported content type accepted.
    None,
}

fn response_mode(accept: Option<&str>) -> AcceptMode {
    let Some(accept) = accept else {
        // Missing Accept: the spec lets a server assume a Streamable HTTP client.
        return AcceptMode::Json;
    };
    let media_ranges: Vec<&str> = accept
        .split(',')
        .map(|part| part.split(';').next().unwrap_or("").trim())
        .filter(|part| !part.is_empty())
        .collect();
    let listed = |want: &str| media_ranges.iter().any(|m| m.eq_ignore_ascii_case(want));
    let wildcard = media_ranges
        .iter()
        .any(|m| m.eq_ignore_ascii_case("*/*") || m.eq_ignore_ascii_case("application/*"));
    if listed("application/json") {
        return AcceptMode::Json;
    }
    if listed("text/event-stream") {
        return AcceptMode::Sse;
    }
    // A generic HTTP client (reqwest, curl, browsers) sends `*/*`; RFC 9110
    // matches it against application/json, so the direct JSON response stands.
    if wildcard {
        AcceptMode::Json
    } else {
        AcceptMode::None
    }
}

/// Streamable HTTP accepts `application/json` (parameters like `charset`
/// ignored) and the JSON-RPC alias `application/json-rpc`.
fn content_type_is_json(value: &str) -> bool {
    let media = value.split(';').next().unwrap_or("").trim();
    media.eq_ignore_ascii_case("application/json")
        || media.eq_ignore_ascii_case("application/json-rpc")
}

fn require_session(
    state: &Arc<StreamableState>,
    given: Option<String>,
) -> Result<String, (u16, &'static str)> {
    let session_id = given.ok_or((400u16, "Bad Request"))?;
    let mut sessions = state.sessions.lock().expect("session lock");
    if !sessions.contains_key(&session_id) {
        return Err((404, "Not Found"));
    }
    // Liveness touch (born doubles as last-seen; TTL sweep only needs a bound).
    if let Some(session) = sessions.get_mut(&session_id) {
        session.born = Instant::now();
    }
    Ok(session_id)
}

fn insert_session(state: &Arc<StreamableState>) -> String {
    let mut sessions = state.sessions.lock().expect("session lock");
    // Opportunistic TTL sweep.
    let now = Instant::now();
    sessions.retain(|_, s| now.duration_since(s.born) < SESSION_TTL);
    let id = new_session_id();
    sessions.insert(
        id.clone(),
        Session {
            initialized: false,
            born: now,
        },
    );
    id
}

/// 128-bit opaque hex session id (RFC token charset, unguessable).
fn new_session_id() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("getrandom always succeeds on supported targets");
    let mut id = String::with_capacity(32);
    for byte in bytes {
        // INVARIANT: 入参是 0..=15 的半字节(>>4 / &0x0f),基数 16 下 from_digit 必为 Some。
        id.push(
            std::char::from_digit(u32::from(byte) >> 4, 16)
                .expect("high nibble 0..=15 is always a valid base-16 digit"),
        );
        id.push(
            std::char::from_digit(u32::from(byte) & 0x0f, 16)
                .expect("low nibble 0..=15 is always a valid base-16 digit"),
        );
    }
    id
}

/// Forces the typed request's numeric id to 0, returning the original (possibly
/// string / null) JSON id so the response echoes it verbatim.
fn coerce_request(mut message: Value) -> Result<(Value, MCPRequest), ()> {
    let id_value = message.get("id").cloned().unwrap_or(Value::Null);
    message["id"] = json!(0u64);
    let typed = serde_json::from_value::<MCPRequest>(message).map_err(|_| ())?;
    Ok((id_value, typed))
}

/// Serializes an [`MCPResponse`] while preserving the request's original id.
fn envelope_with_id(id_value: &Value, response: &MCPResponse) -> Value {
    let mut envelope = json!({
        "jsonrpc": "2.0",
        "id": id_value,
    });
    if let Some(error) = &response.error {
        envelope["error"] = serde_json::to_value(error).unwrap_or(Value::Null);
    } else {
        envelope["result"] = response.result.clone().unwrap_or(Value::Null);
    }
    envelope
}

/// Extra headers identifying a session response.
struct SessionHeaders<'a> {
    session_id: &'a str,
    /// `initialize` additionally carries the negotiated protocol version.
    protocol_version: bool,
}

/// Writes a JSON or SSE-framed JSON-RPC response.
async fn write_envelope(
    sock: &mut TcpStream,
    status: u16,
    as_sse: bool,
    session: Option<SessionHeaders<'_>>,
    envelope: &Value,
) -> io::Result<()> {
    let payload = serde_json::to_vec(envelope).unwrap_or_else(|_| b"{}".to_vec());
    let reason = reason_phrase(status);

    if as_sse {
        let mut body = Vec::with_capacity(payload.len() + 32);
        body.extend_from_slice(b"event: message\r\ndata: ");
        body.extend_from_slice(&payload);
        body.extend_from_slice(b"\r\n\r\n");
        sock.write_all(format!("HTTP/1.1 {status} {reason}\r\n").as_bytes())
            .await?;
        sock.write_all(b"Content-Type: text/event-stream\r\nCache-Control: no-cache\r\n")
            .await?;
        if let Some(session) = &session {
            sock.write_all(format!("Mcp-Session-Id: {}\r\n", session.session_id).as_bytes())
                .await?;
            if session.protocol_version {
                sock.write_all(format!("MCP-Protocol-Version: {MCP_VERSION}\r\n").as_bytes())
                    .await?;
            }
        }
        sock.write_all(b"Connection: close\r\n").await?;
        sock.write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())
            .await?;
        sock.write_all(&body).await?;
    } else {
        sock.write_all(format!("HTTP/1.1 {status} {reason}\r\n").as_bytes())
            .await?;
        sock.write_all(b"Content-Type: application/json\r\n")
            .await?;
        if let Some(session) = &session {
            sock.write_all(format!("Mcp-Session-Id: {}\r\n", session.session_id).as_bytes())
                .await?;
            if session.protocol_version {
                sock.write_all(format!("MCP-Protocol-Version: {MCP_VERSION}\r\n").as_bytes())
                    .await?;
            }
        }
        sock.write_all(b"Connection: close\r\n").await?;
        sock.write_all(format!("Content-Length: {}\r\n\r\n", payload.len()).as_bytes())
            .await?;
        sock.write_all(&payload).await?;
    }
    sock.flush().await
}

/// Writes a bare status response (no JSON-RPC body) with caller-chosen headers.
async fn write_plain(
    sock: &mut TcpStream,
    status: u16,
    reason: &str,
    extra_headers: &[(&str, &str)],
    body: &[u8],
) -> io::Result<()> {
    sock.write_all(format!("HTTP/1.1 {status} {reason}\r\n").as_bytes())
        .await?;
    if !body.is_empty() {
        sock.write_all(b"Content-Type: text/plain; charset=utf-8\r\n")
            .await?;
    }
    for (name, value) in extra_headers {
        sock.write_all(format!("{name}: {value}\r\n").as_bytes())
            .await?;
    }
    sock.write_all(b"Connection: close\r\n").await?;
    sock.write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())
        .await?;
    sock.write_all(body).await?;
    sock.flush().await
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Status",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_negotiation_prefers_json() {
        assert!(matches!(
            response_mode(Some("application/json, text/event-stream")),
            AcceptMode::Json
        ));
        assert!(matches!(
            response_mode(Some("text/event-stream")),
            AcceptMode::Sse
        ));
        assert!(matches!(response_mode(Some("text/html")), AcceptMode::None));
        assert!(matches!(response_mode(None), AcceptMode::Json));
        // Quality parameters / whitespace must not confuse token matching.
        assert!(matches!(
            response_mode(Some("text/event-stream;q=0.9 , application/json ;q=0.8")),
            AcceptMode::Json
        ));
        // Generic HTTP clients send wildcard ranges; direct JSON is acceptable.
        assert!(matches!(response_mode(Some("*/*")), AcceptMode::Json));
        assert!(matches!(
            response_mode(Some("application/*")),
            AcceptMode::Json
        ));
        // Explicit SSE still wins over a bare wildcard when listed first.
        assert!(matches!(
            response_mode(Some("text/event-stream, */*;q=0.1")),
            AcceptMode::Sse
        ));
    }

    #[test]
    fn json_content_type_accepts_charset_and_alias() {
        assert!(content_type_is_json("application/json"));
        assert!(content_type_is_json("application/json; charset=utf-8"));
        assert!(content_type_is_json("application/json-rpc"));
        assert!(!content_type_is_json("text/plain"));
    }

    #[test]
    fn string_ids_are_echoed_verbatim() {
        let message = json!({"jsonrpc": "2.0", "id": "req-42", "method": "ping"});
        let (id_value, typed) = coerce_request(message).unwrap();
        assert_eq!(id_value, json!("req-42"));
        assert_eq!(typed.id, 0);
        let response = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(0),
            result: Some(json!({})),
            error: None,
        };
        let envelope = envelope_with_id(&id_value, &response);
        assert_eq!(envelope["id"], json!("req-42"));
    }

    #[test]
    fn session_ids_are_unique_hex_tokens() {
        let a = new_session_id();
        let b = new_session_id();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }
}
