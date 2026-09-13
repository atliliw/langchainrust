//! Official MCP Streamable HTTP transport (B1, 0.22.4).
//!
//! One HTTP endpoint accepts every JSON-RPC message as a POST
//! (`Content-Type: application/json`, `Accept: application/json,
//! text/event-stream`). The server answers a request either with a direct
//! `application/json` body or with an `text/event-stream` whose `message`
//! events carry JSON-RPC messages (notifications may precede the response;
//! the stream closes right after the response). Notifications get HTTP 202.
//!
//! Session semantics (2025-03-26 spec):
//! - the initialize response may assign a session via `Mcp-Session-Id`;
//! - the id is then echoed on every subsequent POST;
//! - an unknown id comes back as HTTP 404/400 and the client must
//!   re-initialize ([`MCPError::session_lost`], -32006).
//!
//! Unauthenticated requests receive HTTP 401 with a
//! `WWW-Authenticate: Bearer resource_metadata="…"` challenge (OAuth 2.1,
//! RFC 9728). With a [`BearerTokenProvider`] the transport invalidates the
//! rejected token and retries exactly once; the final error carries the
//! parsed [`OAuthChallenge`] in `MCPError.data`.
//!
//! The standalone GET SSE stream (server-to-client pushes with
//! `Last-Event-ID` resumption) is not implemented in 0.22.4 — clients
//! advertise no capabilities that need it.

use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;

use crate::oauth2::{BearerTokenProvider, OAuthChallenge, StaticBearerToken};
use crate::protocol::{
    notification_message, MCPError, MCPRequest, MCPResponse, MCP_ERROR_UNAUTHORIZED,
};

/// Request header the server assigns on `initialize`.
const SESSION_HEADER: &str = "Mcp-Session-Id";

/// Streamable HTTP transport: stateless HTTP requests carrying one optional
/// server-assigned session id.
pub struct StreamableHttpTransport {
    url: String,
    /// No global timeout: an SSE POST response is itself a stream; per-request
    /// deadlines are enforced by the high-level client (`tokio::time::timeout`).
    http: Client,
    token_provider: Option<Arc<dyn BearerTokenProvider>>,
    session: Mutex<Option<String>>,
}

impl std::fmt::Debug for StreamableHttpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamableHttpTransport")
            .field("url", &self.url)
            .field("session", &self.session)
            .field("has_token_provider", &self.token_provider.is_some())
            .finish()
    }
}

impl StreamableHttpTransport {
    /// Creates a transport for a Streamable HTTP MCP endpoint (e.g.
    /// `https://host/mcp`) without authentication.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            http: Client::new(),
            token_provider: None,
            session: Mutex::new(None),
        }
    }

    /// Creates a transport that attaches a bearer token from `provider` to
    /// every POST and runs one invalidate/retry cycle on HTTP 401.
    pub fn with_token_provider(
        url: impl Into<String>,
        provider: Arc<dyn BearerTokenProvider>,
    ) -> Self {
        Self {
            url: url.into(),
            http: Client::new(),
            token_provider: Some(provider),
            session: Mutex::new(None),
        }
    }

    /// Convenience constructor for a fixed bearer token.
    pub fn with_static_bearer(url: impl Into<String>, token: impl Into<String>) -> Self {
        Self::with_token_provider(url, Arc::new(StaticBearerToken(token.into())))
    }

    /// The endpoint URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The current session id assigned by the server, if any.
    pub fn session_id(&self) -> Option<String> {
        self.session.lock().unwrap().clone()
    }

    /// Clears the assigned session (the client re-initializes afterwards).
    pub fn clear_session(&self) {
        *self.session.lock().unwrap() = None;
    }

    /// POSTs one JSON-RPC request and returns the matching response, reading
    /// either a direct JSON body or an SSE stream.
    pub async fn request(&self, req: &MCPRequest) -> Result<MCPResponse, MCPError> {
        let body = serde_json::to_value(req)
            .map_err(|e| MCPError::new(-32603, format!("failed to encode request: {e}")))?;
        let expected_id = req.id;

        let mut attempt = 0u8;
        loop {
            let mut token_used: Option<String> = None;
            match self.exchange(&body, &mut token_used).await? {
                Exchange::Accepted => {
                    return Err(MCPError::new(
                        -32000,
                        "server answered a JSON-RPC request with HTTP 202 (no response)",
                    ));
                }
                Exchange::Unauthorized { challenge_header } => {
                    let retryable = self.token_provider.is_some() && attempt == 0;
                    if retryable {
                        if let (Some(provider), Some(token)) =
                            (&self.token_provider, token_used.as_deref())
                        {
                            provider.invalidate(token).await;
                        }
                        attempt += 1;
                        continue;
                    }
                    return Err(unauthorized_error(challenge_header.as_deref()));
                }
                Exchange::Payload(resp) => {
                    return self.read_payload(resp, Some(expected_id)).await;
                }
            }
        }
    }

    /// POSTs a JSON-RPC notification: HTTP 202 (or an accepted 200 stream the
    /// server closes without a response) is success.
    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), MCPError> {
        let body = notification_message(method, params);
        let mut attempt = 0u8;
        loop {
            let mut token_used: Option<String> = None;
            match self.exchange(&body, &mut token_used).await? {
                Exchange::Accepted => return Ok(()),
                Exchange::Unauthorized { challenge_header } => {
                    let retryable = self.token_provider.is_some() && attempt == 0;
                    if retryable {
                        if let (Some(provider), Some(token)) =
                            (&self.token_provider, token_used.as_deref())
                        {
                            provider.invalidate(token).await;
                        }
                        attempt += 1;
                        continue;
                    }
                    return Err(unauthorized_error(challenge_header.as_deref()));
                }
                Exchange::Payload(resp) => {
                    // A 200 response to a notification is legal only as an SSE
                    // stream carrying server requests/notifications; drain it
                    // and treat the absence of a request id as "no response".
                    return self.read_payload(resp, None).await.map(|_| ());
                }
            }
        }
    }

    /// Performs one POST, normalizing the HTTP status into [`Exchange`].
    async fn exchange(
        &self,
        body: &Value,
        token_used: &mut Option<String>,
    ) -> Result<Exchange, MCPError> {
        let mut builder = self
            .http
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        if let Some(session) = self.session_id() {
            builder = builder.header(SESSION_HEADER, session);
        }
        if let Some(provider) = &self.token_provider {
            let token = provider.token().await?;
            builder = builder.header("Authorization", format!("Bearer {token}"));
            *token_used = Some(token);
        }

        let resp = builder
            .json(body)
            .send()
            .await
            .map_err(|e| MCPError::new(-32000, format!("streamable HTTP POST failed: {e}")))?;

        let status = resp.status();
        // 202 precedes the general 2xx branch: Accepted carries no body and is
        // only valid for notifications.
        if status.as_u16() == 202 {
            self.capture_session(&resp);
            return Ok(Exchange::Accepted);
        }
        if status.is_success() {
            self.capture_session(&resp);
            return Ok(Exchange::Payload(resp));
        }
        if status.as_u16() == 401 {
            let challenge_header = resp
                .headers()
                .get(reqwest::header::WWW_AUTHENTICATE)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            return Ok(Exchange::Unauthorized { challenge_header });
        }
        if status.as_u16() == 404 {
            // Unknown/expired Mcp-Session-Id (or a non-MCP path): per spec the
            // client must re-initialize before retrying.
            return Err(MCPError::session_lost());
        }

        let snippet = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>();
        let message = match status.as_u16() {
            400 => format!("server rejected the Streamable HTTP request (HTTP 400): {snippet}"),
            405 => "endpoint does not accept POST (is it a Streamable HTTP MCP endpoint?)".into(),
            406 => {
                "server cannot honor Accept: application/json, text/event-stream (HTTP 406)".into()
            }
            429 => "rate limited by Streamable HTTP server (HTTP 429)".into(),
            other => format!("streamable HTTP POST failed: HTTP {other}: {snippet}"),
        };
        // 429 maps to the rate-limit class used across the crate (-32002);
        // everything else is a transport error (-32000).
        let code = if status.as_u16() == 429 {
            -32002
        } else {
            -32000
        };
        Err(MCPError::new(code, message))
    }

    /// Records the session id whenever the server (re)assigns one.
    fn capture_session(&self, resp: &reqwest::Response) {
        if let Some(value) = resp
            .headers()
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
        {
            if let Ok(mut session) = self.session.lock() {
                *session = Some(value.to_string());
            }
        }
    }

    /// Reads a 200 response body as either one JSON envelope or an SSE stream.
    ///
    /// With `expected_id = None` (notification POST) the body is drained and
    /// any request-shaped message is ignored — no request was outstanding.
    async fn read_payload(
        &self,
        resp: reqwest::Response,
        expected_id: Option<u64>,
    ) -> Result<MCPResponse, MCPError> {
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();

        match content_type.as_str() {
            "application/json" => {
                let response = resp.json::<MCPResponse>().await.map_err(|e| {
                    MCPError::new(-32700, format!("invalid JSON-RPC response: {e}"))
                })?;
                if let Some(want) = expected_id {
                    if response.id != Some(want) {
                        return Err(MCPError::new(
                            -32000,
                            format!(
                                "JSON-RPC response id {} does not match request id {want}",
                                response
                                    .id
                                    .map(|i| i.to_string())
                                    .unwrap_or_else(|| "null".into())
                            ),
                        ));
                    }
                }
                Ok(response)
            }
            "text/event-stream" => self.read_sse(resp, expected_id).await,
            other => Err(MCPError::new(
                -32000,
                format!("unexpected Streamable HTTP Content-Type '{other}'"),
            )),
        }
    }

    /// Consumes an SSE response, dispatching `message` events until the frame
    /// matching `expected_id` arrives (the server closes the stream right
    /// after).
    async fn read_sse(
        &self,
        resp: reqwest::Response,
        expected_id: Option<u64>,
    ) -> Result<MCPResponse, MCPError> {
        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut cursor = 0usize;

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| MCPError::new(-32000, format!("SSE stream failed: {e}")))?;
            buf.extend_from_slice(&chunk);

            // Parse every complete, newly arrived event block.
            while let Some((block_len, block)) = take_event_block(&buf[cursor..]) {
                cursor += block_len;
                let Some((event_type, data)) = parse_sse_event(&block) else {
                    continue;
                };
                if event_type != "message" {
                    // `ping` / custom events carry no JSON-RPC message.
                    continue;
                }
                let Ok(message) = serde_json::from_str::<Value>(&data) else {
                    log::trace!(
                        target: "lc_mcp::transport::streamable_http",
                        "SSE message event with non-JSON data: {data}"
                    );
                    continue;
                };
                let is_response = message.get("id").is_some()
                    && (message.get("result").is_some() || message.get("error").is_some());
                if !is_response {
                    // Server notification or server-initiated request: the
                    // client exposes no sampling/roots capabilities, so such
                    // frames are diagnostics-only on a POST response.
                    log::trace!(
                        target: "lc_mcp::transport::streamable_http",
                        "ignoring non-response SSE frame: {message}"
                    );
                    continue;
                }
                let response = serde_json::from_value::<MCPResponse>(message).map_err(|e| {
                    MCPError::new(-32700, format!("invalid SSE JSON-RPC frame: {e}"))
                })?;
                match expected_id {
                    Some(want) if response.id == Some(want) => return Ok(response),
                    Some(want) => log::trace!(
                        target: "lc_mcp::transport::streamable_http",
                        "SSE response id {:?} != {want}; continuing",
                        response.id
                    ),
                    None => continue,
                }
            }
            // Compact consumed prefixes periodically.
            if cursor > 0 && cursor == buf.len() {
                buf.clear();
                cursor = 0;
            } else if cursor > 4096 {
                buf.drain(..cursor);
                cursor = 0;
            }
        }

        match expected_id {
            Some(want) => Err(MCPError::new(
                -32000,
                format!("SSE stream closed without a response for request id {want}"),
            )),
            None => Ok(empty_accepted_response()),
        }
    }
}

/// Builds the -32001 error for an exhausted/unretryable 401, attaching the
/// parsed OAuth challenge as `data` for the embedding app.
fn unauthorized_error(challenge_header: Option<&str>) -> MCPError {
    let mut error = MCPError::new(
        MCP_ERROR_UNAUTHORIZED,
        "unauthorized: token rejected by Streamable HTTP server".to_string(),
    );
    if let Some(header) = challenge_header {
        if let Some(challenge) = OAuthChallenge::parse(header, Some("Bearer")) {
            error.data = serde_json::to_value(challenge).ok();
        }
    }
    error
}

/// A synthetic success response for notification POSTs whose SSE body carried
/// no request response (the caller discards it).
fn empty_accepted_response() -> MCPResponse {
    MCPResponse {
        jsonrpc: "2.0".to_string(),
        id: None,
        result: Some(Value::Null),
        error: None,
    }
}

/// Normalized HTTP outcome after status-code handling.
enum Exchange {
    /// HTTP 202: notification accepted, no body.
    Accepted,
    /// HTTP 401 with the raw `WWW-Authenticate` header (if present).
    Unauthorized { challenge_header: Option<String> },
    /// HTTP 200 with a JSON or SSE body to consume.
    Payload(reqwest::Response),
}

/// If `bytes` contains a complete SSE event block, returns its UTF-8 text
/// (without the blank-line terminator) and the number of bytes consumed.
///
/// SSE separates events with a blank line: `\n\n` or `\r\n\r\n`; the earliest
/// boundary wins.
fn take_event_block(bytes: &[u8]) -> Option<(usize, String)> {
    let lf = find_subsequence(bytes, b"\n\n");
    let crlf = find_subsequence(bytes, b"\r\n\r\n");
    // The CRLF boundary's first '\n' sits at `c + 1`, so compare against the
    // LF pair by that key to decide which boundary truly comes first.
    let (content_end, block_len) = match (lf, crlf) {
        (Some(l), Some(c)) if c < l => (c, c + 4),
        (Some(l), _) => (l, l + 2),
        (None, Some(c)) => (c, c + 4),
        (None, None) => return None,
    };
    // Tolerate the mixed "\r\n\n" separator by trimming the lone '\r'.
    let end = if content_end > 0 && bytes[content_end - 1] == b'\r' {
        content_end - 1
    } else {
        content_end
    };
    let text = String::from_utf8_lossy(&bytes[..end]).to_string();
    Some((block_len, text))
}

/// Parses one SSE event block into `(event_type, joined_data)` per the SSE
/// parsing rules: multiple `data:` lines join with `\n`, a missing `event`
/// field defaults to `"message"`, comment/blank/other fields are ignored.
fn parse_sse_event(block: &str) -> Option<(String, String)> {
    let mut event_type = "message".to_string();
    let mut data_lines: Vec<&str> = Vec::new();
    for raw_line in block.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => event_type = value.to_string(),
            "data" => data_lines.push(value),
            _ => {} // id/retry/unknown fields: not needed for MCP responses.
        }
    }
    if data_lines.is_empty() {
        return None;
    }
    Some((event_type, data_lines.join("\n")))
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_message_event() {
        let block = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}";
        let (event, data) = parse_sse_event(block).unwrap();
        assert_eq!(event, "message");
        assert!(data.contains("\"id\":1"));
    }

    #[test]
    fn defaults_event_type_and_handles_crlf() {
        let block = "data: hello\r\ndata: world\r";
        let (event, data) = parse_sse_event(block).unwrap();
        assert_eq!(event, "message");
        assert_eq!(data, "hello\nworld");
    }

    #[test]
    fn ignores_comments_and_id_fields() {
        let block = ": keepalive\nid: evt-42\ndata: x";
        let (event, data) = parse_sse_event(block).unwrap();
        assert_eq!(event, "message");
        assert_eq!(data, "x");
    }

    #[test]
    fn block_without_data_is_not_an_event() {
        assert!(parse_sse_event("event: ping").is_none());
        assert!(parse_sse_event("").is_none());
    }

    #[test]
    fn take_event_block_lf_and_crlf() {
        let (len, text) = take_event_block(b"data: a\n\nrest").unwrap();
        assert_eq!(len, "data: a\n\n".len());
        assert_eq!(text, "data: a");

        let (len2, text2) = take_event_block(b"data: b\r\n\r\nrest").unwrap();
        assert_eq!(len2, "data: b\r\n\r\n".len());
        assert_eq!(text2, "data: b");
    }

    #[test]
    fn no_event_block_until_blank_line() {
        assert!(take_event_block(b"data: a\n").is_none());
    }

    #[test]
    fn session_state_roundtrip() {
        let transport = StreamableHttpTransport::new("http://127.0.0.1:1/mcp");
        assert!(transport.session_id().is_none());
        *transport.session.lock().unwrap() = Some("sess-1".into());
        assert_eq!(transport.session_id().as_deref(), Some("sess-1"));
        transport.clear_session();
        assert!(transport.session_id().is_none());
    }

    #[test]
    fn unauthorized_error_attaches_challenge_data() {
        let err = unauthorized_error(Some(
            r#"Bearer resource_metadata="https://host.test/rm",scope="mcp""#,
        ));
        assert_eq!(err.code, MCP_ERROR_UNAUTHORIZED);
        let challenge = OAuthChallenge::from_error(&err).expect("challenge data");
        assert_eq!(
            challenge.resource_metadata.as_deref(),
            Some("https://host.test/rm")
        );
        assert!(err.data.is_some());
    }

    #[test]
    fn unauthorized_error_without_header_has_no_data() {
        let err = unauthorized_error(None);
        assert_eq!(err.code, MCP_ERROR_UNAUTHORIZED);
        assert!(OAuthChallenge::from_error(&err).is_none());
    }
}
