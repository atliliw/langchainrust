// lc-providers/tests/common/mod.rs
//! Offline cassette-playback harness (T4, v0.23.0) + T1 (v0.25.0) wire-fixture recorder.
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
//! provider **sends** — not just how it maps responses.
//!
//! T1 (v0.25.0) upgrades the cassette for wire-fixture capture/replay:
//! - [`MockResponse`] gains arbitrary response `headers` + builders
//!   ([`retry_after_seconds`]/[`retry_after_date`]/[`mcp_protocol_version`]) so
//!   retry/`Retry-After`/MCP-version behaviour is assertable offline.
//! - Routing becomes per-path a [`RoutePattern`]: a `Fixed` response repeats on
//!   every hit; a `Seq` is consumed one entry per request (retry sequences,
//!   Chroma multi-page `clear`) and, once exhausted, answers `501 no fixture`.
//! - A fixture loader ([`load_fixture`]) reads `tests/fixtures/<provider>/<case>.json`
//!   envelopes ("{status, content_type, headers, body}") written by [`capture_fixture`],
//!   which sanitizes the upstream payload (drops auth headers, redacts key-shaped
//!   tokens) so recorded wire bytes are safe to commit. T1's live capture keeps the
//!   real provider behind env-gated `#[ignore]` tests; CI only replays committed
//!   fixtures.
//!
//! `llc-testkit` exposes the same loopback + `assert_json_canonical` helpers as
//! `lc_testkit::wire` for the embeddings/vector-stores/rag crates to reuse. A
//! dev-dependency cycle forbids lc-providers from importing lc-testkit, so this
//! harness keeps its own local copy of the server (see the note in `lc_testkit::wire`).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// A scripted HTTP response for one request path.
#[derive(Clone, Debug)]
pub struct MockResponse {
    pub status: u16,
    pub content_type: &'static str,
    /// Extra response headers (T1). Emitted verbatim by the loopback responder.
    pub headers: Vec<(String, String)>,
    pub body: String,
}

/// A missing-fixture response: any request with no route (or an exhausted `Seq`)
/// answers `501` with this body, so a test that forgot to add a route fails loudly.
pub fn no_fixture() -> MockResponse {
    MockResponse::json(501, r#"{"error":"no fixture"}"#)
}

impl MockResponse {
    /// A JSON response (e.g. a non-streaming completion).
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "application/json",
            headers: Vec::new(),
            body: body.into(),
        }
    }

    /// An SSE response (a streaming completion) — the body is the raw
    /// `data: {...}\n\n` event stream as the provider would receive it.
    pub fn sse_stream(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            headers: Vec::new(),
            body: body.into(),
        }
    }

    /// Appends an arbitrary response header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Adds `Retry-After: <secs>` (HTTP date-seconds form).
    pub fn retry_after_seconds(mut self, secs: u64) -> Self {
        self.headers.push(("Retry-After".into(), format!("{secs}")));
        self
    }

    /// Adds `Retry-After: <IMF-fixdate>` (e.g. `Wed, 21 Oct 2015 07:28:00 GMT`).
    pub fn retry_after_date(mut self, date: impl Into<String>) -> Self {
        self.headers.push(("Retry-After".into(), date.into()));
        self
    }

    /// Adds a protocol header (e.g. `MCP-Protocol-Version: 2025-06-18`).
    pub fn with_protocol_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// How a route serves repeated requests.
#[derive(Clone, Debug)]
pub enum RoutePattern {
    /// Same response every hit (idempotent endpoints, e.g. `/models`).
    Fixed(MockResponse),
    /// One response per request, consumed in order. Exhausted → `501 no fixture`,
    /// signalling the test forgot enough canned responses for a retry sequence.
    Seq(VecDeque<MockResponse>),
}

/// Request path → serving pattern. A mutex guards `Seq` consumption.
pub type Routes = Arc<Mutex<HashMap<String, RoutePattern>>>;

/// The most recent request body the capture server observed (T5).
pub type CapturedBody = Arc<Mutex<Option<String>>>;

pub mod routes {
    use super::{Arc, MockResponse, Mutex, RoutePattern, Routes};
    use std::collections::HashMap;

    /// Builds a route map where each path serves a single fixed response.
    /// Semantics unchanged from v0.23: the same response repeats on every hit.
    pub fn build(pairs: Vec<(String, MockResponse)>) -> Routes {
        let map = HashMap::from_iter(pairs.into_iter().map(|(p, r)| (p, RoutePattern::Fixed(r))));
        Arc::new(Mutex::new(map))
    }

    /// Builds a route map from a single path whose responses are served one per
    /// request (retry sequences, Chroma multi-page `clear`, `429 → 200`).
    pub fn build_seq(path: impl Into<String>, seq: Vec<MockResponse>) -> Routes {
        Arc::new(Mutex::new(HashMap::from([(
            path.into(),
            RoutePattern::Seq(seq.into()),
        )])))
    }

    /// Adds a fixed route to an existing map (compose build + build_seq).
    pub fn add_fixed(routes: &Routes, path: impl Into<String>, resp: MockResponse) {
        routes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(path.into(), RoutePattern::Fixed(resp));
    }

    /// Adds a one-shot sequence route to an existing map.
    pub fn add_seq(routes: &Routes, path: impl Into<String>, seq: Vec<MockResponse>) {
        routes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(path.into(), RoutePattern::Seq(seq.into()));
    }
}

/// A bound mock server plus its request-body capture handle (T5).
pub struct MockServer {
    pub addr: std::net::SocketAddr,
    pub last_body: CapturedBody,
}

/// Binds a loopback listener WITHOUT body capture and spawns the responder.
/// Returns the bound address.
pub async fn spawn_mock_server(routes: Routes) -> std::net::SocketAddr {
    spawn_mock_server_capture(routes).await.addr
}

/// Binds a loopback listener that ALSO records the most recent request body.
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

fn dispatch(routes: &Routes, path: &str) -> MockResponse {
    let mut map = routes.lock().unwrap_or_else(|e| e.into_inner());
    match map.get_mut(path) {
        Some(RoutePattern::Fixed(r)) => r.clone(),
        Some(RoutePattern::Seq(q)) => q.pop_front().unwrap_or_else(no_fixture),
        None => no_fixture(),
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
            // Consume any body bytes already inside `buf`, then drain the rest.
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

    let resp = dispatch(routes, &path);
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
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {len}\r\n",
        status = resp.status,
        reason = reason,
        ct = resp.content_type,
        len = resp.body.len(),
    );
    for (name, value) in &resp.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("Connection: close\r\n\r\n");
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(resp.body.as_bytes()).await?;
    sock.flush().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// T1 fixture envelopes + sanitization
// ---------------------------------------------------------------------------

/// Fixture envelope shape:
/// ```json
/// { "status": 200, "content_type": "application/json",
///   "headers": { "Retry-After": "1" }, "body": "..." }
/// ```
/// Fixtures live at `tests/fixtures/<provider>/<case>.json`(crate-root-relative,
/// matching the test runner's cwd). They are plain `MockResponse`s captured from a
/// real provider and sanitized, so CI replay is hermetic with no API key.
///
/// Loads a committed fixture, panicking with a descriptive message if absent.
pub fn load_fixture(provider: &str, case: &str) -> MockResponse {
    load_fixture_opt(provider, case)
        .unwrap_or_else(|| panic!("fixture missing: tests/fixtures/{provider}/{case}.json"))
}

/// Fixture loader returning `None` when the file is absent (replay tests that must
/// pass before a fixture is recorded — T1 lets the user record in a later step).
pub fn load_fixture_opt(provider: &str, case: &str) -> Option<MockResponse> {
    let path = format!("tests/fixtures/{provider}/{case}.json");
    let raw = std::fs::read_to_string(&path).ok()?;
    let env: FixtureEnvelope =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("malformed fixture {path}: {e}"));
    Some(env.into_mock())
}

/// Sanitizing writer: serializes a `MockResponse` into a fixture envelope upon a
/// once-only live capture. Drops credential headers and redacts key-shaped tokens
/// from the body so recorded bytes are safe to commit.
pub fn capture_fixture(provider: &str, case: &str, resp: &MockResponse) -> std::io::Result<()> {
    let mut clean = resp.clone();
    clean.headers.retain(|(name, _)| !is_auth_header(name));
    clean.body = sanitize_body(&clean.body);
    let dir = format!("tests/fixtures/{provider}");
    std::fs::create_dir_all(&dir)?;
    let path = format!("{dir}/{case}.json");
    let env = FixtureEnvelope::from_mock(&clean);
    let json = serde_json::to_string_pretty(&env)?;
    std::fs::write(path, json)?;
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FixtureEnvelope {
    status: u16,
    content_type: String,
    /// Nullable so `{content_type:""}` in hand-written fixtures still round-trips.
    #[serde(default)]
    headers: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    body: String,
}

impl FixtureEnvelope {
    fn from_mock(r: &MockResponse) -> Self {
        let headers = r
            .headers
            .iter()
            .cloned()
            .collect::<std::collections::BTreeMap<_, _>>();
        Self {
            status: r.status,
            content_type: r.content_type.to_string(),
            headers,
            body: r.body.clone(),
        }
    }

    fn into_mock(self) -> MockResponse {
        let headers = self.headers.into_iter().collect::<Vec<_>>();
        MockResponse {
            status: self.status,
            content_type: Box::leak(self.content_type.into_boxed_str()),
            headers,
            body: self.body,
        }
    }
}

/// True for header names that must never be persisted (secrets live only in env).
fn is_auth_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "authorization" | "proxy-authorization" | "x-api-key" | "api-key" | "cookie"
    )
}

/// Best-effort secret redaction for captured bodies: `sk-…` and `Bearer …` tokens,
/// plus a bare hit against the crates' common key env vars. Diesel-free single
/// scan; the wire-recording checklist (docs/internal/v0.25.0/T1_FIXTURE_RECORDING.md)
/// still mandates a full regex sweep before committing — this is a first line, not
/// a guarantee.
fn sanitize_body(body: &str) -> String {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < bytes.len() {
        if body[i..].starts_with("sk-") {
            let start = i + 3;
            let mut j = start;
            while j < bytes.len() && bytes[j].is_ascii_alphanumeric() {
                j += 1;
            }
            if j - start >= 8 {
                out.push_str("[REDACTED]");
            } else {
                out.push_str(&body[i..j]); // not long enough to be a key; keep as-is
            }
            i = j;
        } else if body[i..].starts_with("Bearer ") {
            out.push_str("Bearer ");
            let mut j = i + 7;
            while j < bytes.len() && bytes[j] != b' ' && bytes[j] != b'\n' && bytes[j] != b'\r' {
                j += 1;
            }
            if j - (i + 7) >= 8 {
                out.push_str("[REDACTED]");
            } else {
                out.push_str(&body[i + 7..j]);
            }
            i = j;
        } else {
            // advance one UTF-8 scalar (guard Chinese/japanese caption text)
            let ch = body[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"] {
        if let Ok(v) = std::env::var(var) {
            if v.len() >= 8 {
                out = out.replace(&v, "[REDACTED]");
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Same helpers as load_fixture/capture_fixture but with an explicit root — used by
// the harness self-check above and by live-capture tests that stage to temp dirs.
// ---------------------------------------------------------------------------
// Live-capture forwarder (the T1 "recorder" the user runs with a real key)
// ---------------------------------------------------------------------------

/// A loopback port that, per request, relays to a real upstream provider
/// (preserving method/path/less-credential headers/body), captures the real
/// response, sanitizes it, writes a fixture envelope to
/// `tests/fixtures/<provider>/<case>.json`, and returns the raw upstream bytes to
/// the caller. Configure a provider's `base_url` to [`addr`](LiveRecorder) and
/// the cassette records the wire shape exactly as the client sees it. It is
/// invoked only by env-gated `#[ignore]` recording tests — CI never runs it.
pub struct LiveRecorder {
    pub addr: std::net::SocketAddr,
}

pub async fn spawn_live_recorder(upstream: &str, provider: &str, case: &str) -> LiveRecorder {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback for live capture");
    let addr = listener.local_addr().expect("resolved local addr");
    let upstream = upstream.to_string();
    let provider = provider.to_string();
    let case = case.to_string();
    tokio::spawn(async move { record_loop(listener, upstream, provider, case).await });
    LiveRecorder { addr }
}

async fn record_loop(listener: TcpListener, upstream: String, provider: String, case: String) {
    loop {
        let Ok((mut sock, _)) = listener.accept().await else {
            return;
        };
        let upstream = upstream.clone();
        let provider = provider.clone();
        let case = case.clone();
        tokio::spawn(async move {
            if let Err(e) = record_one(&mut sock, &upstream, &provider, &case).await {
                eprintln!("live-capture relay error: {e}");
            }
        });
    }
}

async fn record_one(
    sock: &mut TcpStream,
    upstream: &str,
    provider: &str,
    case: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // --- parse the inbound request (method, path, headers, body) ---
    let mut buf = Vec::new();
    let mut scratch = [0u8; 1024];
    let mut method = "POST".to_string();
    let mut path = "/".to_string();
    let mut inbound_headers = Vec::<(String, String)>::new();
    // Assigned inside the loop on every `break` path; `return Ok(())` diverges,
    // so after the loop `header_end` is always initialized.
    let header_end;
    loop {
        let n = sock.read(&mut scratch).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&scratch[..n]);
        if let Some(h) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            header_end = h + 4;
            let head = String::from_utf8_lossy(&buf[..h]);
            if let Some(req_line) = head.lines().next() {
                let mut it = req_line.split_whitespace();
                method = it.next().unwrap_or("POST").to_string();
                path = it.next().unwrap_or("/").to_string();
            }
            for line in head.lines().skip(1) {
                if let Some((k, v)) = line.split_once(':') {
                    inbound_headers.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
            // drain body
            let mut have = buf.len() - header_end;
            let cl: usize = head
                .lines()
                .find_map(|l| {
                    let lower = l.to_ascii_lowercase();
                    lower
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap_or(0))
                })
                .unwrap_or(0);
            while have < cl {
                let n = sock.read(&mut scratch).await?;
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&scratch[..n]);
                have = buf.len() - header_end;
            }
            break;
        }
    }
    let body = String::from_utf8_lossy(&buf[header_end..]).to_string();

    // --- forward to upstream with a fresh, proxy-free client ---
    let client = reqwest::Client::builder().no_proxy().build()?;
    let url = format!("{upstream}{path}");
    let mut reqb = match method.as_str() {
        "GET" => client.get(&url),
        _ => client.post(&url),
    };
    for (k, v) in &inbound_headers {
        if !is_auth_header(k)
            && !k.eq_ignore_ascii_case("host")
            && !k.eq_ignore_ascii_case("content-length")
        {
            reqb = reqb.header(k, v);
        }
    }
    let req = reqb.body(body).build()?;
    let resp = client.execute(req).await?;
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| "application/json".to_string());
    let extra_headers = resp
        .headers()
        .iter()
        .filter(|(k, _)| !is_auth_header(k.as_str()))
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect::<Vec<_>>();
    let bytes = resp.bytes().await?;
    let body_str = String::from_utf8_lossy(&bytes).to_string();

    let recorded = MockResponse {
        status,
        content_type: Box::leak(content_type.clone().into_boxed_str()),
        headers: extra_headers,
        body: body_str.clone(),
    };
    capture_fixture(provider, case, &recorded)?;

    // --- return the raw upstream response to the provider client ---
    let reason = "OK";
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {len}\r\n",
        status = status,
        reason = reason,
        ct = content_type,
        len = body_str.len(),
    );
    for (k, v) in &recorded.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("Connection: close\r\n\r\n");
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(body_str.as_bytes()).await?;
    sock.flush().await?;
    Ok(())
}

fn load_fixture_with_root(
    root: &std::path::Path,
    provider: &str,
    case: &str,
) -> Option<MockResponse> {
    let dir = root.join("tests/fixtures");
    let dir = dir.join(provider);
    let raw = std::fs::read_to_string(dir.join(format!("{case}.json"))).ok()?;
    let env: FixtureEnvelope = serde_json::from_str(&raw).ok()?;
    Some(env.into_mock())
}

fn capture_fixture_with_root(
    root: &std::path::Path,
    provider: &str,
    case: &str,
    resp: &MockResponse,
) -> std::io::Result<()> {
    let mut clean = resp.clone();
    clean.headers.retain(|(name, _)| !is_auth_header(name));
    clean.body = sanitize_body(&clean.body);
    let dir = root.join("tests/fixtures").join(provider);
    std::fs::create_dir_all(&dir)?;
    let json = serde_json::to_string_pretty(&FixtureEnvelope::from_mock(&clean))?;
    std::fs::write(dir.join(format!("{case}.json")), json)
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Tests for the harness itself (offline, no network)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod harness_tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    async fn request(addr: std::net::SocketAddr, path: &str) -> (u16, String) {
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        let mut read = [0u8; 2048];
        loop {
            let n = tokio::io::AsyncReadExt::read(&mut s, &mut read)
                .await
                .unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&read[..n]);
        }
        let text = String::from_utf8_lossy(&buf);
        // "HTTP/1.1 <code> <reason>" — grab the three-digit status code.
        let status = text
            .get(9..12)
            .and_then(|c| c.parse::<u16>().ok())
            .unwrap_or(0);
        let body_at = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(text.len());
        (status, text[body_at..].to_string())
    }

    #[tokio::test]
    async fn seq_serves_in_order_then_501() {
        let routes = routes::build_seq(
            "/chat",
            vec![
                MockResponse::json(429, "{}").retry_after_seconds(1),
                MockResponse::json(200, r#"{"ok":true}"#),
            ],
        );
        let addr = spawn_mock_server(routes).await;
        assert_eq!(request(addr, "/chat").await.0, 429);
        assert_eq!(request(addr, "/chat").await.0, 200);
        assert_eq!(request(addr, "/chat").await.0, 501); // exhausted
    }

    #[tokio::test]
    async fn seq_then_fixed_compose() {
        // Compose one route map from a fixed + a one-shot sequence, exercising the
        // add_fixed / add_seq mutators.
        let routes = routes::build(vec![("/ping".into(), MockResponse::json(200, "pong"))]);
        routes::add_seq(
            &routes,
            "/chat",
            vec![
                MockResponse::json(429, "{}").retry_after_seconds(1),
                MockResponse::json(200, r#"{"ok":true}"#),
            ],
        );
        routes::add_fixed(
            &routes,
            "/models",
            MockResponse::json(200, r#"{"model":"gpt"}"#),
        );
        let addr = spawn_mock_server(routes).await;
        assert_eq!(request(addr, "/ping").await.1, "pong");
        assert_eq!(request(addr, "/chat").await.0, 429);
        assert_eq!(request(addr, "/chat").await.0, 200);
        assert_eq!(request(addr, "/chat").await.0, 501);
        assert_eq!(request(addr, "/models").await.1, r#"{"model":"gpt"}"#);
    }

    #[tokio::test]
    async fn fixed_repeats() {
        let routes = routes::build(vec![("/ping".into(), MockResponse::json(200, "pong"))]);
        let addr = spawn_mock_server(routes).await;
        assert_eq!(request(addr, "/ping").await.0, 200);
        assert_eq!(request(addr, "/ping").await.0, 200);
        assert_eq!(request(addr, "/ping").await.1, "pong");
    }

    #[tokio::test]
    async fn custom_headers_emitted() {
        let routes = routes::build(vec![(
            "/mcp".into(),
            MockResponse::sse_stream("data: x")
                .with_protocol_header("MCP-Protocol-Version", "2025-06-18")
                .retry_after_date("Wed, 21 Oct 2015 07:28:00 GMT"),
        )]);
        let addr = spawn_mock_server(routes).await;
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        let req = "GET /mcp HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".to_string();
        s.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        let mut read = [0u8; 2048];
        loop {
            let n = tokio::io::AsyncReadExt::read(&mut s, &mut read)
                .await
                .unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&read[..n]);
        }
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("MCP-Protocol-Version: 2025-06-18"), "{text}");
        assert!(
            text.contains("Retry-After: Wed, 21 Oct 2015 07:28:00 GMT"),
            "{text}"
        );
        assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
    }

    #[cfg(test)]
    #[test]
    fn fixture_roundtrip_and_sanitize() {
        // sanitize_body: key-shaped tokens redacted, non-tokens untouched.
        let dirty = r#"{"roles":{"assistant"},"key":"sk-abcdef1234567890","msg":"Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0In0"}"#;
        let clean = sanitize_body(dirty);
        assert!(!clean.contains("sk-abcdef1234567890"), "{clean}");
        assert!(clean.contains("Bearer [REDACTED]"), "{clean}");
        assert!(clean.contains("\"msg\""), "{clean}"); // surrounding json kept

        // capture_fixture → load_fixture round-trips status/content_type/headers/body.
        let temp = std::env::temp_dir();
        let provider = "harness_selfcheck".to_string();
        let case = "roundtrip";
        let mut dir = temp.join("lc_providers_fixture_selfcheck");
        dir.push(&provider);
        // point the loader/writer at the temp dir via an override var used by the
        // harness; default path stays crate-relative for real fixtures.
        std::env::set_var("LC_FIXTURE_ROOT", dir.clone());
        let resp = MockResponse::json(200, r#"{"done":true}"#)
            .with_header("x-random", "v")
            .retry_after_seconds(5);
        capture_fixture_with_root(&dir, &provider, case, &resp).unwrap();
        let loaded = load_fixture_with_root(&dir, &provider, case);
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.status, 200);
        assert_eq!(loaded.body, r#"{"done":true}"#);
        assert!(loaded
            .headers
            .iter()
            .any(|(k, v)| k == "Retry-After" && v == "5"));
        let _ = std::fs::remove_dir_all(temp.join("lc_providers_fixture_selfcheck"));
    }
}
