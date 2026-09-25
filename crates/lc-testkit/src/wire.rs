// lc-testkit/src/wire.rs
//! A reusable loopback HTTP/1.1 mock server for wire-level replay (T1, v0.25.0).
//!
//! `lc-providers` keeps its own copy of this server in `tests/common/mod.rs`
//! (a dev-dependency cycle forbids lc-providers importing `lc_testkit`). The
//! embeddings / vector-stores / rag crates reuse **this** copy by adding
//! `lc-testkit = { path = "../lc-testkit", features = ["wire"] }` to their
//! `[dev-dependencies]`, then pointing a client's `base_url` at a spawned port.
//! Wire fixtures are recorded with the lc-providers recorder and replayed here
//! herd conventionally — see `docs/internal/v0.25.0/T1_FIXTURE_RECORDING.md`.
//!
//! Routing mirrors the provider cassette: a `Fixed` response repeats on each
//! request; a `Seq` is consumed one entry per request (retry sequences, multi-page
//! `clear`) and answers `501 no fixture` once exhausted. Extra headers travel so
//! `Retry-After`/`MCP-Protocol-Version` behaviour is assertable offline.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// A canned response served by the loopback server.
#[derive(Clone, Debug)]
pub struct WireResponse {
    pub status: u16,
    pub content_type: &'static str,
    /// Extra headers emitted verbatim (e.g. `Retry-After`, `MCP-Protocol-Version`).
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl WireResponse {
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "application/json",
            headers: Vec::new(),
            body: body.into(),
        }
    }

    pub fn sse_stream(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            headers: Vec::new(),
            body: body.into(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// How a path serves repeated requests. `pub` only because it appears in the
/// public [`WireRoutes`] alias; it is not part of the supported surface.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum Pattern {
    Fixed(WireResponse),
    Seq(VecDeque<WireResponse>),
}

/// Servable route map. Clone the [`Arc`] to share into `spawn_server`.
pub type WireRoutes = Arc<Mutex<HashMap<String, Pattern>>>;

/// Mirrors `lc_providers::tests::common::routes`.
pub fn fixed(pairs: Vec<(String, WireResponse)>) -> WireRoutes {
    Arc::new(Mutex::new(HashMap::from_iter(
        pairs.into_iter().map(|(p, r)| (p, Pattern::Fixed(r))),
    )))
}

/// Builds a map where one path serves a one-shot retry sequence.
pub fn seq(path: impl Into<String>, responses: Vec<WireResponse>) -> WireRoutes {
    Arc::new(Mutex::new(HashMap::from([(
        path.into(),
        Pattern::Seq(responses.into()),
    )])))
}

fn dispatch(routes: &WireRoutes, path: &str) -> WireResponse {
    let mut map = routes.lock().unwrap_or_else(|e| e.into_inner());
    match map.get_mut(path) {
        Some(Pattern::Fixed(r)) => r.clone(),
        Some(Pattern::Seq(q)) => q
            .pop_front()
            .unwrap_or_else(|| WireResponse::json(501, r#"{"error":"no fixture"}"#)),
        None => WireResponse::json(501, r#"{"error":"no fixture"}"#),
    }
}

/// Binds `127.0.0.1:0`, spawns the responder, returns the bound address.
pub async fn spawn_server(routes: WireRoutes) -> std::net::SocketAddr {
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback wire server");
    let addr = listener.local_addr().expect("resolved local addr");

    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let routes = routes.clone();
            tokio::spawn(async move {
                let _ = serve_conn(&mut sock, &routes).await;
            });
        }
    });

    addr
}

async fn serve_conn(sock: &mut tokio::net::TcpStream, routes: &WireRoutes) -> std::io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = Vec::new();
    let mut scratch = [0u8; 2048];
    let path;
    // Reads request headers; body is discarded (route key is the path).
    loop {
        let n = sock.read(&mut scratch).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&scratch[..n]);
        if let Some(h) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..h]);
            path = head
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("/")
                .to_string();
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
        503 => "Service Unavailable",
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    async fn get(addr: std::net::SocketAddr, path: &str) -> (u16, String) {
        let mut s = TcpStream::connect(addr).await.unwrap();
        let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await.unwrap();
        let mut all = Vec::new();
        let mut rd = [0u8; 2048];
        loop {
            let n = s.read(&mut rd).await.unwrap();
            if n == 0 {
                break;
            }
            all.extend_from_slice(&rd[..n]);
        }
        let text = String::from_utf8_lossy(&all);
        let status = text
            .get(9..12)
            .and_then(|c| c.parse::<u16>().ok())
            .unwrap_or(0);
        let body_at = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(text.len());
        (status, text[body_at..].to_string())
    }

    #[tokio::test]
    async fn fixed_repeats_and_missing_501() {
        let routes = fixed(vec![("/ping".into(), WireResponse::json(200, "pong"))]);
        let addr = spawn_server(routes).await;
        assert_eq!(get(addr, "/ping").await, (200, "pong".to_string()));
        assert_eq!(get(addr, "/ping").await, (200, "pong".to_string()));
        assert_eq!(get(addr, "/nope").await.0, 501);
    }

    #[tokio::test]
    async fn seq_consumed_then_501() {
        let routes = seq(
            "/chat",
            vec![
                WireResponse::json(429, "{}").with_header("Retry-After", "1"),
                WireResponse::json(200, r#"{"ok":true}"#),
            ],
        );
        let addr = spawn_server(routes).await;
        let (s1, _) = get(addr, "/chat").await;
        let (s2, _) = get(addr, "/chat").await;
        let (s3, _) = get(addr, "/chat").await;
        assert_eq!((s1, s2, s3), (429, 200, 501));
    }
}
