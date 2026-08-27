//! MCP SSE server: exposes `MCPServer` as an HTTP/SSE network service.
//!
//! `MCPServer` itself is a pure request handler (`handle_request`); this layer wires it to a real network so any
//! MCP client (`MCPClient::connect(MCPConfig::sse(...))` / Cursor / Claude Desktop etc.) can connect and call the
//! registered tools. The protocol follows client behavior:
//! - `GET /sse` → 200 + SSE long connection: first sends `event: endpoint` announcing the POST address,
//!   then periodically sends a `: keep-alive` heartbeat to hold the connection (a timed-out client disconnects
//!   and reconnects);
//! - `POST /message` → JSON-RPC requests with an `id` go to `MCPServer::handle_request`, processed and answered
//!   directly with JSON (clients prefer parsing the direct response); notifications without an `id` (such as
//!   `notifications/initialized`) are dispatched to `handle_notification` then answered with 202.
//!
//! Deployment: the caller is responsible for binding the listener — bind `127.0.0.1:0` for local debugging,
//! `0.0.0.0:PORT` for remote deployment, and pass in a `public_base` the clients can really reach
//! (see [`MCPServer::serve_sse`](crate::MCPServer::serve_sse)).

use crate::protocol::{MCPError, MCPRequest, MCPResponse};
use crate::server::MCPServer;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Serves MCP over SSE on an already-bound TCP listener, returning the SSE entry URL clients connect to.
///
/// - `listener`: a listener already bound to an address (tests bind `127.0.0.1:0`, deployment binds
///   `0.0.0.0:PORT`);
/// - `public_base`: the base address clients use to reach this server (e.g. `http://your-server-ip:8788`).
///   The POST address the server sends to clients is built from it; when deployed remotely it must be an address
///   clients can really reach, otherwise they cannot call back to POST after receiving the `endpoint` event.
///
/// Returns immediately after startup; the accept loop runs on a background task until the process exits.
pub(crate) fn serve(server: Arc<MCPServer>, listener: TcpListener, public_base: String) -> String {
    let sse_url = format!("{public_base}/sse");
    let message_url = format!("{public_base}/message");

    tokio::spawn(async move {
        loop {
            let (sock, _) = match listener.accept().await {
                Ok(sock) => sock,
                Err(_) => break,
            };
            let server = server.clone();
            let message_url = message_url.clone();
            tokio::spawn(async move {
                handle_connection(server, sock, &message_url).await;
            });
        }
    });

    sse_url
}

/// Handles one connection: dispatches by request line to the SSE long connection or POST handling.
async fn handle_connection(server: Arc<MCPServer>, mut sock: TcpStream, message_url: &str) {
    let (first_line, body) = read_http_request(&mut sock).await;
    if first_line.starts_with("GET ") {
        serve_sse_stream(&mut sock, message_url).await;
    } else if first_line.starts_with("POST ") {
        handle_post(&server, &mut sock, &body).await;
    }
}

/// SSE long connection: first sends 200 + the endpoint event (announcing the POST address), then holds with
/// periodic heartbeats.
async fn serve_sse_stream(sock: &mut TcpStream, message_url: &str) {
    if sock
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n",
        )
        .await
        .is_err()
    {
        return;
    }
    let endpoint = format!("event: endpoint\ndata: {message_url}\n\n");
    if sock.write_all(endpoint.as_bytes()).await.is_err() {
        return;
    }
    loop {
        if sock.write_all(b": keep-alive\n\n").await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// POST: requests with an `id` go to `MCPServer`, processed and answered directly with JSON; notifications
/// without an `id` are answered with 202.
async fn handle_post(server: &MCPServer, sock: &mut TcpStream, body: &str) {
    // Lenient parse: notifications have no id, requests have one, matching `serve_stdio`'s ServerMessage
    let msg: InboundMessage = match serde_json::from_str(body) {
        Ok(m) => m,
        Err(e) => {
            // JSON-RPC 2.0: when the request could not be parsed, the response id MUST be null
            let resp = MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: Some(MCPError::new(-32700, format!("parse error: {e}"))),
            };
            write_json_response(sock, &resp).await;
            return;
        }
    };

    let Some(id) = msg.id else {
        // Notification (no id): the client only cares whether it was sent → 202
        server.handle_notification(&msg.method, msg.params).await;
        let _ = sock
            .write_all(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n")
            .await;
        return;
    };

    let req = MCPRequest {
        jsonrpc: "2.0".to_string(),
        id,
        method: msg.method,
        params: msg.params,
    };
    let resp = server.handle_request(req).await;
    write_json_response(sock, &resp).await;
}

/// Wraps a JSON-RPC response as an HTTP 200 JSON response and writes it back.
async fn write_json_response(sock: &mut TcpStream, resp: &MCPResponse) {
    let resp_body = match serde_json::to_string(resp) {
        Ok(s) => s,
        Err(e) => format!(
            r#"{{"jsonrpc":"2.0","id":null,"error":{{"code":-32603,"message":"internal error: {e}"}}}}"#
        ),
    };
    let out = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        resp_body.len(),
        resp_body
    );
    let _ = sock.write_all(out.as_bytes()).await;
}

/// HTTP header cap: connections exceeding it are dropped (prevents slowloris from accumulating memory without bound).
const MAX_HEADER_SIZE: usize = 64 * 1024;
/// HTTP body cap: bodies exceeding the declared Content-Length are rejected.
const MAX_BODY_SIZE: usize = 1024 * 1024;
/// Single-read timeout: a client that stops sending bytes is disconnected.
const READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Reads one HTTP request, returning (request line, body).
///
/// Simplified implementation: first reads up to `\r\n\r\n`, then reads the full body by `Content-Length`.
/// With size caps and a read timeout: a malicious client that sends a few bytes and never finishes, or declares a
/// huge Content-Length and sends slowly, gets disconnected instead of accumulating without bound (DoS protection).
/// Any protection triggered returns empty strings; the caller closes the connection naturally when GET/POST does
/// not match.
async fn read_http_request(sock: &mut TcpStream) -> (String, String) {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = match tokio::time::timeout(READ_TIMEOUT, sock.read(&mut tmp)).await {
            Ok(Ok(n)) => n,
            // Read timeout / read error → disconnect
            _ => return (String::new(), String::new()),
        };
        if n == 0 {
            return (String::new(), String::new());
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > MAX_HEADER_SIZE {
            return (String::new(), String::new()); // header over the cap
        }
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
            // An oversized body declaration is rejected outright
            if content_length > MAX_BODY_SIZE {
                return (String::new(), String::new());
            }
            let body_start = pos + 4;
            let mut body: Vec<u8> = buf[body_start..].to_vec();
            while body.len() < content_length {
                let n = match tokio::time::timeout(READ_TIMEOUT, sock.read(&mut tmp)).await {
                    Ok(Ok(n)) => n,
                    _ => return (String::new(), String::new()),
                };
                if n == 0 {
                    return (String::new(), String::new());
                }
                body.extend_from_slice(&tmp[..n]);
            }
            let first_line = header.lines().next().unwrap_or("").to_string();
            let body = String::from_utf8_lossy(&body[..body.len().min(content_length)]).to_string();
            return (first_line, body);
        }
    }
}

/// A lenient inbound message: notifications have no id
#[derive(Deserialize)]
struct InboundMessage {
    #[serde(default)]
    id: Option<u64>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    /// Builds a loopback client/server connection pair.
    async fn tcp_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, server) = tokio::join!(TcpStream::connect(addr), listener.accept(),);
        (client.unwrap(), server.unwrap().0)
    }

    #[tokio::test]
    async fn oversized_header_is_rejected() {
        let (mut client, mut server) = tcp_pair().await;
        client.set_nodelay(true).unwrap();

        // Send a header over MAX_HEADER_SIZE with no \r\n\r\n:
        // read_http_request should return empty immediately once the buffer exceeds the cap, not accumulate.
        let junk = vec![b'a'; MAX_HEADER_SIZE + 16];
        let (write_res, read_res) =
            tokio::join!(client.write_all(&junk), read_http_request(&mut server));
        write_res.unwrap();
        assert_eq!(read_res, (String::new(), String::new()));
    }

    #[tokio::test]
    async fn normal_request_is_parsed() {
        let (mut client, mut server) = tcp_pair().await;
        client.set_nodelay(true).unwrap();

        let req = b"POST /message HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        let (write_res, read_res) =
            tokio::join!(client.write_all(req), read_http_request(&mut server));
        write_res.unwrap();
        assert_eq!(
            read_res,
            ("POST /message HTTP/1.1".to_string(), "{}".to_string())
        );
    }
}
