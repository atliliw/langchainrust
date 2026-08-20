//! MCP SSE 服务器:把 `MCPServer` 暴露为 HTTP/SSE 网络服务。
//!
//! `MCPServer` 本身是纯请求处理器(`handle_request`),这一层负责把它接到真实
//! 网络上,让任何 MCP 客户端(`MCPClient::connect(MCPConfig::sse(...))` / Cursor /
//! Claude Desktop 等)都能连上来调用注册的工具。协议按客户端行为实现:
//! - `GET /sse` → 200 + SSE 长连接:先发 `event: endpoint` 告知 POST 地址,
//!   再周期发 `: keep-alive` 心跳保持连接(客户端超时会断开重连);
//! - `POST /message` → 有 `id` 的 JSON-RPC 请求交给 `MCPServer::handle_request`
//!   处理并直接回 JSON(客户端优先解析直接响应);无 `id` 的通知(如
//!   `notifications/initialized`)分发给 `handle_notification` 后回 202。
//!
//! 部署场景:调用方负责绑好 listener——本地联调绑 `127.0.0.1:0`,
//! 远程部署绑 `0.0.0.0:PORT`,并把客户端真能访问的 `public_base` 传进来
//! (详见 [`MCPServer::serve_sse`](crate::MCPServer::serve_sse))。

use crate::protocol::{MCPError, MCPRequest, MCPResponse};
use crate::server::MCPServer;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// 在一个已绑定的 TCP listener 上提供 MCP SSE 服务,返回客户端要连的 SSE 入口 URL。
///
/// - `listener`:已绑定好地址的 listener(测试绑 `127.0.0.1:0`,部署绑 `0.0.0.0:PORT`);
/// - `public_base`:客户端访问本服务器的基地址(如 `http://your-server-ip:8788`)。
///   服务端发给客户端的 POST 地址由它拼出,部署在远程时必须是客户端真能访问的地址,
///   否则客户端收到 `endpoint` 事件后无法回连 POST。
///
/// 启动后立即返回,接收循环在后台任务里运行直到进程退出。
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

/// 处理一条连接:按请求行分派到 SSE 长连接或 POST 处理。
async fn handle_connection(server: Arc<MCPServer>, mut sock: TcpStream, message_url: &str) {
    let (first_line, body) = read_http_request(&mut sock).await;
    if first_line.starts_with("GET ") {
        serve_sse_stream(&mut sock, message_url).await;
    } else if first_line.starts_with("POST ") {
        handle_post(&server, &mut sock, &body).await;
    }
}

/// SSE 长连接:先发 200 + endpoint 事件(告知 POST 地址),再周期心跳保持。
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

/// POST:有 `id` 的请求交给 `MCPServer` 处理并直接回 JSON;无 `id` 的通知回 202。
async fn handle_post(server: &MCPServer, sock: &mut TcpStream, body: &str) {
    // 宽松解析:通知无 id、请求有 id,与 `serve_stdio` 的 ServerMessage 一致
    let msg: InboundMessage = match serde_json::from_str(body) {
        Ok(m) => m,
        Err(e) => {
            // JSON-RPC 2.0:请求无法解析时,响应 id 必须为 null
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
        // 通知(无 id):客户端只看发送是否成功 → 202
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

/// 把一个 JSON-RPC 响应包成 HTTP 200 JSON 响应写回。
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

/// 读取一个 HTTP 请求,返回 (请求行, body)。
///
/// 简化实现:先读到 `\r\n\r\n`,再按 `Content-Length` 读全 body。
async fn read_http_request(sock: &mut TcpStream) -> (String, String) {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = sock.read(&mut tmp).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
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
            let body_start = pos + 4;
            let mut body: Vec<u8> = buf[body_start..].to_vec();
            while body.len() < content_length {
                let n = sock.read(&mut tmp).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            let first_line = header.lines().next().unwrap_or("").to_string();
            let body = String::from_utf8_lossy(&body[..body.len().min(content_length)]).to_string();
            return (first_line, body);
        }
    }
    (String::new(), String::new())
}

/// 宽松入站消息:通知无 id
#[derive(Deserialize)]
struct InboundMessage {
    #[serde(default)]
    id: Option<u64>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}
