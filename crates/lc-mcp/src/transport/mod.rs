//! MCP transport layer: Stdio + SSE
//!
//! P0 fixes:
//! - P0-1: SSE changed from a one-shot `text()` body read to a long connection + background line-by-line
//!   streaming read, continuously consuming server-pushed events (progress/logging, etc.).
//! - P0-2: after a Stdio child-process crash, background monitoring + exponential-backoff auto-reconnect, with
//!   queryable connection state.

mod sse;
mod stdio;

pub use sse::SseTransport;
pub use stdio::StdioTransport;

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

use crate::protocol::{MCPError, MCPRequest, MCPResponse};
use async_trait::async_trait;

/// Default timeout for SSE endpoint discovery (30 seconds).
const SSE_DISCOVER_TIMEOUT: Duration = Duration::from_secs(30);

/// SSE heartbeat interval: when the read loop receives no data (including `: keep-alive` comment lines) within
/// this duration, the connection is judged broken and a reconnect is triggered (P1-2).
const SSE_HEARTBEAT: Duration = Duration::from_secs(30);

/// Child-process reconnect backoff cap (ms).
const MAX_RECONNECT_BACKOFF_MS: u64 = 30_000;

/// MCP transport-layer events (server pushes / connection state changes).
#[derive(Debug, Clone)]
pub enum MCPEvent {
    /// The connection has been established.
    Connected,
    /// The connection has been lost (child process exited / SSE interrupted).
    Disconnected,
    /// A message pushed by the server (SSE `event:`/`data:` lines).
    Message {
        /// The SSE event name, e.g. `logging`, `progress`.
        method: String,
        /// The parsed data (when it parses as JSON).
        params: Option<serde_json::Value>,
    },
}

/// MCP transport abstraction
#[async_trait]
pub trait MCPTransport: Send + Sync {
    /// Sends a request and waits for the response
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError>;
    /// Sends a notification (does not wait for a response)
    async fn notify(&self, method: &str, params: Option<serde_json::Value>)
        -> Result<(), MCPError>;
    /// Closes the connection
    async fn close(&self) -> Result<(), MCPError>;
    /// Whether the connection is alive (child process alive / SSE long connection held).
    fn is_connected(&self) -> bool;
    /// Reconnects and waits for recovery (triggered by upper layers after a disconnect).
    async fn reconnect(&self) -> Result<(), MCPError>;
    /// Subscribes to server-pushed events.
    fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent>;
}

/// Child-process reconnect exponential backoff: 0.5s → 1s → 2s → 4s → ... cap 30s.
fn backoff_delay(attempt: u32) -> Duration {
    let ms = 500u64
        .checked_shl(attempt.min(6))
        .unwrap_or(MAX_RECONNECT_BACKOFF_MS);
    Duration::from_millis(ms.min(MAX_RECONNECT_BACKOFF_MS))
}

/// In-process transport: connects a [`MCPClient`](crate::client::MCPClient) directly to an
/// [`MCPServer`](crate::MCPServer), without child process / network — convenient for embedded integration and
/// testing (P2-6).
///
/// Requests are handled in place via [`MCPServer::handle_request`](crate::MCPServer::handle_request);
/// notifications (`notifications/initialized` etc.) need no response and are ignored; the event channel only
/// broadcasts `Connected` once (no server pushes).
pub struct InMemoryTransport {
    server: Arc<crate::MCPServer>,
    event_tx: broadcast::Sender<MCPEvent>,
}

impl InMemoryTransport {
    /// Wraps an in-process MCP Server.
    pub fn new(server: Arc<crate::MCPServer>) -> Self {
        let (event_tx, _) = broadcast::channel(64);
        let _ = event_tx.send(MCPEvent::Connected);
        // P2-9 streaming tool output: subscribe to the server's `publish_partial`, turn each incremental
        // chunk into a `notifications/tool_partial` push event, which the client event listener routes to
        // `subscribe_tool_stream`. Silently dropped when no client is listening.
        let mut partial_rx = server.subscribe_partials();
        let fwd = event_tx.clone();
        tokio::spawn(async move {
            loop {
                match partial_rx.recv().await {
                    Ok(partial) => {
                        let params = serde_json::to_value(&partial).ok();
                        let evt = MCPEvent::Message {
                            method: "notifications/tool_partial".to_string(),
                            params,
                        };
                        if fwd.send(evt).is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
        Self { server, event_tx }
    }
}

#[async_trait]
impl MCPTransport for InMemoryTransport {
    async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
        Ok(self.server.handle_request(req).await)
    }

    async fn notify(
        &self,
        _method: &str,
        _params: Option<serde_json::Value>,
    ) -> Result<(), MCPError> {
        Ok(())
    }

    async fn close(&self) -> Result<(), MCPError> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        true
    }

    async fn reconnect(&self) -> Result<(), MCPError> {
        Ok(())
    }

    fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent> {
        self.event_tx.subscribe()
    }
}

#[cfg(test)]
mod tests;
