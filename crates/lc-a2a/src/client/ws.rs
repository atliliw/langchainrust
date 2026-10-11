//! A2A client WebSocket transport (V0.27, #186).
//!
//! Mirror of the SSE stream (`crate::client::sse`): reads serialized
//! [`TaskPushNotification`]s off a WebSocket connection opened against the
//! server's `/ws` endpoint. The server emits one text frame per filtered
//! notification with the exact same authz semantics as `/events`, so a caller
//! can swap transports without changing how it consumes events.
//!
//! `next()` auto-answers protocol ping frames so an idle subscription stays
//! alive, and deserializes each text frame into a notification — skipping
//! pongs and binary frames the server never sends on this stream.

#![cfg(feature = "ws")]

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::ClientRequestBuilder;
use tokio_tungstenite::tungstenite::http::Uri;
use tokio_tungstenite::tungstenite::protocol::Message as TungstenMessage;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use crate::protocol::TaskPushNotification;

use super::A2AError;

/// A live WebSocket subscription carrying [`TaskPushNotification`] frames.
///
/// Produced by [`crate::client::A2AClient::connect_ws`]; call [`Self::next`]
/// to pull each notification as it arrives.
pub struct A2AWebSocketStream {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl A2AWebSocketStream {
    /// Open a WebSocket subscription against `ws_url`, forwarding the client's
    /// bearer token and `traceparent` so the server's authz (same as `/events`)
    /// applies to this connection.
    pub(crate) async fn connect(
        ws_url: &str,
        auth_token: Option<&str>,
        traceparent: Option<String>,
    ) -> Result<Self, A2AError> {
        let uri: Uri = ws_url
            .parse()
            .map_err(|e| A2AError::Http(format!("invalid ws url {ws_url}: {e}")))?;
        let mut builder = ClientRequestBuilder::new(uri);
        if let Some(token) = auth_token {
            builder = builder.with_header("Authorization", format!("Bearer {token}"));
        }
        if let Some(tp) = traceparent {
            builder = builder.with_header("traceparent", tp);
        }
        let (ws, _resp) = connect_async(builder)
            .await
            .map_err(|e| A2AError::Http(format!("ws connect to {ws_url} failed: {e}")))?;
        Ok(Self { ws })
    }

    /// Pull the next notification, `None` when the server closes the stream.
    pub async fn next(&mut self) -> Option<Result<TaskPushNotification, A2AError>> {
        loop {
            let Some(ws_msg) = self.ws.next().await else {
                return None;
            };
            let msg = match ws_msg {
                Ok(msg) => msg,
                Err(e) => return Some(Err(A2AError::Http(format!("ws stream error: {e}")))),
            };
            let text = match msg {
                TungstenMessage::Text(frame) => frame.to_string(),
                // Keep an idle connection alive by answering protocol pings.
                TungstenMessage::Ping(payload) => {
                    let _ = self.ws.send(TungstenMessage::Pong(payload)).await;
                    continue;
                }
                // A clean close ends the subscription; treat like stream EOF.
                TungstenMessage::Close(_) => return None,
                // Pongs/binary frames carry no notification on this stream.
                _ => continue,
            };
            return Some(match serde_json::from_str::<TaskPushNotification>(&text) {
                Ok(notification) => Ok(notification),
                Err(e) => Err(A2AError::Parse(format!(
                    "failed to parse task notification: {e}"
                ))),
            });
        }
    }
}