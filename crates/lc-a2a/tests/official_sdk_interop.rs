//! Official A2A SDK (`a2a-rs`) interop: a spec-compliant client drives OUR
//! server over its HTTP JSON-RPC endpoint.
//!
//! This test is gated behind the `official-sdk` feature (and a `[[test]]`
//! `required-features`) because `a2a-rs` requires rustc >= 1.96 — above the
//! workspace MSRV 1.85 and the local default toolchain. It runs in CI on a
//! newer toolchain; the default build never compiles it.
//!
//! Run: `cargo test -p lc-a2a --features official-sdk --test official_sdk_interop`
//!
//! The server here uses a stub chain (no model) so the round-trip is fully
//! offline. The point is wire compatibility: a2a-rs sends `message/send`, and
//! our server must route it (not answer `-32601` method_not_found).

#![cfg(feature = "official-sdk")]

use std::collections::HashMap;
use std::sync::Arc;

use a2a_rs::domain::{Message, SendCompletion};
use a2a_rs::{JsonRpcClient, Transport};
use lc_a2a::A2AServer;
use lc_chains::base::{BaseChain, ChainError, ChainResult};
use serde_json::Value;

/// A stub chain that echoes the input, so no model/network is needed.
struct EchoChain;

#[async_trait::async_trait]
impl BaseChain for EchoChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }
    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get("input").and_then(Value::as_str).unwrap_or("");
        Ok(HashMap::from([("output".to_string(), Value::String(input.to_string()))]))
    }
    fn name(&self) -> &str {
        "echo-chain"
    }
}

/// Start our `A2AServer` on an ephemeral port and return its base URL.
async fn start_server() -> String {
    let server = A2AServer::new(Arc::new(EchoChain));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = server.serve_on(listener).await;
    });
    format!("http://127.0.0.1:{port}")
}

/// The official client must be able to send a message to our server and receive
/// a task back — not a `method_not_found` JSON-RPC error.
#[tokio::test]
async fn official_sdk_client_drives_our_server() {
    let url = start_server().await;
    let client = JsonRpcClient::new(url);

    let task = client
        .send_task_message(
            None,
            &Message::user_text("hello".to_string(), "m1".to_string()),
            None,
            None,
            SendCompletion::WhenSettled,
        )
        .await
        .expect("a2a-rs client must reach our server and get a task back");

    assert!(!task.id.is_empty(), "task must carry an id");
    // EchoChain echoes the input; the task should settle to a completed state
    // (or at minimum be a task we can read back — proving the wire round-trip).
    let fetched = client
        .get_task(&task.id, None)
        .await
        .expect("a2a-rs client must be able to get_task from our server");
    assert_eq!(fetched.id, task.id);
}