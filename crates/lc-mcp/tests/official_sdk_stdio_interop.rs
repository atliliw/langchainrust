//! B1 interop gate — our official stdio client (`StdioMcpClient`) against a
//! **real MCP server built on the official TypeScript SDK**
//! (`tests/official_sdk/ts_stdio_echo_server.mjs`).
//!
//! Ignored by default because it needs Node.js and the SDK installed:
//!
//! ```text
//! cd crates/lc-mcp/tests/official_sdk
//! npm install
//! # then, from the repo root:
//! cargo test -p lc-mcp --test official_sdk_stdio_interop -- --ignored --nocapture
//! ```
//!
//! `tests/official_sdk/run_interop.ps1` performs both directions
//! (official TS/Python clients ↔ our Streamable HTTP server, and this test).

use std::path::PathBuf;

use lc_mcp::{StdioCommand, StdioMcpClient};
use serde_json::json;

fn sdk_server_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("official_sdk")
        .join("ts_stdio_echo_server.mjs")
}

#[tokio::test]
#[ignore = "requires Node.js and `npm install` in tests/official_sdk; see README"]
async fn official_ts_sdk_stdio_server_roundtrip() {
    let script = sdk_server_script();
    assert!(
        script.exists(),
        "missing official-SDK fixture: {}",
        script.display()
    );

    // VersionPolicy::Degrade: the 2026 TS SDK announces a newer protocol
    // version than our 2024-11-05 pin; the handshake must still complete.
    let script_arg = script.to_str().expect("harness path is valid UTF-8");
    let client = StdioMcpClient::connect(StdioCommand::new("node").arg(script_arg))
        .await
        .expect("handshake with the official TS SDK stdio server");

    let server_info = client.initialize_result();
    assert_eq!(
        server_info
            .pointer("/serverInfo/name")
            .and_then(|v| v.as_str()),
        Some("ts-echo-stdio"),
        "initialize result carries the SDK server info: {server_info}"
    );

    let tools = client.list_tools().await.expect("tools/list");
    assert!(
        tools.iter().any(|t| t.name == "echo"),
        "official SDK server exposes its echo tool: {tools:?}"
    );

    let marker = "rust-client-marker-0xCAFE";
    let result = client
        .call_tool("echo", json!({"msg": marker}))
        .await
        .expect("tools/call");
    assert!(!result.is_error, "echo call flagged isError");
    assert!(
        result.text().contains(marker),
        "echo payload must contain {marker}: {0}",
        result.text()
    );

    client.ping().await.expect("ping against the SDK server");

    // Unknown tools surface the SDK server's JSON-RPC error verbatim.
    let err = client
        .call_tool("does-not-exist", json!({}))
        .await
        .expect_err("unknown tool must error");
    assert_eq!(
        err.code, -32601,
        "SDK server returns Method not found: {err}"
    );

    client.close().await.expect("shutdown");
}
