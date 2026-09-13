//! B1 stdio interop matrix: the framework's official-track stdio client
//! (`StdioMcpClient`) talking to a spec-compliant MCP stdio server subprocess
//! (`examples/stdio_echo_server.rs`).
//!
//! The fixture is a plain JSON-RPC stdio server with no framework coupling —
//! the same role an official TS/Python SDK server plays in CI; swapping the
//! spawn target requires no test changes beyond [`echo_server`].

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use lc_core::tools::ToolError;
use lc_core::BaseTool;
use lc_mcp::{MCPToolAdapter, StdioCommand, StdioMcpClient, VersionPolicy, MCP_VERSION};

/// Resolves the pre-built example binary under the active target directory
/// (honors `CARGO_TARGET_DIR`, which this repo points outside the cloud-sync
/// folder during builds; falls back to the workspace `target/`).
fn fixture(exe: &str) -> PathBuf {
    let file_name = if cfg!(windows) {
        format!("{exe}.exe")
    } else {
        exe.to_string()
    };
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("target")
        });
    // `cargo test` builds examples under debug by default; allow release too.
    for profile in ["debug", "release"] {
        let candidate = target.join(profile).join("examples").join(&file_name);
        if candidate.exists() {
            return candidate;
        }
    }
    panic!(
        "example binary '{file_name}' not found under {target:?} — examples must be built before \
         the interop tests run"
    );
}

/// Spawn spec for the demo echo server.
fn echo_server() -> StdioCommand {
    StdioCommand::new(fixture("stdio_echo_server"))
}

#[tokio::test]
async fn initialize_handshake_negotiates_protocol() {
    let client = StdioMcpClient::connect(echo_server())
        .await
        .expect("handshake should complete");

    let info = client.protocol_info();
    assert_eq!(info.requested, MCP_VERSION);
    assert_eq!(info.negotiated, MCP_VERSION);
    assert_eq!(info.server_version, MCP_VERSION);
    assert!(info.supported, "server version must be in supported list");

    let server_info = client
        .initialize_result()
        .get("serverInfo")
        .expect("serverInfo");
    assert_eq!(
        server_info.get("name").and_then(|v| v.as_str()),
        Some("echo-stdio")
    );

    client.close().await.unwrap();
}

#[tokio::test]
async fn tools_list_and_tools_call_roundtrip() {
    let client = StdioMcpClient::connect(echo_server())
        .await
        .expect("handshake");

    let tools = client.list_tools().await.expect("tools/list");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");
    assert!(!tools[0].description.is_empty());

    let result = client
        .call_tool("echo", serde_json::json!({"msg": "hello stdio"}))
        .await
        .expect("tools/call");
    assert!(!result.is_error, "echo must not error: {result:?}");
    let text = result.text();
    assert!(text.contains("hello stdio"), "got: {text}");

    client.ping().await.expect("ping");
    client.close().await.unwrap();
}

#[tokio::test]
async fn unknown_tool_returns_jsonrpc_error() {
    let client = StdioMcpClient::connect(echo_server())
        .await
        .expect("handshake");

    let err = client
        .call_tool("does_not_exist", serde_json::json!({}))
        .await
        .expect_err("unknown tool must error");
    assert_eq!(err.code, -32601, "{err}");
    assert!(err.to_string().contains("unknown tool"), "{err}");

    // The session stays usable after a JSON-RPC-level error.
    let result = client
        .call_tool("echo", serde_json::json!({"msg": "after error"}))
        .await
        .expect("session survives an error response");
    assert!(result.text().contains("after error"));

    client.close().await.unwrap();
}

#[tokio::test]
async fn unknown_method_returns_method_not_found() {
    let client = StdioMcpClient::connect(echo_server())
        .await
        .expect("handshake");

    let err = client
        .send("prompts/list", None)
        .await
        .expect_err("server implements tools only");
    assert_eq!(err.code, -32601, "{err}");

    client.close().await.unwrap();
}

#[tokio::test]
async fn server_exit_after_initialize_is_connection_lost() {
    let command = StdioCommand::new(fixture("stdio_echo_server")).arg("--die-after-init");
    let client = StdioMcpClient::connect(command)
        .await
        .expect("initialize reply arrives before exit");

    // The child exits right after initialize; the next request must surface
    // the connection-lost class (-32000), not a timeout or hang.
    let err = client
        .send("tools/list", None)
        .await
        .expect_err("dead server must fail the request");
    assert_eq!(err.code, -32000, "{err}");
    assert!(err.is_connection_lost(), "{err}");

    client.close().await.unwrap(); // still idempotent on a dead child
}

#[tokio::test]
async fn version_reject_fails_handshake_on_incompatible_server() {
    let command =
        StdioCommand::new(fixture("stdio_echo_server")).args(["--negotiate-version", "1999-01-01"]);

    let err = StdioMcpClient::connect_with(command, VersionPolicy::Reject, Duration::from_secs(10))
        .await
        .expect_err("an unsupported negotiated version must fail under Reject");
    assert_eq!(err.code, lc_mcp::MCP_ERROR_VERSION_UNSUPPORTED, "{err}");
}

#[tokio::test]
async fn version_degrade_pins_library_version_and_keeps_working() {
    let command =
        StdioCommand::new(fixture("stdio_echo_server")).args(["--negotiate-version", "1999-01-01"]);

    let client =
        StdioMcpClient::connect_with(command, VersionPolicy::Degrade, Duration::from_secs(10))
            .await
            .expect("Degrade continues against an incompatible server");

    let info = client.protocol_info();
    assert_eq!(info.server_version, "1999-01-01");
    assert!(!info.supported);
    assert_eq!(info.negotiated, MCP_VERSION);

    // The pinned session is fully usable.
    let tools = client.list_tools().await.expect("tools/list after degrade");
    assert_eq!(tools.len(), 1);

    client.close().await.unwrap();
}

/// End-to-end on the stdio track: a spawned MCP subprocess tool is callable
/// through the same `BaseTool` adapter the rest of the framework uses, and the
/// A16 fail-closed execution gate applies regardless of transport.
#[tokio::test]
async fn stdio_tool_runs_through_base_tool_adapter() {
    let client = Arc::new(
        StdioMcpClient::connect(echo_server())
            .await
            .expect("handshake"),
    );
    let definition = client
        .list_tools()
        .await
        .expect("tools/list")
        .pop()
        .expect("one tool");
    assert_eq!(definition.name, "echo");

    // Fail-closed default (A16): no policy declared → refused before dispatch.
    let gated = MCPToolAdapter::from_client(client.clone(), definition.clone());
    let err = gated
        .run(serde_json::json!({"msg": "denied"}).to_string())
        .await
        .expect_err("the default gate must deny unattended stdio calls");
    assert!(
        matches!(err, ToolError::PermissionDenied(_)),
        "expected PermissionDenied, got {err:?}"
    );

    // Explicit unattended opt-in: the subprocess actually executes the tool.
    let adapter =
        MCPToolAdapter::from_client(client.clone(), definition).allow_unattended_execution();
    assert_eq!(adapter.name(), "echo");
    let out = adapter
        .run(serde_json::json!({"msg": "via adapter"}).to_string())
        .await
        .expect("tool runs over stdio");
    assert!(out.contains("via adapter"), "got: {out}");

    client.close().await.unwrap();
}

/// Namespaced stdio adapter: `server:tool` for the LLM, original name on wire.
#[tokio::test]
async fn stdio_namespaced_adapter_strips_prefix_on_wire() {
    let client = Arc::new(
        StdioMcpClient::connect(echo_server())
            .await
            .expect("handshake"),
    );
    let definition = client.list_tools().await.unwrap().pop().unwrap();

    let adapter = MCPToolAdapter::namespaced_from_client(client.clone(), "local", definition)
        .allow_unattended_execution();
    assert_eq!(adapter.name(), "local:echo");
    let out = adapter
        .run(serde_json::json!({"msg": "ns"}).to_string())
        .await
        .expect("namespaced call");
    assert!(out.contains("ns"), "got: {out}");

    client.close().await.unwrap();
}

#[tokio::test]
async fn request_timeout_is_enforced() {
    // The demo server answers everything; use a near-zero timeout against the
    // initialize call itself by connecting through the low-level API path:
    // connect_with with a 1ns timeout must fail rather than hang forever.
    // (The child is shut down on failure, so no orphan process remains.)
    let result = StdioMcpClient::connect_with(
        echo_server(),
        VersionPolicy::Degrade,
        Duration::from_nanos(1),
    )
    .await;
    let err = result.expect_err("1ns timeout cannot complete an initialize round trip");
    assert_eq!(err.code, lc_mcp::MCP_ERROR_REQUEST_TIMEOUT, "{err}");
}
