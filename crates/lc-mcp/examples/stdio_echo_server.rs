//! Spec-based demo MCP server over official stdio transport (B1, 0.22.4).
//!
//! A minimal, dependency-light Model Context Protocol server speaking
//! newline-delimited JSON-RPC 2.0 on stdin/stdout:
//!
//! - `initialize` → protocol/capability negotiation
//! - `notifications/initialized` → acknowledged (no response)
//! - `ping` → empty result
//! - `tools/list` → one tool, `echo`
//! - `tools/call` → `echo` echoes its `msg` argument; unknown tools get -32601
//!
//! All diagnostics go to **stderr** so stdout carries protocol frames only.
//!
//! Run it manually against any MCP stdio client, or use it as the fixture for
//! the official-SDK interop matrix:
//!
//! ```text
//! cargo run -p lc-mcp --example stdio_echo_server
//! ```
//!
//! Modes (used by the interop tests):
//! - `--die-after-init`: exit(0) immediately after replying to `initialize`,
//!   simulating a server crash to exercise connection-lost handling.
//! - `--negotiate-version <v>`: announce protocol version `v` in the
//!   `initialize` result regardless of what the client requested, simulating
//!   a server pinned to an incompatible version.

use std::io::{BufRead, BufReader, Write};

use serde_json::{json, Map, Value};

const PROTOCOL_VERSION: &str = "2024-11-05";

/// Protocol versions this fixture build implements.
const SERVER_PROTOCOL_VERSIONS: &[&str] = &[PROTOCOL_VERSION];

fn main() {
    let mut die_after_init = false;
    let mut forced_version: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--die-after-init" => die_after_init = true,
            "--negotiate-version" => forced_version = args.next(),
            _ => {}
        }
    }

    let mut stdin = BufReader::new(std::io::stdin()).lines();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    eprintln!("[echo-stdio] server starting (die_after_init={die_after_init})");

    while let Some(Ok(line)) = stdin.next() {
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[echo-stdio] parse error: {e}");
                send(
                    &mut out,
                    Value::Null,
                    Err((-32700, "Parse error".to_string())),
                );
                continue;
            }
        };

        let id = message.get("id").cloned();
        let method = message.get("method").and_then(|m| m.as_str()).unwrap_or("");

        // Notifications get no response.
        match method {
            "notifications/initialized" => {
                eprintln!("[echo-stdio] initialized notification");
                continue;
            }
            "notifications/cancelled" => continue,
            _ => {}
        }

        let Some(id) = id else {
            // Unknown notification: ignore.
            continue;
        };

        match method {
            "initialize" => {
                eprintln!("[echo-stdio] initialize");
                send(
                    &mut out,
                    id,
                    Ok(initialize_result(&message, forced_version.as_deref())),
                );
                let _ = out.flush();
                if die_after_init {
                    eprintln!("[echo-stdio] exiting after initialize (fixture mode)");
                    std::process::exit(0);
                }
            }
            "ping" => send(&mut out, id, Ok(json!({}))),
            "tools/list" => {
                eprintln!("[echo-stdio] tools/list");
                send(&mut out, id, Ok(json!({"tools": [echo_tool()]})));
            }
            "tools/call" => {
                let params = message.get("params").cloned().unwrap_or(Value::Null);
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                eprintln!("[echo-stdio] tools/call name={name}");
                if name != "echo" {
                    send(&mut out, id, Err((-32601, format!("unknown tool: {name}"))));
                    continue;
                }
                let msg = params
                    .get("arguments")
                    .and_then(|a| a.get("msg"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("");
                send(
                    &mut out,
                    id,
                    Ok(json!({
                        "content": [{"type": "text", "text": format!("echo: {msg}")}],
                        "isError": false,
                    })),
                );
            }
            other => {
                eprintln!("[echo-stdio] unknown method: {other}");
                send(
                    &mut out,
                    id,
                    Err((-32601, format!("Method not found: {other}"))),
                );
            }
        }
        let _ = out.flush();
    }

    eprintln!("[echo-stdio] stdin EOF, exiting");
}

/// Negotiates per the official MCP handshake: when the client's requested
/// version is one this server speaks, echo it; otherwise announce the
/// server's own pinned version and let the client decide. `forced_version`
/// overrides the result (incompatible-server fixture).
fn initialize_result(request: &Value, forced_version: Option<&str>) -> Value {
    let negotiated = match forced_version {
        Some(v) => v,
        None => {
            let requested = request
                .pointer("/params/protocolVersion")
                .and_then(|v| v.as_str())
                .unwrap_or(PROTOCOL_VERSION);
            // Echo the requested version when this build speaks it, otherwise
            // announce the pinned version (the client decides whether to
            // continue — mirroring real official-SDK servers).
            SERVER_PROTOCOL_VERSIONS
                .iter()
                .copied()
                .find(|v| *v == requested)
                .unwrap_or(PROTOCOL_VERSION)
        }
    };
    json!({
        "protocolVersion": negotiated,
        "capabilities": {"tools": {"listChanged": false}},
        "serverInfo": {"name": "echo-stdio", "version": "0.22.4"},
    })
}

fn echo_tool() -> Value {
    let mut schema = Map::new();
    schema.insert("type".into(), json!("object"));
    schema.insert(
        "properties".into(),
        json!({"msg": {"type": "string", "description": "message to echo"}}),
    );
    json!({
        "name": "echo",
        "description": "Echoes its msg argument back as text.",
        "inputSchema": Value::Object(schema),
    })
}

/// Writes one JSON-RPC frame (single line, newline terminated).
fn send(out: &mut impl Write, id: Value, result: Result<Value, (i32, String)>) {
    let payload = match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }),
    };
    let frame = serde_json::to_string(&payload).expect("frame serialization");
    let _ = writeln!(out, "{frame}");
}
