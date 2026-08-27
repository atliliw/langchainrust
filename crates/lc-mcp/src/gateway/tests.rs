use std::time::Duration;

use super::*;
use crate::sandbox::{ParamRule, ServerSandbox};
use crate::test_support::{start_fake_sse_server, PostMode};
use crate::tool_timeout::ToolSpec;
use crate::types::MCPConfig;
use serde_json::json;

/// Fixed-window rate limit: over-quota calls in a window are rejected, quota restores after the window expires.
#[test]
fn test_rate_limiter_window_blocks_then_recovers() {
    let mut rl = RateLimiter::new(2, Duration::from_millis(30));
    assert!(rl.allow());
    assert!(rl.allow());
    assert!(!rl.allow(), "3rd call within window should be rejected");
    assert_eq!(rl.remaining(), 0);
    std::thread::sleep(Duration::from_millis(50));
    assert!(rl.allow(), "quota should be restored after window expires");
}

/// The limiter allows at least 1 call.
#[test]
fn test_rate_limiter_min_one() {
    let mut rl = RateLimiter::new(0, Duration::from_secs(60));
    assert!(rl.allow(), "max_calls must be at least 1");
}

/// register is lazy: no connection, no tool pull; the unified registry stays empty.
#[tokio::test]
async fn test_register_is_lazy_and_empty_registry() {
    let gw = MCPGateway::new();
    gw.register(GatewayServerSpec::new(
        "bad",
        MCPConfig::stdio("no_such_cmd_xyz", vec![]),
    ))
    .await
    .expect("register should not spawn a connection");
    assert_eq!(gw.server_count().await, 1);
    assert!(
        gw.tools().await.is_empty(),
        "registry should be empty before sync"
    );
}

/// Re-registering the same server name errors out.
#[tokio::test]
async fn test_register_duplicate_rejected() {
    let gw = MCPGateway::new();
    let spec = GatewayServerSpec::new("dup", MCPConfig::stdio("no_such_cmd_xyz", vec![]));
    gw.register(spec.clone())
        .await
        .expect("first register should succeed");
    let err = gw.register(spec).await.unwrap_err();
    assert!(err.to_string().contains("already registered"), "{}", err);
}

/// sync pulls tools from a fake SSE server, and `server:tool` appears in the unified registry.
#[tokio::test]
async fn test_sync_populates_namespaced_registry() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let gw = MCPGateway::new();
    gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
        .await
        .expect("register should succeed");

    let namespaced = gw.sync("fs").await.expect("sync should succeed");
    assert_eq!(namespaced.len(), 1);
    assert_eq!(namespaced[0].full_name, "fs:echo");
    assert_eq!(namespaced[0].server, "fs");
    assert_eq!(namespaced[0].definition.name, "echo");

    let tools = gw.tools().await;
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].full_name, "fs:echo");
}

/// sync is idempotent: re-syncing does not duplicate registry entries.
#[tokio::test]
async fn test_sync_is_idempotent() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let gw = MCPGateway::new();
    gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
        .await
        .unwrap();
    let first = gw.sync("fs").await.unwrap();
    let second = gw.sync("fs").await.unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(
        second.len(),
        1,
        "idempotent: second sync does not duplicate"
    );
    assert_eq!(gw.tools().await.len(), 1);
}

/// Unified entry: call("server:tool") routes to the server by raw name (auto-syncs when not manually synced).
#[tokio::test]
async fn test_call_dispatches_with_auto_sync() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let gw = MCPGateway::new();
    gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
        .await
        .unwrap();

    let out = gw
        .call("fs:echo", json!({}))
        .await
        .expect("dispatch by full name");
    assert_eq!(
        out, "echo",
        "should reach the server with the raw tool name and echo back"
    );
}

/// An unregistered tool returns ToolNotFound.
#[tokio::test]
async fn test_call_unknown_tool_not_found() {
    let gw = MCPGateway::new();
    let err = gw.call("ghost:read_file", json!({})).await.unwrap_err();
    assert!(
        matches!(err, ToolError::ToolNotFound(ref n) if n == "ghost:read_file"),
        "{}",
        err
    );
}

/// Sandbox blocks: violating parameters are caught at the Gateway entry, never reach the server, and are audited.
#[tokio::test]
async fn test_call_sandbox_blocks_and_audits() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let sandbox = Arc::new(ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
        field: "path".to_string(),
        prefix: "file:///tmp/".to_string(),
    }));
    let gw = MCPGateway::new();
    gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)).with_sandbox(sandbox))
        .await
        .unwrap();

    let err = gw
        .call("fs:echo", json!({ "path": "file:///etc/passwd" }))
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::InvalidInput(ref m) if m.contains("least-privilege")),
        "{}",
        err
    );

    let log = gw.audit_log();
    assert!(!log.is_empty(), "blocked call should be recorded in audit");
    assert!(!log[0].allowed);
    assert_eq!(log[0].tool, "fs:echo");
    assert!(log[0]
        .reason
        .as_deref()
        .unwrap()
        .contains("least-privilege"));
}

/// Rate limit: over-quota calls in a window are rejected and audited; one record for allow, one for block.
#[tokio::test]
async fn test_call_rate_limit_blocks_and_audits() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let gw = MCPGateway::new();
    gw.register(
        GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url))
            .with_rate_limit(1, Duration::from_secs(60)),
    )
    .await
    .unwrap();

    let first = gw.call("fs:echo", json!({})).await;
    assert!(first.is_ok(), "1st call within window should be allowed");
    let err = gw.call("fs:echo", json!({})).await.unwrap_err();
    assert!(err.to_string().contains("rate limit"), "{}", err);

    let log = gw.audit_log();
    assert_eq!(log.len(), 2, "one record for allow and one for block");
    assert!(log[0].allowed);
    assert!(!log[1].allowed);
    assert!(log[1].reason.as_deref().unwrap().contains("rate limit"));
}

/// Static layer + dynamic layer: after pin, select hits the full-name tool.
#[tokio::test]
async fn test_select_over_synced_registry() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let gw = MCPGateway::new();
    gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
        .await
        .unwrap();
    gw.sync("fs").await.unwrap();
    assert!(
        gw.pin("fs:echo").await,
        "pin of a synced tool should succeed"
    );
    assert!(
        !gw.pin("fs:ghost").await,
        "pin of an unsynced tool should fail"
    );

    let picked = gw.select("echo tool", 5, usize::MAX).await;
    assert_eq!(picked.len(), 1);
    assert_eq!(
        picked[0].name, "fs:echo",
        "discovery layer returns the full name"
    );
}

/// Convert to BaseTool: adapters carry the namespace + timeout + sandbox and can be called normally.
#[tokio::test]
async fn test_as_base_tools_builds_adapters() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let gw = MCPGateway::new();
    gw.register(
        GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url))
            .with_timeout(ToolSpec::new("echo", Duration::from_secs(5))),
    )
    .await
    .unwrap();
    gw.sync("fs").await.unwrap();

    let tools = gw
        .as_base_tools()
        .await
        .expect("building adapters should succeed");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "fs:echo");
    let out = tools[0]
        .run("{}".into())
        .await
        .expect("adapter should be callable");
    assert_eq!(out, "echo");
}

/// Breaker delegation: health probe -> Down, reap_unhealthy removes it.
#[tokio::test]
async fn test_gateway_health_and_reap() {
    let gw = MCPGateway::new();
    gw.register(
        GatewayServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
            .with_max_failures(1),
    )
    .await
    .unwrap();

    let h = gw
        .health("bad")
        .await
        .expect("health probe should not error");
    assert_eq!(
        h.status,
        crate::HealthStatus::Down,
        "1 failure triggers circuit breaker"
    );
    let removed = gw.reap_unhealthy().await;
    assert_eq!(removed, vec!["bad".to_string()]);
}

/// Audit ring cap: only the newest max_audit entries are kept.
#[tokio::test]
async fn test_audit_cap_keeps_newest() {
    let fake = start_fake_sse_server(PostMode::Quiet).await;
    let gw = MCPGateway::new().with_max_audit(1);
    gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
        .await
        .unwrap();
    gw.call("fs:echo", json!({})).await.expect("1st call");
    gw.call("fs:echo", json!({})).await.expect("2nd call");

    let log = gw.audit_log();
    assert_eq!(log.len(), 1, "ring buffer keeps only the newest 1");
    assert_eq!(log[0].tool, "fs:echo");
}
