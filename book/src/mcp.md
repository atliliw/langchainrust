# MCP Protocol

The Model Context Protocol (MCP) is Anthropic's open standard for connecting LLM applications to external tools and data sources. LangChainRust provides both a client and server implementation.

## Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| `MCPClient` | Client | Connect to any MCP server, list/call tools |
| `MCPServer` | Server | Expose `BaseTool` implementations via MCP |
| `MCPToolAdapter` | Adapter | Wrap MCP tools as `BaseTool` for agent use |
| `ConnectionManager` | Client | 100+ Server lazy startup / idle reaping / connection pool |
| `ToolNamespace` | Client | `server:tool` name uniquification + conflict policy |
| `ToolDiscovery` | Client | Static (pinned) + dynamic (query top-k) tool selection |
| `ToolSpec` | Client | Per-tool timeout with progress-reset + hard cap |
| `ServerHealth` / `CircuitBreaker` | Client | Per-server health probe (`list_tools`) + circuit breaker |
| `ServerSandbox` | Client | Per-server security isolation (param least-privilege + egress whitelist + audit) |
| `SamplingGuard` | Client | Sampling recursion protection (depth + token budget + timeout) |
| `MCPGateway` | Client | Unified registry + on-demand dispatch (rate limiting + audit) |
| `PartialContent` / `ToolStream` | Client | Streaming tool output (partial chunks + per-tool subscribe) |
| `TenantGateway` | Client | Multi-tenant isolation (per-tenant registry + audit) |
| `ToolOrchestrator` / `ToolStep` | Client | Tool orchestration (dependency DAG + parallel/serial + `${id}` template args) |
| `VersionPolicy` / `ProtocolInfo` | Protocol | Protocol version negotiation (degrade or reject on unsupported) |
| `StdioTransport` | Transport | Child process stdin/stdout JSON-RPC |
| `SseTransport` | Transport | HTTP SSE + POST JSON-RPC |
| `MCPConfig` | Config | `Stdio` or `Sse` connection configuration |

## MCP Client

```rust
use langchainrust::{MCPClient, MCPConfig};

// Connect via stdio (spawn a child process)
let config = MCPConfig::stdio(
    "npx",
    vec!["@anthropic/mcp-server-filesystem".to_string(), "/tmp".to_string()],
);
let client = MCPClient::connect(config).await?;

// List available tools
let tools = client.list_tools().await?;
for tool in &tools {
    println!("{}: {}", tool.name, tool.description);
}

// Call a tool
let result = client.call_tool("read_file", serde_json::json!({"path": "/tmp/hello.txt"})).await?;
println!("{}", result.text());

// Convert all MCP tools to BaseTool for agent use
// (auto-discovers tools via tools/list if the cache is empty)
let base_tools: Vec<Arc<dyn BaseTool>> = client.as_tools().await?;
```

## MCP Server

```rust
use langchainrust::{MCPServer, BaseTool, Calculator};
use std::sync::Arc;

let server = MCPServer::new()
    .with_tool(Arc::new(Calculator::new()) as Arc<dyn BaseTool>)
    .with_tool(Arc::new(DateTimeTool::new()) as Arc<dyn BaseTool>)
    .with_server_info("my-mcp-server", "1.0.0");

// Serve via stdio (for use by MCP clients like Claude Desktop)
server.serve_stdio().await?;
```

## SSE Transport

```rust
use langchainrust::MCPConfig;

// Connect to a remote MCP server via SSE
let config = MCPConfig::sse("http://localhost:3000/sse");
let client = MCPClient::connect(config).await?;
let tools = client.list_tools().await?;
```

## Multi-Server Management (100+ Servers)

Directly calling `MCPClient::connect` for every server spawns hundreds of child
processes / long-lived connections — exhausting memory and file descriptors. Use
`ConnectionManager` to host a managed registry, and `ToolNamespace` to keep each
server's tools uniquely named.

### Connection Manager (lazy start / idle reaping / pooling)

```rust
use langchainrust::{ConnectionManager, ServerSpec, MCPConfig};

let manager = ConnectionManager::new();
// register is lazy: no child process / connection is spawned yet
manager.register(
    ServerSpec::new("fs", MCPConfig::stdio("npx", vec![
        "@anthropic/mcp-server-filesystem".into(), "/tmp".into(),
    ])),
).await?;
// stateful servers marked keep_alive are never reaped while idle
manager.register(
    ServerSpec::new("db", MCPConfig::sse("http://localhost:8080/sse")).keep_alive(),
).await?;

// first client() call spawns the connection; later calls reuse it
let client = manager.client("fs").await?;
```

Idle servers (non-`keep_alive`, idle beyond `max_idle`) are closed in the
background. `manager.reap_idle()` triggers a sweep manually; `manager.shutdown()`
closes everything and stops the reaper task.

### Tool Namespace (name uniquification + conflict policy)

Different servers often expose same-named tools (several with `read_file`). The
`ToolNamespace` registry uniquifies each tool as `server_name:tool_name`, with an
explicit conflict strategy:

- `ToolConflict::Prefix` — colliding tools are **all exposed**, each under its
  own `server:` prefix;
- `ToolConflict::Reject` — a same-named tool from another server is **rejected**
  at registration.

```rust
use langchainrust::{ToolNamespace, ToolConflict};

let mut ns = ToolNamespace::new();
// both servers expose "read_file" → distinct names fs:read_file / db:read_file
ns.register("fs", fs_tools, ToolConflict::Prefix)?;
ns.register("db", db_tools, ToolConflict::Prefix)?;

// route a call back to the owning server / raw tool name
let (server, raw) = ns.resolve("fs:read_file").expect("registered");
```

The namespaced adapter exposes `server:tool` to the LLM while calling the raw
tool name on the server side:

```rust
use langchainrust::MCPToolAdapter;

let client = manager.client("fs").await?;
let adapter = MCPToolAdapter::namespaced(client, "fs", read_file_def);
assert_eq!(adapter.name(), "fs:read_file");  // what the LLM sees
```

### Static + Dynamic Tool Discovery

100+ servers can declare hundreds of thousands of tokens of tool schemas —
far beyond any context window. `ToolDiscovery` avoids injecting everything at
once by splitting tools into two layers:

- **Static layer**: 20-50 high-frequency tools pinned as always-on;
- **Dynamic layer**: tools retrieved per-query by relevance (top-k), like a RAG
  step over the tool registry. Relevance is scored by
  [`KeywordScorer`] (token overlap, zero dependencies) by default; implement
  [`ToolScorer`] for vector-based scoring and inject it with `with_scorer`.

```rust
use langchainrust::{ToolDiscovery, ToolScorer};

let mut discovery = ToolDiscovery::new();
for def in all_tools { discovery.register(def); }       // full registry
discovery.pin("get_time");                              // static layer: always injected
discovery.pin("search_db");

// per-query injection: pinned tools + top-2 query-relevant tools (deduped)
let injected = discovery.select("find files modified today", /*top_k*/ 2, /*static_limit*/ 50);
```

### Per-Tool Timeout with Progress Reset

Long-running tools may legitimately outlive a fixed timeout. Rather than killing
them, `ToolSpec` gives each tool a default timeout that is **reset whenever a
`notifications/progress` notification arrives** (the tool is still alive), plus a
hard cap that bounds total time regardless of progress — so a "half-dead but
still reporting progress" tool cannot hold a connection forever.

```rust
use langchainrust::{MCPToolAdapter, ToolSpec};
use std::time::Duration;

let adapter = MCPToolAdapter::new(client, def)
    .with_timeout(ToolSpec::new("read_file", Duration::from_secs(30)));
// default timeout 30s, reset by each progress notification;
// hard cap defaults to 3× (90s), or override with .with_max_timeout(..)
```

### Health Probe + Circuit Breaker

With 100+ servers a single one can go down at any time. Each registered server
carries a [`CircuitBreaker`]: `list_tools` doubles as the health probe, `N`
consecutive failures trip the breaker open (the server is "摘除" — incoming
`client()` calls fail fast instead of hammering a dead server), and after an
exponential backoff the breaker opens a half-open probe window so a recovered
server can reconnect.

```rust
use langchainrust::{
    ConnectionManager, ServerSpec, ServerHealth, HealthStatus, MCPConfig,
};

let manager = ConnectionManager::new();
manager.register(
    ServerSpec::new("fs", MCPConfig::sse("http://localhost:8080/sse"))
        .with_max_failures(3), // 3 次连续失败 → 熔断
).await?;

// 主动探活:list_tools 即探活;返回健康快照
let health: ServerHealth = manager.health("fs").await?;
match health.status {
    HealthStatus::Healthy => { /* 正常 */ }
    HealthStatus::Degraded => { /* 有失败但未达阈值 */ }
    HealthStatus::Down => { /* 已熔断 */ }
}

// 熔断期间 client() 快速失败,不再往坏 Server 上打请求
// let client = manager.client("fs").await?;  // Err "熔断中,退避期拒绝连接"

// 手动摘除所有熔断的 Server(返回被摘除的名字)
let removed = manager.reap_unhealthy().await;
```

Default `max_failures` is 3. The breaker backs off 0.5s → 1s → 2s → … (cap 30s)
between reconnect attempts. Health is the liveness gate only — tool calls still
fail fast through [`ToolSpec`] timeouts.

### Per-Server Security Sandbox

100+ servers come from very different origins; each one must be narrowed to its
own least-privilege boundary. [`ServerSandbox`] bundles:

- **Parameter-level least privilege** — [`ParamRule`] constrains tool-call
  arguments: a filesystem server only allows `file:///tmp/` prefixes, formats
  only allow enum values, and path-traversal substrings (`..`) are rejected.
  Violating calls are blocked *before* the request is sent to the server.
- **Outbound network whitelist** — [`EgressPolicy`] declares which hosts the
  server may contact; an empty whitelist denies all egress (fail-closed).
- **Full audit log** — every allowed/blocked call is recorded (server, tool,
  arguments, decision, reason), ring-buffered with a cap.

```rust
use langchainrust::{MCPToolAdapter, ServerSandbox, ParamRule, EgressPolicy};
use std::sync::Arc;

let sandbox = Arc::new(
    ServerSandbox::new("fs")
        .with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(), // 只允许 tmp 前缀
        })
        .with_param_rule(ParamRule::RejectContains {
            field: "path".to_string(),
            forbidden: vec!["..".to_string()],  // 拒绝路径穿越
        })
        .allow_host("example.com"),             // 出站白名单
);

let adapter = MCPToolAdapter::new(client, def).with_sandbox(sandbox);
// run() 先过沙箱,拦截则返回 InvalidInput 并记审计,不进 Server
// let out = adapter.run(r#"{"path": "file:///etc/passwd"}"#).await?; // Err
```

Same-server tools share one sandbox (cheap `Arc` clone), so they all write into
the same audit log. Read it back with `sandbox.audit_log()`.

### Sampling Recursion Protection

MCP Sampling ("Agent 调工具 → 工具请求 Sampling → LLM 调工具 → 再请求 Sampling")
can recurse unboundedly. [`SamplingGuard`] bounds the whole chain on the Host
side with three constraints:

- **Depth limit** — nested Sampling may not exceed `max_depth` (default 3);
- **Cumulative token budget** — each request's `max_tokens` accumulates toward a
  chain-wide budget;
- **Timeout / deadline** — the whole chain must finish within `total_timeout` or
  an explicit `deadline`.

Call `enter(request.max_tokens)` before every `sampling/createMessage`. The
returned [`SamplingLease`] holds one nesting level for the duration of the call
(atomic counters, safe across `await`) and releases it on `Drop`.

```rust
use langchainrust::{SamplingGuard};
use std::time::Duration;

let guard = SamplingGuard::new(/*max_depth*/ 3, /*token_budget*/ 10_000)
    .with_timeout(Duration::from_secs(60));
// guard.enter(req.max_tokens)?;   // 深度/预算/超时任一超限即 Err
// ... 执行 LLM 推理 ...
// lease Drop 自动释放嵌套深度
```

The counter is `AtomicUsize` (not a `Mutex`) so re-entrant nested sampling across
`await` cannot deadlock, and a rejected `enter` occupies neither depth nor budget.

### MCP Gateway (unified registry + on-demand dispatch)

`MCPGateway` is the single entry point that composes P2-1~P2-6 into one registry:
it hosts the [`ConnectionManager`] (lazy connect / idle reaping / circuit breaker),
the [`ToolNamespace`] (per-server name isolation), the [`ToolDiscovery`] (static +
dynamic selection), per-server sandboxes and timeouts, plus rate limiting and a
unified audit log. Registering a server is lazy — no child process or connection
is spawned until the first `sync` / `call`.

```rust
use langchainrust::{MCPGateway, GatewayServerSpec, MCPConfig};
use std::time::Duration;

let gateway = MCPGateway::new();
// register is lazy: nothing connects yet
gateway.register(
    GatewayServerSpec::new("fs", MCPConfig::sse("http://localhost:8080/sse"))
        .with_rate_limit(10, Duration::from_secs(60)) // max 10 calls / minute
        .with_timeout(Duration::from_secs(30)),
).await?;
gateway.register(
    GatewayServerSpec::new("db", MCPConfig::stdio("npx", vec![
        "@anthropic/mcp-server-db".into(), "--db", "/tmp/app.db".into(),
    ])),
).await?;

// sync connects + pulls tools + populates namespace/discovery (idempotent)
gateway.sync("fs").await?;
gateway.sync_all().await?;

// per-query injection: pinned static tools + top-k relevant dynamic tools
let selected = gateway.select("find files modified today", /*top_k*/ 2, /*static_limit*/ 50);

// on-demand dispatch: "server:tool" resolves → rate-limits → breaker-gated
// client → sandbox → (timeout) → call → audit. Auto-syncs on a first-time miss.
let out = gateway.call("fs:read_file", serde_json::json!({"path": "/tmp/a.txt"})).await?;

// hang the whole registry on an agent as BaseTools (namespaced `server:tool`)
let base_tools = gateway.as_base_tools().await?;
```

Every call passes through a unified audit log (server, tool, decision, reason),
ring-buffered with a cap (`with_max_audit`); read it with `gateway.audit_log()`.
Ops helpers mirror the underlying manager: `health`, `reap_unhealthy`,
`reap_idle`, `release`, and `shutdown`.

### Streaming Tool Output

Long-running tools can push results incrementally instead of making the caller
wait for a single final response. The server streams `notifications/tool_partial`
chunks over the existing connection; each chunk is a [`PartialContent`]:
tool name, monotonic `seq`, an optional progress ratio, and a `final` marker on
the last chunk. Chunks carry a full [`MCPContent`] (text / image / resource),
so multi-type content (P1-7) streams through the same path.

```rust
use langchainrust::{MCPClient, MCPConfig};
use std::time::Duration;

let client = MCPClient::connect(MCPConfig::sse("http://localhost:8080/sse")).await?;

// subscribe BEFORE calling the tool — only deliveries after this point arrive
let mut stream = client.subscribe_tool_stream("long_running_tool");

// each chunk: .tool / .seq / .content / .progress / .is_final()
while let Some(chunk) = stream.next().await? {
    println!("[{:.0}%] {}", chunk.progress.unwrap_or(0.0) * 100.0, chunk.render_text());
    if chunk.is_final { break; }
}

// or collect everything up to the final chunk in one call
let client2 = MCPClient::connect(MCPConfig::sse("http://localhost:8080/sse")).await?;
let mut all = client2.subscribe_tool_stream("long_running_tool");
let chunks: Vec<_> = all.collect(Duration::from_secs(120)).await?; // last one is_final
```

`ToolStream` filters to one tool's chunks; other tools' pushes are ignored.
The channel is a ring buffer — if pushes outpace consumption,
`next()` returns `Err(ToolStreamError::Lagged)` instead of silently dropping
data. On the server side, [`MCPServer::publish_partial`] pushes a chunk to
connected hosts; the `InMemoryTransport` forwards them transparently, so the
same `subscribe_tool_stream` works in-process or over SSE.

### Multi-Tenancy (per-tenant isolation)

One process serving many customers (SaaS) must never leak a tenant's tools,
connections, or audit trail to another. [`TenantGateway`] wraps an [`MCPGateway`]
per tenant: `register` is lazy (nothing connects until a tenant's first
`sync_all` / `call`), and every operation is routed by `tenant_id`.

```rust
use langchainrust::{TenantGateway, GatewayServerSpec, MCPConfig};

let tenants = TenantGateway::new();
// register is scoped to the tenant — tenant "b" never sees tenant "a"'s tools
tenants.register("a", GatewayServerSpec::new("fs", MCPConfig::sse("http://localhost:8080/sse"))).await?;

tenants.sync_all("a").await?;
// on-demand dispatch: auto-syncs on a first-time miss, then calls
let out = tenants.call("a", "fs:read_file", serde_json::json!({"path": "/tmp/a.txt"})).await?;

// audit logs are per-tenant — no cross-tenant visibility
let audit = tenants.audit_log("a");
// teardown: removing a tenant drops its registry and connections
tenants.remove_tenant("a");
```

`TenantGateway::tenant_ids()` lists live tenants; a missing tenant is created
lazily on first use, so there is no setup step to forget.

### Tool Orchestration (dependency DAG)

A multi-step task is often a DAG of tool calls — the output of one tool feeds
the next. [`ToolOrchestrator`] declares steps with dependencies, then executes:

- **Validation first** — duplicate ids, unknown dependencies, and dependency
  cycles are rejected *before* any tool runs (Kahn topological sort);
- **Round-based parallelism** — every step whose dependencies are satisfied
  runs in the same round, with concurrency capped by `with_max_concurrency`
  (default 4);
- **Argument templating** — `${id}` substitutes a previous step's whole JSON
  output, `${id.field}` extracts one field, so downstream args are computed
  from upstream results.

```rust
use langchainrust::{ToolOrchestrator, ToolStep, MCPGateway};
use serde_json::json;

let orch = ToolOrchestrator::new()
    .with_max_concurrency(4)
    .add_step(ToolStep::new("a", "fs:read_file", json!({"path": "/tmp/orders.csv"})))
    .add_step(ToolStep::new("b", "db:query", json!({"sql": "SELECT ..."})).after("a"))
    .add_step(ToolStep::new("sum", "calc:total", json!({
        "rows": "${b.sum}",
        "file": "${a.content}"
    })).after("b"));

// MCPGateway implements ToolCaller — plug the whole DAG into the registry
let results = orch.execute(&gateway).await?;
let total = &results["sum"]["total"];
```

Any [`ToolCaller`] implementation drives the orchestrator; steps reference
`server:tool` full names. Failure of one step fails the round — later steps
that depended on it are skipped and the error propagates as `OrchestrateError`.

### Protocol Version Negotiation (vNext)

MCP protocol versions evolve. On `initialize` the client declares its version;
the server answers with the version it will speak. `langchainrust` records the
negotiation result and lets you choose how to treat mismatches:

- `SUPPORTED_PROTOCOL_VERSIONS` — the versions this library implements;
- [`VersionPolicy::Degrade`] (default) — server version not supported: keep
  going on the library's own version, recording `supported = false`;
- [`VersionPolicy::Reject`] — server version not supported: handshake fails,
  the connection is refused;
- [`ProtocolInfo`] — the locked negotiation result (`requested`,
  `server_version`, `negotiated`, `supported`), read back via
  `protocol_info()` / `protocol_version()`.

```rust
use langchainrust::{MCPClient, MCPConfig, VersionPolicy};

// default: degrade — unknown versions connect and fall back to this version
let client = MCPClient::connect(MCPConfig::sse("http://localhost:8080/sse")).await?;
assert_eq!(client.protocol_version().as_deref(), Some("2024-11-05"));

// strict: unknown versions are rejected at handshake
let strict = MCPClient::connect_with_policy(
    MCPConfig::sse("http://localhost:8080/sse"),
    VersionPolicy::Reject,
).await; // Err on unsupported versions
```

`connect_with_policy` / `with_transport_policy` carry the policy into the
handshake; the negotiated version is locked after connect. The server side is
symmetric: `initialize` echoes the requested version when supported, otherwise
answers with its own version.

## Protocol Details

- **Version**: `2024-11-05`
- **Format**: JSON-RPC 2.0 over stdio or SSE
- **Handshake**: Client sends `initialize`, server responds with capabilities
- **Methods**: `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, `completion/complete`, `sampling/createMessage`

## Sub-Protocol Support

| Sub-protocol | Client | Server |
|-------------|--------|--------|
| Tools | `list_tools`, `call_tool` | `serve_stdio` |
| Resources | `list_resources`, `read_resource` | -- |
| Prompts | `list_prompts`, `get_prompt` | -- |
| Sampling | `create_message` | -- |
| Completion | `complete` | -- |
| Roots | `list_roots` | -- |
| Elicitation | `elicit` | -- |
