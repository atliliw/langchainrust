# MCP Protocol

The Model Context Protocol (MCP) is Anthropic's open standard for connecting LLM applications to external tools and data sources. LangChainRust provides both a client and server implementation. As of 0.22.0 the crate is **stateless single-track**: every request is a self-contained JSON-RPC HTTP POST tagged with `Mcp-Method` / `Mcp-Name` routing headers and a `_meta` block; there is no handshake and no session. The legacy handshake-based `MCPClient` (SSE/stdio transports) was removed in 0.22.0.

## Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| `StatelessMcpClient` | Client | Connect to any stateless MCP server, list/call tools |
| `MCPServer` | Server | Expose `BaseTool` implementations via MCP (`handle_request` / `serve_http` / `serve_stdio`) |
| `MCPToolAdapter` | Adapter | Wrap MCP tools as `BaseTool` for agent use |
| `ConnectionManager` | Client | 100+ Server lazy startup / idle reaping / managed registry |
| `ToolNamespace` | Client | `server:tool` name uniquification + conflict policy |
| `ToolDiscovery` | Client | Static (pinned) + dynamic (query top-k) tool selection |
| `ToolSpec` | Client | Per-tool timeout with hard cap |
| `ServerHealth` / `CircuitBreaker` | Client | Per-server health probe (`list_tools`) + circuit breaker |
| `ServerSandbox` | Client | Per-server security isolation (param least-privilege + egress whitelist + audit) |
| `SamplingGuard` | Client | Sampling recursion protection (depth + token budget + timeout) |
| `MCPGateway` | Client | Unified registry + on-demand dispatch (rate limiting + audit) |
| `TenantGateway` | Client | Multi-tenant isolation (per-tenant registry + audit) |
| `ToolOrchestrator` / `ToolStep` | Client | Tool orchestration (dependency DAG + parallel/serial + `${id}` template args) |
| `VersionPolicy` / `ProtocolInfo` | Protocol | Protocol version handling (degrade or reject on unsupported) |
| `StatelessTransport` | Transport | Pure HTTP POST JSON-RPC, no session |
| `MethodRateLimiter` | Client | Client-side per-method rate limiting (fails fast, no network round trip) |
| `TokenValidator` / `AuthScheme` | Auth | Bearer/JWT token validation (server side) + auth headers (client side) |
| `McpTaskHandle` | Client | MCP Tasks placeholder (persistence deferred to 0.22.1) |

## MCP Client

```rust
use langchainrust::mcp::{MCPToolAdapter, StatelessMcpClient};

// Connect to a stateless MCP endpoint — no handshake, this never fails
let client = StatelessMcpClient::connect("http://localhost:3001/mcp");

// List available tools
let tools = client.list_tools().await?;
for tool in &tools {
    println!("{}: {}", tool.name, tool.description);
}

// Call a tool
let result = client.call_tool("read_file", serde_json::json!({"path": "/tmp/hello.txt"})).await?;
println!("{}", result.text());

// Convert all MCP tools to BaseTool for agent use
let base_tools: Vec<Arc<dyn BaseTool>> = client
    .list_tools()
    .await?
    .into_iter()
    .map(|def| Arc::new(MCPToolAdapter::new(client.clone(), def)) as Arc<dyn BaseTool>)
    .collect();
```

Client-side extras:

- `.with_mrtr(MrtrConfig { max_round_trips })` — bound the `input_required`
  multi-round request loop; the client resends the original request with the
  continuation token (MRTR), up to the limit (`-32003` when exceeded);
- `.with_answer_provider(provider)` — collects answers for `input_required`
  questions (required for servers that use MRTR);
- `.with_method_rate_limiter(MethodRateLimiter)` — per-method client-side rate
  limit; a hit returns `-32002` without a network round trip;
- `client.meta()` — the pinned self-contained `_meta` (protocol version
  `2026-07-28` + client identity + optional `requestState`).

## MCP Server

```rust
use langchainrust::{MCPServer, BaseTool, Calculator};
use std::sync::Arc;

let server = MCPServer::new()
    .with_tool(Arc::new(Calculator::new()) as Arc<dyn BaseTool>)
    .with_tool(Arc::new(DateTimeTool::new()) as Arc<dyn BaseTool>)
    .with_server_info("my-mcp-server", "1.0.0");

// Serve via stdio (for external hosts like Claude Desktop)
server.serve_stdio().await?;
```

Or serve it as a deployable stateless HTTP service (see
`examples/mcp_http_server.rs`):

```rust
use std::sync::Arc;
use tokio::net::TcpListener;

let listener = TcpListener::bind("0.0.0.0:8788").await?;
let url = Arc::new(server).serve_http(listener); // "http://0.0.0.0:8788/mcp"
println!("clients connect to: {url}");
```

`server.handle_request(req)` is also available for custom transports: one-shot
JSON-RPC handling with no network at all.

## Stateless HTTP Track

Every request is self-contained: the client POSTs a JSON-RPC envelope plus
`_meta` (protocol version / client identity / optional `requestState`) and
tags it with the `Mcp-Method` / `Mcp-Name` headers so gateways can route and
throttle without parsing the body. There is no handshake, no session id, and
no sticky routing — any HTTP client that can POST JSON can talk to the server.
Auth is per-request bearer/JWT (`connect_with_auth` on the client,
`TokenValidator` implementations on the server/gateway side).

## Multi-Server Management (100+ Servers)

Holding one dedicated client per server still costs hundreds of long-lived
handles. Use `ConnectionManager` to host a managed registry, and
`ToolNamespace` to keep each server's tools uniquely named.

### Connection Manager (lazy start / idle reaping / pooling)

```rust
use langchainrust::{ConnectionManager, ServerSpec};

let manager = ConnectionManager::new();
// register is lazy: nothing is built yet
manager.register(
    ServerSpec::new("fs", "http://mcp-fs.internal:8080/mcp"),
).await?;
// stateful servers marked keep_alive are never reaped while idle
manager.register(
    ServerSpec::new("db", "http://mcp-db.internal:8080/mcp").keep_alive(),
).await?;

// first client() call builds the client; later calls reuse it
let client = manager.client("fs").await?;
```

Idle servers (non-`keep_alive`, idle beyond `max_idle`) are reaped in the
background. `manager.reap_idle()` triggers a sweep manually;
`manager.shutdown()` closes everything and stops the reaper task.

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

### Per-Tool Timeout with Hard Cap

Long-running tools must not hold callers hostage. `ToolSpec` gives each tool a
default timeout plus a hard cap that bounds total time regardless of anything
else.

```rust
use langchainrust::{MCPToolAdapter, ToolSpec};
use std::time::Duration;

let adapter = MCPToolAdapter::new(client, def)
    .with_timeout(ToolSpec::new("read_file", Duration::from_secs(30)));
// default timeout 30s; hard cap defaults to 3× (90s), or override with .with_max_timeout(..)
```

### Health Probe + Circuit Breaker

With 100+ servers a single one can go down at any time. Each registered server
carries a [`CircuitBreaker`]: `list_tools` doubles as the health probe, `N`
consecutive failures trip the breaker open (the server is removed — incoming
`client()` calls fail fast instead of hammering a dead server), and after an
exponential backoff the breaker opens a half-open probe window so a recovered
server can reconnect.

```rust
use langchainrust::{
    ConnectionManager, ServerSpec, ServerHealth, HealthStatus,
};

let manager = ConnectionManager::new();
manager.register(
    ServerSpec::new("fs", "http://localhost:8080/mcp")
        .with_max_failures(3), // 3 consecutive failures → trip
).await?;

// proactive probe: list_tools is the probe; returns a health snapshot
let health: ServerHealth = manager.health("fs").await?;
match health.status {
    HealthStatus::Healthy => { /* normal */ }
    HealthStatus::Degraded => { /* failures below threshold */ }
    HealthStatus::Down => { /* breaker open */ }
}

// during the open state client() fails fast
// manually remove all tripped servers (returns their names)
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
            prefix: "file:///tmp/".to_string(), // only allow tmp prefixes
        })
        .with_param_rule(ParamRule::RejectContains {
            field: "path".to_string(),
            forbidden: vec!["..".to_string()],  // reject path traversal
        })
        .allow_host("example.com"),             // egress whitelist
);

let adapter = MCPToolAdapter::new(client, def).with_sandbox(sandbox);
// run() checks the sandbox first; a block returns InvalidInput and records audit
// let out = adapter.run(r#"{"path": "file:///etc/passwd"}"#).await?; // Err
```

Same-server tools share one sandbox (cheap `Arc` clone), so they all write into
the same audit log. Read it back with `sandbox.audit_log()`.

### Sampling Recursion Protection

MCP Sampling ("Agent calls tool → tool requests Sampling → LLM calls tool →
another Sampling request") can recurse unboundedly. [`SamplingGuard`] bounds
the whole chain on the Host side with three constraints:

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
// guard.enter(req.max_tokens)?;   // depth/budget/timeout: any breach → Err
// ... run LLM inference ...
// lease Drop releases the nesting depth
```

The counter is `AtomicUsize` (not a `Mutex`) so re-entrant nested sampling across
`await` cannot deadlock, and a rejected `enter` occupies neither depth nor budget.

### MCP Gateway (unified registry + on-demand dispatch)

`MCPGateway` is the single entry point that composes P2-1~P2-6 into one registry:
it hosts the [`ConnectionManager`] (lazy connect / idle reaping / circuit breaker),
the [`ToolNamespace`] (per-server name isolation), the [`ToolDiscovery`] (static +
dynamic selection), per-server sandboxes and timeouts, plus rate limiting and a
unified audit log. Registering a server is lazy — nothing is built until the
first `sync` / `call`.

```rust
use langchainrust::{MCPGateway, GatewayServerSpec};
use std::time::Duration;

let gateway = MCPGateway::new();
// register is lazy: nothing connects yet
gateway.register(
    GatewayServerSpec::new("fs", "http://localhost:8080/mcp")
        .with_rate_limit(10, Duration::from_secs(60)) // max 10 calls / minute
        .with_timeout(Duration::from_secs(30)),
).await?;
gateway.register(
    GatewayServerSpec::new("db", "http://localhost:8081/mcp"),
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

### Multi-Tenancy (per-tenant isolation)

One process serving many customers (SaaS) must never leak a tenant's tools,
connections, or audit trail to another. [`TenantGateway`] wraps an [`MCPGateway`]
per tenant: `register` is lazy (nothing is built until a tenant's first
`sync_all` / `call`), and every operation is routed by `tenant_id`.

```rust
use langchainrust::{TenantGateway, GatewayServerSpec};

let tenants = TenantGateway::new();
// register is scoped to the tenant — tenant "b" never sees tenant "a"'s tools
tenants.register("a", GatewayServerSpec::new("fs", "http://localhost:8080/mcp")).await?;

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

### Protocol Version Handling (vNext)

MCP protocol versions evolve. On the stateless track there is no handshake —
the client pins its `_meta.protocol_version` (default `2026-07-28`) and the
server can inspect it per request. `langchainrust` records the negotiation
result and lets you choose how to treat mismatches:

- `SUPPORTED_PROTOCOL_VERSIONS` — the versions this library implements;
- [`VersionPolicy::Degrade`] (default) — unknown version: keep going on the
  library's own version, recording `supported = false`;
- [`VersionPolicy::Reject`] — unknown version: the request is refused;
- [`ProtocolInfo`] — the negotiation result (`requested`, `server_version`,
  `negotiated`, `supported`).

```rust
use langchainrust::mcp::StatelessMcpClient;

let client = StatelessMcpClient::connect("http://localhost:8080/mcp");
assert_eq!(client.meta().protocol_version, "2026-07-28");
```

## Protocol Details

- **Version**: `2026-07-28` (stateless single-track)
- **Format**: JSON-RPC 2.0 over HTTP POST
- **Transport**: stateless, self-contained requests; no handshake, no session
- **Methods**: `initialize`, `tools/list`, `tools/call`, `server/discover`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, `completion/complete`, `sampling/createMessage`

## Sub-Protocol Support

| Sub-protocol | Client | Server |
|-------------|--------|--------|
| Tools | `list_tools`, `call_tool` | `serve_stdio`, `serve_http` |
| Resources | `list_resources`, `read_resource` | -- |
| Prompts | `list_prompts`, `get_prompt` | -- |
| Sampling | `create_message` | -- |
| Completion | `complete` | -- |
| Roots | `list_roots` | -- |
| Elicitation | `elicit` | -- |
