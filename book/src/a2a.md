# A2A Protocol

The [A2A](https://github.com/google/A2A) (Agent-to-Agent) protocol is Google's open standard for agent interoperability — letting agents built by different teams and vendors discover and call each other. LangChainRust provides both a server and client implementation over JSON-RPC 2.0.

Layering: **A2A handles "Agent ↔ Agent" communication; MCP handles "Agent ↔ tools / data sources."** They compose: MCP supplies tools, agents collaborate over A2A.

## Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| `A2AServer` | Server | Expose your agent as an A2A service (handler functions, pluggable into any HTTP framework) |
| `A2AClient` | Client | Discover and call remote agents (`send_task` / `get_task` / `cancel_task`) |
| `AgentCard` | Protocol | Agent self-description card; v1.0.1 `supportedInterfaces[]` |
| `AgentInterface` / `A2ATransport` | Protocol | Per-interface `(protocolVersion, transport, url)` + optional tenant tag |
| `negotiate` / `is_v101` | Protocol | Client/server transport + version negotiation |
| `sign_agent_card` / `verify_card_signature` | Security | Card signing (RFC 8785-lite canonicalization, HMAC) stored in `card.signature` |
| `sign_card_jws` / `verify_card_jws` | Security | Compact JWS HS256 card signing (header-friendly) |
| `AgentRegistry` / `RegistryClient` | Discovery | Registry-based multi-agent discovery |
| `FederationGateway` | Gateway | Cross-org federation with call policies and data contracts |
| `ResilientA2AClient` | Client | Retries / timeouts / circuit breaking wrapper |
| `SkillRouter` / `SkillMapRouter` | Client | Route tasks to agents by skill |

## Agent Card (v1.0.1)

A v1.0.1 card declares multiple interfaces instead of a single protocol binding:

```rust
use langchainrust::a2a::protocol::{AgentCard, AgentInterface, A2ATransport, A2A_VERSION_V101};

let card = AgentCard::new("agent-a", "A helpful agent", "https://a.example")
    .with_supported_interface(AgentInterface::new(
        A2A_VERSION_V101,           // "1.0.1", a separate protocolVersion per interface
        A2ATransport::HttpJson,     // JsonRpc | HttpJson | Grpc
        "https://a.example/a2a",
    ))
    .with_supported_interface(
        AgentInterface::new("1.0.1", A2ATransport::JsonRpc, "https://a.example/a2a/jsonrpc")
            .with_tenant("acme"),   // enterprise multi-tenancy (optional)
    );

assert!(card.is_v101());
```

The old v0.3 fields are kept for reader compatibility; new cards should fill `supportedInterfaces` and treat the old fields as fallback.

### Negotiation

The client picks a mutually supported interface given the transport it wants and the versions it speaks; a miss is a clear error:

```rust
let picked = card.negotiate(A2ATransport::HttpJson, &["1.0.1", "1.0"])?;
```

## Card Signing

The Agent Card drives discovery — a tampered card (swapped endpoint, downgraded protocol) is a supply-chain attack. Signing canonicalizes the card JSON (RFC 8785-lite: recursive key ordering) and applies HMAC-SHA256.

```rust
use langchainrust::a2a::client::{sign_agent_card, verify_card_signature, sign_card_jws, verify_card_jws};
use langchainrust::a2a::protocol::AgentCard;

let secret = b"shared-secret-between-registrar-and-clients";

// In-place signing, hex-encoded into card.signature
let mut card = AgentCard::new("agent-a", "A helpful agent", "https://a.example");
sign_agent_card(&mut card, secret)?;

// Verify — tampering with any field fails
verify_card_signature(&card, secret)?;

// Compact JWS token (header-friendly)
let jws = sign_card_jws(&card, secret)?;
verify_card_jws(&card, &jws, secret)?;
```

Boundaries: HS256 symmetric only (ES256 planned 0.22.1); canonicalization is RFC 8785-lite, not full JCS; expiry rides the card's `expiresAt` field.

## A2A Server (Expose Your Agent)

`A2AServer` provides handler functions you plug into any HTTP framework (axum, actix, warp) — it does NOT start its own listener.

```rust
use langchainrust::a2a::A2AServer;
use langchainrust::LLMChain;
use std::sync::Arc;

let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
let server = A2AServer::new(chain)
    .with_card(card);

// In your HTTP handlers:
// GET  /.well-known/agent-card.json → server.get_agent_card()
// POST /                       → server.handle_a2a_request(body).await
```

Endpoints: `GET /.well-known/agent-card.json` (discovery) and `POST /` (JSON-RPC requests). Tasks from `tasks/send` are stored in an in-memory `RwLock<HashMap>` — wrap with a database-backed `TaskStore` for production.

## A2A Client (Call Remote Agent)

```rust
use langchainrust::a2a::{A2AClient, A2AMessage};

let client = A2AClient::new("http://remote-agent:8080".to_string()).unwrap();

// Discover
let card = client.get_agent_card().await?;

// Dispatch / poll / cancel
let task = client.send_task(A2AMessage::user("hello")).await?;
let task = client.get_task(&task.id).await?;
let task = client.cancel_task(&task.id).await?;
```

## Protocol Flow

| Step | Operation | Role |
|---|---|---|
| 1. Discover | `get_agent_card` fetches the remote Agent Card | Client |
| 2. Dispatch | `send_task` submits a task (`A2AMessage`) | Client → Server |
| 3. Progress | `get_task` queries status and result by task ID | Client |
| 4. Cancel | `cancel_task` cancels an unfinished task | Client |

## Status Matrix

| Capability | Status |
|---|---|
| `tasks/send` / `tasks/get` / `tasks/cancel` | Implemented |
| Agent Card discovery | Implemented |
| v1.0.1 `supportedInterfaces` + negotiation | Implemented (0.22.0) |
| Card signing JWS HS256 | Implemented (0.22.0; ES256 / full JCS → 0.22.1) |
| API key / Static Bearer auth | Implemented |
| gRPC binding | → 0.22.1 (tonic codegen needs protoc) |
| OAuth / OIDC | Planned (0.22.1) |
| Task state machine / streaming push / TLS | Planned |

A deployment walkthrough lives at `docs/internal/deployment/a2a-http-server-deployment.md`; the runnable example is `crates/lc/examples/a2a_http_server.rs`.
