//! Example / deployable A2A (Agent-to-Agent) HTTP server.
//!
//! Exposes an LLMChain (real model) as a networked A2A agent via `A2AServer::serve_on`; any
//! A2A client (this repo's `A2AClient` / A2A SDKs in other languages) can call it by protocol:
//!
//! | Route | Description |
//! |---|---|
//! | `GET /.well-known/agent-card.json` | Discover the agent: name/description/skills/address |
//! | `POST /` | `tasks/send` / `tasks/get` / `tasks/cancel` (JSON-RPC) |
//! | `GET /events` | SSE task-progress push (needs `with_streaming` enabled) |
//!
//! # Run (local testing)
//!
//! ```powershell
//! cargo run -p langchainrust --example a2a_http_server
//! ```
//!
//! # Build a standalone executable (for deployment)
//!
//! ```powershell
//! cargo build --release -p langchainrust --example a2a_http_server
//! # Artifact: target/release/examples/a2a_http_server.exe, copy it to the remote server and run
//! ```
//!
//! # Runtime configuration (environment variables, not hardcoded)
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `A2A_HOST` | `127.0.0.1` | Bind address (default local-only; for remote access set it explicitly to `0.0.0.0` and configure auth / network whitelist yourself) |
//! | `A2A_PORT` | `8080` | Listening port |
//! | `A2A_API_KEY` | built-in key | API key for Alibaba Cloud MaaS |
//! | `A2A_BASE_URL` | built-in URL | OpenAI-compatible endpoint for Alibaba Cloud MaaS |
//! | `A2A_MODEL` | `qwen3.7-max-2026-06-08` | Model name (if the account lacks this model, switch to an accessible one, e.g. `qwen3.5-ocr`) |
//! | `A2A_QUESTION_TEMPLATE` | `Answer in one sentence: {question}` | Prompt template sent to the LLM |
//! | `A2A_AUTH_TOKEN` | empty | When set, all requests must carry `Authorization: Bearer <token>` |
//!
//! For remote deployment set `A2A_HOST` to `0.0.0.0` and configure a network whitelist /
//! reverse-proxy auth. After startup the agent card lives at
//! `http://<host>:<port>/.well-known/agent-card.json`.
//!
//! ```powershell
//! $env:A2A_PORT = "8080"
//! .\target\release\examples\a2a_http_server.exe
//! ```
//!
//! # Test with this repo's client (in another process)
//!
//! ```powershell
//! cargo test -p langchainrust --test a2a f04_remote_http_roundtrip
//! ```

use std::sync::Arc;

use langchainrust::{A2AServer, LLMChain, OpenAIChat, OpenAIConfig};
use tokio::net::TcpListener;

/// A model pointing at a real endpoint: key/url/model name can all be overridden with
/// environment variables; the defaults are self-explanatory.
fn real_llm() -> OpenAIChat {
    OpenAIChat::new(OpenAIConfig {
        api_key: std::env::var("A2A_API_KEY")
            .unwrap_or_else(|_| "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string()),
        base_url: std::env::var("A2A_BASE_URL").unwrap_or_else(|_| {
            "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                .to_string()
        }), // Alibaba Cloud MaaS endpoint
        model: std::env::var("A2A_MODEL").unwrap_or_else(|_| "qwen3.7-max-2026-06-08".to_string()),
        streaming: false,
        temperature: Some(0.3),
        max_tokens: Some(500),
        ..Default::default()
    })
}

#[tokio::main]
async fn main() {
    // 1. Read the configuration (environment variables with defaults). Default binds to the
    //    local loopback only; this server has no built-in auth. Binding 0.0.0.0 would expose
    //    the service to anyone who can reach this port, so it must be an explicit choice, and
    //    you must add auth yourself.
    let host = std::env::var("A2A_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("A2A_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);

    // 2. Build an LLMChain (real model + prompt template) as the A2A server's skill
    let template = std::env::var("A2A_QUESTION_TEMPLATE")
        .unwrap_or_else(|_| "Answer in one sentence: {question}".to_string());
    let chain = LLMChain::new(real_llm(), template);

    // 3. Build the A2A server: wrap the chain; optionally require a bearer auth token
    let server = A2AServer::new(Arc::new(chain) as Arc<dyn langchainrust::BaseChain>);
    let server = match std::env::var("A2A_AUTH_TOKEN").ok() {
        Some(token) if !token.is_empty() => server.with_auth_token(&token),
        _ => server,
    };

    // 4. Bind the listen address (default 127.0.0.1 = local only; 0.0.0.0 = all interfaces, must be explicit)
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to bind {host}:{port}: {e}");
            std::process::exit(1);
        });
    let bound = listener.local_addr().unwrap();

    // 5. Print the endpoints and start serving (axum HTTP layer; the process stays alive via
    //    serve_on's request loop)
    println!("A2A agent started ✅");
    println!("  agent card: http://{bound}/.well-known/agent-card.json");
    println!("  task endpoint: POST http://{bound}/  (tasks/send / tasks/get / tasks/cancel)");
    println!("  skill: one-sentence Q&A (qwen3.7-max)");
    println!("press Ctrl+C to stop.");

    server.serve_on(listener).await.unwrap_or_else(|e| {
        eprintln!("A2A service exited with error: {e}");
        std::process::exit(1);
    });
}
