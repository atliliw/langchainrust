//! Example / deployable streaming agent served over Server-Sent Events (B8, v0.22.4).
//!
//! Exposes a [`StreamingFunctionCallingAgent`](langchainrust::agents::StreamingFunctionCallingAgent)
//! (real streaming model) as an HTTP endpoint that pushes agent events to clients as
//! they happen, instead of buffering one JSON response:
//!
//! | Route | Description |
//! |---|---|
//! | `POST /agent/stream` | JSON `{"input":"..."}` in → `text/event-stream` of agent events out (programmatic / `fetch()` clients) |
//! | `GET /agent/stream?input=...` | Same run for browser-native `EventSource` (GET-only) |
//!
//! # Event stream
//!
//! Named SSE events, each with a monotonic numeric `id`:
//!
//! - `event: text` — one streamed token, `data: {"content":"..."}`
//! - `event: tool_call` — function-call state transition
//!   (`started` / `arguments_streaming` / `arguments_complete` / `executing` /
//!   `completed` / `failed`)
//! - `event: final_answer` — the complete answer
//! - `event: error` — model/stream failure (`{"message":"..."}`), still followed by
//! - `event: done` — unnumbered terminal frame
//!
//! Idle connections get a `: keep-alive` comment every 20s. On reconnect a client may
//! send `Last-Event-ID: <n>` (header on POST, or `last_event_id` in the JSON body) and
//! frames at or below that id are suppressed within the new run.
//!
//! # Run (local testing)
//!
//! ```powershell
//! cargo run -p langchainrust --example agent_sse_server
//! ```
//!
//! # Consume it
//!
//! ```powershell
//! # POST (any HTTP client):
//! curl -N -X POST http://127.0.0.1:8090/agent/stream `
//!   -H "content-type: application/json" `
//!   -d '{\"input\":\"Write a haiku about Rust\"}'
//!
//! # Native browser EventSource (GET route):
//! #   new EventSource("http://127.0.0.1:8090/agent/stream?input=hello")
//! ```
//!
//! # Runtime configuration (environment variables)
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `AGENT_SSE_HOST` | `127.0.0.1` | Bind address (default local-only; `0.0.0.0` is an explicit, auth-less choice) |
//! | `AGENT_SSE_PORT` | `8090` | Listening port |
//! | `AGENT_SSE_API_KEY` | built-in key | API key for the OpenAI-compatible endpoint |
//! | `AGENT_SSE_BASE_URL` | built-in URL | OpenAI-compatible endpoint (Alibaba Cloud MaaS) |
//! | `AGENT_SSE_MODEL` | `qwen3.7-max-2026-06-08` | Model name (switch to an accessible one if needed) |
//!
//! Default bind is loopback only; the endpoint has no built-in auth. Binding `0.0.0.0`
//! exposes it to anyone who can reach the port, so add reverse-proxy auth for remote use.
//! The built-in CORS layer only allows `http://localhost*` / `http://127.0.0.1*` origins.

use std::sync::Arc;

use langchainrust::agents::{
    serve_agent_sse_on, AgentStreamFactory, StreamingFunctionCallingAgent,
};
use langchainrust::{OpenAIChat, OpenAIConfig};
use tokio::net::TcpListener;

/// A streaming model pointing at a real endpoint; key/url/model are all overridable
/// through environment variables.
fn real_llm() -> OpenAIChat {
    OpenAIChat::new(OpenAIConfig {
        api_key: std::env::var("AGENT_SSE_API_KEY")
            .unwrap_or_else(|_| "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string()),
        base_url: std::env::var("AGENT_SSE_BASE_URL").unwrap_or_else(|_| {
            "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                .to_string()
        }),
        model: std::env::var("AGENT_SSE_MODEL")
            .unwrap_or_else(|_| "qwen3.7-max-2026-06-08".to_string()),
        streaming: true,
        temperature: Some(0.3),
        max_tokens: Some(1024),
        ..Default::default()
    })
}

#[tokio::main]
async fn main() {
    // 1. Configuration from the environment (loopback default; 0.0.0.0 is explicit).
    let host = std::env::var("AGENT_SSE_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("AGENT_SSE_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8090);

    // 2. One shared, cheaply-clonable streaming agent.
    let agent = Arc::new(StreamingFunctionCallingAgent::new(real_llm()));

    // 3. A per-request factory: each HTTP request gets a fresh event stream.
    let factory: Arc<dyn AgentStreamFactory> = Arc::new(move |input: String| {
        let agent = agent.clone();
        async move { agent.invoke_stream(input).await }
    });

    // 4. Bind (default 127.0.0.1 = local only; 0.0.0.0 must be chosen explicitly).
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to bind {host}:{port}: {e}");
            std::process::exit(1);
        });
    let bound = listener.local_addr().unwrap();

    println!("Streaming agent SSE server started ✅");
    println!("  POST http://{bound}/agent/stream   body: {{\"input\":\"...\"}}");
    println!("  GET  http://{bound}/agent/stream?input=...   (native EventSource)");
    println!("events: text* → final_answer | error, then done; 20s keep-alive");
    println!("press Ctrl+C to stop.");

    // 5. Serve until the process is stopped; dropping an SSE response cancels the run.
    serve_agent_sse_on(factory, listener)
        .await
        .unwrap_or_else(|e| {
            eprintln!("SSE service exited with error: {e}");
            std::process::exit(1);
        });
}
