//! Serve a chat model behind the OpenAI-compatible endpoint.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! export OPENAI_MODEL=gpt-4o-mini   # optional
//! cargo run -p langchainrust --example openai_server --features openai-server
//! ```

use std::sync::Arc;

use langchainrust::agents::{openai_compat_router, ModelBackend};
use langchainrust::{OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = Arc::new(OpenAIChat::new(OpenAIConfig::from_env_result()?));
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let app = openai_compat_router(Arc::new(ModelBackend::new(llm)), model);

    let addr = std::env::var("LC_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("OpenAI-compatible server on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
