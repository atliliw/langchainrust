//! Quick Start — create an Agent in 3 lines of code
//!
//! Demonstrates the new langchainrust v0.7.2 API:
//! - `LLMClient::from_env()` — zero-config provider auto-detection
//! - `AgentBuilder` — fluent builder for creating Agents
//! - `FunctionCallingAgent` — now supports any LLM provider
//!
//! # How to run
//!
//! ```bash
//! # Auto-detection (set any one environment variable)
//! export OPENAI_API_KEY="sk-..."
//! cargo run --example quick_start
//!
//! # Or specify a provider explicitly
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run --example quick_start
//! ```

use langchainrust::{AgentBuilder, AgentExecutor, BaseAgent, LLMClient};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========================================
    // Method 1: LLMClient::from_env() — auto-detect
    // ========================================
    println!("=== Method 1: LLMClient::from_env() ===");

    match LLMClient::from_env() {
        Ok(llm) => {
            println!("✓ Detected LLM provider: {}", llm.model_name());

            // Create an Agent with AgentBuilder — 3 lines of code
            let agent = AgentBuilder::new()
                .llm_from_arc(llm.into_inner())
                .system("You are a helpful assistant. Answer concisely.")
                .build()?;

            let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, vec![]);

            let result = executor.invoke("What is 2+2?".into()).await?;
            println!("Answer: {}", result);
        }
        Err(e) => {
            println!("✗ No LLM provider detected: {}", e);
            println!("  Set one of the following environment variables:");
            println!("  - OPENAI_API_KEY");
            println!("  - ANTHROPIC_API_KEY");
            println!("  - OLLAMA_BASE_URL");
        }
    }

    // ========================================
    // Method 2: specify a provider explicitly
    // ========================================
    println!("\n=== Method 2: explicit provider ===");

    // Use OpenAI
    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if !openai_key.is_empty() {
        use langchainrust::OpenAIConfig;

        let config = OpenAIConfig::new(&openai_key).with_model("gpt-4o-mini");
        let llm = LLMClient::openai(config);

        let agent = AgentBuilder::new()
            .llm_from_arc(llm.into_inner())
            .system("You are a math tutor.")
            .build()?;

        println!(
            "✓ OpenAI Agent created: {}",
            agent.system_prompt().unwrap_or("none")
        );
    }

    // ========================================
    // Method 3: read config from env, then override
    // ========================================
    println!("\n=== Method 3: from_env_result() + override ===");

    if !openai_key.is_empty() {
        use langchainrust::OpenAIConfig;

        let config = OpenAIConfig::from_env_result()?.with_model("gpt-4o-mini");
        let llm = LLMClient::openai(config);

        println!("✓ Model: {}", llm.model_name());
    }

    println!("\n=== Done ===");
    Ok(())
}
