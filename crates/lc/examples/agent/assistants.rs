//! Assistants API example
//!
//! Shows how to use OpenAIAssistant for a conversation with tool calls.
//!
//! # Run
//! ```bash
//! cargo run --example assistants
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)

use langchainrust::{BaseTool, Calculator, OpenAIAssistant, OpenAIConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OpenAI Assistants API example ===\n");

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "sk-test".to_string());

    let config = OpenAIConfig {
        api_key,
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-4o".to_string(),
        ..Default::default()
    };

    // Create an Assistant with tools
    let mut registry = langchainrust::ToolRegistry::new();
    registry.register(Arc::new(Calculator::new()) as Arc<dyn BaseTool>);

    println!("Creating an Assistant with the calculator tool...");
    let assistant = OpenAIAssistant::create_with_tools(
        config,
        "gpt-4o",
        "You are a math assistant that can use a calculator to help users compute.",
        registry,
    )
    .await?;

    println!("Assistant ID: {}", assistant.assistant_id());
    println!("\nWhen the user asks something that needs computation, the Assistant will:");
    println!("1. enter the requires_action state");
    println!("2. call the calculator tool to compute");
    println!("3. send the result back to OpenAI");
    println!("4. return the final answer");

    // Real call (needs a real API key)
    // let answer = assistant.run_once("Compute (23 + 45) * 7").await?;
    // println!("Answer: {}", answer);

    println!("\nNote: uncomment the code above and set the API key to run a real call.");
    Ok(())
}
