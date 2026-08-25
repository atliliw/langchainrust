//! Anthropic Extended Thinking example
//!
//! Shows Claude's "think before you speak" mode: configure budget_tokens, get the chain of
//! thought in thinking_content.
//!
//! # Run
//! ```bash
//! cargo run --example agent_extended_thinking
//! ```
//!
//! # Environment variables
//! - `ANTHROPIC_API_KEY`: Anthropic API key (required)

use langchainrust::language_models::providers::anthropic::{AnthropicChat, AnthropicConfig};
use langchainrust::{BaseChatModel, Message, ThinkingConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure the Anthropic LLM with Extended Thinking
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("please set the ANTHROPIC_API_KEY environment variable");

    let llm = AnthropicChat::new(AnthropicConfig {
        api_key,
        model: "claude-sonnet-5-20250514".to_string(),
        thinking: ThinkingConfig::enabled(10000), // think at most 10000 tokens
        ..Default::default()
    });

    // 2. Send a complex reasoning question
    let messages = vec![Message::human(
        "There are 3 switches in one room controlling 3 lights in the room next door.\
         You may enter the next room only once. How do you determine which switch controls which light?",
    )];

    let result = llm.chat(messages, None).await?;

    // 3. Print the reasoning process and the final answer
    if let Some(thinking) = &result.thinking_content {
        println!("=== Thinking process ===");
        println!("{}", thinking);
        println!();
    }

    println!("=== Final answer ===");
    println!("{}", result.content);

    Ok(())
}
