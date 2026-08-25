//! Token Counter example
//!
//! Shows how to track token usage with TiktokenCounter and TokenTrackingLLM.
//!
//! # Run
//! ```bash
//! cargo run --example token_counter
//! ```

use langchainrust::{TiktokenCounter, TokenCounter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Token Counter example ===\n");

    // TiktokenCounter uses the same tokenization algorithm as OpenAI
    let counter = TiktokenCounter::new()?;

    // Count tokens on different texts (the counter handles CJK too)
    let text1 = "Hello, world!";
    let text2 = "This sentence is a Chinese paragraph used to test tokenization.";
    let text3 =
        "The quick brown fox jumps over the lazy dog. This is a longer sentence for testing.";

    println!("text: \"{}\"", text1);
    println!("tokens: {}\n", counter.count_tokens(text1));

    println!("text: \"{}\"", text2);
    println!("tokens: {}\n", counter.count_tokens(text2));

    println!("text: \"{}\"", text3);
    println!("tokens: {}\n", counter.count_tokens(text3));

    // TokenTrackingLLM features
    println!("TokenTrackingLLM features:");
    println!("- Wraps any BaseChatModel and automatically tracks per-call token usage");
    println!("- Reports prompt_tokens / completion_tokens / total_tokens");
    println!("- Supports cost estimation by model pricing");
    println!("- Optional token budget stops automatically when exceeded");

    println!("\nUsage:");
    println!("  let tracked = TokenTrackingLLM::new(llm);");
    println!("  let result = tracked.chat(messages, None).await?;");
    println!("  let usage = tracked.usage(); // cumulative usage");

    Ok(())
}
