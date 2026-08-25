//! LLMChain example
//!
//! Shows LLMChain calling an LLM with template variables.
//!
//! # Run
//! ```bash
//! cargo run --example chains_llm_chain
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)

use langchainrust::{BaseChain, LLMChain, OpenAIChat, OpenAIConfig};
use serde_json::Value;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("please set the OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    let chain =
        LLMChain::new(llm, "Explain this topic in one sentence: {topic}").with_input_key("topic");

    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert(
        "topic".to_string(),
        Value::String("Rust ownership".to_string()),
    );

    let result = chain.invoke(inputs).await?;
    println!("Answer: {}", result.get("text").unwrap());
    Ok(())
}
