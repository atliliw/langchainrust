//! SequentialChain example
//!
//! Shows a sequential chain: Chain1's output feeds Chain2's input.
//!
//! # Run
//! ```bash
//! cargo run --example chains_sequential_chain
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)

use langchainrust::{BaseChain, LLMChain, OpenAIChat, OpenAIConfig, SequentialChain};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

fn make_llm() -> OpenAIChat {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("please set the OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Chain 1: list the topic's key features
    let chain1 =
        LLMChain::new(make_llm(), "List 3 key features of {topic}.").with_output_key("features");
    // Chain 2: summarize the features
    let chain2 = LLMChain::new(make_llm(), "Summarize these features briefly: {features}");

    let pipeline = SequentialChain::new()
        .add_chain(Arc::new(chain1), vec!["topic"], vec!["features"])
        .add_chain(Arc::new(chain2), vec!["features"], vec!["summary"]);

    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert("topic".to_string(), Value::String("Rust".to_string()));

    let results = pipeline.invoke(inputs).await?;
    println!("Features: {}", results.get("features").unwrap());
    println!("Summary: {}", results.get("summary").unwrap());
    Ok(())
}
