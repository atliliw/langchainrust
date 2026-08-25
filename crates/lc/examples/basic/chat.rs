//! Basic chat example
//!
//! Demonstrates a simple conversation with OpenAI.
//!
//! # Run
//! ```bash
//! cargo run --example basic_chat
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional, defaults to official)

use langchainrust::schema::Message;
use langchainrust::{BaseChatModel, OpenAIChat, OpenAIConfig};

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

    let messages = vec![
        Message::system("You are a Rust expert. Answer concisely."),
        Message::human("What is Rust's ownership mechanism? Answer in one sentence."),
    ];

    let response = llm.chat(messages, None).await?;
    println!("Answer:\n{}", response.content);

    Ok(())
}
