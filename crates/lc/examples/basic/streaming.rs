//! Streaming output example
//!
//! Shows how to emit a response token-by-token with `stream_chat`,
//! suitable for real-time display in a chat UI.
//!
//! # Run
//! ```bash
//! cargo run --example basic_streaming
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional)

use futures_util::StreamExt;
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
        streaming: true,
        ..Default::default()
    });

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::human("Count from 1 to 5."),
    ];

    let mut stream = llm.stream_chat(messages, None).await?;
    while let Some(chunk) = stream.next().await {
        if let Ok(token) = chunk {
            print!("{}", token);
        }
    }
    println!();

    Ok(())
}
