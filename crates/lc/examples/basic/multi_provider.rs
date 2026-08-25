//! Multiple LLM provider example
//!
//! Shows how to switch between OpenAI / Ollama / DeepSeek,
//! calling different backends with the same messages.
//!
//! # Run
//! ```bash
//! # Default: openai
//! cargo run --example basic_multi_provider
//! # Switch to ollama (requires Ollama running locally)
//! $env:PROVIDER="ollama"; cargo run --example basic_multi_provider
//! # Switch to deepseek
//! $env:PROVIDER="deepseek"; cargo run --example basic_multi_provider
//! ```
//!
//! # Environment variables
//! - `PROVIDER`: openai (default) / ollama / deepseek
//! - `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`: key for the corresponding provider

use langchainrust::schema::Message;
use langchainrust::{
    BaseChatModel, DeepSeekChat, DeepSeekConfig, OllamaChat, OllamaConfig, OpenAIChat, OpenAIConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = std::env::var("PROVIDER").unwrap_or_else(|_| "openai".to_string());

    let messages = vec![
        Message::system("You are a helpful assistant. Answer in one sentence."),
        Message::human("Introduce yourself in one sentence."),
    ];

    let answer = match provider.as_str() {
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY").expect("please set OPENAI_API_KEY");
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            let llm = OpenAIChat::new(OpenAIConfig {
                api_key,
                base_url,
                model: "gpt-4o-mini".to_string(),
                ..Default::default()
            });
            llm.chat(messages, None).await?.content
        }
        "ollama" => {
            let llm = OllamaChat::with_config(OllamaConfig {
                base_url: std::env::var("OLLAMA_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
                model: "llama3.2".to_string(),
                ..Default::default()
            });
            llm.chat(messages, None).await?.content
        }
        "deepseek" => {
            let api_key = std::env::var("DEEPSEEK_API_KEY").expect("please set DEEPSEEK_API_KEY");
            let llm = DeepSeekChat::new(DeepSeekConfig {
                api_key,
                model: "deepseek-chat".to_string(),
                ..Default::default()
            });
            llm.chat(messages, None).await?.content
        }
        other => {
            return Err(
                format!("unknown provider: {other} (options: openai/ollama/deepseek)").into(),
            )
        }
    };

    println!("[{provider}] Answer:\n{answer}");
    Ok(())
}
