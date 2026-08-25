//! Batch API example
//!
//! Demonstrates the BatchClient batch inference flow: submit → poll → fetch results.
//! Suitable for offline evaluation and batch translation/summarization (~50% cost saving).
//!
//! # Run
//! ```bash
//! cargo run --example basic_batch_api
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional)

use langchainrust::core::batch::{BatchClient, BatchProvider, BatchRequest};
use langchainrust::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("please set the OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    // 1. Create the batch client
    let client = BatchClient::new(BatchProvider::OpenAI, &api_key).with_base_url(&base_url);

    // 2. Prepare the batch requests
    let requests = vec![
        BatchRequest {
            custom_id: "translate-1".into(),
            messages: vec![Message::human(
                "Translate the following English into Chinese: Hello, World!",
            )],
            model: "gpt-4o-mini".into(),
            temperature: Some(0.3),
            max_tokens: None,
        },
        BatchRequest {
            custom_id: "translate-2".into(),
            messages: vec![Message::human(
                "Translate the following English into Chinese: Rust is awesome!",
            )],
            model: "gpt-4o-mini".into(),
            temperature: Some(0.3),
            max_tokens: None,
        },
        BatchRequest {
            custom_id: "summarize-1".into(),
            messages: vec![Message::human(
                "Summarize in one sentence: Rust is a systems programming language focused on memory safety and concurrent performance.",
            )],
            model: "gpt-4o-mini".into(),
            temperature: Some(0.3),
            max_tokens: None,
        },
    ];

    println!("Submitting {} batch requests...", requests.len());

    // 3. Submit and wait (auto-poll every 5s, up to 5 minutes)
    let results = client.submit_and_wait(requests, 5_000, 300_000).await?;

    // 4. Print the results
    println!("\n=== Batch results ===");
    for result in &results {
        match &result.result {
            Ok(llm_result) => {
                println!("[{}] {}", result.custom_id, llm_result.content);
            }
            Err(e) => {
                println!("[{}] error: {}", result.custom_id, e);
            }
        }
    }

    Ok(())
}
