//! OpenAI Responses API example
//!
//! Shows the built-in tools of ResponsesModel: WebSearch + CodeInterpreter.
//! "Model + tools" in a single request, no multi-turn interaction needed.
//!
//! # Run
//! ```bash
//! cargo run --example basic_responses_api
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional)

use langchainrust::language_models::openai::responses::{
    BuiltinTool, ResponsesConfig, ResponsesModel,
};
use langchainrust::{BaseChatModel, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure the Responses API
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("please set the OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let config = ResponsesConfig {
        api_key,
        model: "gpt-4o".to_string(),
        base_url,
        builtin_tools: vec![
            BuiltinTool::WebSearch,       // the model searches the web automatically
            BuiltinTool::CodeInterpreter, // the model writes and runs code automatically
        ],
        ..Default::default()
    };
    let model = ResponsesModel::new(config);

    // 2. Use the WebSearch built-in tool
    let messages = vec![Message::human(
        "Who won the 2024 Nobel Prize in Physics, and why?",
    )];
    let result = model.chat(messages, None).await?;
    println!("=== WebSearch result ===");
    println!("{}", result.content);

    // 3. Use the CodeInterpreter built-in tool
    let messages = vec![Message::human(
        "Compute the first 20 terms of the Fibonacci sequence and their average.",
    )];
    let result = model.chat(messages, None).await?;
    println!("\n=== CodeInterpreter result ===");
    println!("{}", result.content);

    Ok(())
}
