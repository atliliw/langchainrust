//! 基础聊天示例
//!
//! 展示如何使用 OpenAI 进行一次简单对话。
//!
//! # 运行
//! ```bash
//! cargo run --example basic_chat
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选,默认官方)

use langchainrust::schema::Message;
use langchainrust::{BaseChatModel, OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    let messages = vec![
        Message::system("你是一个 Rust 专家,回答简洁。"),
        Message::human("什么是 Rust 的所有权机制?一句话回答。"),
    ];

    let response = llm.chat(messages, None).await?;
    println!("回答:\n{}", response.content);

    Ok(())
}
