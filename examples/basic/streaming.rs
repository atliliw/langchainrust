//! 流式输出示例
//!
//! 展示如何使用 `stream_chat` 逐 token 输出响应,
//! 适合聊天界面实时显示。
//!
//! # 运行
//! ```bash
//! cargo run --example basic_streaming
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选)

use futures_util::StreamExt;
use langchainrust::{BaseChatModel, OpenAIChat, OpenAIConfig};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
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
        Message::system("你是一个 helpful assistant。"),
        Message::human("从 1 数到 5。"),
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
