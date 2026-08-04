//! Anthropic Extended Thinking 示例
//!
//! 展示 Claude "先想后说"模式:配置 budget_tokens,拿到思考链 thinking_content。
//!
//! # 运行
//! ```bash
//! cargo run --example agent_extended_thinking
//! ```
//!
//! # 环境变量
//! - `ANTHROPIC_API_KEY`:Anthropic API 密钥(必需)

use langchainrust::language_models::providers::anthropic::{AnthropicChat, AnthropicConfig};
use langchainrust::{BaseChatModel, Message, ThinkingConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 配置 Anthropic LLM + Extended Thinking
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("请设置 ANTHROPIC_API_KEY 环境变量");

    let llm = AnthropicChat::new(AnthropicConfig {
        api_key,
        model: "claude-sonnet-5-20250514".to_string(),
        thinking: ThinkingConfig::enabled(10000), // 最多思考 10000 token
        ..Default::default()
    });

    // 2. 发送复杂推理问题
    let messages = vec![Message::human(
        "一个房间里有 3 个开关,控制隔壁房间的 3 盏灯。\
         你只能进隔壁房间一次。如何确定每个开关对应哪盏灯?",
    )];

    let result = llm.chat(messages, None).await?;

    // 3. 输出思考过程和最终回答
    if let Some(thinking) = &result.thinking_content {
        println!("=== 思考过程 ===");
        println!("{}", thinking);
        println!();
    }

    println!("=== 最终回答 ===");
    println!("{}", result.content);

    Ok(())
}
