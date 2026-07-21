//! Token Counter 示例
//!
//! 展示如何使用 TiktokenCounter 和 TokenTrackingLLM 追踪 token 用量。
//!
//! # 运行
//! ```bash
//! cargo run --example token_counter
//! ```

use langchainrust::{TiktokenCounter, TokenCounter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Token Counter 示例 ===\n");

    // TiktokenCounter 使用与 OpenAI 相同的分词算法
    let counter = TiktokenCounter::default();

    // 计算 token 数量
    let text1 = "Hello, world!";
    let text2 = "这是一段中文文本，用于测试分词器。";
    let text3 = "The quick brown fox jumps over the lazy dog. This is a longer sentence for testing.";

    println!("文本: \"{}\"", text1);
    println!("Token 数: {}\n", counter.count_tokens(text1));

    println!("文本: \"{}\"", text2);
    println!("Token 数: {}\n", counter.count_tokens(text2));

    println!("文本: \"{}\"", text3);
    println!("Token 数: {}\n", counter.count_tokens(text3));

    // TokenTrackingLLM 功能
    println!("TokenTrackingLLM 功能:");
    println!("- 包装任意 BaseChatModel,自动追踪每次调用的 token 用量");
    println!("- 统计 prompt_tokens / completion_tokens / total_tokens");
    println!("- 支持按模型定价计算费用");
    println!("- 可设置 token 预算,超限自动停止");

    println!("\n使用方式:");
    println!("  let tracked = TokenTrackingLLM::new(llm);");
    println!("  let result = tracked.chat(messages, None).await?;");
    println!("  let usage = tracked.usage(); // 获取累计用量");

    Ok(())
}
