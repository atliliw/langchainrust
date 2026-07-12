//! Token Counter 集成测试 - 需要 API Key
//!
//! 测试真实 LLM 调用的 token 统计与成本估算。
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_token_counter -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::schema::Message;
use langchainrust::{ModelPricing, TokenTrackingLLM};

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_token_tracking_with_real_llm() {
    let llm = TestConfig::get().openai_chat();
    let tracked = TokenTrackingLLM::for_openai(llm).expect("创建 TokenTrackingLLM 失败");

    let messages = vec![
        Message::system("你是一个助手,一句话回答。"),
        Message::human("什么是 Rust?"),
    ];
    let response = tracked.chat(messages, None).await.expect("chat 失败");
    println!("回答: {}", response.content);

    let usage = tracked.get_usage();
    println!(
        "prompt: {}, completion: {}, total: {}",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    );
    assert!(usage.total_tokens > 0, "总 token 应大于 0");

    let cost = tracked.estimate_cost(&ModelPricing::gpt4o_mini());
    println!("估算成本: ${:.6}", cost);
    assert!(cost > 0.0);
}
