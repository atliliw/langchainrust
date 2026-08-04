//! LLMChain 示例
//!
//! 展示 LLMChain 用模板变量调用 LLM。
//!
//! # 运行
//! ```bash
//! cargo run --example chains_llm_chain
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)

use langchainrust::{BaseChain, LLMChain, OpenAIChat, OpenAIConfig};
use serde_json::Value;
use std::collections::HashMap;

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

    let chain =
        LLMChain::new(llm, "Explain this topic in one sentence: {topic}").with_input_key("topic");

    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert(
        "topic".to_string(),
        Value::String("Rust 所有权".to_string()),
    );

    let result = chain.invoke(inputs).await?;
    println!("回答: {}", result.get("text").unwrap());
    Ok(())
}
