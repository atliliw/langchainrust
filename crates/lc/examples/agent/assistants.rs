//! Assistants API 示例
//!
//! 展示如何使用 OpenAIAssistant 进行带工具调用的对话。
//!
//! # 运行
//! ```bash
//! cargo run --example assistants
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)

use langchainrust::{BaseTool, Calculator, OpenAIAssistant, OpenAIConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OpenAI Assistants API 示例 ===\n");

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "sk-test".to_string());

    let config = OpenAIConfig {
        api_key,
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-4o".to_string(),
        ..Default::default()
    };

    // 创建带工具的 Assistant
    let mut registry = langchainrust::ToolRegistry::new();
    registry.register(Arc::new(Calculator::new()) as Arc<dyn BaseTool>);

    println!("创建带计算器工具的 Assistant...");
    let assistant = OpenAIAssistant::create_with_tools(
        config,
        "gpt-4o",
        "你是一个数学助手,可以使用计算器帮助用户计算。",
        registry,
    )
    .await?;

    println!("Assistant ID: {}", assistant.assistant_id());
    println!("\n当用户提问需要计算时,Assistant 会:");
    println!("1. 进入 requires_action 状态");
    println!("2. 调用计算器工具执行计算");
    println!("3. 将结果回传给 OpenAI");
    println!("4. 返回最终回答");

    // 实际调用(需要真实 API key)
    // let answer = assistant.run_once("计算 (23 + 45) * 7").await?;
    // println!("回答: {}", answer);

    println!("\n提示: 取消注释上方代码并设置 API key 即可运行真实调用。");
    Ok(())
}
