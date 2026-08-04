//! OpenAI Responses API 示例
//!
//! 展示 ResponsesModel 的内置工具:WebSearch + CodeInterpreter。
//! 一条请求完成"模型+工具",无需多轮交互。
//!
//! # 运行
//! ```bash
//! cargo run --example basic_responses_api
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选)

use langchainrust::language_models::openai::responses::{
    BuiltinTool, ResponsesConfig, ResponsesModel,
};
use langchainrust::{BaseChatModel, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 配置 Responses API
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let config = ResponsesConfig {
        api_key,
        model: "gpt-4o".to_string(),
        base_url,
        builtin_tools: vec![
            BuiltinTool::WebSearch,       // 模型自动搜索互联网
            BuiltinTool::CodeInterpreter, // 模型自动写代码并执行
        ],
        ..Default::default()
    };
    let model = ResponsesModel::new(config);

    // 2. 使用 WebSearch 内置工具
    let messages = vec![Message::human("2024 年诺贝尔物理学奖颁给了谁?为什么?")];
    let result = model.chat(messages, None).await?;
    println!("=== WebSearch 结果 ===");
    println!("{}", result.content);

    // 3. 使用 CodeInterpreter 内置工具
    let messages = vec![Message::human(
        "计算斐波那契数列的前 20 项,并求它们的平均值。",
    )];
    let result = model.chat(messages, None).await?;
    println!("\n=== CodeInterpreter 结果 ===");
    println!("{}", result.content);

    Ok(())
}
