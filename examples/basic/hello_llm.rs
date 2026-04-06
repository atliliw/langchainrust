// examples/basic/hello_llm.rs
//! 基础示例 1: 简单的 LLM 调用
//!
//! 运行: cargo run --example hello_llm
//!
//! 功能: 演示如何创建 OpenAI 客户端并进行简单的对话

use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 基础示例 1: Hello LLM ===\n");
    
    // 1. 创建配置
    // 注意: 实际使用时请替换为您自己的 API Key
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "your-api-key-here".to_string()),
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        temperature: Some(0.7),
        max_tokens: Some(500),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        organization: None,
    };
    
    println!("配置信息:");
    println!("  模型: {}", config.model);
    println!("  基础 URL: {}", config.base_url);
    println!();
    
    // 2. 创建 LLM 客户端
    let llm = OpenAIChat::new(config);
    
    // 3. 创建消息
    let messages = vec![
        Message::system("你是一个友好的助手，用简洁的中文回答问题。"),
        Message::human("什么是 Rust 语言？用一两句话介绍。"),
    ];
    
    println!("发送消息: 什么是 Rust 语言？\n");
    
    // 4. 调用 LLM
    match llm.chat(messages, None).await {
        Ok(response) => {
            println!("LLM 回复:\n{}\n", response.content);
        }
        Err(e) => {
            eprintln!("错误: {}", e);
            eprintln!("提示: 请确保设置了正确的 OPENAI_API_KEY 环境变量");
        }
    }
    
    println!("=== 示例完成 ===");
    Ok(())
}