// examples/basic/streaming.rs
//! 基础示例 2: 流式输出
//!
//! 运行: cargo run --example streaming
//!
//! 功能: 演示如何使用流式 API 逐字接收 LLM 输出

use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;
use langchainrust::language_models::openai::OpenAIError;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 基础示例 2: 流式输出 ===\n");
    
    // 1. 创建配置（启用流式）
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "your-api-key-here".to_string()),
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true,  // 启用流式输出
        temperature: Some(0.7),
        max_tokens: Some(500),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        organization: None,
    };
    
    let llm = OpenAIChat::new(config);
    
    // 2. 创建消息
    let messages = vec![
        Message::system("你是一个创意作家，擅长写短篇故事。"),
        Message::human("请写一个关于机器人学习人类情感的短故事，大约100字。"),
    ];
    
    println!("发送请求（流式输出）...\n");
    println!("--- 开始输出 ---\n");
    
    // 3. 流式调用
    match llm.stream_chat(messages, None).await {
        Ok(mut stream) => {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(token) => {
                        // 逐字打印
                        print!("{}", token);
                        // 强制刷新输出
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();
                    }
                    Err(OpenAIError::Http(msg)) => {
                        eprint!("\n[HTTP错误: {}]", msg);
                    }
                    Err(OpenAIError::Api(msg)) => {
                        eprint!("\n[API错误: {}]", msg);
                    }
                    Err(OpenAIError::Parse(msg)) => {
                        eprint!("\n[解析错误: {}]", msg);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("无法创建流: {}", e);
            eprintln!("提示: 请确保设置了正确的 OPENAI_API_KEY 环境变量");
        }
    }
    
    println!("\n\n--- 输出结束 ---\n");
    println!("=== 示例完成 ===");
    Ok(())
}