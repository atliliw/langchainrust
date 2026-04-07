//! LLM 测试 - 需要 API Key
//!
//! 测试 OpenAIChat 的基础聊天和流式输出功能

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use futures_util::StreamExt;

/// 测试基础聊天功能
///
/// 测试内容：
/// - 发送单轮对话请求
/// - 验证返回内容非空
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_chat_single_turn() {
    let llm = TestConfig::get().openai_chat();
    
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::human("What is Rust? Answer in one sentence."),
    ];
    
    let response = llm.chat(messages, None).await.unwrap();
    
    println!("Response: {}", response.content);
    assert!(!response.content.is_empty());
}

/// 测试多轮对话
///
/// 测试内容：
/// - 发送包含历史记录的多轮对话
/// - 验证 LLM 能记住上下文中的名字
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_chat_multi_turn() {
    let llm = TestConfig::get().openai_chat();
    
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::human("My name is Alice."),
        Message::ai("Nice to meet you, Alice!"),
        Message::human("What's my name?"),
    ];
    
    let response = llm.chat(messages, None).await.unwrap();
    
    println!("Response: {}", response.content);
    assert!(response.content.to_lowercase().contains("alice"));
}

/// 测试流式输出
///
/// 测试内容：
/// - 使用 stream_chat 获取流式响应
/// - 逐个 token 接收并拼接完整响应
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_chat_streaming() {
    let llm = TestConfig::get().openai_chat();
    
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::human("Count from 1 to 5."),
    ];
    
    let mut stream = llm.stream_chat(messages, None).await.unwrap();
    
    let mut full_response = String::new();
    
    while let Some(chunk) = stream.next().await {
        if let Ok(token) = chunk {
            print!("{}", token);
            full_response.push_str(&token);
        }
    }
    
    println!("\nFull response: {}", full_response);
    assert!(!full_response.is_empty());
}