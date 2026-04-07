//! 整合测试 - Chain + Memory 工作流

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{BaseChain, LLMChain, SequentialChain, ChatMessageHistory};
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// 测试 LLMChain + Memory
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_chain_with_memory() {
    let config = TestConfig::get();
    let llm = config.openai_chat();
    let mut history = ChatMessageHistory::new();
    
    let chain = LLMChain::new(config.openai_chat(), "You are a helpful assistant. {question}")
        .with_input_key("question");
    
    println!("=== 第一轮对话 ===");
    
    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert("question".to_string(), Value::String("My favorite color is blue.".to_string()));
    
    let result = chain.invoke(inputs).await.unwrap();
    let response = result.get("text").unwrap().as_str().unwrap().to_string();
    
    println!("User: My favorite color is blue.");
    println!("Assistant: {}", response);
    
    history.add_message(Message::human("My favorite color is blue."));
    history.add_message(Message::ai(&response));
    
    println!("\n=== 第二轮对话（测试记忆） ===");
    
    let mut messages = vec![Message::system("Remember what the user tells you.")];
    messages.extend(history.messages().iter().cloned());
    messages.push(Message::human("What's my favorite color?"));
    
    let response2 = llm.chat(messages, None).await.unwrap();
    
    println!("User: What's my favorite color?");
    println!("Assistant: {}", response2.content);
    
    assert!(response2.content.to_lowercase().contains("blue"));
}

/// 测试 SequentialChain 数据流转
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_sequential_chain_data_flow() {
    let config = TestConfig::get();
    
    let chain1 = LLMChain::new(config.openai_chat(), "Extract 3 key topics from: {question}.")
        .with_input_key("question")
        .with_output_key("topics");
    
    let chain2 = LLMChain::new(config.openai_chat(), "Summarize: {topics}");
    
    let pipeline = SequentialChain::new()
        .add_chain(Arc::new(chain1), vec!["question"], vec!["topics"])
        .add_chain(Arc::new(chain2), vec!["topics"], vec!["summary"]);
    
    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert("question".to_string(), Value::String("AI and Machine Learning".to_string()));
    
    println!("=== SequentialChain 数据流转测试 ===");
    
    let results = pipeline.invoke(inputs).await.unwrap();
    
    println!("中间输出 (topics): {:?}", results.get("topics"));
    println!("最终输出 (summary): {:?}", results.get("summary"));
    
    assert!(results.contains_key("topics"));
    assert!(results.contains_key("summary"));
}