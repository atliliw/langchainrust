//! Chain 测试 - 需要 API Key
//!
//! 测试 LLMChain 和 SequentialChain 的功能

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{BaseChain, LLMChain, SequentialChain};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// 测试 LLMChain 基础功能
///
/// 测试内容：
/// - 创建 LLMChain 并使用默认的 "question" 输入键
/// - 验证返回结果包含 "text" 输出键
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_llm_chain_basic() {
    let llm = TestConfig::get().openai_chat();
    
    // LLMChain 默认使用 "question" 作为输入键
    let chain = LLMChain::new(llm, "Explain this topic in one sentence: {question}");
    
    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert("question".to_string(), Value::String("Machine Learning".to_string()));
    
    let result = chain.invoke(inputs).await.unwrap();
    
    println!("Result: {:?}", result);
    assert!(result.contains_key("text"));
}

/// 测试 LLMChain 多变量模板
///
/// 测试内容：
/// - 使用多个模板变量 {style} 和 {question}
/// - 验证变量替换和 LLM 调用正确
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_llm_chain_with_variables() {
    let llm = TestConfig::get().openai_chat();
    
    let chain = LLMChain::new(
        llm,
        "Write a {style} description of {question} in one paragraph.",
    );
    
    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert("style".to_string(), Value::String("technical".to_string()));
    inputs.insert("question".to_string(), Value::String("Rust programming language".to_string()));
    
    let result = chain.invoke(inputs).await.unwrap();
    
    println!("Result: {:?}", result);
    assert!(result.contains_key("text"));
}

/// 测试 LLMChain 自定义输入键
///
/// 测试内容：
/// - 使用 with_input_key 修改默认输入键为 "topic"
/// - 验证自定义输入键生效
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_llm_chain_custom_input_key() {
    let llm = TestConfig::get().openai_chat();
    
    // 使用 with_input_key 修改默认输入键
    let chain = LLMChain::new(llm, "Explain this topic: {topic}")
        .with_input_key("topic");
    
    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert("topic".to_string(), Value::String("Machine Learning".to_string()));
    
    let result = chain.invoke(inputs).await.unwrap();
    
    println!("Result: {:?}", result);
    assert!(result.contains_key("text"));
}

/// 测试 SequentialChain 顺序执行
///
/// 测试内容：
/// - 链式执行多个 Chain
/// - 第一个 Chain 输出作为第二个 Chain 输入
/// - 验证最终结果包含所有中间输出
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_sequential_chain() {
    let config = TestConfig::get();
    
    let llm1 = config.openai_chat();
    let llm2 = config.openai_chat();
    
    // Chain 1: 列出主题的关键特性
    let chain1 = LLMChain::new(llm1, "List 3 key features of {topic}.")
        .with_output_key("features");
    
    // Chain 2: 总结特性
    let chain2 = LLMChain::new(llm2, "Summarize these features briefly: {features}");
    
    let pipeline = SequentialChain::new()
        .add_chain(Arc::new(chain1), vec!["topic"], vec!["features"])
        .add_chain(Arc::new(chain2), vec!["features"], vec!["summary"]);
    
    let mut inputs: HashMap<String, Value> = HashMap::new();
    inputs.insert("topic".to_string(), Value::String("Rust".to_string()));
    
    let results = pipeline.invoke(inputs).await.unwrap();
    
    println!("Results: {:?}", results);
    assert!(results.contains_key("features"));
    assert!(results.contains_key("summary"));
}