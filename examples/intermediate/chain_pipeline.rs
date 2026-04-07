// examples/intermediate/chain_pipeline.rs
//! 中级示例 3: Chain 链式调用
//!
//! 运行: cargo run --example chain_pipeline
//!
//! 功能: 演示如何使用 LLMChain 和 SequentialChain

use langchainrust::{
    OpenAIChat, OpenAIConfig,
    LLMChain, LLMChainBuilder, SequentialChain, BaseChain,
};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 中级示例 3: Chain 链式调用 ===\n");
    
    // 创建 LLM
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
    
    // ========== 1. 简单 LLMChain ==========
    println!("--- 1. 简单 LLMChain ---\n");
    
    let llm1 = OpenAIChat::new(config.clone());
    let chain = LLMChain::new(llm1, "你是一个{role}。请回答: {question}");
    
    // 执行
    let mut inputs = HashMap::new();
    inputs.insert("role".to_string(), serde_json::Value::String("编程专家".to_string()));
    inputs.insert("question".to_string(), serde_json::Value::String("什么是 Rust 的所有权？用一句话解释。".to_string()));
    
    println!("问题: 什么是 Rust 的所有权？\n");
    
    match chain.invoke(inputs).await {
        Ok(result) => {
            if let Some(text) = result.get("text") {
                println!("回答: {}\n", text);
            }
        }
        Err(e) => {
            eprintln!("错误: {}", e);
            eprintln!("提示: 请确保设置了正确的 OPENAI_API_KEY 环境变量");
        }
    }
    
    // ========== 2. SequentialChain 多步骤链 ==========
    println!("--- 2. SequentialChain 多步骤链 ---\n");
    
    // 步骤 1: 分析主题
    let llm2 = OpenAIChat::new(config.clone());
    let analyze_chain = Arc::new(LLMChain::new(llm2, "你是一个分析师。请简要分析以下主题: {topic}"));
    
    // 步骤 2: 生成总结
    let llm3 = OpenAIChat::new(config.clone());
    let summarize_chain = Arc::new(LLMChain::new(llm3, "你是一个总结专家。请根据以下分析生成一个简洁的总结: {analysis}"));
    
    // 创建顺序链
    let sequential = SequentialChain::new()
        .add_chain(analyze_chain, vec!["topic"], vec!["analysis"])
        .add_chain(summarize_chain, vec!["analysis"], vec!["summary"]);
    
    println!("执行两步链: 分析 -> 总结\n");
    println!("主题: 人工智能的未来\n");
    
    let mut seq_inputs = HashMap::new();
    seq_inputs.insert("topic".to_string(), serde_json::Value::String("人工智能的未来发展趋势".to_string()));
    
    match sequential.invoke(seq_inputs).await {
        Ok(results) => {
            println!("--- 执行结果 ---");
            if let Some(analysis) = results.get("analysis") {
                println!("\n步骤 1 - 分析:\n{}\n", analysis);
            }
            if let Some(summary) = results.get("summary") {
                println!("步骤 2 - 总结:\n{}\n", summary);
            }
        }
        Err(e) => {
            eprintln!("错误: {}", e);
        }
    }
    
    // ========== 3. 使用 LLMChainBuilder ==========
    println!("--- 3. 使用 LLMChainBuilder ---\n");
    
    let llm4 = OpenAIChat::new(config);
    let builder_chain = LLMChainBuilder::new(llm4, "请用{style}的风格解释{topic}")
        .input_key("topic")
        .output_key("answer")
        .name("style_chain")
        .build();
    
    let mut builder_inputs = HashMap::new();
    builder_inputs.insert("style".to_string(), serde_json::Value::String("通俗易懂".to_string()));
    builder_inputs.insert("topic".to_string(), serde_json::Value::String("量子计算".to_string()));
    
    println!("问题: 用通俗易懂的风格解释量子计算\n");
    
    match builder_chain.invoke(builder_inputs).await {
        Ok(result) => {
            if let Some(answer) = result.get("answer") {
                println!("回答: {}\n", answer);
            }
        }
        Err(e) => {
            eprintln!("错误: {}", e);
        }
    }
    
    println!("=== 示例完成 ===");
    Ok(())
}