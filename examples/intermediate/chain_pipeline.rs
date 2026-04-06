// examples/intermediate/chain_pipeline.rs
//! 中级示例 3: Chain 链式调用
//!
//! 运行: cargo run --example chain_pipeline
//!
//! 功能: 演示如何使用 LLMChain 和 SequentialChain

use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    LLMChain, LLMChainBuilder, SequentialChain,
    ConversationBufferMemory,
};
use langchainrust::prompts::ChatPromptTemplate;
use langchainrust::schema::Message;
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
    
    let llm = Arc::new(OpenAIChat::new(config));
    
    // ========== 1. 简单 LLMChain ==========
    println!("--- 1. 简单 LLMChain ---\n");
    
    // 创建提示词模板
    let template = ChatPromptTemplate::new(vec![
        Message::system("你是一个{role}。"),
        Message::human("{question}"),
    ]);
    
    // 创建 LLMChain
    let chain = LLMChainBuilder::new()
        .llm(llm.clone())
        .template(template)
        .build()?;
    
    // 执行
    let mut inputs = HashMap::new();
    inputs.insert("role".to_string(), "编程专家".to_string());
    inputs.insert("question".to_string(), "什么是 Rust 的所有权？用一句话解释。".to_string());
    
    println!("问题: 什么是 Rust 的所有权？\n");
    
    match chain.invoke(inputs).await {
        Ok(result) => {
            println!("回答: {}\n", result);
        }
        Err(e) => {
            eprintln!("错误: {}", e);
        }
    }
    
    // ========== 2. SequentialChain 多步骤链 ==========
    println!("--- 2. SequentialChain 多步骤链 ---\n");
    
    // 步骤 1: 分析主题
    let analyze_template = ChatPromptTemplate::new(vec![
        Message::system("你是一个分析师。请简要分析以下主题。"),
        Message::human("主题: {topic}"),
    ]);
    
    let analyze_chain = LLMChainBuilder::new()
        .llm(llm.clone())
        .template(analyze_template)
        .output_key("analysis")
        .build()?;
    
    // 步骤 2: 生成总结
    let summarize_template = ChatPromptTemplate::new(vec![
        Message::system("你是一个总结专家。请根据以下分析生成一个简洁的总结。"),
        Message::human("分析结果: {analysis}"),
    ]);
    
    let summarize_chain = LLMChainBuilder::new()
        .llm(llm.clone())
        .template(summarize_template)
        .output_key("summary")
        .build()?;
    
    // 创建顺序链
    let sequential = SequentialChain::new()
        .add_chain(analyze_chain)
        .add_chain(summarize_chain);
    
    println!("执行两步链: 分析 -> 总结\n");
    println!("主题: 人工智能的未来\n");
    
    let mut seq_inputs = HashMap::new();
    seq_inputs.insert("topic".to_string(), "人工智能的未来发展趋势".to_string());
    
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
    
    // ========== 3. Chain + Memory ==========
    println!("--- 3. Chain + Memory ---\n");
    
    let memory = Arc::new(ConversationBufferMemory::new());
    
    // 创建带记忆的 Chain
    let memory_template = ChatPromptTemplate::new(vec![
        Message::system("你是一个友好的助手。"),
        Message::human("{history}"),
        Message::human("{question}"),
    ]);
    
    let memory_chain = LLMChainBuilder::new()
        .llm(llm.clone())
        .template(memory_template)
        .memory(memory.clone())
        .build()?;
    
    // 多轮对话
    let questions = vec![
        "我叫小红",
        "我喜欢 Python 编程",
        "我刚才说我叫什么名字？",
    ];
    
    println!("带记忆的多轮对话:\n");
    
    for question in questions {
        println!("用户: {}", question);
        
        let mut inputs = HashMap::new();
        inputs.insert("question".to_string(), question.to_string());
        
        match memory_chain.invoke(inputs).await {
            Ok(response) => {
                println!("助手: {}\n", response);
            }
            Err(e) => {
                eprintln!("错误: {}", e);
            }
        }
    }
    
    println!("=== 示例完成 ===");
    Ok(())
}