//! RetrievalQA 测试
//!
//! 测试一站式检索问答 Chain 的核心功能：
//! 1. 检索相关文档 - 从向量存储检索最相关文档
//! 2. 组装 Prompt - 将上下文和问题组合成 prompt
//! 3. LLM 生成答案 - 基于上下文生成答案
//! 4. 返回来源文档 - 可选返回检索到的文档

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{RetrievalQA, BaseChain, SimilarityRetriever, Document};
use langchainrust::retrieval::RetrieverTrait;
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value;

/// 测试 RetrievalQA 基础问答
///
/// 功能验证：
/// - 检索相关文档
/// - 组装 prompt（上下文 + 问题）
/// - LLM 生成答案
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_retrieval_qa_basic() {
    let config = TestConfig::get();
    
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));
    
    let retriever = Arc::new(SimilarityRetriever::new(store.clone(), embeddings.clone()));
    
    retriever.add_documents(vec![
        Document::new("Rust 是一门系统编程语言，注重安全、并发和性能。"),
        Document::new("Python 是一门脚本语言，语法简洁，适合快速开发。"),
        Document::new("JavaScript 主要用于 Web 开发，是浏览器唯一支持的编程语言。"),
    ]).await.unwrap();
    
    let qa = RetrievalQA::new(config.openai_chat(), retriever)
        .with_k(2)
        .with_verbose(true);
    
    println!("\n=== 测试：RetrievalQA 基础问答 ===");
    
    let inputs = HashMap::from([
        ("query".to_string(), Value::String("什么是 Rust？".to_string()))
    ]);
    
    let result = qa.invoke(inputs).await.unwrap();
    
    let answer = result.get("result").unwrap().as_str().unwrap();
    println!("答案: {}", answer);
    
    assert!(!answer.is_empty(), "答案不应为空");
}

/// 测试 RetrievalQA 返回来源文档
///
/// 功能验证：
/// - with_return_source_documents(true) 配置返回来源
/// - result 包含 source_documents 字段
/// - source_documents 包含检索到的文档列表
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_retrieval_qa_with_sources() {
    let config = TestConfig::get();
    
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));
    
    let retriever = Arc::new(SimilarityRetriever::new(store.clone(), embeddings.clone()));
    
    retriever.add_documents(vec![
        Document::new("LangChain 是一个 LLM 应用开发框架。"),
        Document::new("RustLangChain 是 LangChain 的 Rust 实现。"),
    ]).await.unwrap();
    
    let qa = RetrievalQA::new(config.openai_chat(), retriever)
        .with_k(2)
        .with_return_source_documents(true)
        .with_verbose(true);
    
    println!("\n=== 测试：返回来源文档 ===");
    
    let inputs = HashMap::from([
        ("query".to_string(), Value::String("什么是 LangChain？".to_string()))
    ]);
    
    let result = qa.invoke(inputs).await.unwrap();
    
    assert!(result.contains_key("result"), "应包含答案");
    assert!(result.contains_key("source_documents"), "应包含来源文档");
    
    let sources = result.get("source_documents").unwrap().as_array().unwrap();
    println!("检索到 {} 个来源文档", sources.len());
    
    assert!(!sources.is_empty(), "应至少返回一个来源文档");
}

/// 测试 RetrievalQA 自定义 prompt
///
/// 功能验证：
/// - with_prompt_template() 自定义 prompt 模板
/// - 自定义模板正确应用到 LLM 调用
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_retrieval_qa_custom_prompt() {
    let config = TestConfig::get();
    
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));
    
    let retriever = Arc::new(SimilarityRetriever::new(store.clone(), embeddings.clone()));
    
    retriever.add_documents(vec![
        Document::new("Go 语言由 Google 开发，注重并发编程。"),
    ]).await.unwrap();
    
    let custom_prompt = "请根据以下参考信息简要回答问题（不超过50字）：

参考信息：
{context}

问题：{question}

简短回答：";
    
    let qa = RetrievalQA::new(config.openai_chat(), retriever)
        .with_prompt_template(custom_prompt)
        .with_k(1)
        .with_verbose(true);
    
    println!("\n=== 测试：自定义 Prompt ===");
    
    let inputs = HashMap::from([
        ("query".to_string(), Value::String("Go 语言是谁开发的？".to_string()))
    ]);
    
    let result = qa.invoke(inputs).await.unwrap();
    let answer = result.get("result").unwrap().as_str().unwrap();
    
    println!("答案: {}", answer);
    assert!(!answer.is_empty());
}

/// 测试 RetrievalQA query 简化接口
///
/// 功能验证：
/// - query() 方法直接传入字符串
/// - 返回答案字符串而非 HashMap
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_retrieval_qa_query_interface() {
    let config = TestConfig::get();
    
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));
    
    let retriever = Arc::new(SimilarityRetriever::new(store.clone(), embeddings.clone()));
    
    retriever.add_documents(vec![
        Document::new("TypeScript 是 JavaScript 的类型超集，增加了静态类型检查。"),
    ]).await.unwrap();
    
    let qa = RetrievalQA::new(config.openai_chat(), retriever)
        .with_verbose(true);
    
    println!("\n=== 测试：query 简化接口 ===");
    
    let answer = qa.query("TypeScript 和 JavaScript 的关系是什么？").await.unwrap();
    
    println!("答案: {}", answer);
    assert!(!answer.is_empty());
}