//! RAG (检索增强生成) 测试 - 需要 API Key
//!
//! 测试完整的 RAG 流程：文档索引 → 检索 → 生成答案

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{
    Document, InMemoryVectorStore, RecursiveCharacterSplitter,
    SimilarityRetriever, RetrieverTrait, TextSplitter,
};
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use std::sync::Arc;

/// 测试 RAG 检索流程
///
/// 测试内容：
/// - 创建长文档并使用 RecursiveCharacterSplitter 分割
/// - 使用 OpenAIEmbeddings 为文档生成向量
/// - 根据查询检索最相关的文档块
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_rag_retrieval() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // 创建长文档
    let doc = Document::new(
        "Rust is a systems programming language focused on safety, speed, and concurrency. \
         It prevents common programming errors through its ownership system. \
         Rust achieves memory safety without garbage collection."
    );
    
    // 分割文档为小块
    let splitter = RecursiveCharacterSplitter::new(50, 10);
    let chunks: Vec<Document> = splitter.split_text(&doc.page_content())
        .into_iter()
        .map(Document::new)
        .collect();
    
    // 索引文档到向量存储
    retriever.add_documents(chunks).await.unwrap();
    
    // 检索相关文档
    let relevant_docs = retriever.retrieve("What makes Rust safe?", 2).await.unwrap();
    
    println!("Retrieved {} documents", relevant_docs.len());
    for (i, doc) in relevant_docs.iter().enumerate() {
        println!("Doc {}: {}", i, doc.page_content());
    }
    
    assert!(!relevant_docs.is_empty());
}

/// 测试完整 RAG 流程（检索 + 生成）
///
/// 测试内容：
/// - 索引多个知识文档到向量存储
/// - 根据问题检索相关文档作为上下文
/// - 使用 LLM 基于上下文生成答案
/// - 验证答案包含正确信息（2015）
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_rag_with_llm_generation() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();
    
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // 索引知识库文档
    let docs = vec![
        Document::new("Rust was created by Mozilla."),
        Document::new("Rust 1.0 was released in 2015."),
        Document::new("Rust focuses on memory safety."),
    ];
    
    retriever.add_documents(docs).await.unwrap();
    
    // 检索相关文档
    let relevant_docs = retriever.retrieve("When was Rust released?", 2).await.unwrap();
    
    // 构建上下文字符串
    let context = relevant_docs.iter()
        .map(|d| d.page_content())
        .collect::<Vec<_>>()
        .join("\n");
    
    // 使用 LLM 基于上下文生成答案
    let messages = vec![
        Message::system("Answer based on the context provided."),
        Message::human(&format!("Context:\n{}\n\nQuestion: When was Rust 1.0 released?", context)),
    ];
    
    let response = llm.chat(messages, None).await.unwrap();
    
    println!("Answer: {}", response.content);
    assert!(response.content.contains("2015"));
}