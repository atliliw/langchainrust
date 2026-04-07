//! 整合测试 - RAG + LLM 完整流程

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{
    Document, InMemoryVectorStore, RecursiveCharacterSplitter,
    SimilarityRetriever, RetrieverTrait, TextSplitter, VectorStore,
};
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use std::sync::Arc;

/// 测试完整 RAG 流程
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_rag_full_pipeline() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();
    
    println!("=== 1. 创建知识文档 ===");
    
    let knowledge_base = vec![
        Document::new("Rust 是一门系统编程语言，由 Mozilla 研发。Rust 专注于安全性、速度和并发性。"),
        Document::new("Rust 1.0 于 2015 年 5 月 15 日发布。Rust 的创始人是 Graydon Hoare。"),
        Document::new("Rust 的所有权系统包含三个核心概念：所有权、借用和生命周期。"),
        Document::new("Rust 的内存安全不需要垃圾回收器。Rust 广泛应用于 WebAssembly 和嵌入式系统。"),
    ];
    
    println!("创建了 {} 个知识文档", knowledge_base.len());
    
    println!("\n=== 2. 分割文档 ===");
    
    let splitter = RecursiveCharacterSplitter::new(100, 20);
    let chunks: Vec<Document> = knowledge_base.iter()
        .flat_map(|doc| {
            splitter.split_text(&doc.page_content())
                .into_iter()
                .map(Document::new)
                .collect::<Vec<_>>()
        })
        .collect();
    
    println!("分割后共 {} 个文档块", chunks.len());
    
    println!("\n=== 3. 索引到向量存储 ===");
    
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    retriever.add_documents(chunks).await.unwrap();
    
    let doc_count = store.count().await;
    println!("向量存储中共有 {} 个文档", doc_count);
    
    println!("\n=== 4. 测试检索和生成 ===");
    
    let test_queries = vec![
        ("Rust 是什么时候发布的？", "2015"),
        ("Rust 的所有权系统有什么特点？", "所有权"),
        ("Rust 有垃圾回收器吗？", "没有"),
    ];
    
    for (query, expected_keyword) in test_queries {
        println!("\n--- 问题: {} ---", query);
        
        let relevant_docs = retriever.retrieve(query, 3).await.unwrap();
        println!("检索到 {} 个相关文档", relevant_docs.len());
        
        let context = relevant_docs.iter()
            .map(|d| d.page_content())
            .collect::<Vec<_>>()
            .join("\n\n");
        
        let messages = vec![
            Message::system("根据资料回答问题。"),
            Message::human(&format!("资料：\n{}\n\n问题：{}", context, query)),
        ];
        
        let response = llm.chat(messages, None).await.unwrap();
        println!("答案: {}", response.content);
        
        assert!(response.content.contains(expected_keyword));
    }
}