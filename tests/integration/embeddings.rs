//! Embeddings 测试 - 需要 API Key
//!
//! 测试 OpenAIEmbeddings 的文本嵌入功能

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::Embeddings;

/// 测试单个文本嵌入
///
/// 测试内容：
/// - 对单个文本生成嵌入向量
/// - 验证向量维度大于 0
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_embed_single_text() {
    let embeddings = TestConfig::get().embeddings();
    
    let vector = embeddings.embed_query("Rust is a systems programming language.").await.unwrap();
    
    println!("Embedding dimension: {}", vector.len());
    assert!(!vector.is_empty());
}

/// 测试批量文本嵌入
///
/// 测试内容：
/// - 对多个文本同时生成嵌入向量
/// - 验证返回向量数量与输入一致
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_embed_multiple_texts() {
    let embeddings = TestConfig::get().embeddings();
    
    let texts: Vec<&str> = vec![
        "Rust is fast.",
        "Python is easy.",
        "JavaScript is versatile.",
    ];
    
    let vectors = embeddings.embed_documents(&texts).await.unwrap();
    
    println!("Vectors count: {}", vectors.len());
    assert_eq!(vectors.len(), 3);
    assert!(!vectors[0].is_empty());
}

/// 测试嵌入模型维度
///
/// 测试内容：
/// - 获取模型输出的向量维度
/// - 验证维度值有效（如 ada-002 为 1536）
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_embedding_dimension() {
    let embeddings = TestConfig::get().embeddings();
    
    let dim = embeddings.dimension();
    println!("Model dimension: {}", dim);
    assert!(dim > 0);
}