// tests/bm25/unified_hybrid.rs
//! UnifiedHybridIndex 集成测试

use langchainrust::{
    UnifiedHybridIndex, HybridIndexConfig, Document,
    MockEmbeddings, Embeddings,
};
use std::sync::Arc;

fn create_embeddings(dim: usize) -> Arc<dyn Embeddings> {
    Arc::new(MockEmbeddings::new(dim))
}

#[tokio::test]
async fn test_unified_hybrid_index_creation() {
    let embeddings = create_embeddings(3);
    let index = UnifiedHybridIndex::new(embeddings.clone(), 3);

    assert_eq!(index.config.chunk_size, 500);
    assert_eq!(index.config.bm25_k, 10);
}

#[tokio::test]
async fn test_add_document() {
    let embeddings = create_embeddings(3);
    let index = UnifiedHybridIndex::new(embeddings.clone(), 3);

    let doc = Document::new("这是一段测试文本用于验证功能").with_id("test_001");

    let id = index.add_document(doc).await.expect("添加文档失败");

    assert_eq!(id, "test_001");
}

#[tokio::test]
async fn test_retrieve() {
    let embeddings = create_embeddings(3);
    let config = HybridIndexConfig::new()
        .with_chunk_size(50)
        .with_top_k(5, 5);
    let index = UnifiedHybridIndex::with_config(embeddings.clone(), 3, config);

    index.add_documents(vec![
        Document::new("Rust是一门系统编程语言").with_id("doc_001"),
        Document::new("Python是一门脚本语言").with_id("doc_002"),
    ]).await.expect("添加文档失败");

    let results = index.retrieve("编程语言", 3).await.expect("检索失败");

    assert!(results.len() <= 3);
}

#[tokio::test]
async fn test_clear() {
    let embeddings = create_embeddings(3);
    let index = UnifiedHybridIndex::new(embeddings.clone(), 3);

    index.add_document(Document::new("测试文档")).await.expect("添加失败");

    index.clear().await.expect("清空失败");

    assert_eq!(index.document_count().await, 0);
}