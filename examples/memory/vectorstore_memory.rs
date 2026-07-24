//! VectorStore Memory 示例
//!
//! 展示如何使用 VectorStoreRetrieverMemory 进行语义检索历史记忆。
//!
//! # 运行
//! ```bash
//! cargo run --example vectorstore_memory
//! ```

use langchainrust::vector_stores::Document;
use langchainrust::{Embeddings, InMemoryVectorStore, MockEmbeddings, VectorStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VectorStore Memory 示例 ===\n");

    // 创建内存向量存储
    let store = InMemoryVectorStore::new();
    let embeddings = MockEmbeddings::new(4);

    // 添加文档到向量存储
    let docs = vec![
        Document::new("Rust 是一种系统编程语言,注重安全和性能").with_id("1"),
        Document::new("Python 是一种脚本语言,适合快速开发").with_id("2"),
        Document::new("LangChain 是构建 LLM 应用的框架").with_id("3"),
    ];

    let mut emb_vecs = Vec::new();
    for d in &docs {
        let emb = embeddings.embed_query(&d.content).await?;
        emb_vecs.push(emb);
    }
    let ids = store.add_documents(docs, emb_vecs).await?;
    println!("添加了 {} 个文档: {:?}", ids.len(), ids);

    // 语义搜索
    let query_emb = embeddings.embed_query("编程语言").await?;
    let results = store.similarity_search(&query_emb, 2).await?;
    println!("\n搜索 '编程语言' Top 2:");
    for r in &results {
        println!("  [{:.3}] {}", r.score, r.document.content);
    }

    println!("\nVectorStoreRetrieverMemory 使用向量存储保存对话历史,");
    println!("通过语义相似度检索相关记忆,实现长期记忆的智能召回。");
    Ok(())
}
