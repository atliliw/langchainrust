//! BM25 关键词检索示例
//!
//! 展示 ChunkedBM25Retriever 的关键词检索(无需 LLM / 向量库)。
//!
//! # 运行
//! ```bash
//! cargo run --example rag_bm25_search
//! ```

use langchainrust::retrieval::bm25::ChunkedBM25Retriever;
use langchainrust::retrieval::ChunkedDocumentStore;
use langchainrust::Document;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_documents(vec![
        Document::new("Rust 是一门系统编程语言,由 Mozilla 开发,注重安全和性能。")
            .with_id("rust_intro"),
        Document::new("Rust 的核心特性包括所有权系统、借用检查和零成本抽象。")
            .with_id("rust_features"),
        Document::new("机器学习是 AI 的核心技术,使计算机从数据中学习。")
            .with_id("ml_def"),
    ]);

    let results = retriever.search("Rust 语言特点", 3);
    println!("检索结果(按相关度排序):");
    for (i, r) in results.iter().enumerate() {
        println!("  [{}] {}", i + 1, r.content());
    }
    Ok(())
}
