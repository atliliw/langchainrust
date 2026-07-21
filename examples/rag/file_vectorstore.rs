//! FileVectorStore 示例
//!
//! 展示如何使用 FileVectorStore 进行向量持久化存储。
//!
//! # 运行
//! ```bash
//! cargo run --example file_vectorstore
//! ```

use langchainrust::{FileVectorStore, VectorStore, Document, MockEmbeddings};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FileVectorStore 示例 ===\n");

    let path = PathBuf::from("./example_vectors.json");
    let dim = 4;

    // 创建文件向量存储
    let store = FileVectorStore::new(path.clone(), dim)?;
    println!("创建 FileVectorStore: {:?}", path);
    println!("向量维度: {}", store.dimension());

    // 添加文档
    let docs = vec![
        Document::new("Rust 注重安全和性能").with_id("rust"),
        Document::new("Python 适合快速开发").with_id("python"),
        Document::new("Go 适合并发服务").with_id("go"),
    ];

    let _embeddings = MockEmbeddings::new(dim);
    let emb: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    let ids = store.add_documents(docs, emb).await?;
    println!("添加了 {} 个文档: {:?}", ids.len(), ids);
    println!("当前文档数: {}", store.count().await);

    // 语义搜索
    let query = vec![0.9, 0.1, 0.0, 0.0]; // 接近 Rust
    let results = store.similarity_search(&query, 2).await?;
    println!("\n搜索 Top 2:");
    for r in &results {
        println!("  [{:.3}] {}", r.score, r.document.content);
    }

    // 持久化验证
    println!("\n文件已自动持久化到磁盘。");
    println!("重启后创建新 FileVectorStore(path, dim) 即可加载已有数据。");

    // 清理
    store.clear().await?;
    println!("\n已清空存储。");

    // 删除示例文件
    let _ = std::fs::remove_file(&path);

    Ok(())
}
