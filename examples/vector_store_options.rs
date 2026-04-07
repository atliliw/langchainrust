// examples/vector_store_options.rs
//! 向量存储选项示例
//! 
//! 演示如何在不同类型的向量存储之间切换

use langchainrust::{
    Document, InMemoryVectorStore, VectorStoreProvider, 
    VectorStoreType, VectorStoreBuilder,
    MockEmbeddings, SimilarityRetriever, RetrieverTrait
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LangChainRust - 向量存储选项示例 ===\n");

    // 选项1: 简单内存存储（默认）
    println!("1. 使用内存向量存储 (默认，无需外部依赖):");
    let memory_store = VectorStoreBuilder::in_memory().build().await?;
    example_retrieval(memory_store.clone(), "内存").await?;
    println!();

    // 选项2: 文件持久化存储（稍后实现）
    println!("2. 使用文件持久化向量存储:");
    let file_store = VectorStoreBuilder::file_backed().build().await?;
    example_retrieval(file_store, "文件").await?;
    println!();

    // 选项3: Qdrant 向量存储 
    println!("3. 使用 Qdrant 向量存储 (等待连接真实服务):");
    // 注意：你需要在此处提供真实启动的 Qdrant 服务地址
    let qdrant_store = VectorStoreBuilder::qdrant("http://your-qdrant-host:6334", "your_collection").build().await?;
    println!("   已配置 Qdrant 连接，实际连接将在调用时初始化");
    println!("   请在 Linux 上启动 Qdrant 并提供真实的 IP:PORT");
    println!();

    println!("=== 各种存储的适用场景 ===");
    println!("• 内存存储: 测试、Demo、临时应用");
    println!("• 文件持久化: 个人知识库、单机应用");
    println!("• Qdrant: 多用户应用、大数据集、生产环境");
    println!();
    println!("💡 优势: 统一的 Rust API，相同的上层代码适用于不同的后端!");

    Ok(())
}

async fn example_retrieval(
    store: Arc<dyn langchainrust::VectorStore>, 
    store_type: &str
) -> Result<(), Box<dyn std::error::Error>> {
    // 创建示例文档
    let docs = vec![
        Document::new("Rust 是一门系统编程语言，注重安全性").with_metadata("topic", "programming"),
        Document::new("向量数据库使用嵌入模型存储相似度信息").with_metadata("topic", "database"),
        Document::new("人工智能是计算机科学的一个分支").with_metadata("topic", "ai"),
    ];

    // 模拟嵌入（在实际使用中，这些是由 OpenAIEmbeddings 等生成的）
    let embeddings: Vec<Vec<f32>> = vec![
        vec![0.9, 0.1, 0.2],  // Rust 相关向量
        vec![0.2, 0.8, 0.1],  // 数据库相关向量  
        vec![0.1, 0.2, 0.9],  // AI 相关向量
    ];

    // 添加到存储
    let ids = store.add_documents(docs, embeddings).await?;
    println!("   ✓ 添加了 {} 个文档到 {} 存储", ids.len(), store_type);

    // 示例查询向量（模拟 "Rust 编程语言" 的嵌入）
    let query_embedding = [0.85, 0.15, 0.25];  // 偏向 Rust 的查询向量
    let results = store.similarity_search(&query_embedding, 2).await?;

    println!("   ✓ 检索到 {} 个相关文档", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("     [{}] '{:.30}...' (相似度: {:.3})", 
                 i + 1, 
                 result.document.content.chars().take(30).collect::<String>(), 
                 result.score);
    }

    println!("   ✓ 总文档数: {}", store.count().await);

    Ok(())
}