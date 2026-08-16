//! Qdrant 向量数据库使用教程
//!
//! 运行方式:
//! cargo test --test qdrant_tutorial --features qdrant-integration -- --nocapture
//!
//! 你的 Qdrant 地址: http://192.168.10.100:6334

#![cfg(feature = "qdrant-integration")]

use langchainrust::embeddings::MockEmbeddings;
use langchainrust::retrieval::{RetrieverTrait, SimilarityRetriever};
use langchainrust::vector_stores::{
    Document, QdrantConfig, QdrantDistance, QdrantVectorStore, VectorStore,
};
use std::sync::Arc;

const QDRANT_URL: &str = "http://192.168.10.100:6334";
const VECTOR_SIZE: usize = 128;

/// 教程 1：连接 Qdrant 数据库
///
/// 创建配置并连接到 Qdrant 服务，验证连接成功
#[tokio::test]
async fn t01_connection() {
    let config = QdrantConfig::new(QDRANT_URL, "tutorial_basic")
        .with_vector_size(VECTOR_SIZE)
        .with_distance(QdrantDistance::Cosine);

    let store = QdrantVectorStore::new(config).await.unwrap();
    println!("✅ 连接成功: {}", QDRANT_URL);
    println!("   文档数: {}", store.count().await);
}

/// 教程 2：添加文档到向量数据库
///
/// 创建文档（可带元数据），生成向量，存储到数据库
#[tokio::test]
async fn t02_add_documents() {
    let config = QdrantConfig::new(QDRANT_URL, "tutorial_add").with_vector_size(VECTOR_SIZE);
    let store = QdrantVectorStore::new(config).await.unwrap();
    let _ = store.clear().await;

    let docs = vec![
        Document::new("Rust 编程语言").with_metadata("category", "programming"),
        Document::new("向量数据库").with_metadata("category", "database"),
    ];

    // 生成向量（实际用 OpenAI Embeddings）
    let embeddings: Vec<Vec<f32>> = vec![vec![1.0; VECTOR_SIZE], vec![0.0; VECTOR_SIZE]];

    let ids = store.add_documents(docs, embeddings).await.unwrap();
    println!("✅ 添加 {} 个文档", ids.len());
}

/// 教程 3：使用自定义 ID
///
/// 通过 with_id() 设置文档 ID，便于后续管理
#[tokio::test]
async fn t03_custom_id() {
    let config = QdrantConfig::new(QDRANT_URL, "tutorial_custom_id").with_vector_size(VECTOR_SIZE);
    let store = QdrantVectorStore::new(config).await.unwrap();
    let _ = store.clear().await;

    let docs = vec![
        Document::new("文档 A").with_id("doc-a"),
        Document::new("文档 B").with_id("doc-b"),
    ];

    let embeddings = vec![vec![1.0; VECTOR_SIZE], vec![0.5; VECTOR_SIZE]];
    let ids = store.add_documents(docs, embeddings).await.unwrap();
    println!("✅ 自定义 ID: {:?}", ids);

    let doc = store.get_document("doc-a").await.unwrap();
    println!("   获取 doc-a: {:?}", doc.map(|d| d.content));
}

/// 教程 4：相似度搜索
///
/// 提供查询向量，返回最相似的文档
#[tokio::test]
async fn t04_similarity_search() {
    let config = QdrantConfig::new(QDRANT_URL, "tutorial_search").with_vector_size(VECTOR_SIZE);
    let store = QdrantVectorStore::new(config).await.unwrap();
    let _ = store.clear().await;

    let docs = vec![
        Document::new("机器学习").with_id("ml"),
        Document::new("深度学习").with_id("dl"),
    ];

    // 设计不同方向的向量
    let embeddings = vec![
        vec![1.0, 0.0]
            .into_iter()
            .cycle()
            .take(VECTOR_SIZE)
            .collect(),
        vec![0.0, 1.0]
            .into_iter()
            .cycle()
            .take(VECTOR_SIZE)
            .collect(),
    ];

    store.add_documents(docs, embeddings).await.unwrap();

    // 查询向量
    let query: Vec<f32> = vec![0.9, 0.1]
        .into_iter()
        .cycle()
        .take(VECTOR_SIZE)
        .collect();
    let results = store.similarity_search(&query, 2).await.unwrap();

    println!("✅ 搜索结果:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "   {}. {} (分数: {:.4})",
            i + 1,
            r.document.content,
            r.score
        );
    }
}

/// 教程 5：获取文档的向量
///
/// 通过 ID 获取存储在数据库中的向量数据
#[tokio::test]
async fn t05_get_embedding() {
    let config = QdrantConfig::new(QDRANT_URL, "tutorial_get_vec").with_vector_size(8);
    let store = QdrantVectorStore::new(config).await.unwrap();
    let _ = store.clear().await;

    let doc = Document::new("测试文档").with_id("test-doc");
    let custom_embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    println!("存储向量: {:?}", custom_embedding);
    store
        .add_documents(vec![doc], vec![custom_embedding.clone()])
        .await
        .unwrap();

    let embedding = store.get_embedding("test-doc").await.unwrap();
    match embedding {
        Some(vec) => {
            println!("✅ 获取向量: {:?}", vec);
            println!("   验证: {}", vec == custom_embedding);
        }
        None => println!("❌ 向量不存在"),
    }
}

/// 教程 6：使用 Retriever 自动处理向量
///
/// Retriever 封装了向量生成，只需提供文本
#[tokio::test]
async fn t06_retriever() {
    let config = QdrantConfig::new(QDRANT_URL, "tutorial_retriever").with_vector_size(VECTOR_SIZE);
    let store = Arc::new(QdrantVectorStore::new(config).await.unwrap());
    let _ = store.clear().await;

    let embeddings = Arc::new(MockEmbeddings::new(VECTOR_SIZE));
    let retriever = SimilarityRetriever::new(store.clone(), embeddings);

    retriever
        .add_documents(vec![
            Document::new("Rust 教程").with_metadata("type", "lang"),
            Document::new("Qdrant 数据库").with_metadata("type", "db"),
        ])
        .await
        .unwrap();

    println!("✅ 通过 Retriever 添加 {} 个文档", store.count().await);

    let results = retriever.retrieve("编程", 2).await.unwrap();
    for r in &results {
        println!("   - {}", r.content);
    }
}

/// 教程 7：完整 RAG 流程
///
/// 演示检索增强生成的完整流程：
/// 1. 索引知识库
/// 2. 用户提问
/// 3. 检索相关文档
/// 4. 构建上下文（传给 LLM）
#[tokio::test]
async fn t07_rag_flow() {
    let config = QdrantConfig::new(QDRANT_URL, "rag_demo").with_vector_size(VECTOR_SIZE);
    let store = Arc::new(QdrantVectorStore::new(config).await.unwrap());
    let _ = store.clear().await;

    let embeddings = Arc::new(MockEmbeddings::new(VECTOR_SIZE));
    let retriever = SimilarityRetriever::new(store.clone(), embeddings);

    let knowledge = vec![
        Document::new("Rust 专注内存安全").with_id("doc-1"),
        Document::new("Cargo 是包管理器").with_id("doc-2"),
    ];

    retriever.add_documents(knowledge).await.unwrap();
    println!("📚 索引 {} 个文档", store.count().await);

    let question = "Rust 的特点？";
    let relevant = retriever.retrieve(question, 2).await.unwrap();

    println!("\n❓ 问题: {}", question);
    println!("🔍 相关文档:");
    for doc in &relevant {
        println!("   - {}", doc.content);
    }

    println!("\n✅ RAG 流程完成");
}

/// 教程 8：元数据的使用
///
/// 演示如何存储和检索文档的元数据
#[tokio::test]
async fn t08_metadata() {
    let config = QdrantConfig::new(QDRANT_URL, "tutorial_metadata").with_vector_size(VECTOR_SIZE);
    let store = QdrantVectorStore::new(config).await.unwrap();
    let _ = store.clear().await;

    let doc = Document::new("Rust 教程")
        .with_metadata("author", "张三")
        .with_metadata("category", "programming")
        .with_metadata("date", "2024-01-01");

    store
        .add_documents(vec![doc], vec![vec![1.0; VECTOR_SIZE]])
        .await
        .unwrap();

    let results = store
        .similarity_search(&[1.0; VECTOR_SIZE], 1)
        .await
        .unwrap();
    let metadata = &results[0].document.metadata;

    println!("✅ 文档元数据:");
    println!("   author: {:?}", metadata.get("author"));
    println!("   category: {:?}", metadata.get("category"));
    println!("   date: {:?}", metadata.get("date"));
}
