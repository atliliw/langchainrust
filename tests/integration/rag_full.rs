//! 完整的 LLM + RAG 测试
//!
//! 测试内容：
//! 1. 使用 OpenAI Embeddings 生成真实向量
//! 2. InMemoryVectorStore 和 QdrantVectorStore 两种数据库
//! 3. 完整的 RAG 流程：索引 → 检索 → LLM 生成答案
//!
//! 运行方式：
//! # InMemoryVectorStore 测试
//! cargo test --test integration_rag_full -- --nocapture
//!
//! # Qdrant 测试（需要启用 feature）
//! cargo test --test integration_rag_full --features qdrant-integration -- --nocapture
//!
//! 配置：
//! - 设置环境变量 OPENAI_API_KEY 或在 tests/common/mod.rs 中配置
//! - Qdrant 地址在下方 QDRANT_URL 配置

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{
    Document, InMemoryVectorStore, RecursiveCharacterSplitter,
    SimilarityRetriever, RetrieverTrait, TextSplitter,
    Embeddings, VectorStore,
};
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use std::sync::Arc;

#[cfg(feature = "qdrant-integration")]
use langchainrust::vector_stores::{QdrantVectorStore, QdrantConfig, QdrantDistance};

// ============================================================================
// 配置
// ============================================================================

/// Qdrant 服务地址
#[cfg(feature = "qdrant-integration")]
const QDRANT_URL: &str = "http://192.168.10.100:6334";

// ============================================================================
// 测试 1：InMemoryVectorStore + OpenAI Embeddings（真实向量）
// ============================================================================

/// 测试：使用 OpenAI Embeddings 生成向量并存储到 InMemoryVectorStore
///
/// 这是最基础的测试，验证：
/// - OpenAI Embeddings API 正常工作
/// - 向量生成并存储
/// - 相似度搜索返回正确结果
#[tokio::test]
#[ignore = "需要配置 OPENAI_API_KEY"]
async fn test_inmemory_embeddings_real() {
    println!("\n=== 测试 1：InMemoryVectorStore + 真实 OpenAI Embeddings ===\n");
    
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    
    println!("📦 Embedding 模型: {}", embeddings.model_name());
    println!("   向量维度: {}", embeddings.dimension());
    
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // 知识库文档
    let docs = vec![
        Document::new("Rust 是一门系统编程语言，专注于内存安全和并发性能。")
            .with_metadata("category", "language")
            .with_metadata("lang", "rust"),
        Document::new("Python 是一门解释型语言，广泛用于数据科学和机器学习。")
            .with_metadata("category", "language")
            .with_metadata("lang", "python"),
        Document::new("JavaScript 是 Web 开发的主要语言，用于前端和后端。")
            .with_metadata("category", "language")
            .with_metadata("lang", "javascript"),
        Document::new("向量数据库用于存储和检索高维向量，支持语义搜索。")
            .with_metadata("category", "database")
            .with_metadata("type", "vector"),
    ];
    
    println!("\n📚 索引 {} 个文档...", docs.len());
    
    // 添加文档（自动调用 embeddings 生成向量）
    retriever.add_documents(docs).await.unwrap();
    
    println!("✅ 索引完成，共 {} 个向量", store.count().await);
    
    // 检索测试
    let query = "什么是 Rust 语言的特点？";
    println!("\n🔍 查询: {}", query);
    
    let results = retriever.retrieve_with_scores(query, 3).await.unwrap();
    
    println!("\n📄 检索结果:");
    for (i, r) in results.iter().enumerate() {
        println!("   {}. {} (相似度: {:.4})", 
            i + 1, 
            r.document.content,
            r.score
        );
        println!("      元数据: {:?}", r.document.metadata);
    }
    
    // 验证：第一个结果应该是 Rust 相关文档
    assert!(!results.is_empty(), "应该有检索结果");
    assert!(results[0].document.metadata.get("lang") == Some(&"rust".to_string()),
        "第一个结果应该是 Rust 文档");
}

// ============================================================================
// 测试 2：完整 RAG 流程 - InMemoryVectorStore + LLM
// ============================================================================

/// 测试：完整的 RAG 流程（检索 + 生成）
///
/// 流程：
/// 1. 索引知识库文档
/// 2. 用户提问
/// 3. 检索相关文档作为上下文
/// 4. LLM 基于上下文生成答案
#[tokio::test]
#[ignore = "需要配置 OPENAI_API_KEY"]
async fn test_rag_inmemory_full_pipeline() {
    println!("\n=== 测试 2：完整 RAG 流程 (InMemoryVectorStore + LLM) ===\n");
    
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();
    
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // 创建知识库
    let knowledge_docs = vec![
        Document::new("Rust 语言由 Mozilla 研发，2015 年发布 1.0 版本。")
            .with_metadata("source", "wiki"),
        Document::new("Rust 的所有权系统确保内存安全，无需垃圾回收。")
            .with_metadata("source", "wiki"),
        Document::new("Cargo 是 Rust 的包管理器和构建工具。")
            .with_metadata("source", "wiki"),
        Document::new("Rust 支持异步编程，使用 async/await 语法。")
            .with_metadata("source", "wiki"),
        Document::new("Rust 编译器非常严格，能捕获许多潜在错误。")
            .with_metadata("source", "wiki"),
    ];
    
    println!("📚 索引知识库...");
    retriever.add_documents(knowledge_docs).await.unwrap();
    println!("✅ 已索引 {} 个文档", store.count().await);
    
    // 用户提问
    let question = "Rust 是什么时候发布的？它的主要特点是什么？";
    
    println!("\n❓ 用户问题: {}", question);
    
    // 检索相关文档
    println!("\n🔍 检索相关文档...");
    let relevant_docs = retriever.retrieve_with_scores(question, 3).await.unwrap();
    
    println!("\n📄 检索到的上下文:");
    for (i, doc) in relevant_docs.iter().enumerate() {
        println!("   {}. {} (相似度: {:.4})", i + 1, doc.document.content, doc.score);
    }
    
    // 构建上下文
    let context = relevant_docs.iter()
        .map(|r| r.document.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    
    // 构建 RAG Prompt
    let prompt = format!(
        "你是一个助手，请根据以下上下文回答问题。如果上下文中没有相关信息，请诚实说明。\n\n\
         上下文:\n{}\n\n\
         问题: {}\n\n\
         请用中文回答:",
        context, question
    );
    
    println!("\n🤖 调用 LLM 生成答案...");
    
    let messages = vec![
        Message::system("你是一个知识助手，根据提供的上下文准确回答问题。"),
        Message::human(&prompt),
    ];
    
    let response = llm.chat(messages, None).await.unwrap();
    
    println!("\n✅ LLM 答案:\n{}\n", response.content);
    
    // 验证答案包含关键信息
    assert!(response.content.contains("2015"), 
        "答案应该包含发布年份 2015");
}

// ============================================================================
// 测试 3：文档分割 + RAG
// ============================================================================

/// 测试：长文档分割后进行 RAG
///
/// 场景：将长文档分割成小块，检索最相关的块
#[tokio::test]
#[ignore = "需要配置 OPENAI_API_KEY"]
async fn test_rag_with_document_splitting() {
    println!("\n=== 测试 3：文档分割 + RAG ===\n");
    
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();
    
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // 创建长文档
    let long_doc = Document::new(
        "Rust 是一门现代系统编程语言，由 Mozilla 研发。它于 2015 年发布 1.0 版本。\
         Rust 的核心特性是所有权系统，这确保了内存安全而无需垃圾回收器。\
         Rust 编译器非常严格，能在编译时捕获许多常见错误，如空指针引用和数据竞争。\
         Cargo 是 Rust 的官方包管理器，用于依赖管理和项目构建。\
         Rust 支持泛型、trait、宏等高级特性，同时保持高性能。\
         Rust 在系统编程、Web 服务、嵌入式开发等领域广泛应用。\
         Rust 的异步编程模型使用 async/await 语法，基于 Future trait。\
         Rust 社区活跃，文档完善，学习曲线较陡峭但值得投入。"
    );
    
    println!("📄 原始文档长度: {} 字符", long_doc.content.len());
    
    // 分割文档
    let splitter = RecursiveCharacterSplitter::new(100, 20);
    let chunks: Vec<Document> = splitter.split_text(&long_doc.content)
        .into_iter()
        .map(|text| Document::new(text).with_metadata("source", "long_doc"))
        .collect();
    
    println!("✂️ 分割为 {} 个块", chunks.len());
    
    for (i, chunk) in chunks.iter().enumerate() {
        let preview: String = chunk.content.chars().take(50).collect();
        println!("   块 {}: {}...", i + 1, preview);
    }
    
    // 索引所有块
    println!("\n📚 索引所有文档块...");
    retriever.add_documents(chunks).await.unwrap();
    
    // 检索特定问题的相关块
    let question = "Cargo 是什么？有什么功能？";
    println!("\n❓ 问题: {}", question);
    
    let results = retriever.retrieve_with_scores(question, 2).await.unwrap();
    
    println!("\n📄 最相关的块:");
    for (i, r) in results.iter().enumerate() {
        println!("   {}. {}", i + 1, r.document.content);
        println!("      相似度: {:.4}", r.score);
    }
    
    // LLM 基于检索结果回答
    let context = results.iter()
        .map(|r| r.document.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    
    let messages = vec![
        Message::system("根据提供的上下文回答问题。"),
        Message::human(&format!("上下文:\n{}\n\n问题: {}", context, question)),
    ];
    
    println!("\n🤖 LLM 回答:");
    let response = llm.chat(messages, None).await.unwrap();
    println!("{}\n", response.content);
    
    assert!(response.content.contains("Cargo") || response.content.contains("包管理"),
        "答案应该提及 Cargo");
}

// ============================================================================
// 测试 4：Qdrant + OpenAI Embeddings + LLM
// ============================================================================

#[cfg(feature = "qdrant-integration")]
#[tokio::test]
#[ignore = "需要配置 OPENAI_API_KEY 和 Qdrant 服务"]
async fn test_rag_qdrant_full_pipeline() {
    println!("\n=== 测试 4：完整 RAG 流程 (Qdrant + OpenAI Embeddings + LLM) ===\n");
    
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();
    
    println!("📡 连接 Qdrant: {}", QDRANT_URL);
    
    // 创建 Qdrant 配置
    let qdrant_config = QdrantConfig::new(QDRANT_URL, "test_rag_full")
        .with_vector_size(embeddings.dimension())
        .with_distance(QdrantDistance::Cosine);
    
    let store = Arc::new(QdrantVectorStore::new(qdrant_config).await.unwrap());
    let _ = store.clear().await; // 清空旧数据
    
    println!("✅ Qdrant 连接成功");
    
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // 索引知识库
    let docs = vec![
        Document::new("向量数据库是一种专门用于存储和检索向量数据的数据库系统。")
            .with_id("vec-1")
            .with_metadata("topic", "database"),
        Document::new("Qdrant 是一个高性能的向量数据库，支持过滤和分页。")
            .with_id("vec-2")
            .with_metadata("topic", "qdrant"),
        Document::new("RAG 是检索增强生成技术，结合了检索和生成模型。")
            .with_id("vec-3")
            .with_metadata("topic", "rag"),
        Document::new("Embeddings 模型将文本转换为向量，用于语义搜索。")
            .with_id("vec-4")
            .with_metadata("topic", "embeddings"),
    ];
    
    println!("\n📚 索引 {} 个文档到 Qdrant...", docs.len());
    retriever.add_documents(docs).await.unwrap();
    println!("✅ 索引完成，共 {} 个向量", store.count().await);
    
    // 检索测试
    let question = "什么是 RAG？它是如何工作的？";
    println!("\n❓ 问题: {}", question);
    
    let results = retriever.retrieve_with_scores(question, 2).await.unwrap();
    
    println!("\n📄 Qdrant 检索结果:");
    for (i, r) in results.iter().enumerate() {
        println!("   {}. ID: {}", i + 1, r.document.id.as_ref().unwrap_or(&"N/A".to_string()));
        println!("      内容: {}", r.document.content);
        println!("      相似度: {:.4}", r.score);
    }
    
    // LLM 生成答案
    let context = results.iter()
        .map(|r| r.document.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    
    let messages = vec![
        Message::system("你是一个技术助手，根据上下文准确回答问题。"),
        Message::human(&format!(
            "根据以下信息回答:\n{}\n\n问题: {}", 
            context, question
        )),
    ];
    
    println!("\n🤖 LLM 答案:");
    let response = llm.chat(messages, None).await.unwrap();
    println!("{}\n", response.content);
    
    // 验证可以获取向量
    println!("📊 验证向量存储...");
    let embedding = store.get_embedding("vec-3").await.unwrap();
    if let Some(vec) = embedding {
        println!("   vec-3 向量维度: {}", vec.len());
        println!("   前5个值: {:?}", &vec[..5]);
        assert_eq!(vec.len(), embeddings.dimension(), "向量维度应该匹配");
    }
    
    println!("\n✅ Qdrant RAG 测试完成");
}

// ============================================================================
// 测试 5：对比测试 - InMemory vs Qdrant
// ============================================================================

#[cfg(feature = "qdrant-integration")]
#[tokio::test]
#[ignore = "需要配置 OPENAI_API_KEY 和 Qdrant 服务"]
async fn test_compare_memory_vs_qdrant() {
    println!("\n=== 测试 5：对比 InMemory vs Qdrant ===\n");
    
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    
    // 相同的测试文档
    let docs = vec![
        Document::new("Rust 专注于内存安全。"),
        Document::new("Python 适合数据科学。"),
        Document::new("JavaScript 用于 Web 开发。"),
    ];
    
    // InMemoryVectorStore
    let mem_store = Arc::new(InMemoryVectorStore::new());
    let mem_retriever = SimilarityRetriever::new(mem_store.clone(), embeddings.clone());
    mem_retriever.add_documents(docs.clone()).await.unwrap();
    
    // QdrantVectorStore
    let qdrant_config = QdrantConfig::new(QDRANT_URL, "test_compare")
        .with_vector_size(embeddings.dimension())
        .with_distance(QdrantDistance::Cosine);
    let qdrant_store = Arc::new(QdrantVectorStore::new(qdrant_config).await.unwrap());
    let _ = qdrant_store.clear().await;
    let qdrant_retriever = SimilarityRetriever::new(qdrant_store.clone(), embeddings.clone());
    qdrant_retriever.add_documents(docs.clone()).await.unwrap();
    
    println!("✅ InMemory 文档数: {}", mem_store.count().await);
    println!("✅ Qdrant 文档数: {}", qdrant_store.count().await);
    
    // 同一个查询
    let query = "什么语言适合做网站？";
    
    println!("\n🔍 查询: {}", query);
    
    // InMemory 结果
    let mem_results = mem_retriever.retrieve_with_scores(query, 3).await.unwrap();
    println!("\n📊 InMemory 结果:");
    for (i, r) in mem_results.iter().enumerate() {
        println!("   {}. {} ({:.4})", i + 1, r.document.content, r.score);
    }
    
    // Qdrant 结果
    let qdrant_results = qdrant_retriever.retrieve_with_scores(query, 3).await.unwrap();
    println!("\n📊 Qdrant 结果:");
    for (i, r) in qdrant_results.iter().enumerate() {
        println!("   {}. {} ({:.4})", i + 1, r.document.content, r.score);
    }
    
    // 两种数据库返回相同的最相关文档
    println!("\n✅ 验证: 两种数据库返回一致");
    assert_eq!(
        mem_results[0].document.content,
        qdrant_results[0].document.content,
        "两种数据库应返回相同的最相关文档"
    );
}

// ============================================================================
// 测试 6：多轮对话 RAG
// ============================================================================

/// 测试：多轮对话中的 RAG
///
/// 场景：用户进行多轮问答，每次都基于知识库检索
#[tokio::test]
#[ignore = "需要配置 OPENAI_API_KEY"]
async fn test_rag_multi_turn_conversation() {
    println!("\n=== 测试 6：多轮对话 RAG ===\n");
    
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();
    
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // 知识库
    let docs = vec![
        Document::new("LangChain 是一个 LLM 应用开发框架。"),
        Document::new("LangChain 支持 Python 和 JavaScript。"),
        Document::new("langchainrust 是 LangChain 的 Rust 实现。"),
        Document::new("LangChain 提供 Chain、Agent、Memory 等组件。"),
    ];
    
    retriever.add_documents(docs).await.unwrap();
    
    // 第一轮对话
    let q1 = "LangChain 是什么？";
    println!("\n❓ 第一轮: {}", q1);
    
    let r1 = retriever.retrieve(q1, 2).await.unwrap();
    let ctx1 = r1.iter().map(|d| d.content.as_str()).collect::<Vec<_>>().join("\n");
    
    let resp1 = llm.chat(vec![
        Message::system("根据上下文回答，简洁明了。"),
        Message::human(&format!("上下文:\n{}\n问题: {}", ctx1, q1)),
    ], None).await.unwrap();
    
    println!("🤖 回答: {}\n", resp1.content);
    
    // 第二轮对话（追问）
    let q2 = "它支持哪些编程语言？";
    println!("\n❓ 第二轮: {}", q2);
    
    let r2 = retriever.retrieve(q2, 2).await.unwrap();
    let ctx2 = r2.iter().map(|d| d.content.as_str()).collect::<Vec<_>>().join("\n");
    
    let resp2 = llm.chat(vec![
        Message::system("根据上下文回答，简洁明了。"),
        Message::human(&format!("上下文:\n{}\n问题: {}", ctx2, q2)),
    ], None).await.unwrap();
    
    println!("🤖 回答: {}\n", resp2.content);
    
    // 验证答案准确性
    assert!(resp2.content.contains("Python") || resp2.content.contains("JavaScript") || resp2.content.contains("Rust"),
        "答案应该提及支持的编程语言");
}