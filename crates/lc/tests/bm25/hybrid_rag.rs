// tests/bm25/hybrid_rag.rs
//! 混合检索 (BM25 + 向量) + LLM 完整 RAG 流程测试

// HybridRetriever 已弃用(P1-1),但本文件仍在覆盖其行为;
// 新代码请迁移到 UnifiedHybridIndex。
#![allow(deprecated)]

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::retrieval::bm25::{AutoMergingConfig, ChunkedBM25Retriever};
use langchainrust::retrieval::ChunkedDocumentStore;
use langchainrust::retrieval::{reciprocal_rank_fusion, HybridRetriever};
use langchainrust::{BaseChatModel, Document, InMemoryVectorStore, Message};
use langchainrust::{RetrieverTrait, SimilarityRetriever};
use std::sync::Arc;
use tempfile::NamedTempFile;

fn build_rag_messages(query: &str, contexts: &[String]) -> Vec<Message> {
    let context_text = contexts.join("\n\n");

    vec![
        Message::system("基于提供的上下文回答问题。如果上下文中没有相关信息，请说\"我不知道\"。"),
        Message::human(format!(
            "上下文：\n{}\n\n问题：{}\n\n请根据上下文回答问题。",
            context_text, query
        )),
    ]
}

/// 测试：RRF 融合算法
/// 验证：BM25 和 向量检索结果正确融合
#[test]
fn test_rrf_fusion() {
    let bm25_docs = vec![
        Document::new("Rust是一门系统编程语言，注重安全和性能。").with_id("doc1"),
        Document::new("Python是一门高级编程语言，适合数据科学。").with_id("doc2"),
        Document::new("Go是一门并发编程语言，由Google开发。").with_id("doc3"),
    ];

    let vector_docs = vec![
        Document::new("Rust是一门系统编程语言，注重安全和性能。").with_id("doc1"),
        Document::new("JavaScript是一门前端脚本语言。").with_id("doc4"),
        Document::new("Python是一门高级编程语言，适合数据科学。").with_id("doc2"),
    ];

    println!("=== 测试 RRF 融合算法 ===");
    println!("\nBM25 检索结果:");
    for (i, doc) in bm25_docs.iter().enumerate() {
        println!(
            "  [{}] id={}, 内容={}",
            i,
            doc.id.clone().unwrap_or_default(),
            doc.content
        );
    }

    println!("\n向量检索结果:");
    for (i, doc) in vector_docs.iter().enumerate() {
        println!(
            "  [{}] id={}, 内容={}",
            i,
            doc.id.clone().unwrap_or_default(),
            doc.content
        );
    }

    let results = reciprocal_rank_fusion(bm25_docs, vector_docs, 60);

    println!("\nRRF 融合结果:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  [{}] id={}, rrf_score={:.4}",
            i,
            r.document.id.clone().unwrap_or_default(),
            r.score
        );
    }

    println!("\n分析:");
    println!("  - doc1 和 doc2 在两个列表中都出现，RRF分数更高");
    println!("  - doc3 只在BM25出现，doc4 只在向量检索出现");
}

/// 测试：HybridRetriever 使用
/// 验证：混合检索器能正确执行检索
#[test]
fn test_hybrid_retriever_usage() {
    let hybrid = HybridRetriever::new();

    let bm25_docs = vec![
        Document::new("机器学习是AI的核心技术。").with_id("ml_doc"),
        Document::new("深度学习使用神经网络。").with_id("dl_doc"),
    ];

    let vector_docs = vec![
        Document::new("机器学习是AI的核心技术。").with_id("ml_doc"),
        Document::new("自然语言处理是NLP。").with_id("nlp_doc"),
    ];

    println!("=== 测试 HybridRetriever ===");

    let results = hybrid.retrieve(bm25_docs, vector_docs);

    println!("混合检索结果数: {}", results.len());
    for r in &results {
        println!(
            "  id={}, score={:.4}",
            r.document.id.clone().unwrap_or_default(),
            r.score
        );
    }
}

/// 测试：BM25 + 向量检索 + LLM 完整混合 RAG 流程
/// 验证：混合检索结果作为上下文，LLM正确回答
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_hybrid_rag_with_llm() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();

    println!("=== 测试 BM25 + 向量 + LLM 混合 RAG ===");

    // [1] 创建共享的 DocumentStore
    let doc_store = Arc::new(ChunkedDocumentStore::new());

    // [2] 知识库文档
    let knowledge_docs = vec![
        Document::new("Rust是一门系统编程语言，由Mozilla开发。Rust通过所有权系统实现内存安全，无需垃圾回收器。")
            .with_id("rust_doc"),
        Document::new("Python是一门高级编程语言，由Guido van Rossum开发。Python以简洁语法著称，适合数据科学。")
            .with_id("python_doc"),
        Document::new("Go语言由Google开发，专注于简洁性和并发编程。Go编译速度快，适合分布式系统。")
            .with_id("go_doc"),
        Document::new("JavaScript是一门前端脚本语言，用于Web开发。Node.js让JavaScript也能做后端。")
            .with_id("js_doc"),
    ];

    // [3] BM25 检索器初始化（使用共享 store）
    let bm25_config = AutoMergingConfig::new()
        .with_leaf_size(100)
        .with_threshold(0.5);
    let mut bm25_retriever = ChunkedBM25Retriever::with_config(doc_store.clone(), bm25_config);

    println!("BM25 检索器初始化完成");

    // [4] 向量检索器初始化
    let vector_store = Arc::new(InMemoryVectorStore::new());
    let vector_retriever = SimilarityRetriever::new(vector_store.clone(), embeddings.clone());

    vector_retriever
        .add_documents(knowledge_docs.clone())
        .await
        .unwrap();

    println!("向量检索器初始化完成，文档数: {}", knowledge_docs.len());

    // [5] 用户查询
    let query = "Rust语言的内存安全是如何实现的？";
    println!("\n用户查询: {}", query);

    // [6] BM25 检索
    let bm25_results = bm25_retriever.search_async(query, 5).await;
    let bm25_docs: Vec<Document> = bm25_results
        .iter()
        .map(|r| Document::new(r.content()).with_id(r.parent_id.clone()))
        .collect();

    println!("\nBM25 检索结果:");
    for (i, result) in bm25_results.iter().enumerate() {
        println!(
            "  [{}] 分数={}, 是否合并={}, 内容={}",
            i,
            result.score,
            result.is_merged(),
            result.content()
        );
    }

    // [7] 向量检索
    let vector_docs = vector_retriever.retrieve(query, 5).await.unwrap();

    println!("\n向量检索结果:");
    for (i, doc) in vector_docs.iter().enumerate() {
        println!(
            "  [{}] id={}, 内容={}",
            i,
            doc.id.clone().unwrap_or_default(),
            doc.content
        );
    }

    // [8] RRF 融合
    let hybrid = HybridRetriever::new();
    let fused_results = hybrid.retrieve(bm25_docs, vector_docs);

    println!("\nRRF 融合结果:");
    for (i, r) in fused_results.iter().enumerate() {
        println!(
            "  [{}] id={}, rrf_score={:.4}",
            i,
            r.document.id.clone().unwrap_or_default(),
            r.score
        );
    }

    // [9] 构建上下文
    let contexts: Vec<String> = fused_results
        .iter()
        .take(3)
        .map(|r| r.document.content.clone())
        .collect();

    println!("\n最终上下文 (top-3):");
    for (i, ctx) in contexts.iter().enumerate() {
        println!("  [{}] {}", i, ctx);
    }

    // [10] LLM 回答
    let messages = build_rag_messages(query, &contexts);
    let response = llm.chat(messages, None).await.unwrap();

    println!("\nLLM 回答:");
    println!("  {}", response.content);
}

/// 测试：不同查询词的混合检索效果
/// 验证：关键词查询BM25优势，语义查询向量优势，混合综合
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_hybrid_rag_query_types() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();

    println!("=== 测试不同查询类型的混合检索 ===");

    // 创建共享的 DocumentStore
    let doc_store = Arc::new(ChunkedDocumentStore::new());

    let knowledge_docs = vec![
        Document::new("机器学习监督学习无监督学习强化学习是AI的核心技术。").with_id("ml_doc"),
        Document::new("深度学习使用神经网络自动提取特征，应用于图像识别。").with_id("dl_doc"),
        Document::new("自然语言处理让计算机理解人类语言，应用包括翻译问答。").with_id("nlp_doc"),
    ];

    let mut bm25_retriever = ChunkedBM25Retriever::new(doc_store.clone());

    let vector_store = Arc::new(InMemoryVectorStore::new());
    let vector_retriever = SimilarityRetriever::new(vector_store.clone(), embeddings.clone());
    vector_retriever
        .add_documents(knowledge_docs.clone())
        .await
        .unwrap();

    let queries = vec![
        ("监督学习 无监督学习", "关键词查询 - BM25应该优势"),
        ("如何让计算机理解语言", "语义查询 - 向量应该优势"),
        ("机器学习算法类型", "混合查询 - 综合效果"),
    ];

    let hybrid = HybridRetriever::new();

    for (query, query_type) in queries {
        println!("\n---");
        println!("查询: {} ({})", query, query_type);

        let bm25_results = bm25_retriever.search(query, 3);
        let bm25_docs: Vec<Document> = bm25_results
            .iter()
            .map(|r| Document::new(r.content()).with_id(r.parent_id.clone()))
            .collect();

        println!("BM25结果数: {}", bm25_docs.len());

        let vector_docs = vector_retriever.retrieve(query, 3).await.unwrap();
        println!("向量结果数: {}", vector_docs.len());

        let fused_results = hybrid.retrieve(bm25_docs, vector_docs);

        println!("融合结果:");
        for r in &fused_results {
            println!(
                "  id={}, score={:.4}",
                r.document.id.clone().unwrap_or_default(),
                r.score
            );
        }

        let contexts: Vec<String> = fused_results
            .iter()
            .take(2)
            .map(|r| r.document.content.clone())
            .collect();

        let messages = build_rag_messages(query, &contexts);
        let response = llm.chat(messages, None).await.unwrap();

        println!("LLM回答: {}", response.content);
    }
}

/// 测试：持久化后的混合检索
/// 验证：BM25索引持久化后仍能正常混合检索
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_hybrid_rag_with_persistence() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();

    println!("=== 测试持久化后的混合检索 ===");

    // 创建共享的 DocumentStore
    let doc_store = Arc::new(ChunkedDocumentStore::new());

    let knowledge_docs = vec![
        Document::new("人工智能AI是计算机科学分支。").with_id("ai_doc"),
        Document::new("机器学习ML是AI核心技术。").with_id("ml_doc"),
    ];

    // [1] BM25 创建并保存
    let bm25_retriever = ChunkedBM25Retriever::new(doc_store.clone());

    let temp_file = NamedTempFile::new().unwrap();
    bm25_retriever.save(temp_file.path()).unwrap();

    println!("BM25索引保存到: {}", temp_file.path().display());

    // [2] 加载 BM25（需要传入同一个 store）
    let loaded_bm25 = ChunkedBM25Retriever::load(doc_store.clone(), temp_file.path()).unwrap();

    println!("BM25索引加载成功");

    // [3] 向量检索器
    let vector_store = Arc::new(InMemoryVectorStore::new());
    let vector_retriever = SimilarityRetriever::new(vector_store.clone(), embeddings.clone());
    vector_retriever
        .add_documents(knowledge_docs.clone())
        .await
        .unwrap();

    // [4] 查询
    let query = "什么是人工智能？";
    println!("\n查询: {}", query);

    // [5] 混合检索
    let mut loaded_bm25_mut = loaded_bm25;
    let bm25_results = loaded_bm25_mut.search(query, 3);
    let bm25_docs: Vec<Document> = bm25_results
        .iter()
        .map(|r| Document::new(r.content()).with_id(r.parent_id.clone()))
        .collect();

    let vector_docs = vector_retriever.retrieve(query, 3).await.unwrap();

    let hybrid = HybridRetriever::new();
    let fused_results = hybrid.retrieve(bm25_docs, vector_docs);

    println!("融合结果数: {}", fused_results.len());

    // [6] LLM回答
    let contexts: Vec<String> = fused_results
        .iter()
        .map(|r| r.document.content.clone())
        .collect();

    let messages = build_rag_messages(query, &contexts);
    let response = llm.chat(messages, None).await.unwrap();

    println!("LLM回答: {}", response.content);
}
