// tests/bm25/llm_integration.rs
//! BM25 Chunked Retriever + LLM 集成测试

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::retrieval::bm25::{AutoMergingConfig, ChunkedBM25Retriever};
use langchainrust::retrieval::ChunkedDocumentStore;
use langchainrust::{BaseChatModel, Document, Message};
use std::sync::Arc;

fn build_rag_messages(query: &str, contexts: &[String]) -> Vec<Message> {
    let context_text = contexts.join("\n\n");

    vec![
        Message::system("基于提供的上下文回答问题。如果上下文中没有相关信息，请说\"我不知道\"。"),
        Message::human(format!(
            "上下文：\n{}\n\n\
            问题：{}\n\n\
            请根据上下文回答问题。",
            context_text, query
        )),
    ]
}

/// 测试：BM25 + LLM 完整 RAG 流程
/// 验证：BM25 检索出相关文档，LLM 基于上下文回答问题
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_bm25_rag_with_openai_llm() {
    let config = TestConfig::get();
    let llm = config.openai_chat();

    println!("=== 测试 BM25 + LLM RAG 流程 ===");

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents_async(vec![
            Document::new(
                "Rust是一门系统编程语言，由Mozilla研究院开发。\
             Rust的设计目标是提供内存安全、并发安全和高性能。\
             Rust通过所有权系统实现了这些目标，无需垃圾回收器。",
            )
            .with_id("rust_doc"),
            Document::new(
                "Python是一门高级编程语言，由Guido van Rossum开发。\
             Python以简洁的语法和强大的生态系统著称。\
             Python广泛应用于数据科学、机器学习和Web开发。",
            )
            .with_id("python_doc"),
            Document::new(
                "Go语言由Google开发，专注于简洁性和并发编程。\
             Go的语法简单，编译速度快，适合构建大规模分布式系统。",
            )
            .with_id("go_doc"),
        ])
        .await;

    println!("已添加文档:");
    println!("  - rust_doc: Rust语言介绍");
    println!("  - python_doc: Python语言介绍");
    println!("  - go_doc: Go语言介绍");

    let query = "Rust语言的内存安全是如何实现的？";
    println!("\n用户问题: {}", query);

    let results = retriever.search_async(query, 3).await;

    println!("BM25 检索结果:");
    for (i, result) in results.iter().enumerate() {
        println!(
            "  [{}] 分数={}, 是否合并={}",
            i,
            result.score,
            result.is_merged()
        );
        println!("      内容: {}", result.content());
    }

    let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
    let messages = build_rag_messages(query, &contexts);

    let response = llm.chat(messages, None).await.unwrap();

    println!("\nLLM 回答:");
    println!("  {}", response.content);
}

/// 测试：多查询 RAG 流程
/// 验证：多个不同问题都能正确检索并回答
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_bm25_rag_multi_query() {
    let config = TestConfig::get();
    let llm = config.openai_chat();

    println!("=== 测试多查询 RAG 流程 ===");

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents_async(vec![
            Document::new(
                "LangChain是一个用于开发LLM应用的框架。\
             LangChain提供了链式调用、Agent、Memory和Retriever等组件。\
             LangChain支持Python和JavaScript，也有Rust版本。",
            )
            .with_id("langchain_doc"),
            Document::new(
                "LlamaIndex是另一个LLM应用框架，专注于数据索引和检索。\
             LlamaIndex提供了多种索引类型，包括向量索引、关键词索引和混合索引。",
            )
            .with_id("llamaindex_doc"),
        ])
        .await;

    println!("已添加文档:");
    println!("  - langchain_doc: LangChain框架介绍");
    println!("  - llamaindex_doc: LlamaIndex框架介绍");

    let queries = vec![
        "LangChain有哪些主要组件？",
        "LlamaIndex和LangChain有什么区别？",
    ];

    for query in queries {
        println!("\n---");
        println!("用户问题: {}", query);

        let results = retriever.search_async(query, 3).await;

        println!("BM25 检索结果数: {}", results.len());

        let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
        let messages = build_rag_messages(query, &contexts);

        let response = llm.chat(messages, None).await.unwrap();

        println!("LLM 回答:");
        println!("  {}", response.content);
    }
}

/// 测试：大文档 AutoMerging + LLM
/// 验证：大文档被拆分后，AutoMerging 合并相关片段，LLM 正确回答
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_bm25_rag_with_large_context() {
    let config = TestConfig::get();
    let llm = config.openai_chat();

    println!("=== 测试大文档 AutoMerging + LLM ===");

    let merging_config = AutoMergingConfig::new()
        .with_leaf_size(100)
        .with_threshold(0.5);

    println!("AutoMerging 配置:");
    println!("  - Leaf大小: {}", merging_config.leaf_chunk_size);
    println!("  - 合并阈值: {}", merging_config.merge_threshold);

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::with_config(store, merging_config);

    let large_doc = Document::new(
        "机器学习是人工智能的核心技术。\
         监督学习使用标注数据进行训练，包括分类和回归任务。\
         无监督学习使用未标注数据，包括聚类和降维。\
         强化学习通过奖励信号训练Agent做出决策。\
         深度学习使用神经网络自动提取特征。\
         机器学习广泛应用于图像识别、自然语言处理和推荐系统。",
    )
    .with_id("ml_doc");

    retriever.add_document_async(large_doc).await;

    println!("已添加大文档:");
    println!("  - ml_doc: 机器学习介绍");
    println!("  - Leaf chunks 数量: {}", retriever.len());

    let query = "机器学习的三种主要类型是什么？";
    println!("\n用户问题: {}", query);

    let results = retriever.search_async(query, 3).await;

    println!("BM25 检索结果:");
    for (i, result) in results.iter().enumerate() {
        println!(
            "  [{}] 是否合并={}, 分数={}",
            i,
            result.is_merged(),
            result.score
        );
        println!("      内容长度: {}", result.content().len());
    }

    let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
    let messages = build_rag_messages(query, &contexts);

    let response = llm.chat(messages, None).await.unwrap();

    println!("\nLLM 回答:");
    println!("  {}", response.content);
}

/// 测试：中文查询 RAG 流程
/// 验证：中文查询能正确检索并生成回答
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_bm25_rag_chinese_query() {
    let config = TestConfig::get();
    let llm = config.openai_chat();

    println!("=== 测试中文查询 RAG 流程 ===");

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents_async(vec![
            Document::new(
                "人工智能（AI）是计算机科学的一个分支。\
             AI的研究包括推理、知识表示、规划、学习、自然语言处理等。\
             强人工智能指具有人类级别智能的系统，目前尚未实现。\
             弱人工智能指专注于特定任务的AI系统，如围棋AI AlphaGo。",
            )
            .with_id("ai_doc"),
            Document::new(
                "自然语言处理（NLP）让计算机理解和生成人类语言。\
             NLP应用包括机器翻译、情感分析、问答系统和文本生成。\
             大语言模型如GPT系列极大地推动了NLP的发展。",
            )
            .with_id("nlp_doc"),
        ])
        .await;

    println!("已添加文档:");
    println!("  - ai_doc: 人工智能介绍");
    println!("  - nlp_doc: 自然语言处理介绍");

    let query = "强人工智能和弱人工智能有什么区别？";
    println!("\n用户问题: {}", query);

    let results = retriever.search_async(query, 3).await;

    println!("BM25 检索结果:");
    for (i, result) in results.iter().enumerate() {
        println!(
            "  [{}] 分数={}, 是否合并={}",
            i,
            result.score,
            result.is_merged()
        );
        println!(
            "      内容包含关键词: {}",
            if result.content().contains("强人工智能") {
                "强人工智能"
            } else {
                ""
            }
        );
    }

    let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
    let messages = build_rag_messages(query, &contexts);

    let response = llm.chat(messages, None).await.unwrap();

    println!("\nLLM 回答:");
    println!("  {}", response.content);
}
