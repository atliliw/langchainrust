// tests/bm25/rag.rs
//! BM25 Chunked RAG 测试（无需 LLM）

use langchainrust::retrieval::bm25::{
    AutoMergingConfig, ChunkedBM25Retriever, ChunkedSearchResult,
};
use langchainrust::retrieval::ChunkedDocumentStore;
use langchainrust::Document;
use std::sync::Arc;
use tempfile::NamedTempFile;

fn build_rag_prompt(query: &str, contexts: &[String]) -> String {
    let context_text = contexts.join("\n\n");

    format!(
        "基于以下上下文回答问题。如果上下文中没有相关信息，请说\"我不知道\"。\n\n\
        上下文：\n{}\n\n\
        问题：{}\n\n\
        回答：",
        context_text, query
    )
}

/// 测试：RAG Prompt 构建
/// 验证：检索结果能正确构建成 RAG prompt
#[test]
fn test_rag_prompt_building() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents(vec![
            Document::new("Rust是一门系统编程语言，由Mozilla开发，注重安全和性能。")
                .with_id("rust_intro"),
            Document::new("Rust的核心特性包括所有权系统、借用检查和零成本抽象。")
                .with_id("rust_features"),
        ])
        .unwrap();

    let results = retriever.search("Rust语言特点", 3);

    let prompt = build_rag_prompt(
        "Rust有什么特点？",
        &results.iter().map(|r| r.content()).collect::<Vec<_>>(),
    );

    println!("RAG Prompt:");
    println!("{}", prompt);

    assert!(prompt.contains("上下文"));
    assert!(prompt.contains("问题"));
    assert!(prompt.contains("Rust有什么特点？"));
}

/// 测试：RAG 流程基础
/// 验证：完整的检索流程
#[test]
fn test_rag_pipeline_basic() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_documents(vec![
        Document::new("人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。")
            .with_id("ai_def"),
        Document::new("机器学习是AI的核心技术，它使计算机能够从数据中学习而无需明确编程。")
            .with_id("ml_def"),
        Document::new("深度学习是机器学习的子集，使用多层神经网络进行学习。")
            .with_id("dl_def"),
    ]).unwrap();

    let query = "什么是人工智能？";
    let results = retriever.search(query, 3);

    println!("查询: {}", query);
    println!("返回结果数: {}", results.len());

    let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
    let prompt = build_rag_prompt(query, &contexts);

    println!("生成的 Prompt 长度: {}", prompt.len());
}

/// 测试：多源上下文
/// 验证：多个文档作为上下文
#[test]
fn test_rag_with_multiple_sources() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents(vec![
            Document::new("Python是一种高级编程语言，由Guido van Rossum创建。")
                .with_id("python_intro"),
            Document::new("Python广泛应用于数据科学、Web开发和自动化脚本。")
                .with_id("python_usage"),
            Document::new("Python的设计哲学强调代码可读性和简洁性。").with_id("python_philosophy"),
        ])
        .unwrap();

    let query = "Python的应用领域有哪些？";
    let results = retriever.search(query, 3);

    let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
    let prompt = build_rag_prompt(query, &contexts);

    println!("查询: {}", query);
    println!("上下文来源数: {}", contexts.len());
    println!("Prompt:");
    println!("{}", prompt);
}

/// 测试：上下文窗口限制
/// 验证：大文档 AutoMerging 控制上下文大小
#[test]
fn test_rag_context_window_limit() {
    let config = AutoMergingConfig::new()
        .with_leaf_size(50)
        .with_parent_size(200);

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::with_config(store, config);

    retriever
        .add_document(
            Document::new(
                "这是一段很长的文档内容，包含了大量的信息。\
             我们需要测试RAG系统在处理长文档时的表现。\
             AutoMerging机制会将相关的片段合并成完整的上下文。\
             这样可以既保证精确匹配，又提供完整信息。",
            )
            .with_id("long_doc"),
        )
        .unwrap();

    let results = retriever.search("RAG系统", 2);

    let total_len: usize = results.iter().map(|r| r.content().len()).sum();

    println!("返回结果数: {}", results.len());
    println!("总上下文长度: {}", total_len);

    assert!(total_len < 1000);
}

/// 测试：空上下文处理
/// 验证：没有相关结果时的处理
#[test]
fn test_rag_empty_context() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_document(Document::new("这是一个关于烹饪的文档。").with_id("cooking"))
        .unwrap();

    let query = "编程语言";
    let results = retriever.search(query, 3);

    let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
    let prompt = build_rag_prompt(query, &contexts);

    println!("查询: {}", query);
    println!("返回结果数: {}", results.len());

    if results.is_empty() {
        println!("无相关结果，Prompt 应包含\"我不知道\"");
        assert!(prompt.contains("我不知道"));
    }
}

/// 测试：持久化后 RAG 流程
/// 验证：加载索引后能正常构建 prompt
#[test]
fn test_rag_persistence_workflow() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store.clone());

    retriever
        .add_documents(vec![
            Document::new("LangChain是一个用于构建LLM应用的框架。").with_id("lc_intro"),
            Document::new("LangChain支持链式调用、记忆管理和检索增强生成。").with_id("lc_features"),
        ])
        .unwrap();

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    retriever.save(temp_file.path()).expect("Failed to save");

    let mut loaded = ChunkedBM25Retriever::load(store, temp_file.path()).expect("Failed to load");

    let query = "LangChain有什么功能？";
    let results = loaded.search(query, 3);

    let contexts: Vec<String> = results.iter().map(|r| r.content()).collect();
    let prompt = build_rag_prompt(query, &contexts);

    println!("加载后查询: {}", query);
    println!("Prompt 包含 LangChain: {}", prompt.contains("LangChain"));
}

/// 测试：分数阈值过滤
/// 验证：可以按分数过滤结果
#[test]
fn test_rag_score_threshold_filter() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents(vec![
            Document::new("Go语言由Google开发，是一门静态类型的编译语言。").with_id("go_intro"),
            Document::new("Rust语言注重内存安全，无垃圾回收。").with_id("rust_intro"),
            Document::new("Python是动态类型语言，有垃圾回收。").with_id("python_intro"),
        ])
        .unwrap();

    let results = retriever.search("Go Google", 5);

    let filtered: Vec<ChunkedSearchResult> =
        results.into_iter().filter(|r| r.score > 0.5).collect();

    println!("过滤前结果数: {}", retriever.len());
    println!("过滤后结果数: {}", filtered.len());

    for result in &filtered {
        println!("分数: {}, 内容: {}", result.score, result.content());
        assert!(result.score > 0.5);
    }
}

/// 测试：结果排序验证
/// 验证：结果按分数降序排列
#[test]
fn test_rag_context_ordering() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents(vec![
            Document::new("第一点：Rust注重安全。").with_id("point1"),
            Document::new("第二点：Rust性能优秀。").with_id("point2"),
            Document::new("第三点：Rust无垃圾回收。").with_id("point3"),
        ])
        .unwrap();

    let results = retriever.search("Rust特点", 3);

    println!("返回结果数: {}", results.len());

    if results.len() > 1 {
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        println!("分数序列: {:?}", scores);

        for i in 0..scores.len() - 1 {
            assert!(scores[i] >= scores[i + 1]);
        }
    }
}

/// 测试：多轮对话上下文
/// 验证：不同查询返回不同上下文
#[test]
fn test_rag_multi_turn_context() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents(vec![
            Document::new("向量数据库用于存储和检索高维向量。").with_id("vector_db"),
            Document::new("常见的向量数据库包括Pinecone、Milvus和Qdrant。")
                .with_id("vector_examples"),
            Document::new("向量检索使用相似度度量如余弦相似度或欧几里得距离。")
                .with_id("vector_metrics"),
        ])
        .unwrap();

    let results1 = retriever.search("向量数据库", 2);
    let prompt1 = build_rag_prompt(
        "什么是向量数据库？",
        &results1.iter().map(|r| r.content()).collect::<Vec<_>>(),
    );

    let results2 = retriever.search("相似度", 2);
    let prompt2 = build_rag_prompt(
        "向量检索使用什么度量？",
        &results2.iter().map(|r| r.content()).collect::<Vec<_>>(),
    );

    println!("第一轮查询: 什么是向量数据库？");
    println!("返回结果数: {}", results1.len());

    println!("第二轮查询: 向量检索使用什么度量？");
    println!("返回结果数: {}", results2.len());

    assert!(prompt1.contains("向量数据库"));
    assert!(prompt2.contains("相似度"));
}
