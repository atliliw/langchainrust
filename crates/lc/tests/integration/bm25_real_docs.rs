// tests/integration/bm25_real_docs.rs
//! BM25 真实文档数据测试
//!
//! 使用 tests/data/ 目录下的真实文档测试 BM25 检索功能：
//! - 编程语言文档（中英文）
//! - LangChainRust 框架文档
//! - BM25 算法文档

use langchainrust::{BM25Retriever, Document};
use std::fs;

// ============================================================================
// 测试数据加载
// ============================================================================

/// 从文件加载文档
///
/// 每行作为一个独立文档
fn load_documents_from_file(path: &str) -> Vec<Document> {
    let content =
        fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to load file: {}", path));

    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Document::new(line.trim()))
        .collect()
}

// ============================================================================
// 英文文档测试
// ============================================================================

/// 测试英文编程语言文档检索
///
/// 使用 tests/data/programming_languages_en.txt
/// 验证 BM25 能正确检索编程语言相关文档
#[test]
fn test_bm25_english_programming_languages() {
    let documents = load_documents_from_file("tests/data/programming_languages_en.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 验证文档数量（7种编程语言）
    assert!(!retriever.is_empty(), "应加载至少 1 个文档");

    // 搜索 "systems programming" 应返回 Rust 文档
    let results = retriever.search("systems programming language", 3);
    assert!(!results.is_empty(), "应返回匹配结果");

    // Rust 文档应排在前列（包含 "systems programming language"）
    let rust_found = results
        .iter()
        .any(|r| r.document.content.contains("Rust") && r.document.content.contains("systems"));
    assert!(rust_found, "应找到 Rust 相关文档");
}

/// 测试英文关键词精确匹配
///
/// BM25 应精确匹配关键词，而非语义相似
#[test]
fn test_bm25_exact_keyword_match() {
    let documents = load_documents_from_file("tests/data/programming_languages_en.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 精确搜索 "garbage collection"
    let results = retriever.search("garbage collection", 2);

    // Rust 文档包含 "without garbage collection"
    let rust_doc = results
        .iter()
        .find(|r| r.document.content.contains("garbage collection"));

    assert!(rust_doc.is_some(), "应找到包含 garbage collection 的文档");
}

// ============================================================================
// 中文文档测试
// ============================================================================

/// 测试中文编程语言文档检索
///
/// 使用 tests/data/programming_languages_zh.txt
/// 验证 BM25 中文分词和检索功能
#[test]
fn test_bm25_chinese_programming_languages() {
    let documents = load_documents_from_file("tests/data/programming_languages_zh.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 验证文档加载
    assert!(!retriever.is_empty(), "应加载中文文档");

    // 搜索 "系统级编程"
    let results = retriever.search("系统级编程", 3);
    assert!(!results.is_empty(), "应返回中文匹配结果");

    // Rust 文档包含 "系统级编程"
    let rust_found = results.iter().any(|r| r.document.content.contains("Rust"));
    assert!(rust_found, "应找到 Rust 中文文档");
}

/// 测试中文关键词检索
///
/// 中文单字和双字组合应正确匹配
#[test]
fn test_bm25_chinese_keywords() {
    let documents = load_documents_from_file("tests/data/programming_languages_zh.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 搜索 "垃圾回收"
    let results = retriever.search("垃圾回收", 2);

    // Rust 文档包含 "无需垃圾回收"
    assert!(
        results
            .iter()
            .any(|r| r.document.content.contains("垃圾回收")),
        "应找到垃圾回收相关文档"
    );

    // 搜索 "微服务"
    let go_results = retriever.search("微服务", 2);
    assert!(
        go_results.iter().any(|r| r.document.content.contains("Go")),
        "微服务应关联到 Go 文档"
    );
}

/// 测试中文混合文档
///
/// 使用 tests/data/programming_short_zh.txt（短文档）
#[test]
fn test_bm25_chinese_short_documents() {
    let documents = load_documents_from_file("tests/data/programming_short_zh.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 搜索 "机器学习"
    let results = retriever.search("机器学习", 2);

    assert!(
        results
            .iter()
            .any(|r| r.document.content.contains("Python")),
        "机器学习应关联到 Python"
    );

    // 搜索 "前端开发"
    let web_results = retriever.search("前端开发", 2);
    assert!(
        web_results
            .iter()
            .any(|r| r.document.content.contains("JavaScript")),
        "前端开发应关联到 JavaScript"
    );
}

// ============================================================================
// LangChainRust 框架文档测试
// ============================================================================

/// 测试框架文档检索
///
/// 使用 tests/data/langchainrust_docs.txt
/// 验证对框架特定功能的检索
#[test]
fn test_bm25_langchainrust_docs() {
    let documents = load_documents_from_file("tests/data/langchainrust_docs.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 搜索 "LangGraph"
    let results = retriever.search("LangGraph workflow", 5);

    assert!(
        results
            .iter()
            .any(|r| r.document.content.contains("LangGraph")),
        "应找到 LangGraph 文档"
    );

    // 搜索 "memory system"
    let memory_results = retriever.search("memory system", 3);
    assert!(
        memory_results
            .iter()
            .any(|r| r.document.content.contains("Memory")),
        "应找到 Memory 文档"
    );
}

/// 测试框架功能关键词检索
///
/// 验证专业功能术语检索
#[test]
fn test_bm25_framework_features() {
    let documents = load_documents_from_file("tests/data/langchainrust_docs.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 搜索 "Human-in-the-loop"
    let results = retriever.search("Human in the loop", 3);

    assert!(
        results.iter().any(|r| r.document.content.contains("Human")),
        "应找到 Human-in-the-loop 文档"
    );

    // 搜索 "Qdrant"
    let vector_results = retriever.search("Qdrant vector store", 2);
    assert!(
        vector_results
            .iter()
            .any(|r| r.document.content.contains("Qdrant")),
        "应找到 Qdrant 文档"
    );
}

// ============================================================================
// BM25 算法文档测试
// ============================================================================

/// 测试 BM25 算法文档检索
///
/// 使用 tests/data/bm25_docs.txt
/// 验证算法术语检索效果
#[test]
fn test_bm25_algorithm_docs() {
    let documents = load_documents_from_file("tests/data/bm25_docs.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 搜索 "IDF inverse document frequency"
    let results = retriever.search("IDF inverse document frequency", 3);

    assert!(
        results.iter().any(|r| r.document.content.contains("IDF")),
        "应找到 IDF 文档"
    );

    // 搜索 "k1 parameter"
    let param_results = retriever.search("k1 parameter", 2);
    assert!(
        param_results
            .iter()
            .any(|r| r.document.content.contains("k1")),
        "应找到 k1 参数文档"
    );
}

/// 测试算法原理检索
///
/// 搜索 BM25 算法核心原理
#[test]
fn test_bm25_algorithm_principles() {
    let documents = load_documents_from_file("tests/data/bm25_docs.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(documents);

    // 搜索 "term frequency saturation"
    let results = retriever.search("term frequency saturation", 3);

    assert!(
        results
            .iter()
            .any(|r| r.document.content.contains("saturation")),
        "应找到词频饱和文档"
    );

    // 搜索 "document length normalization"
    let norm_results = retriever.search("document length normalization", 2);
    assert!(
        norm_results
            .iter()
            .any(|r| r.document.content.contains("length")),
        "应找到文档长度归一化文档"
    );
}

// ============================================================================
// 混合场景测试
// ============================================================================

/// 测试多文件文档集合
///
/// 加载多个文件构建综合索引
#[test]
fn test_bm25_multi_file_collection() {
    let mut all_documents = Vec::new();

    // 加载多个数据文件
    all_documents.extend(load_documents_from_file(
        "tests/data/programming_languages_en.txt",
    ));
    all_documents.extend(load_documents_from_file(
        "tests/data/langchainrust_docs.txt",
    ));

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(all_documents);

    // 验证文档总数
    let total_docs = retriever.len();
    assert!(total_docs > 20, "应加载超过 20 个文档");

    // 搜索 "Rust"
    let results = retriever.search("Rust", 5);

    // 应同时返回编程语言文档和框架文档中的 Rust 相关内容
    assert!(!results.is_empty(), "应返回 Rust 相关结果");
}

/// 测试长文档与短文档检索
///
/// 验证文档长度归一化效果
#[test]
fn test_bm25_document_length_effect() {
    let short_docs = load_documents_from_file("tests/data/programming_short_zh.txt");
    let long_docs = load_documents_from_file("tests/data/programming_languages_zh.txt");

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(short_docs);
    retriever.add_documents_sync(long_docs);

    // 搜索 "Python"
    let results = retriever.search("Python", 5);

    // 验证检索结果
    assert!(!results.is_empty(), "应返回 Python 相关文档");

    // 检查评分排序
    for i in 0..results.len().saturating_sub(1) {
        assert!(
            results[i].score >= results[i + 1].score,
            "结果应按评分降序排列"
        );
    }
}

// ============================================================================
// 边界测试
// ============================================================================

/// 测试空文档文件
///
/// 空文件应返回空文档列表
#[test]
fn test_bm25_empty_file_handling() {
    // 创建空文档列表
    let empty_docs: Vec<Document> = Vec::new();

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(empty_docs);

    assert!(retriever.is_empty(), "空文档集合应返回 is_empty = true");

    let results = retriever.search("test", 5);
    assert!(results.is_empty(), "空索引搜索应返回空结果");
}

/// 测试单文档检索
///
/// 单个文档的检索效果
#[test]
fn test_bm25_single_document() {
    let single_doc = vec![Document::new(
        "This is a single test document about Rust programming",
    )];

    let mut retriever = BM25Retriever::new();
    retriever.add_documents_sync(single_doc);

    assert_eq!(retriever.len(), 1, "应有 1 个文档");

    let results = retriever.search("Rust", 1);
    assert_eq!(results.len(), 1, "应返回 1 个结果");
}
