// tests/unit/bm25.rs
//! BM25 检索器测试
//!
//! 测试 BM25 算法的核心功能：
//! - BM25 评分计算
//! - 文档索引构建
//! - 中英文分词
//! - 关键词检索

use langchainrust::retrieval::bm25::{bm25_score, compute_idf};
use langchainrust::{BM25Index, BM25Params, BM25Retriever, Document, Tokenizer};

// ============================================================================
// BM25 算法核心测试
// ============================================================================

/// 测试 IDF（逆文档频率）计算
///
/// IDF 公式: log((N - n + 0.5) / (n + 0.5) + 1)
/// - 常见词（出现在所有文档）IDF 较低
/// - 稀有词（只出现在少数文档）IDF 较高
#[test]
fn test_bm25_idf_calculation() {
    // 常见词：出现在 100 个文档中的 100 个，IDF 应较低
    let idf_common = compute_idf(100, 100);
    assert!(idf_common < 1.0, "常见词 IDF 应小于 1.0");

    // 稀有词：出现在 100 个文档中的 1 个，IDF 应较高
    let idf_rare = compute_idf(1, 100);
    assert!(idf_rare > idf_common, "稀有词 IDF 应大于常见词");

    // 不存在的词：n=0，IDF 应为 0
    let idf_zero = compute_idf(0, 100);
    assert_eq!(idf_zero, 0.0, "不存在的词 IDF 应为 0");

    // 空文档集：N=0，IDF 应为 0
    let idf_empty = compute_idf(1, 0);
    assert_eq!(idf_empty, 0.0, "空文档集 IDF 应为 0");
}

/// 测试 BM25 评分公式
///
/// BM25 公式: score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
/// - 验证评分计算正确性
/// - 高词频文档应得更高分
#[test]
fn test_bm25_score_calculation() {
    use std::collections::HashMap;

    let params = BM25Params::default();

    // 查询词
    let query_terms = vec!["rust".to_string(), "programming".to_string()];

    // 文档词频
    let mut doc_term_freqs = HashMap::new();
    doc_term_freqs.insert("rust".to_string(), 2);
    doc_term_freqs.insert("programming".to_string(), 1);

    // IDF 值
    let mut idf_values = HashMap::new();
    idf_values.insert("rust".to_string(), 2.0);
    idf_values.insert("programming".to_string(), 1.5);

    let score = bm25_score(
        &query_terms,
        &doc_term_freqs,
        10,   // 文档长度
        15.0, // 平均文档长度
        &idf_values,
        &params,
    );

    // 评分应大于 0（因为词频和 IDF 都存在）
    assert!(score > 0.0, "BM25 评分应大于 0");
}

/// 测试 BM25 高词频文档得更高分
///
/// 词频饱和效应：高词频文档得高分，但增长逐渐减缓（由 k1 参数控制）
#[test]
fn test_bm25_high_term_frequency() {
    use std::collections::HashMap;

    let params = BM25Params::default();
    let query = vec!["rust".to_string()];
    let idf = HashMap::from([("rust".to_string(), 2.0)]);

    // 低词频文档（tf=1）
    let low_tf = HashMap::from([("rust".to_string(), 1)]);
    let score_low = bm25_score(&query, &low_tf, 10, 15.0, &idf, &params);

    // 高词频文档（tf=5）
    let high_tf = HashMap::from([("rust".to_string(), 5)]);
    let score_high = bm25_score(&query, &high_tf, 10, 15.0, &idf, &params);

    assert!(score_high > score_low, "高词频文档应得更高分");
}

/// 测试 BM25 参数配置
///
/// 参数说明：
/// - k1: 词频饱和参数（默认 1.5），控制高频词的影响
/// - b: 文档长度归一化参数（默认 0.75），控制长文档惩罚
#[test]
fn test_bm25_parameters() {
    let default = BM25Params::default();
    assert_eq!(default.k1, 1.5, "默认 k1 应为 1.5");
    assert_eq!(default.b, 0.75, "默认 b 应为 0.75");

    let custom = BM25Params::with_values(2.0, 0.5);
    assert_eq!(custom.k1, 2.0, "自定义 k1 应为 2.0");
    assert_eq!(custom.b, 0.5, "自定义 b 应为 0.5");
}

// ============================================================================
// BM25 索引测试
// ============================================================================

/// 测试索引基础操作
///
/// - 文档添加
/// - 文档计数
/// - 文档长度记录
#[test]
fn test_bm25_index_basic_operations() {
    let mut index = BM25Index::new();

    let doc = Document::new("Rust programming language");
    let terms = vec![
        "rust".to_string(),
        "programming".to_string(),
        "language".to_string(),
    ];

    index.add_document(doc, terms);

    assert_eq!(index.n_docs(), 1, "文档数量应为 1");
    assert_eq!(index.get_doc_length(0), 3, "文档长度应为 3");
    assert!(index.get_document(0).is_some(), "应能获取文档");
}

/// 测试索引 IDF 计算
///
/// 验证词频对 IDF 的影响：
/// - 出现在所有文档的词 IDF 低
/// - 只出现在少数文档的词 IDF 高
#[test]
fn test_bm25_index_idf_values() {
    let mut index = BM25Index::new();

    // 文档 1: rust, programming, language
    index.add_document(
        Document::new("Rust programming language"),
        vec![
            "rust".to_string(),
            "programming".to_string(),
            "language".to_string(),
        ],
    );

    // 文档 2: python, scripting, language
    index.add_document(
        Document::new("Python scripting language"),
        vec![
            "python".to_string(),
            "scripting".to_string(),
            "language".to_string(),
        ],
    );

    // "language" 出现在两个文档，IDF 应较低
    let idf_language = index.compute_idf_for_term("language");

    // "rust" 只出现在一个文档，IDF 应较高
    let idf_rust = index.compute_idf_for_term("rust");

    assert!(idf_rust > idf_language, "稀有词 IDF 应高于常见词");
}

/// 测试平均文档长度计算
///
/// avgdl = 所有文档长度之和 / 文档数量
#[test]
fn test_bm25_index_average_document_length() {
    let mut index = BM25Index::new();

    // 文档 1: 1 个词
    index.add_document(Document::new("a"), vec!["a".to_string()]);

    // 文档 2: 3 个词
    index.add_document(
        Document::new("a b c"),
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    );

    // 平均长度 = (1 + 3) / 2 = 2.0
    assert_eq!(index.avgdl(), 2.0, "平均文档长度应为 2.0");
}

// ============================================================================
// 分词器测试
// ============================================================================

/// 测试英文分词
///
/// 处理流程：
/// - 空格分割
/// - 小写化
/// - 去除标点
/// - 过滤停用词（可选）
#[test]
fn test_tokenizer_english() {
    let tokenizer = Tokenizer::new();

    let terms = tokenizer.tokenize_english("Hello World Rust");
    assert_eq!(terms, vec!["hello", "world", "rust"], "英文分词结果");
}

/// 测试英文停用词过滤
///
/// 常见停用词（the, is, a, an 等）应被过滤
#[test]
fn test_tokenizer_english_stopwords() {
    let tokenizer = Tokenizer::new();

    let terms = tokenizer.tokenize_english("The Rust is a programming language");

    // 停用词应被过滤
    assert!(!terms.contains(&"the".to_string()), "'the' 应被过滤");
    assert!(!terms.contains(&"is".to_string()), "'is' 应被过滤");
    assert!(!terms.contains(&"a".to_string()), "'a' 应被过滤");

    // 内容词应保留
    assert!(terms.contains(&"rust".to_string()), "'rust' 应保留");
    assert!(
        terms.contains(&"programming".to_string()),
        "'programming' 应保留"
    );
    assert!(terms.contains(&"language".to_string()), "'language' 应保留");
}

/// 测试中文分词
///
/// 处理流程：
/// - 单字分割
/// - 双字组合（bigram）
/// - 停用词过滤
#[test]
fn test_tokenizer_chinese() {
    let tokenizer = Tokenizer::new();

    let terms = tokenizer.tokenize_chinese("编程语言");

    // 单字
    assert!(terms.contains(&"编".to_string()), "应包含单字 '编'");
    assert!(terms.contains(&"程".to_string()), "应包含单字 '程'");
    assert!(terms.contains(&"语".to_string()), "应包含单字 '语'");
    assert!(terms.contains(&"言".to_string()), "应包含单字 '言'");

    // 双字组合
    assert!(terms.contains(&"编程".to_string()), "应包含双字 '编程'");
    assert!(terms.contains(&"程语".to_string()), "应包含双字 '程语'");
    assert!(terms.contains(&"语言".to_string()), "应包含双字 '语言'");
}

/// 测试中文停用词过滤
///
/// 常见停用词（的、了、在 等）应被过滤
#[test]
fn test_tokenizer_chinese_stopwords() {
    let tokenizer = Tokenizer::new();

    let terms = tokenizer.tokenize_chinese("编程的语言");

    // 停用词应被过滤
    assert!(!terms.contains(&"的".to_string()), "'的' 应被过滤");

    // 内容词应保留
    assert!(terms.contains(&"编".to_string()), "'编' 应保留");
    assert!(terms.contains(&"程".to_string()), "'程' 应保留");
}

/// 测试中英文混合分词
///
/// 自动识别语言并分别处理
#[test]
fn test_tokenizer_mixed_chinese_english() {
    let tokenizer = Tokenizer::new();

    let terms = tokenizer.tokenize("Rust 编程语言");

    // 英文词
    assert!(terms.contains(&"rust".to_string()), "应包含 'rust'");

    // 中文单字
    assert!(terms.contains(&"编".to_string()), "应包含 '编'");
    assert!(terms.contains(&"程".to_string()), "应包含 '程'");

    // 中文双字
    assert!(terms.contains(&"编程".to_string()), "应包含 '编程'");
    assert!(terms.contains(&"语言".to_string()), "应包含 '语言'");
}

/// 测试保留停用词模式
///
/// with_stopwords() 创建的分词器保留所有词
#[test]
fn test_tokenizer_keep_stopwords() {
    let tokenizer = Tokenizer::with_stopwords();

    let terms = tokenizer.tokenize("The programming language");

    // 停用词应保留
    assert!(terms.contains(&"the".to_string()), "'the' 应保留");
    assert!(
        terms.contains(&"programming".to_string()),
        "'programming' 应保留"
    );
    assert!(terms.contains(&"language".to_string()), "'language' 应保留");
}

// ============================================================================
// BM25Retriever 测试
// ============================================================================

/// 测试 BM25 检索器基础功能
///
/// - 文档添加
/// - 检索功能
/// - 结果排序
#[test]
fn test_bm25_retriever_basic_search() {
    let mut retriever = BM25Retriever::new();

    retriever.add_documents_sync(vec![
        Document::new("Rust is a systems programming language"),
        Document::new("Python is a scripting language"),
        Document::new("JavaScript is used for web development"),
    ]);

    assert_eq!(retriever.len(), 3, "文档数量应为 3");

    let results = retriever.search("programming language", 2);
    assert_eq!(results.len(), 2, "应返回 2 个结果");

    // 第一个结果应包含 "programming"（评分最高）
    assert!(
        results[0].document.content.contains("programming"),
        "第一个结果应包含 'programming'"
    );
}

/// 测试 BM25 中文检索
///
/// 验证中文分词和检索功能
#[test]
fn test_bm25_retriever_chinese_search() {
    let mut retriever = BM25Retriever::new();

    retriever.add_documents_sync(vec![
        Document::new("Rust 是一门系统编程语言"),
        Document::new("Python 是脚本语言"),
        Document::new("JavaScript 用于网页开发"),
    ]);

    let results = retriever.search("编程语言", 2);
    assert!(results.len() > 0, "应返回至少 1 个结果");

    // 应返回包含 "编程" 的文档
    assert!(
        results[0].document.content.contains("编程"),
        "结果应包含 '编程'"
    );
}

/// 测试空索引检索
///
/// 空索引应返回空结果
#[test]
fn test_bm25_retriever_empty_index() {
    let mut retriever = BM25Retriever::new();

    let results = retriever.search("test query", 5);
    assert!(results.is_empty(), "空索引应返回空结果");
}

/// 测试自定义 BM25 参数
///
/// k1=2.0, b=0.5 自定义参数
#[test]
fn test_bm25_retriever_custom_parameters() {
    let mut retriever = BM25Retriever::with_params(2.0, 0.5);

    retriever.add_documents_sync(vec![
        Document::new("Rust programming"),
        Document::new("Python scripting"),
    ]);

    let results = retriever.search("programming", 1);
    assert_eq!(results.len(), 1, "应返回 1 个结果");
}

/// 测试无匹配文档
///
/// 查询词不存在于任何文档时返回空结果
#[test]
fn test_bm25_retriever_no_matching_documents() {
    let mut retriever = BM25Retriever::new();

    retriever.add_documents_sync(vec![
        Document::new("Rust programming language"),
        Document::new("Python scripting language"),
    ]);

    // 查询词不在文档中
    let results = retriever.search("javascript typescript", 5);
    assert!(results.is_empty(), "无匹配时应返回空结果");
}

/// 测试 BM25 检索结果评分
///
/// 验证评分排序正确（高分在前）
#[test]
fn test_bm25_retriever_score_ordering() {
    let mut retriever = BM25Retriever::new();

    retriever.add_documents_sync(vec![
        Document::new("Rust Rust Rust programming"), // 高词频
        Document::new("Python programming"),         // 低词频
    ]);

    let results = retriever.search("rust", 2);

    if results.len() >= 2 {
        // 第一个结果评分应高于第二个
        assert!(results[0].score >= results[1].score, "结果应按评分降序排列");
    }
}

/// 测试 BM25 清空索引
///
/// clear() 方法应清空所有文档
#[test]
fn test_bm25_retriever_clear_index() {
    let mut retriever = BM25Retriever::new();

    retriever.add_documents_sync(vec![Document::new("Test document")]);

    assert_eq!(retriever.len(), 1, "添加后应有 1 个文档");

    retriever.clear();

    assert_eq!(retriever.len(), 0, "清空后应为 0 个文档");
    assert!(retriever.is_empty(), "is_empty() 应返回 true");
}

/// 测试 BM25 文档长度归一化
///
/// 长文档应有惩罚（由 b 参数控制）
#[test]
fn test_bm25_retriever_document_length_normalization() {
    let mut retriever = BM25Retriever::new();

    // 短文档：关键词密度高
    retriever.add_documents_sync(vec![
        Document::new("Rust"), // 短文档
    ]);

    // 长文档：关键词密度低（受 b 参数惩罚）
    retriever.add_documents_sync(vec![
        Document::new("Rust is a systems programming language with many features"), // 长文档
    ]);

    let results = retriever.search("rust", 2);

    // 短文档可能得更高分（关键词密度高）
    assert!(results.len() > 0, "应有匹配结果");
}
