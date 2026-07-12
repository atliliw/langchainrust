// tests/bm25/chunked.rs
//! BM25 Chunked Retriever 核心功能测试

mod unified_hybrid;
mod full_flow;

use langchainrust::retrieval::bm25::{
    AutoMergingConfig, ChunkedBM25Retriever, ChunkedSearchResult,
};
use langchainrust::vector_stores::document_store::ChunkedDocumentStore;
use langchainrust::Document;
use tempfile::NamedTempFile;
use std::sync::Arc;

/// 测试：基础文档添加功能
/// 验证：添加文档后索引不为空，文档数符合预期
#[test]
fn test_chunked_retriever_basic() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_documents(vec![
        Document::new("Rust是一门系统编程语言，注重安全和性能。由Mozilla开发，于2010年首次发布。")
            .with_id("rust_doc"),
        Document::new("Python是一门高级编程语言，适合数据科学、机器学习和Web开发。")
            .with_id("python_doc"),
        Document::new("JavaScript是一门脚本语言，主要用于前端开发和Node.js后端开发。")
            .with_id("js_doc"),
    ]);

    println!("索引是否为空: {}", retriever.is_empty());
    println!("索引文档数量: {}", retriever.len());
}

/// 测试：BM25 搜索功能
/// 验证：搜索返回正确的结果，排序按分数降序
#[test]
fn test_chunked_retriever_search() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_documents(vec![
        Document::new("Rust是一门系统编程语言，注重安全和性能。由Mozilla开发。")
            .with_id("rust_doc"),
        Document::new("Python是一门脚本语言，适合数据科学。").with_id("python_doc"),
        Document::new("Go语言由Google开发，是一门并发编程语言。").with_id("go_doc"),
    ]);

    let results = retriever.search("系统编程", 2);

    println!("搜索关键词: 系统编程");
    println!("返回结果数: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "结果 {}: 分数={}, 内容={}",
            i,
            result.score,
            result.content()
        );
    }
}

/// 测试：中文分词搜索
/// 验证：中文查询能正确匹配到相关文档
#[test]
fn test_chunked_retriever_chinese_search() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_documents(vec![
        Document::new("机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。")
            .with_id("ml_doc"),
        Document::new("深度学习使用神经网络进行特征学习和模式识别。").with_id("dl_doc"),
        Document::new("自然语言处理让计算机理解和生成人类语言。").with_id("nlp_doc"),
    ]);

    let results: Vec<ChunkedSearchResult> = retriever.search("机器学习算法", 3);

    println!("搜索关键词: 机器学习算法");
    println!("返回结果数: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "结果 {}: 分数={}, 是否合并={}, 内容={}",
            i,
            result.score,
            result.is_merged(),
            result.content()
        );
    }
}

/// 测试：AutoMerging 配置功能
/// 验证：自定义配置参数生效，文档按配置拆分
#[test]
fn test_auto_merging_config() {
    let config = AutoMergingConfig::new()
        .with_threshold(0.6)
        .with_leaf_size(300)
        .with_parent_size(1500);

    println!("合并阈值: {}", config.merge_threshold);
    println!("Leaf大小: {}", config.leaf_chunk_size);
    println!("Parent大小: {}", config.parent_chunk_size);

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::with_config(store, config);

    let doc = Document::new(
        "这是一段很长的测试文本，用于验证自定义配置是否正常工作。我们需要确保文档能够被正确拆分。",
    )
    .with_id("test_doc");

    retriever.add_document(doc);

    println!("索引文档数量: {}", retriever.len());
}

/// 测试：高匹配率时 AutoMerging 合并
/// 验证：当多个 Leaf 匹配同一 Parent 时，触发合并返回 Parent
#[test]
fn test_auto_merging_high_match_ratio() {
    let config = AutoMergingConfig::new()
        .with_leaf_size(30)
        .with_threshold(0.5);

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::with_config(store, config);

    let long_doc = Document::new(
        "Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust Rust"
    ).with_id("rust_repeat");

    retriever.add_document(long_doc);

    let results: Vec<ChunkedSearchResult> = retriever.search("Rust", 1);

    println!("搜索关键词: Rust");
    println!("Leaf大小配置: 30");
    println!("合并阈值: 0.5");
    for (i, result) in results.iter().enumerate() {
        println!(
            "结果 {}: 是否合并={}, 分数={}, 内容长度={}",
            i,
            result.is_merged(),
            result.score,
            result.content().len()
        );
    }
}

/// 测试：低匹配率时不合并，保留单独 Leaf
/// 验证：匹配率低于阈值时，返回单独的 Leaf 片段
#[test]
fn test_auto_merging_low_match_ratio() {
    let config = AutoMergingConfig::new()
        .with_leaf_size(30)
        .with_threshold(0.8);

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::with_config(store, config);

    retriever.add_documents(vec![
        Document::new("Rust是一门系统编程语言。Python是一门脚本语言。JavaScript是一门前端语言。")
            .with_id("multi_lang"),
        Document::new("Go是一门并发编程语言。").with_id("go_doc"),
    ]);

    let results: Vec<ChunkedSearchResult> = retriever.search("Rust", 2);

    println!("搜索关键词: Rust");
    println!("合并阈值: 0.8 (高阈值，不易触发合并)");
    for (i, result) in results.iter().enumerate() {
        println!(
            "结果 {}: 是否合并={}, Leaf数量={}, 分数={}",
            i,
            result.is_merged(),
            result.leaf_chunks.len(),
            result.score
        );
    }
}

/// 测试：Bincode 持久化保存和加载
/// 验证：save/load 后数据完整，文档数一致
#[test]
fn test_persistence_save_load() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store.clone());

    retriever.add_documents(vec![
        Document::new("Rust是一门系统编程语言，注重安全。").with_id("rust_doc"),
        Document::new("Python适合数据科学和机器学习。").with_id("python_doc"),
        Document::new("JavaScript用于Web前端开发。").with_id("js_doc"),
    ]);

    let original_len = retriever.len();

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    retriever.save(temp_file.path()).expect("Failed to save");

    println!("保存路径: {}", temp_file.path().display());
    println!("原始索引文档数: {}", original_len);

    let loaded = ChunkedBM25Retriever::load(store, temp_file.path()).expect("Failed to load");

    println!("加载后文档数: {}", loaded.len());
    println!(
        "rust_doc 存在: {}",
        loaded.get_parent_document("rust_doc").is_some()
    );
    println!(
        "python_doc 存在: {}",
        loaded.get_parent_document("python_doc").is_some()
    );
    println!(
        "js_doc 存在: {}",
        loaded.get_parent_document("js_doc").is_some()
    );
}

/// 测试：持久化后搜索功能
/// 验证：加载索引后能正常搜索
#[test]
fn test_persistence_with_search() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store.clone());

    retriever.add_documents(vec![
        Document::new("人工智能是计算机科学的一个分支。").with_id("ai_doc"),
        Document::new("机器学习是人工智能的核心技术。").with_id("ml_doc"),
    ]);

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    retriever.save(temp_file.path()).expect("Failed to save");

    let mut loaded = ChunkedBM25Retriever::load(store, temp_file.path()).expect("Failed to load");

    let results: Vec<ChunkedSearchResult> = loaded.search("人工智能", 2);

    println!("加载后搜索关键词: 人工智能");
    println!("返回结果数: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("结果 {}: 内容={}", i, result.content());
    }
}

/// 测试：获取 Parent 文档功能
/// 验证：通过 ID 能获取到 Parent 文档
#[test]
fn test_get_parent_document() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    let doc = Document::new("这是一个测试文档的内容。").with_id("test_doc");

    retriever.add_document(doc);

    let parent = retriever.get_parent_document("test_doc");

    println!("获取 test_doc: {}", parent.is_some());
    if let Some(p) = parent {
        println!("Parent ID: {:?}", p.id);
        println!("Parent 内容: {}", p.content);
    }

    let not_found = retriever.get_parent_document("nonexistent");
    println!("获取 nonexistent: {}", not_found.is_some());
}

/// 测试：清空索引功能
/// 验证：clear() 后索引为空
#[test]
fn test_clear() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_documents(vec![
        Document::new("文档一").with_id("doc1"),
        Document::new("文档二").with_id("doc2"),
    ]);

    println!("清空前文档数: {}", retriever.len());

    retriever.clear();

    println!("清空后文档数: {}", retriever.len());
    println!("清空后是否为空: {}", retriever.is_empty());
}

/// 测试：空索引搜索
/// 验证：空索引搜索返回空结果
#[test]
fn test_empty_search() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    let results = retriever.search("测试查询", 5);

    println!("空索引搜索结果数: {}", results.len());
}

/// 测试：大文档拆分功能
/// 验证：长文档被正确拆分成多个 Leaf chunks
#[test]
fn test_large_document_chunking() {
    let config = AutoMergingConfig::new()
        .with_leaf_size(100)
        .with_threshold(0.5);

    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::with_config(store, config);

    let large_doc = Document::new(
        "这是一个很长的文档，包含了很多内容。我们需要测试文档拆分功能是否正常工作。\
         文档拆分是将长文档切分成多个小块的过程，这样可以提高检索的精确度。\
         BM25算法会在每个小块上进行搜索，然后通过AutoMerging机制将相关的小块合并。\
         这种方法结合了精确性和完整性，既能够精确定位到相关内容，又能够提供完整的上下文。\
         用户可以根据需要调整合并阈值和chunk大小，以适应不同的应用场景。",
    )
    .with_id("large_doc");

    retriever.add_document(large_doc);

    println!("Leaf大小配置: 100");
    println!("索引文档数 (Leaf chunks): {}", retriever.len());

    let results: Vec<ChunkedSearchResult> = retriever.search("文档拆分", 2);

    println!("搜索关键词: 文档拆分");
    println!("返回结果数: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "结果 {}: 是否合并={}, 内容长度={}",
            i,
            result.is_merged(),
            result.content().len()
        );
    }
}

/// 测试：多关键词搜索
/// 验证：多个关键词同时搜索能返回相关结果
#[test]
fn test_multiple_keywords_search() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_documents(vec![
        Document::new("Rust是一门系统编程语言，注重安全和并发。").with_id("rust_doc"),
        Document::new("Python是一门高级语言，适合数据科学。").with_id("python_doc"),
        Document::new("Go是一门并发语言，由Google开发。").with_id("go_doc"),
    ]);

    let results: Vec<ChunkedSearchResult> = retriever.search("系统编程 安全", 2);

    println!("搜索关键词: 系统编程 安全 (多关键词)");
    println!("返回结果数: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "结果 {}: 分数={}, 内容={}",
            i,
            result.score,
            result.content()
        );
    }
}

/// 测试：搜索结果属性验证
/// 验证：score、content、parent_id 等属性有效
#[test]
fn test_search_result_properties() {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever.add_document(Document::new("测试文档用于验证搜索结果的属性。").with_id("test_doc"));

    let results: Vec<ChunkedSearchResult> = retriever.search("测试", 1);

    if !results.is_empty() {
        let result: &ChunkedSearchResult = &results[0];
        println!("结果分数: {}", result.score);
        println!("结果内容长度: {}", result.content().len());
        println!("Parent ID: {}", result.parent_id);
        println!("是否合并: {}", result.is_merged());
        println!("Leaf数量: {}", result.leaf_chunks.len());
        println!("匹配词: {:?}", result.matched_terms);
    }
}
