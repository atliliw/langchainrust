//! 对话检索 + 文档处理 Chain 单元测试
//!
//! 测试以下 Chain 的核心功能：
//! - ConversationRetrievalChain: 带记忆的检索增强对话
//! - StuffDocumentsChain: 文档填充处理
//! - RefineDocumentsChain: 迭代优化处理
//! - MapReduceDocumentsChain: 并行映射 + 合并处理
//! - MapRerankDocumentsChain: 并行映射 + 评分排序

use langchainrust::language_models::{OpenAIChat, OpenAIConfig};
use langchainrust::{
    BaseChain, ChainError, ConversationBufferMemory, ConversationRetrievalChain, Document,
    MapReduceDocumentsChain, MapRerankDocumentsChain, RefineDocumentsChain, StuffDocumentsChain,
};
use std::collections::HashMap;
use std::sync::Arc;

/// 创建测试用的配置
fn create_test_config() -> OpenAIConfig {
    OpenAIConfig {
        api_key: "sk-test".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        organization: None,
        frequency_penalty: None,
        max_tokens: None,
        presence_penalty: None,
        temperature: None,
        top_p: None,
        tools: None,
        tool_choice: None,
    }
}

/// 创建测试用的 LLM
fn create_test_llm() -> OpenAIChat {
    OpenAIChat::new(create_test_config())
}

// ============================================================
// ConversationRetrievalChain 测试
// ============================================================

/// 测试 ConversationRetrievalChain 创建
///
/// 验证：基本配置正确初始化。
#[test]
fn test_conversation_retrieval_new() {
    let llm = create_test_llm();
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));
    let retriever = Arc::new(langchainrust::SimilarityRetriever::new(store, embeddings));
    let memory = ConversationBufferMemory::new();

    let chain = ConversationRetrievalChain::new(llm, retriever, memory);

    assert_eq!(chain.input_keys(), vec!["query"]);
    assert_eq!(chain.output_keys(), vec!["result"]);
    assert_eq!(chain.name(), "conversation_retrieval");
}

/// 测试 ConversationRetrievalChain 配置方法
///
/// 验证：所有配置方法正确生效。
#[test]
fn test_conversation_retrieval_with_options() {
    let llm = create_test_llm();
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));
    let retriever = Arc::new(langchainrust::SimilarityRetriever::new(store, embeddings));
    let memory = ConversationBufferMemory::new();

    let chain = ConversationRetrievalChain::new(llm, retriever, memory)
        .with_system_prompt("你是一个 Rust 专家")
        .with_k(5)
        .with_input_key("question")
        .with_output_key("answer")
        .with_return_source_documents(true)
        .with_verbose(true);

    assert_eq!(chain.input_keys(), vec!["question"]);
    assert_eq!(chain.output_keys(), vec!["answer", "source_documents"]);
}

/// 测试 ConversationRetrievalChain 缺少输入
///
/// 验证：输入缺失时返回 MissingInput 错误。
#[tokio::test]
async fn test_conversation_retrieval_missing_input() {
    let llm = create_test_llm();
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));
    let retriever = Arc::new(langchainrust::SimilarityRetriever::new(store, embeddings));
    let memory = ConversationBufferMemory::new();

    let chain = ConversationRetrievalChain::new(llm, retriever, memory);

    let inputs = HashMap::new();
    let result = chain.invoke(inputs).await;

    assert!(result.is_err());
    match result {
        Err(ChainError::MissingInput(_)) => {} // 期望缺少输入错误
        _ => panic!("应当返回 MissingInput 错误"),
    }
}

// ============================================================
// StuffDocumentsChain 测试
// ============================================================

/// 测试 StuffDocumentsChain 创建
///
/// 验证：基本配置正确初始化。
#[test]
fn test_stuff_documents_new() {
    let llm = create_test_llm();
    let chain = StuffDocumentsChain::new(llm);

    assert_eq!(chain.input_keys(), vec!["input", "documents"]);
    assert_eq!(chain.output_keys(), vec!["output"]);
    assert_eq!(chain.name(), "stuff_documents");
}

/// 测试 StuffDocumentsChain 配置方法
#[test]
fn test_stuff_documents_with_options() {
    let llm = create_test_llm();
    let chain = StuffDocumentsChain::new(llm)
        .with_input_key("question")
        .with_output_key("answer")
        .with_max_doc_length(500)
        .with_verbose(true);

    assert_eq!(chain.input_keys(), vec!["question", "documents"]);
    assert_eq!(chain.output_keys(), vec!["answer"]);
}

/// 测试 StuffDocumentsChain 文档格式化
///
/// 验证：多个文档被正确格式化为包含编号的文本。
#[test]
fn test_stuff_documents_format_documents() {
    let llm = create_test_llm();
    let chain = StuffDocumentsChain::new(llm);

    let docs = vec![Document::new("文档一的内容"), Document::new("文档二的内容")];
    let formatted = chain.format_documents(&docs);

    assert!(formatted.contains("Document 1:"));
    assert!(formatted.contains("文档一的内容"));
    assert!(formatted.contains("Document 2:"));
    assert!(formatted.contains("文档二的内容"));
}

/// 测试 StuffDocumentsChain 文档截断
///
/// 验证：超过 max_doc_length 的文档被正确截断。
#[test]
fn test_stuff_documents_truncation() {
    let llm = create_test_llm();
    let chain = StuffDocumentsChain::new(llm).with_max_doc_length(10);

    let docs = vec![Document::new("这是一段超过十个字符的文档内容")];
    let formatted = chain.format_documents(&docs);

    assert!(formatted.contains("[document truncated]"));
    assert!(formatted.len() < 100);
}

/// 测试 StuffDocumentsChain prompt 构建
///
/// 验证：模板中的变量被正确替换。
#[test]
fn test_stuff_documents_build_prompt() {
    let llm = create_test_llm();
    let chain = StuffDocumentsChain::new(llm);

    let prompt = chain.build_prompt("这是上下文", "这是问题");

    assert!(prompt.contains("这是上下文"));
    assert!(prompt.contains("这是问题"));
    assert!(!prompt.contains("{context}")); // 变量已被替换
    assert!(!prompt.contains("{input}"));
}

/// 测试 StuffDocumentsChain 自定义模板
///
/// 验证：自定义模板正确替换自定义变量名。
#[test]
fn test_stuff_documents_custom_template() {
    let llm = create_test_llm();
    let chain = StuffDocumentsChain::new(llm)
        .with_prompt_template("背景：{context}\n问题：{input}")
        .with_document_variable("context");

    let prompt = chain.build_prompt("测试背景", "测试问题");
    assert!(prompt.contains("背景：测试背景"));
    assert!(prompt.contains("问题：测试问题"));
}

/// 测试 StuffDocumentsChain 空文档
///
/// 验证：空文档列表格式化为空字符串。
#[test]
fn test_stuff_documents_empty_docs() {
    let llm = create_test_llm();
    let chain = StuffDocumentsChain::new(llm);

    let docs = vec![];
    let formatted = chain.format_documents(&docs);
    assert!(formatted.is_empty());
}

// ============================================================
// RefineDocumentsChain 测试
// ============================================================

/// 测试 RefineDocumentsChain 创建
///
/// 验证：基本配置正确初始化。
#[test]
fn test_refine_documents_new() {
    let llm = create_test_llm();
    let chain = RefineDocumentsChain::new(llm);

    assert_eq!(chain.input_keys(), vec!["input", "documents"]);
    assert_eq!(chain.output_keys(), vec!["output"]);
    assert_eq!(chain.name(), "refine_documents");
}

/// 测试 RefineDocumentsChain prompt 构建
///
/// 验证：初始 prompt 和优化 prompt 正确替换变量。
#[test]
fn test_refine_documents_build_prompts() {
    let llm = create_test_llm();
    let chain = RefineDocumentsChain::new(llm)
        .with_initial_prompt("初始：{context} - {input}")
        .with_refine_prompt("优化：{context} - {input} - 已有：{existing_answer}")
        .with_document_variable("context");

    // 测试初始 prompt
    let initial = chain.build_initial_prompt("文档内容", "我的问题");
    assert!(initial.contains("初始：文档内容 - 我的问题"));

    // 测试优化 prompt
    let refine = chain.build_refine_prompt("新文档", "问题", "已有答案");
    assert!(refine.contains("优化：新文档 - 问题 - 已有：已有答案"));
}

/// 测试 RefineDocumentsChain 空文档
///
/// 验证：空文档列表返回错误。
#[tokio::test]
async fn test_refine_documents_empty_docs() {
    let llm = create_test_llm();
    let chain = RefineDocumentsChain::new(llm);

    let result = chain.invoke_with_documents(vec![], "测试").await;
    assert!(result.is_err());
}

// ============================================================
// MapReduceDocumentsChain 测试
// ============================================================

/// 测试 MapReduceDocumentsChain 创建
///
/// 验证：基本配置正确初始化。
#[test]
fn test_map_reduce_new() {
    let llm = create_test_llm();
    let chain = MapReduceDocumentsChain::new(llm);

    assert_eq!(chain.input_keys(), vec!["input", "documents"]);
    assert_eq!(chain.output_keys(), vec!["output"]);
    assert_eq!(chain.name(), "map_reduce_documents");
}

/// 测试 MapReduceDocumentsChain prompt 构建
///
/// 验证：Map 和 Reduce 的 prompt 正确替换变量。
#[test]
fn test_map_reduce_build_prompts() {
    let llm = create_test_llm();
    let chain = MapReduceDocumentsChain::new(llm)
        .with_map_prompt("处理：{context} - {input}")
        .with_reduce_prompt("合并：{summaries}\n问题：{input}")
        .with_document_variable("context");

    // 测试 Map prompt
    let map_prompt = chain.build_map_prompt("文档内容", "问题");
    assert!(map_prompt.contains("处理：文档内容 - 问题"));

    // 测试 Reduce prompt
    let reduce_prompt = chain.build_reduce_prompt(&["答案1".into(), "答案2".into()], "原始问题");
    assert!(reduce_prompt.contains("合并："));
    assert!(reduce_prompt.contains("答案1"));
    assert!(reduce_prompt.contains("答案2"));
    assert!(reduce_prompt.contains("原始问题"));
}

/// 测试 MapReduceDocumentsChain 空文档
///
/// 验证：空文档列表返回错误。
#[tokio::test]
async fn test_map_reduce_empty_docs() {
    let llm = create_test_llm();
    let chain = MapReduceDocumentsChain::new(llm);

    let result = chain.invoke_with_documents(vec![], "测试").await;
    assert!(result.is_err());
}

// ============================================================
// BaseChain 接口一致性测试
// ============================================================

/// 测试所有 Chain 遵循 BaseChain 接口
///
/// 验证：所有 Chain 实现了 input_keys / output_keys / name 方法。
#[test]
fn test_all_chains_implement_base_chain() {
    let store = Arc::new(langchainrust::InMemoryVectorStore::new());
    let embeddings = Arc::new(langchainrust::MockEmbeddings::new(64));

    // 验证每个 Chain 的 input_keys 非空
    let stuff = StuffDocumentsChain::new(create_test_llm());
    assert!(!stuff.input_keys().is_empty());
    assert!(!stuff.output_keys().is_empty());

    let refine = RefineDocumentsChain::new(create_test_llm());
    assert!(!refine.input_keys().is_empty());
    assert!(!refine.output_keys().is_empty());

    let map_reduce = MapReduceDocumentsChain::new(create_test_llm());
    assert!(!map_reduce.input_keys().is_empty());
    assert!(!map_reduce.output_keys().is_empty());

    let map_rerank = MapRerankDocumentsChain::new(create_test_llm());
    assert!(!map_rerank.input_keys().is_empty());
    assert!(!map_rerank.output_keys().is_empty());

    let retriever = Arc::new(langchainrust::SimilarityRetriever::new(store, embeddings));
    let memory = ConversationBufferMemory::new();
    let conv_retrieval = ConversationRetrievalChain::new(create_test_llm(), retriever, memory);
    assert!(!conv_retrieval.input_keys().is_empty());
    assert!(!conv_retrieval.output_keys().is_empty());
}

// ============================================================
// MapRerankDocumentsChain 测试
// ============================================================

/// 测试 MapRerankDocumentsChain 创建
///
/// 验证：基本配置正确初始化。
#[test]
fn test_map_rerank_new() {
    let llm = create_test_llm();
    let chain = MapRerankDocumentsChain::new(llm);

    assert_eq!(chain.input_keys(), vec!["input", "documents"]);
    assert_eq!(chain.output_keys(), vec!["output"]);
    assert_eq!(chain.name(), "map_rerank_documents");
}

/// 测试 MapRerankDocumentsChain 配置方法
///
/// 验证：所有配置方法正确生效。
#[test]
fn test_map_rerank_with_options() {
    let llm = create_test_llm();
    let chain = MapRerankDocumentsChain::new(llm)
        .with_top_k(3)
        .with_input_key("question")
        .with_output_key("ranked_results")
        .with_verbose(true);

    assert_eq!(chain.input_keys(), vec!["question", "documents"]);
    assert_eq!(chain.output_keys(), vec!["ranked_results"]);
}

/// 测试 MapRerankDocumentsChain 空文档
///
/// 验证：空文档列表返回错误。
#[tokio::test]
async fn test_map_rerank_empty_docs() {
    let llm = create_test_llm();
    let chain = MapRerankDocumentsChain::new(llm);

    let result = chain.invoke_with_documents(vec![], "测试").await;
    assert!(result.is_err());
}

/// 测试 MapRerankDocumentsChain prompt 构建
///
/// 验证：模板中的变量被正确替换。
#[test]
fn test_map_rerank_build_prompt() {
    let llm = create_test_llm();
    let chain = MapRerankDocumentsChain::new(llm)
        .with_map_prompt("评分：{context}\n问题：{input}")
        .with_document_variable("context");

    // 通过 with_map_prompt 测试替换
    let prompt = chain.build_map_prompt("文档内容", "我的问题");
    assert!(prompt.contains("评分：文档内容"));
    assert!(prompt.contains("问题：我的问题"));
}

/// 测试 MapRerankDocumentsChain 评分提取
///
/// 验证：能从 LLM 输出中正确提取评分和答案。
#[test]
fn test_map_rerank_extract_score() {
    use langchainrust::chains::document_chains::extract_score;
    // 中文格式
    let (score, answer) = extract_score("相关性评分：85\n答案：Rust 是一门系统编程语言");
    assert_eq!(score, 85);
    assert!(answer.contains("Rust"));

    // 英文格式
    let (score2, _answer2) = extract_score("Score: 92\nAnswer: It's a programming language");
    assert_eq!(score2, 92);

    // 无评分格式（默认 50）
    let (score3, _) = extract_score("这是一段普通文本");
    assert_eq!(score3, 50);

    // 评分超过 100 时取 100
    let (score4, _) = extract_score("相关性评分：150");
    assert_eq!(score4, 100);
}
