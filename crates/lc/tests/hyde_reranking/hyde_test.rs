// tests/hyde_reranking/hyde_test.rs
//! HyDE Retriever 测试用例

use langchainrust::{
    Document, HyDEConfig, HyDEError, InMemoryVectorStore, MockEmbeddings, RetrieverTrait,
    SimilarityRetriever,
};
use std::sync::Arc;

fn create_test_retriever() -> Arc<dyn RetrieverTrait> {
    let store = Arc::new(InMemoryVectorStore::new());
    let embeddings = Arc::new(MockEmbeddings::new(128));
    Arc::new(SimilarityRetriever::new(store, embeddings))
}

#[test]
fn test_hyde_config_default() {
    let config = HyDEConfig::default();

    assert_eq!(config.k, 5);
    assert!(config.include_original_query);
    assert!(config.prompt_template.contains("{question}"));
}

#[test]
fn test_hyde_config_custom_k() {
    let config = HyDEConfig::new().with_k(10);

    assert_eq!(config.k, 10);
}

#[test]
fn test_hyde_config_exclude_original_query() {
    let config = HyDEConfig::new().with_include_original_query(false);

    assert!(!config.include_original_query);
}

#[test]
fn test_hyde_config_custom_prompt() {
    let custom_prompt = "Please answer: {question}".to_string();
    let config = HyDEConfig::new().with_prompt(custom_prompt);

    assert!(config.prompt_template.contains("{question}"));
}

#[test]
fn test_hyde_config_chain() {
    let config = HyDEConfig::new()
        .with_k(8)
        .with_include_original_query(false);

    assert_eq!(config.k, 8);
    assert!(!config.include_original_query);
}

#[tokio::test]
async fn test_hyde_error_display() {
    let error = HyDEError::LLMError("test error".to_string());
    assert!(error.to_string().contains("LLM"));

    let error = HyDEError::EmbeddingError("embedding error".to_string());
    assert!(error.to_string().contains("Embedding"));

    let error = HyDEError::RetrieverError("retriever error".to_string());
    assert!(error.to_string().contains("检索"));
}

#[tokio::test]
async fn test_retriever_basic() {
    let retriever = create_test_retriever();

    let docs = vec![
        Document::new("Rust 是系统编程语言"),
        Document::new("Python 是脚本语言"),
    ];

    retriever.add_documents(docs).await.unwrap();

    let results = retriever.retrieve("编程", 2).await.unwrap();
    assert_eq!(results.len(), 2);
}
