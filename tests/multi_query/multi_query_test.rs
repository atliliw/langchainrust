// tests/multi_query/multi_query_test.rs
//! MultiQueryRetriever 测试用例

use langchainrust::{MultiQueryRetriever, MultiQueryConfig, StaticQueryGenerator, Document, RetrieverTrait, RetrieverError, SimilarityRetriever, InMemoryVectorStore, MockEmbeddings, SearchResult};
use std::sync::Arc;
use std::collections::HashMap;

fn get_test_config() -> langchainrust::OpenAIConfig {
    langchainrust::OpenAIConfig::default()
}

fn create_test_retriever() -> Arc<dyn RetrieverTrait> {
    let store = Arc::new(InMemoryVectorStore::new());
    let embeddings = Arc::new(MockEmbeddings::new(128));
    Arc::new(SimilarityRetriever::new(store, embeddings))
}

#[tokio::test]
async fn test_static_query_generator_basic() {
    let generator = StaticQueryGenerator::new();
    let queries = generator.generate("测试查询");
    
    assert!(queries.is_empty() || !queries.contains(&"测试查询".to_string()));
}

#[tokio::test]
async fn test_static_query_generator_synonym() {
    let synonyms: HashMap<String, Vec<String>> = HashMap::from([
        ("数据库".to_string(), vec!["DB".to_string(), "database".to_string()]),
    ]);
    
    let generator = StaticQueryGenerator::new()
        .with_synonym_expansion(synonyms);
    
    let queries = generator.generate("数据库连接超时");
    
    assert!(queries.contains(&"DB连接超时".to_string()));
    assert!(queries.contains(&"database连接超时".to_string()));
}

#[tokio::test]
async fn test_static_query_generator_prefix() {
    let generator = StaticQueryGenerator::new()
        .with_prefix_expansion(vec!["如何".to_string(), "怎么".to_string()]);
    
    let queries = generator.generate("处理数据库错误");
    
    assert!(queries.contains(&"如何 处理数据库错误".to_string()));
    assert!(queries.contains(&"怎么 处理数据库错误".to_string()));
}

#[tokio::test]
async fn test_static_query_generator_combined() {
    let synonyms: HashMap<String, Vec<String>> = HashMap::from([
        ("错误".to_string(), vec!["exception".to_string(), "异常".to_string()]),
    ]);
    
    let generator = StaticQueryGenerator::new()
        .with_synonym_expansion(synonyms)
        .with_prefix_expansion(vec!["如何".to_string()]);
    
    let queries = generator.generate("处理错误");
    
    assert!(queries.len() >= 2);
}

#[test]
fn test_multi_query_config_default() {
    let config = MultiQueryConfig::default();
    
    assert_eq!(config.num_queries, 3);
    assert_eq!(config.k_per_query, 5);
    assert_eq!(config.final_k, 10);
}

#[test]
fn test_multi_query_config_custom() {
    let config = MultiQueryConfig::new()
        .with_num_queries(5)
        .with_k_per_query(10)
        .with_final_k(20);
    
    assert_eq!(config.num_queries, 5);
    assert_eq!(config.k_per_query, 10);
    assert_eq!(config.final_k, 20);
}

#[test]
fn test_multi_query_config_chain() {
    let config = MultiQueryConfig::new()
        .with_num_queries(4)
        .with_k_per_query(8);
    
    assert_eq!(config.num_queries, 4);
    assert_eq!(config.k_per_query, 8);
    assert_eq!(config.final_k, 10);
}

#[tokio::test]
async fn test_retriever_basic() {
    let retriever = create_test_retriever();
    
    let docs = vec![
        Document::new("Rust 是系统编程语言"),
        Document::new("Python 是脚本语言"),
        Document::new("JavaScript 用于网页开发"),
    ];
    
    retriever.add_documents(docs).await.unwrap();
    
    let results = retriever.retrieve("编程语言", 2).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_retriever_with_scores() {
    let retriever = create_test_retriever();
    
    let docs = vec![
        Document::new("文档 A"),
        Document::new("文档 B"),
    ];
    
    retriever.add_documents(docs).await.unwrap();
    
    let results = retriever.retrieve_with_scores("测试", 2).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_static_generator_no_duplicates() {
    let generator = StaticQueryGenerator::new()
        .with_prefix_expansion(vec!["test".to_string()]);
    
    let queries = generator.generate("query");
    
    for q in &queries {
        assert_ne!(q, "query");
    }
}

#[test]
fn test_config_prompt_custom() {
    let custom_prompt = "Generate queries: {question}".to_string();
    let config = MultiQueryConfig::new()
        .with_prompt(custom_prompt.clone());
    
    assert!(config.prompt_template.contains("{question}"));
}