// tests/hyde_reranking/reranking_test.rs
//! Reranking 测试用例

use langchainrust::{
    BM25Reranker, Document, KeywordReranker, Reranker, RerankingConfig, RerankingExecutor,
    SearchResult,
};
use std::collections::HashMap;

#[test]
fn test_reranking_config_default() {
    let config = RerankingConfig::default();

    assert_eq!(config.top_n, 5);
    assert!(config.min_score.is_none());
    assert!(config.preserve_original_score);
}

#[test]
fn test_reranking_config_custom_top_n() {
    let config = RerankingConfig::new().with_top_n(10);

    assert_eq!(config.top_n, 10);
}

#[test]
fn test_reranking_config_min_score() {
    let config = RerankingConfig::new().with_min_score(0.5);

    assert_eq!(config.min_score, Some(0.5));
}

#[test]
fn test_reranking_config_preserve_score() {
    let config = RerankingConfig::new().with_preserve_original_score(false);

    assert!(!config.preserve_original_score);
}

#[test]
fn test_keyword_reranker_basic() {
    let reranker = KeywordReranker::new();

    let query = "Rust programming";
    let documents = vec![
        Document::new("Rust is a programming language"),
        Document::new("Python is a scripting language"),
        Document::new("JavaScript for web development"),
    ];

    let scores = reranker.score(query, &documents).unwrap();

    assert_eq!(scores.len(), 3);
    assert!(scores[0] > 0.0);
}

#[test]
fn test_keyword_reranker_no_match() {
    let reranker = KeywordReranker::new();

    let query = "database";
    let documents = vec![
        Document::new("Rust programming"),
        Document::new("Python scripting"),
    ];

    let scores = reranker.score(query, &documents).unwrap();

    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0], 0.0);
    assert_eq!(scores[1], 0.0);
}

#[test]
fn test_keyword_reranker_empty_query() {
    let reranker = KeywordReranker::new();

    let documents = vec![Document::new("Some content")];

    let scores = reranker.score("", &documents).unwrap();

    assert_eq!(scores[0], 0.0);
}

#[test]
fn test_keyword_reranker_empty_documents() {
    let reranker = KeywordReranker::new();

    let scores = reranker.score("test", &[]).unwrap();

    assert!(scores.is_empty());
}

#[test]
fn test_keyword_reranker_weights() {
    let weights: HashMap<String, f32> =
        HashMap::from([("rust".to_string(), 2.0), ("programming".to_string(), 1.5)]);

    let reranker = KeywordReranker::new().with_keyword_weights(weights);

    let query = "Rust programming";
    let documents = vec![
        Document::new("Rust Rust Rust programming"),
        Document::new("Rust programming"),
    ];

    let scores = reranker.score(query, &documents).unwrap();

    assert!(scores[0] > scores[1]);
}

#[test]
fn test_bm25_reranker_basic() {
    let reranker = BM25Reranker::new();

    let query = "programming language";
    let documents = vec![
        Document::new("Rust is a programming language"),
        Document::new("Python is a programming language"),
        Document::new("Web development"),
    ];

    let scores = reranker.score(query, &documents).unwrap();

    assert_eq!(scores.len(), 3);
    assert!(scores[0] > scores[2]);
    assert!(scores[1] > scores[2]);
}

#[test]
fn test_bm25_reranker_custom_params() {
    let reranker = BM25Reranker::new().with_params(2.0, 0.5);

    let documents = vec![Document::new("Rust programming language")];

    let scores = reranker.score("Rust", &documents).unwrap();

    assert!(scores[0] > 0.0);
}

#[test]
fn test_bm25_reranker_empty() {
    let reranker = BM25Reranker::new();

    let scores = reranker.score("", &[Document::new("test")]).unwrap();
    assert_eq!(scores[0], 0.0);

    let scores = reranker.score("test", &[]).unwrap();
    assert!(scores.is_empty());
}

#[test]
fn test_reranking_executor_basic() {
    let reranker = Box::new(KeywordReranker::new());
    let executor = RerankingExecutor::new(reranker).with_top_n(2);

    let results = vec![
        SearchResult {
            document: Document::new("Rust Rust Rust"),
            score: 0.5,
        },
        SearchResult {
            document: Document::new("Python Python"),
            score: 0.4,
        },
        SearchResult {
            document: Document::new("JavaScript"),
            score: 0.3,
        },
    ];

    let reranked = executor.rerank("Rust", results).unwrap();

    assert_eq!(reranked.len(), 2);
    assert!(reranked[0].document.content.contains("Rust"));
}

#[test]
fn test_reranking_executor_min_score() {
    let reranker = Box::new(KeywordReranker::new());
    let executor = RerankingExecutor::new(reranker)
        .with_top_n(5)
        .with_min_score(2.0);

    let results = vec![
        SearchResult {
            document: Document::new("Rust Rust Rust Rust"),
            score: 0.0,
        },
        SearchResult {
            document: Document::new("No match"),
            score: 0.0,
        },
    ];

    let reranked = executor.rerank("Rust", results).unwrap();

    assert!(reranked.len() <= 1);
}

#[test]
fn test_reranking_executor_preserve_score() {
    let reranker = Box::new(KeywordReranker::new());
    let executor = RerankingExecutor::new(reranker)
        .with_preserve_original_score(false)
        .with_top_n(2);

    let results = vec![
        SearchResult {
            document: Document::new("Rust Rust"),
            score: 0.5,
        },
        SearchResult {
            document: Document::new("No match"),
            score: 0.4,
        },
    ];

    let reranked = executor.rerank("Rust", results).unwrap();

    assert_eq!(reranked.len(), 2);
}

#[test]
fn test_reranking_executor_empty_input() {
    let reranker = Box::new(KeywordReranker::new());
    let executor = RerankingExecutor::new(reranker);

    let reranked = executor.rerank("test", vec![]).unwrap();
    assert!(reranked.is_empty());
}

#[test]
fn test_rerank_documents() {
    let reranker = Box::new(KeywordReranker::new());
    let executor = RerankingExecutor::new(reranker).with_top_n(2);

    let documents = vec![
        Document::new("Rust Rust Rust"),
        Document::new("Python Python"),
        Document::new("JavaScript"),
    ];

    let results = executor.rerank_documents("Rust", documents).unwrap();

    assert_eq!(results.len(), 2);
    assert!(results[0].document.content.contains("Rust"));
}

#[test]
fn test_reranker_trait_impl() {
    let reranker: Box<dyn Reranker> = Box::new(KeywordReranker::new());

    let documents = vec![Document::new("test")];
    let scores = reranker.score("test", &documents).unwrap();

    assert_eq!(scores.len(), 1);
}

#[test]
fn test_bm25_vs_keyword_comparison() {
    let keyword_reranker = KeywordReranker::new();
    let bm25_reranker = BM25Reranker::new();

    let query = "programming language";
    let documents = vec![
        Document::new("Rust programming language"),
        Document::new("Python programming"),
    ];

    let keyword_scores = keyword_reranker.score(query, &documents).unwrap();
    let bm25_scores = bm25_reranker.score(query, &documents).unwrap();

    assert_eq!(keyword_scores.len(), 2);
    assert_eq!(bm25_scores.len(), 2);
}
