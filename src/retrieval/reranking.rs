// src/retrieval/reranking.rs
//! Reranking（重排序）实现
//!
//! 使用评分函数对检索结果重新排序，提升检索精确度。

use crate::vector_stores::{Document, SearchResult};
use std::collections::HashMap;

/// Reranking 错误类型
#[derive(Debug)]
pub enum RerankingError {
    ScoringError(String),
    InvalidInput(String),
}

impl std::fmt::Display for RerankingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RerankingError::ScoringError(msg) => write!(f, "评分错误: {}", msg),
            RerankingError::InvalidInput(msg) => write!(f, "输入无效: {}", msg),
        }
    }
}

impl std::error::Error for RerankingError {}

/// Reranking 配置
pub struct RerankingConfig {
    /// 最终返回的文档数量
    pub top_n: usize,

    /// 最小分数阈值（可选）
    pub min_score: Option<f32>,

    /// 是否保留原始分数
    pub preserve_original_score: bool,
}

impl Default for RerankingConfig {
    fn default() -> Self {
        Self {
            top_n: 5,
            min_score: None,
            preserve_original_score: true,
        }
    }
}

impl RerankingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    pub fn with_preserve_original_score(mut self, preserve: bool) -> Self {
        self.preserve_original_score = preserve;
        self
    }
}

/// Reranking 评分器 trait
pub trait Reranker: Send + Sync {
    fn score(&self, query: &str, documents: &[Document]) -> Result<Vec<f32>, RerankingError>;
}

/// 基于关键词匹配的简单 Reranker
pub struct KeywordReranker {
    /// 关键词权重（可选）
    keyword_weights: HashMap<String, f32>,
}

impl KeywordReranker {
    pub fn new() -> Self {
        Self {
            keyword_weights: HashMap::new(),
        }
    }

    pub fn with_keyword_weights(mut self, weights: HashMap<String, f32>) -> Self {
        self.keyword_weights = weights;
        self
    }

    fn extract_keywords(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .filter(|w| w.len() > 1)
            .map(|w| w.to_lowercase())
            .collect()
    }

    fn count_keyword_matches(&self, keywords: &[String], document: &Document) -> f32 {
        let doc_lower = document.content.to_lowercase();
        let mut score = 0.0;

        for keyword in keywords {
            let count = doc_lower.matches(keyword).count() as f32;
            let weight = self.keyword_weights.get(keyword).unwrap_or(&1.0);
            score += count * weight;
        }

        score
    }
}

impl Default for KeywordReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for KeywordReranker {
    fn score(&self, query: &str, documents: &[Document]) -> Result<Vec<f32>, RerankingError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let keywords = self.extract_keywords(query);

        if keywords.is_empty() {
            return Ok(documents.iter().map(|_| 0.0).collect());
        }

        let scores: Vec<f32> = documents
            .iter()
            .map(|doc| self.count_keyword_matches(&keywords, doc))
            .collect();

        Ok(scores)
    }
}

/// Reranker 执行器
pub struct RerankingExecutor {
    reranker: Box<dyn Reranker>,
    config: RerankingConfig,
}

impl RerankingExecutor {
    pub fn new(reranker: Box<dyn Reranker>) -> Self {
        Self {
            reranker,
            config: RerankingConfig::default(),
        }
    }

    pub fn with_config(mut self, config: RerankingConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_top_n(mut self, n: usize) -> Self {
        self.config.top_n = n;
        self
    }

    pub fn with_min_score(mut self, score: f32) -> Self {
        self.config.min_score = Some(score);
        self
    }

    pub fn with_preserve_original_score(mut self, preserve: bool) -> Self {
        self.config.preserve_original_score = preserve;
        self
    }

    pub fn rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, RerankingError> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let documents: Vec<Document> = results.iter().map(|r| r.document.clone()).collect();
        let scores = self.reranker.score(query, &documents)?;

        let mut reranked: Vec<SearchResult> = results
            .iter()
            .enumerate()
            .map(|(idx, r)| {
                let new_score = if self.config.preserve_original_score {
                    r.score + scores[idx]
                } else {
                    scores[idx]
                };

                SearchResult {
                    document: r.document.clone(),
                    score: new_score,
                }
            })
            .collect();

        if let Some(min_score) = self.config.min_score {
            reranked.retain(|r| r.score >= min_score);
        }

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        reranked.truncate(self.config.top_n);

        Ok(reranked)
    }

    pub fn rerank_documents(
        &self,
        query: &str,
        documents: Vec<Document>,
    ) -> Result<Vec<SearchResult>, RerankingError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let scores = self.reranker.score(query, &documents)?;

        let mut results: Vec<SearchResult> = documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| SearchResult {
                document: doc.clone(),
                score: scores[idx],
            })
            .collect();

        if let Some(min_score) = self.config.min_score {
            results.retain(|r| r.score >= min_score);
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(self.config.top_n);

        Ok(results)
    }
}

/// BM25-style Reranker（简化版）
pub struct BM25Reranker {
    k1: f32,
    b: f32,
}

impl BM25Reranker {
    pub fn new() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }

    pub fn with_params(mut self, k1: f32, b: f32) -> Self {
        self.k1 = k1;
        self.b = b;
        self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|w| w.len() > 1)
            .map(|w| w.to_lowercase())
            .collect()
    }

    fn compute_tf(&self, term: &str, document: &Document) -> f32 {
        let doc_lower = document.content.to_lowercase();
        let freq = doc_lower.matches(term).count() as f32;
        let doc_len = doc_lower.split_whitespace().count() as f32;

        freq / (freq + self.k1 * (1.0 - self.b + self.b * doc_len / 100.0))
    }
}

impl Default for BM25Reranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for BM25Reranker {
    fn score(&self, query: &str, documents: &[Document]) -> Result<Vec<f32>, RerankingError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let query_terms = self.tokenize(query);

        if query_terms.is_empty() {
            return Ok(documents.iter().map(|_| 0.0).collect());
        }

        let avgdl = documents
            .iter()
            .map(|d| d.content.split_whitespace().count() as f32)
            .sum::<f32>()
            / documents.len() as f32;

        let scores: Vec<f32> = documents
            .iter()
            .map(|doc| {
                let doc_len = doc.content.split_whitespace().count() as f32;
                query_terms
                    .iter()
                    .map(|term| {
                        let tf = self.compute_tf(term, doc);
                        tf * (1.0 + self.k1)
                            / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl))
                    })
                    .sum()
            })
            .collect();

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reranking_config_default() {
        let config = RerankingConfig::default();

        assert_eq!(config.top_n, 5);
        assert!(config.min_score.is_none());
        assert!(config.preserve_original_score);
    }

    #[test]
    fn test_reranking_config_custom() {
        let config = RerankingConfig::new()
            .with_top_n(10)
            .with_min_score(0.5)
            .with_preserve_original_score(false);

        assert_eq!(config.top_n, 10);
        assert_eq!(config.min_score, Some(0.5));
        assert!(!config.preserve_original_score);
    }

    #[test]
    fn test_keyword_reranker_basic() {
        let reranker = KeywordReranker::new();

        let query = "Rust programming";
        let documents = vec![
            Document::new("Rust is a programming language"),
            Document::new("Python is also a programming language"),
            Document::new("JavaScript for web"),
        ];

        let scores = reranker.score(query, &documents).unwrap();

        assert_eq!(scores.len(), 3);
        assert!(scores[0] > 0.0);
        assert!(scores[1] > 0.0);
    }

    #[test]
    fn test_keyword_reranker_empty_query() {
        let reranker = KeywordReranker::new();

        let documents = vec![Document::new("Some content")];

        let scores = reranker.score("", &documents).unwrap();

        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn test_reranking_executor_basic() {
        let reranker = Box::new(KeywordReranker::new());
        let executor = RerankingExecutor::new(reranker).with_top_n(2);

        let results = vec![
            SearchResult {
                document: Document::new("Rust programming language"),
                score: 0.5,
            },
            SearchResult {
                document: Document::new("Python scripting"),
                score: 0.4,
            },
            SearchResult {
                document: Document::new("JavaScript web"),
                score: 0.3,
            },
        ];

        let reranked = executor.rerank("Rust programming", results).unwrap();

        assert_eq!(reranked.len(), 2);
    }

    #[test]
    fn test_reranking_executor_min_score() {
        let reranker = Box::new(KeywordReranker::new());
        let executor = RerankingExecutor::new(reranker)
            .with_top_n(5)
            .with_min_score(1.0);

        let results = vec![
            SearchResult {
                document: Document::new("Rust Rust Rust"),
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
    fn test_bm25_reranker_basic() {
        let reranker = BM25Reranker::new();

        let query = "programming language";
        let documents = vec![
            Document::new("Rust is a programming language"),
            Document::new("Python is a programming language too"),
            Document::new("Web development"),
        ];

        let scores = reranker.score(query, &documents).unwrap();

        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[2]);
    }

    #[test]
    fn test_bm25_reranker_params() {
        let reranker = BM25Reranker::new().with_params(2.0, 0.5);

        let documents = vec![Document::new("test content")];

        let scores = reranker.score("test", &documents).unwrap();

        assert!(scores[0] > 0.0);
    }

    #[test]
    fn test_rerank_documents() {
        let reranker = Box::new(KeywordReranker::new());
        let executor = RerankingExecutor::new(reranker).with_top_n(2);

        let documents = vec![
            Document::new("Rust programming"),
            Document::new("Python scripting"),
            Document::new("JavaScript web"),
        ];

        let results = executor.rerank_documents("Rust", documents).unwrap();

        assert_eq!(results.len(), 2);
    }
}
