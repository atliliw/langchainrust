// src/retrieval/reranking.rs
//! Reranking implementation
//!
//! Re-ranks retrieval results with a scoring function, improving retrieval precision.

use lc_vector_stores::{Document, SearchResult};
use std::collections::HashMap;

/// Reranking error type
#[derive(Debug)]
#[non_exhaustive]
pub enum RerankingError {
    /// Scoring error
    ScoringError(String),
    /// Invalid input error
    InvalidInput(String),
}

impl std::fmt::Display for RerankingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RerankingError::ScoringError(msg) => write!(f, "scoring error: {}", msg),
            RerankingError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
        }
    }
}

impl std::error::Error for RerankingError {}

/// True min-max normalization to [0, 1], monotonic over the real scores.
///
/// Unlike magnitude-max (`|s|`) normalization, a negative score maps below
/// every positive one while preserving their relative order — so reranking
/// never flips sign or collapses strong positives toward zero. A degenerate
/// input (all equal, or max <= min) yields all zeros, keeping the sum stable
/// when every candidate ties.
fn normalize_minmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &s in scores {
        if s.is_finite() {
            min = min.min(s);
            max = max.max(s);
        }
    }
    // Non-finite (e.g. BM25 edge-case NaN/Inf) or all-equal input: no range to
    // span; map everything to 0.0 so a NaN never leaks into the combined score.
    if !(max > min) {
        return vec![0.0; scores.len()];
    }
    let span = max - min;
    scores
        .iter()
        .map(|&s| if s.is_finite() { (s - min) / span } else { 0.0 })
        .collect()
}

/// Reranking configuration
pub struct RerankingConfig {
    /// Number of documents returned in the final result
    pub top_n: usize,

    /// Minimum score threshold (optional)
    pub min_score: Option<f32>,

    /// Whether to preserve the original score
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
    /// Creates a `RerankingConfig` with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of documents returned in the final result
    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Sets the minimum score threshold
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Sets whether to preserve the original score
    pub fn with_preserve_original_score(mut self, preserve: bool) -> Self {
        self.preserve_original_score = preserve;
        self
    }
}

/// Reranking scorer trait
pub trait Reranker: Send + Sync {
    /// Scores the given document list, returning a score array that maps one-to-one to the documents
    fn score(&self, query: &str, documents: &[Document]) -> Result<Vec<f32>, RerankingError>;
}

/// A simple keyword-matching Reranker
pub struct KeywordReranker {
    /// Keyword weights (optional)
    keyword_weights: HashMap<String, f32>,
}

impl KeywordReranker {
    /// Creates a default keyword Reranker
    pub fn new() -> Self {
        Self {
            keyword_weights: HashMap::new(),
        }
    }

    /// Sets the keyword-weight mapping
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

/// Reranker executor
pub struct RerankingExecutor {
    reranker: Box<dyn Reranker>,
    config: RerankingConfig,
}

impl RerankingExecutor {
    /// Creates a reranker executor
    pub fn new(reranker: Box<dyn Reranker>) -> Self {
        Self {
            reranker,
            config: RerankingConfig::default(),
        }
    }

    /// Sets the reranking configuration
    pub fn with_config(mut self, config: RerankingConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the number of documents returned in the final result
    pub fn with_top_n(mut self, n: usize) -> Self {
        self.config.top_n = n;
        self
    }

    /// Sets the minimum score threshold
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.config.min_score = Some(score);
        self
    }

    /// Sets whether to preserve the original score
    pub fn with_preserve_original_score(mut self, preserve: bool) -> Self {
        self.config.preserve_original_score = preserve;
        self
    }

    /// Re-ranks the retrieval results, returning the reranked scored results
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
        Self::ensure_score_count(documents.len(), scores.len())?;

        // Normalize scores to [0, 1] range before combining. The prior code divided by the
        // max *magnitude* (`r.score.abs()`), which collapsed strong positives toward zero
        // and gave a lone negative candidate a huge `|norm|` — flipping the ranking/truncation
        // semantics whenever scores carried negative values (BM25/neural scores often do).
        // True min-max over the real scores is monotonic: ordering is preserved, negatives
        // stay below positives, and both contributions land on [0,1] so `min_score` filtering
        // and `top_n` truncation compare like values. A degenerate constant set (max <= min,
        // includes the all-zero / all-equal case) maps every candidate to 0.0.
        let norm_original =
            normalize_minmax(&results.iter().map(|r| r.score).collect::<Vec<f32>>());
        let norm_rerank = normalize_minmax(&scores);

        let mut reranked: Vec<SearchResult> = results
            .iter()
            .enumerate()
            .map(|(idx, r)| {
                let new_score = if self.config.preserve_original_score {
                    norm_original[idx] + norm_rerank[idx]
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

    /// §3.2 (LOW): a custom [`Reranker`] must return exactly one score per
    /// input document; anything else made `scores[idx]` panic on out-of-range.
    /// Fail with a diagnostic instead of an indexing panic.
    fn ensure_score_count(documents: usize, scores: usize) -> Result<(), RerankingError> {
        if documents == scores {
            Ok(())
        } else {
            Err(RerankingError::InvalidInput(format!(
                "reranker returned {scores} scores for {documents} documents"
            )))
        }
    }

    /// Scores and re-ranks a document list directly, returning scored results
    pub fn rerank_documents(
        &self,
        query: &str,
        documents: Vec<Document>,
    ) -> Result<Vec<SearchResult>, RerankingError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let scores = self.reranker.score(query, &documents)?;
        Self::ensure_score_count(documents.len(), scores.len())?;

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

/// BM25-style Reranker (simplified)
pub struct BM25Reranker {
    k1: f32,
    b: f32,
}

impl BM25Reranker {
    /// Creates a BM25 Reranker with default parameters
    pub fn new() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }

    /// Sets the BM25 parameters k1 and b
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

        let n_docs = documents.len() as f32;
        let avgdl = documents
            .iter()
            .map(|d| d.content.split_whitespace().count() as f32)
            .sum::<f32>()
            / n_docs;
        // LOW: a non-empty list of *empty-content* documents yields `avgdl == 0.0`
        // (sum 0 / n_docs), which would then make the term score `doc_len/avgdl =
        // 0/0 = NaN`. `denom <= 0.0` never fires for NaN, so a NaN leaks into the
        // result. Clamping the divisor keeps normalization finite; an empty doc
        // scores 0 (its term freq is 0 anyway).
        let avgdl = avgdl.max(f32::MIN_POSITIVE);

        // Per-query-term inverse document frequency. Standard BM25 uses
        // IDF so that rare terms contribute more than common ones; the
        // prior implementation had no IDF term (rare words were never
        // boosted) and saturates term frequency twice, skewing rankings.
        let lowercase: Vec<String> = documents.iter().map(|d| d.content.to_lowercase()).collect();
        let idfs: Vec<f32> = query_terms
            .iter()
            .map(|term| {
                // Document frequency: how many documents contain this term.
                let df = lowercase
                    .iter()
                    .filter(|doc| doc.contains(term.as_str()))
                    .count() as f32;
                // L6: use the SAME one-plus IDF as `bm25::algorithm::compute_idf` (`+1`
                // after the ratio). This re-ranker is meant to re-rank the primary BM25
                // hit list; with the previous no-`+1` formula the two scorers disagree on
                // the relative weight of rare vs common terms and can reorder results.
                ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln()
            })
            .collect();

        let scores: Vec<f32> = documents
            .iter()
            .zip(&lowercase)
            .map(|(doc, doc_lower)| {
                let doc_len = doc.content.split_whitespace().count() as f32;
                query_terms
                    .iter()
                    .zip(&idfs)
                    .map(|(term, idf)| {
                        let freq = doc_lower.matches(term.as_str()).count() as f32;
                        let denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl);
                        if denom <= 0.0 {
                            0.0
                        } else {
                            idf * (freq * (1.0 + self.k1)) / denom
                        }
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
    fn test_bm25_uses_idf_rare_term_outranks() {
        // Standard BM25 must boost rare terms. "rare" appears in only one of
        // four documents (positive IDF) while "the" appears in three
        // (negative IDF). Without an IDF term, the doc matching only the
        // common "the" would be judged by raw term frequency alone.
        let reranker = BM25Reranker::new();

        let query = "the rare";
        let documents = vec![
            Document::new("the the the"),
            Document::new("the the"),
            Document::new("the"),
            Document::new("rare exotic uncommon phrase"),
        ];

        let scores = reranker.score(query, &documents).unwrap();

        // The doc containing the rare term must score strictly positive and
        // strictly higher than the docs matching only the common "the".
        assert!(
            scores[3] > 0.0,
            "rare-term doc should be positive, got {scores:?}"
        );
        assert!(
            scores[3] > scores[0] && scores[3] > scores[1] && scores[3] > scores[2],
            "rare-term doc should outrank common-term docs, got {scores:?}"
        );
    }

    #[test]
    fn test_bm25_no_terms_scores_zero() {
        let reranker = BM25Reranker::new();
        let documents = vec![Document::new("some content")];

        let scores = reranker.score("", &documents).unwrap();
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn test_bm25_empty_documents() {
        let reranker = BM25Reranker::new();
        let scores = reranker.score("query", &[]).unwrap();
        assert!(scores.is_empty());
    }

    /// LOW: a non-empty list of *empty-content* documents must not produce NaN
    /// scores. The old code computed `avgdl = 0 / n_docs = 0.0`, then
    /// `doc_len/avgdl = 0/0 = NaN` in the term score, and the `denom <= 0.0`
    /// guard never fires on NaN → NaN leaked into the result.
    #[test]
    fn test_bm25_all_empty_content_no_nan() {
        let reranker = BM25Reranker::new();
        let docs: Vec<Document> = vec![Document::new(""), Document::new(""), Document::new("   ")];
        let scores = reranker.score("query", &docs).unwrap();
        assert_eq!(scores.len(), 3);
        for s in scores {
            assert!(s.is_finite(), "NaN leaked into BM25 score: {s:?}");
            assert_eq!(s, 0.0, "empty-content doc should score 0");
        }
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

        // "quantum" appears in only one doc (positive IDF); the matching doc
        // must beat the non-matching docs. With correct BM25, a doc whose only
        // query terms are common across every candidate can legitimately score
        // non-positive — order by relative relevance is what we assert here.
        let query = "quantum";
        let documents = vec![
            Document::new("Rust quantum computing"),
            Document::new("Python programming"),
            Document::new("Web development"),
        ];

        let scores = reranker.score(query, &documents).unwrap();

        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
    }

    #[test]
    fn test_bm25_reranker_params() {
        // Custom k1/b must still produce a positive score because "test" is
        // rare here (1 of 3 docs). A single-document corpus would force
        // df == N and give negative IDF by construction.
        let reranker = BM25Reranker::new().with_params(2.0, 0.5);

        let documents = vec![
            Document::new("test content"),
            Document::new("other text"),
            Document::new("more text"),
        ];

        let scores = reranker.score("test", &documents).unwrap();

        assert!(scores[0] > 0.0);
    }

    /// R-6: a negative score must not be flipped to a positive by `.abs()`
    /// magnitude normalization. A doc with a strongly-negative original score
    /// stays worst-ranked; a positive one stays best, and `min_score` filtering
    /// does not admit a doc purely because its magnitude was large.
    #[test]
    fn rerank_preserves_order_with_negative_scores() {
        // KeywordReranker emits only >= 0, so craft a custom Reranker that
        // returns a deliberately negative mix to exercise the min-max path.
        struct NegReranker;
        impl Reranker for NegReranker {
            fn score(
                &self,
                _query: &str,
                _documents: &[Document],
            ) -> Result<Vec<f32>, RerankingError> {
                Ok(vec![-10.0, 5.0, -2.0])
            }
        }
        let executor = RerankingExecutor::new(Box::new(NegReranker))
            .with_preserve_original_score(true)
            .with_top_n(3);
        // Original scores also span negative → positive to stress both sides.
        let results = vec![
            SearchResult {
                document: Document::new("c"),
                score: -7.0,
            },
            SearchResult {
                document: Document::new("a"),
                score: 8.0,
            },
            SearchResult {
                document: Document::new("b"),
                score: 0.5,
            },
        ];
        let reranked = executor.rerank("q", results).unwrap();
        // idx1 (orig 8 / rerank 5) must be highest; idx2 (0.5, -2) above idx0.
        assert_eq!(reranked[0].document.content, "a");
        assert_eq!(reranked[1].document.content, "b");
        assert_eq!(reranked[2].document.content, "c");
        let scores: Vec<f32> = reranked.iter().map(|r| r.score).collect();
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "must be non-increasing, got {scores:?}"
        );
        // The worst doc normalizes its min original to 0.0 (never promoted by a
        // large |magnitude|), so it lands at the bottom—it must not outrank the
        // positive candidates despite its large negative magnitude.
        assert!(
            scores[2] < scores[1] && scores[2] < scores[0],
            "worst doc must rank last despite large magnitude, got {scores:?}"
        );
    }

    #[test]
    fn normalize_minmax_is_monotonic_and_sign_preserving() {
        // Pure helper: negatives stay below positives, order preserved, span -> [0,1].
        assert_eq!(normalize_minmax(&[0.0, 4.0, 2.0]), vec![0.0, 1.0, 0.5]);
        let out = normalize_minmax(&[-10.0, 5.0, -2.0]);
        // Argmax / argmin correspond to the original order; a negative maps to
        // its true rank in the span (min→0.0), never to a large positive.
        assert_eq!(out, vec![0.0, 1.0, 8.0 / 15.0]);
        // All-equal and all-NaN degenerate to zeros (no range to span).
        assert_eq!(normalize_minmax(&[3.0, 3.0, 3.0]), vec![0.0, 0.0, 0.0]);
        assert_eq!(normalize_minmax(&[f32::NAN, f32::INFINITY]), vec![0.0, 0.0]);
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
