// src/retrieval/hybrid.rs
//! Hybrid retrieval module
//!
//! Combines BM25 keyword retrieval + vector semantic retrieval

use lc_vector_stores::Document;
use std::collections::HashMap;

/// The k parameter in the RRF fusion algorithm (default 60).
pub const RRF_K: usize = 60;

/// Generate a stable document ID from content hash to avoid collisions (H46).
///
/// P2-3: Replaces `DefaultHasher` with FNV-1a 64-bit. `DefaultHasher`'s algorithm is an
/// internal std implementation detail, not guaranteed stable across processes/versions;
/// FNV-1a is a fully-specified deterministic hash, so when `doc.id` is missing, fusion
/// dedup does not drift across processes/versions.
fn doc_content_hash(doc: &Document) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = fnv::FnvHasher::default();
    doc.content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Retrieval result (with score)
#[derive(Debug, Clone)]
pub struct RetrievedDocument {
    /// Document content
    pub document: Document,
    /// Fused score
    pub score: f64,
    /// Retrieval source (BM25 / vector / hybrid)
    pub source: RetrievalSource,
}

/// Retrieval source
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetrievalSource {
    /// From BM25 keyword retrieval
    BM25,
    /// From vector semantic retrieval
    Vector,
    /// From RRF fusion
    Hybrid,
}

/// Filters retrieval results by a minimum score (P1-2).
///
/// Eliminates the duplicated `score > 0.0` ghost-threshold implementation across
/// `unified_hybrid` / `graph_rag::matcher`. `min_score` is compared on the **raw score scale**:
/// the default 0.0 keeps the old behavior (only positive similarities are kept). Cosine
/// similarity ranges over [-1, 1]; with non-normalized embedding models the cosine of a
/// relevant document can be negative, so different models may lower the threshold.
pub fn filter_by_score<T, S: PartialOrd>(scored: Vec<(T, S)>, min_score: S) -> Vec<(T, S)> {
    scored.into_iter().filter(|(_, s)| *s > min_score).collect()
}

/// RRF fusion algorithm
///
/// Formula: RRF_score(d) = Σ 1/(k + rank(d))
///
/// Arguments:
/// - bm25_results: BM25 retrieval results, sorted by score descending
/// - vector_results: vector retrieval results, sorted by similarity descending
/// - k: the RRF parameter, usually 60
///
/// Returns:
/// - the fused document list, sorted by RRF score descending
pub fn reciprocal_rank_fusion(
    bm25_results: Vec<Document>,
    vector_results: Vec<Document>,
    k: usize,
) -> Vec<RetrievedDocument> {
    let mut rrf_scores: HashMap<String, (f64, Document)> = HashMap::new();

    // Process BM25 results
    for (rank, doc) in bm25_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_else(|| doc_content_hash(doc));
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);

        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _existing_doc)| {
                *score += rrf_contribution;
            })
            .or_insert((rrf_contribution, doc.clone()));
    }

    // Process vector results
    for (rank, doc) in vector_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_else(|| doc_content_hash(doc));
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);

        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _)| {
                *score += rrf_contribution;
            })
            .or_insert((rrf_contribution, doc.clone()));
    }

    // Sort by RRF score
    let mut results: Vec<RetrievedDocument> = rrf_scores
        .into_iter()
        .map(|(_, (score, doc))| RetrievedDocument {
            document: doc,
            score,
            source: RetrievalSource::Hybrid,
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_basic() {
        let bm25_docs = vec![
            Document::new("Rust系统编程").with_id("doc1"),
            Document::new("Python数据科学").with_id("doc2"),
            Document::new("Go并发编程").with_id("doc3"),
        ];

        let vector_docs = vec![
            Document::new("Rust系统编程").with_id("doc1"),
            Document::new("JavaScript前端").with_id("doc4"),
            Document::new("Python数据科学").with_id("doc2"),
        ];

        let results = reciprocal_rank_fusion(bm25_docs, vector_docs, 60);

        println!("RRF 融合结果:");
        for (i, r) in results.iter().enumerate() {
            println!(
                "  [{}] doc_id={}, score={:.4}",
                i,
                r.document.id.clone().unwrap_or_default(),
                r.score
            );
        }

        // doc1 appears in both lists, so its score should be highest
        let first_doc_id = results[0].document.id.clone().unwrap_or_default();
        println!("最高分文档: {}", first_doc_id);
    }

    /// P1-2: Shares the filter_by_score utility — the default 0.0 keeps only positive
    /// similarities; lowering the threshold keeps negative-similarity documents (with
    /// non-normalized embedding models a relevant document's cosine can be negative).
    #[test]
    fn test_filter_by_score() {
        let scored = vec![("a", 0.9_f32), ("b", 0.2), ("c", -0.3), ("d", 0.0)];

        // Default threshold 0.0: keep only strictly-greater scores (matches the old `score > 0.0` behavior)
        let filtered = filter_by_score(scored.clone(), 0.0);
        let ids: Vec<&str> = filtered.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec!["a", "b"]);

        // Lowering the threshold keeps negative similarities
        let relaxed = filter_by_score(scored.clone(), -0.5);
        assert_eq!(relaxed.len(), 4);

        // Raising the threshold is stricter
        let strict = filter_by_score(scored.clone(), 0.5);
        let ids: Vec<&str> = strict.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec!["a"]);
    }

    /// P1-2: filter_by_score also works for f64 scores.
    #[test]
    fn test_filter_by_score_f64() {
        let scored = vec![("x", 0.8_f64), ("y", 0.0), ("z", -0.5)];
        let filtered = filter_by_score(scored, 0.0);
        let ids: Vec<&str> = filtered.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec!["x"]);
    }

    /// P2-3: `doc_content_hash` is a deterministic hash — the same content yields the same
    /// result across calls, different content yields different results. FNV-1a is fully
    /// specified and does not drift across processes/versions.
    #[test]
    fn test_doc_content_hash_stable() {
        let content = "Rust 系统编程与并发";
        let doc_a = Document::new(content.to_string());
        let doc_b = Document::new(content.to_string());
        let doc_c = Document::new("Python 数据科学");

        let hash_a1 = doc_content_hash(&doc_a);
        let hash_a2 = doc_content_hash(&doc_b);
        assert_eq!(hash_a1, hash_a2, "相同内容应产生相同哈希");

        let hash_c = doc_content_hash(&doc_c);
        assert_ne!(hash_a1, hash_c, "不同内容应产生不同哈希");
        assert_eq!(hash_a1.len(), 16, "应为 64 位哈希的 16 位十六进制表示");
    }
}
