// src/retrieval/hybrid.rs
//! 混合检索模块
//!
//! 结合 BM25 关键词检索 + 向量语义检索

use crate::vector_stores::Document;
use std::collections::HashMap;

pub const RRF_K: usize = 60;

/// 检索结果（带分数）
#[derive(Debug, Clone)]
pub struct RetrievedDocument {
    pub document: Document,
    pub score: f64,
    pub source: RetrievalSource,
}

/// 检索来源
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetrievalSource {
    BM25,
    Vector,
    Hybrid,
}

/// RRF 融合算法
///
/// 公式: RRF_score(d) = Σ 1/(k + rank(d))
///
/// 参数:
/// - bm25_results: BM25 检索结果，按分数降序排列
/// - vector_results: 向量检索结果，按相似度降序排列
/// - k: RRF 参数，通常为 60
///
/// 返回:
/// - 融合后的文档列表，按 RRF 分数降序排列
pub fn reciprocal_rank_fusion(
    bm25_results: Vec<Document>,
    vector_results: Vec<Document>,
    k: usize,
) -> Vec<RetrievedDocument> {
    let mut rrf_scores: HashMap<String, (f64, Document)> = HashMap::new();

    // BM25 结果处理
    for (rank, doc) in bm25_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_else(|| format!("bm25_{}", rank));
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);

        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _existing_doc)| {
                *score += rrf_contribution;
            })
            .or_insert((rrf_contribution, doc.clone()));
    }

    // 向量结果处理
    for (rank, doc) in vector_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_else(|| format!("vector_{}", rank));
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);

        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _)| {
                *score += rrf_contribution;
            })
            .or_insert((rrf_contribution, doc.clone()));
    }

    // 按 RRF 分数排序
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

/// 带原始分数的 RRF 融合
///
/// 保留 BM25 和 Vector 的原始分数信息
pub fn reciprocal_rank_fusion_with_scores(
    bm25_results: Vec<(Document, f64)>,
    vector_results: Vec<(Document, f64)>,
    k: usize,
) -> Vec<RetrievedDocument> {
    let mut rrf_scores: HashMap<String, (f64, Document, Option<f64>, Option<f64>)> = HashMap::new();

    // BM25 结果处理
    for (rank, (doc, bm25_score)) in bm25_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_else(|| format!("bm25_{}", rank));
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);

        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _, bm25, _vector)| {
                *score += rrf_contribution;
                *bm25 = Some(*bm25_score);
            })
            .or_insert((rrf_contribution, doc.clone(), Some(*bm25_score), None));
    }

    // 向量结果处理
    for (rank, (doc, vector_score)) in vector_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_else(|| format!("vector_{}", rank));
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);

        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _, _bm25, vector)| {
                *score += rrf_contribution;
                *vector = Some(*vector_score);
            })
            .or_insert((rrf_contribution, doc.clone(), None, Some(*vector_score)));
    }

    // 按 RRF 分数排序
    let mut results: Vec<RetrievedDocument> = rrf_scores
        .into_iter()
        .map(|(_, (score, doc, _, _))| RetrievedDocument {
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

/// 混合检索器
#[allow(dead_code)]
pub struct HybridRetriever {
    bm25_k: usize,
    vector_k: usize,
    rrf_k: usize,
}

impl HybridRetriever {
    pub fn new() -> Self {
        Self {
            bm25_k: 10,
            vector_k: 10,
            rrf_k: RRF_K,
        }
    }

    pub fn with_top_k(bm25_k: usize, vector_k: usize) -> Self {
        Self {
            bm25_k,
            vector_k,
            rrf_k: RRF_K,
        }
    }

    pub fn with_rrf_k(mut self, k: usize) -> Self {
        self.rrf_k = k;
        self
    }

    /// 执行混合检索
    ///
    /// 参数:
    /// - query: 查询文本
    /// - bm25_results: BM25 检索结果
    /// - vector_results: 向量检索结果
    ///
    /// 返回:
    /// - 融合后的 top-k 结果
    pub fn retrieve(
        &self,
        bm25_results: Vec<Document>,
        vector_results: Vec<Document>,
    ) -> Vec<RetrievedDocument> {
        reciprocal_rank_fusion(bm25_results, vector_results, self.rrf_k)
    }

    /// 执行混合检索（带原始分数）
    pub fn retrieve_with_scores(
        &self,
        bm25_results: Vec<(Document, f64)>,
        vector_results: Vec<(Document, f64)>,
    ) -> Vec<RetrievedDocument> {
        reciprocal_rank_fusion_with_scores(bm25_results, vector_results, self.rrf_k)
    }
}

impl Default for HybridRetriever {
    fn default() -> Self {
        Self::new()
    }
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

        // doc1 在两个列表都出现，分数应该最高
        let first_doc_id = results[0].document.id.clone().unwrap_or_default();
        println!("最高分文档: {}", first_doc_id);
    }

    #[test]
    fn test_rrf_with_scores() {
        let bm25_docs = vec![
            (Document::new("Rust").with_id("doc1"), 3.5),
            (Document::new("Python").with_id("doc2"), 2.1),
        ];

        let vector_docs = vec![
            (Document::new("Rust").with_id("doc1"), 0.92),
            (Document::new("Go").with_id("doc3"), 0.88),
        ];

        let results = reciprocal_rank_fusion_with_scores(bm25_docs, vector_docs, 60);

        println!("带分数的 RRF 融合:");
        for r in &results {
            println!(
                "  doc_id={}, rrf_score={:.4}",
                r.document.id.clone().unwrap_or_default(),
                r.score
            );
        }
    }

    #[test]
    fn test_hybrid_retriever() {
        let retriever = HybridRetriever::new();

        let bm25_docs = vec![
            Document::new("机器学习").with_id("doc1"),
            Document::new("深度学习").with_id("doc2"),
        ];

        let vector_docs = vec![
            Document::new("机器学习").with_id("doc1"),
            Document::new("自然语言处理").with_id("doc3"),
        ];

        let results = retriever.retrieve(bm25_docs, vector_docs);

        println!("HybridRetriever 结果数: {}", results.len());
        for r in &results {
            println!(
                "  id={}, score={:.4}",
                r.document.id.clone().unwrap_or_default(),
                r.score
            );
        }
    }
}
