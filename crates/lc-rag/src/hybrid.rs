// src/retrieval/hybrid.rs
//! 混合检索模块
//!
//! 结合 BM25 关键词检索 + 向量语义检索

use lc_vector_stores::Document;
use std::collections::HashMap;

pub const RRF_K: usize = 60;

/// Generate a stable document ID from content hash to avoid collisions (H46).
///
/// P2-3: 用 FNV-1a 64 替代 `DefaultHasher`。`DefaultHasher` 的算法是 std 内部
/// 实现细节,不保证跨进程/跨版本稳定;FNV-1a 是完全指定的确定性哈希,同一
/// 内容的 `doc.id` 缺失时融合去重不会漂移。
fn doc_content_hash(doc: &Document) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = fnv::FnvHasher::default();
    doc.content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

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

/// 按最小分数过滤检索结果(P1-2)。
///
/// 消除 `score > 0.0` 幽灵阈值在 `unified_hybrid` / `chunked_hybrid` /
/// `graph_rag::matcher` 三处的重复实现。`min_score` 在**原始分数尺度**上比较:
/// 默认 0.0 保持旧行为(只保留正相似度)。余弦相似度范围 [-1,1],
/// 非归一化嵌入模型下相关文档的余弦可能为负,不同模型可自行调低阈值。
pub fn filter_by_score<T, S: PartialOrd>(scored: Vec<(T, S)>, min_score: S) -> Vec<(T, S)> {
    scored.into_iter().filter(|(_, s)| *s > min_score).collect()
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
        let doc_id = doc.id.clone().unwrap_or_else(|| doc_content_hash(doc));
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
        let doc_id = doc.id.clone().unwrap_or_else(|| doc_content_hash(doc));
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
        let doc_id = doc.id.clone().unwrap_or_else(|| doc_content_hash(doc));
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
        let doc_id = doc.id.clone().unwrap_or_else(|| doc_content_hash(doc));
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

/// 混合检索器（已废弃）
///
/// 旧版 BM25 + 向量 RRF 融合检索器。请迁移到
/// [`UnifiedHybridIndex`](crate::unified_hybrid::UnifiedHybridIndex)，
/// 后者统一了索引、自带向量存储并实现 `RetrieverTrait`。
#[allow(dead_code)]
#[deprecated(
    note = "Use UnifiedHybridIndex instead (see crate::unified_hybrid::UnifiedHybridIndex)"
)]
pub struct HybridRetriever {
    bm25_k: usize,
    vector_k: usize,
    rrf_k: usize,
}

#[allow(deprecated)] // 已弃用类型的内部实现仍需引用自身字段
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

#[allow(deprecated)] // 已弃用类型的 Default 实现
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

    /// P1-2: 共享 filter_by_score 工具函数——默认 0.0 只保留正相似度,
    /// 调低阈值可保留负相似度文档(非归一化嵌入模型下相关文档余弦可为负)。
    #[test]
    fn test_filter_by_score() {
        let scored = vec![("a", 0.9_f32), ("b", 0.2), ("c", -0.3), ("d", 0.0)];

        // 默认阈值 0.0: 严格大于才保留(与旧 `score > 0.0` 行为一致)
        let filtered = filter_by_score(scored.clone(), 0.0);
        let ids: Vec<&str> = filtered.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec!["a", "b"]);

        // 调低阈值可保留负相似度
        let relaxed = filter_by_score(scored.clone(), -0.5);
        assert_eq!(relaxed.len(), 4);

        // 调高阈值更严格
        let strict = filter_by_score(scored.clone(), 0.5);
        let ids: Vec<&str> = strict.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec!["a"]);
    }

    /// P1-2: filter_by_score 对 f64 分数同样适用。
    #[test]
    fn test_filter_by_score_f64() {
        let scored = vec![("x", 0.8_f64), ("y", 0.0), ("z", -0.5)];
        let filtered = filter_by_score(scored, 0.0);
        let ids: Vec<&str> = filtered.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec!["x"]);
    }

    #[allow(deprecated)] // 已弃用类型的兼容性测试
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

    /// P2-3: `doc_content_hash` 为确定性哈希——同内容多次调用结果一致,
    /// 不同内容结果不同。FNV-1a 完全指定,跨进程/跨版本不漂移。
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
