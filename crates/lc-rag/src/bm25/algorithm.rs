// src/retrieval/bm25/algorithm.rs
//! BM25 算法核心实现
//!
//! BM25 公式: score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

/// BM25 参数配置
#[derive(Debug, Clone)]
pub struct BM25Params {
    /// 词频饱和参数，控制高频词的影响 (默认 1.5)
    pub k1: f64,

    /// 文档长度归一化参数，控制长文档惩罚 (默认 0.75)
    pub b: f64,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

impl BM25Params {
    /// 创建默认参数
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建自定义参数
    pub fn with_values(k1: f64, b: f64) -> Self {
        Self { k1, b }
    }
}

/// 计算 IDF (逆文档频率)
///
/// 公式: IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
///
/// # 参数
/// - `n`: 包含该词的文档数量
/// - `N`: 文档总数
///
/// # 返回
/// IDF 值
pub fn compute_idf(n: usize, total_docs: usize) -> f64 {
    if n == 0 || total_docs == 0 {
        return 0.0;
    }

    let numerator = total_docs as f64 - n as f64 + 0.5;
    let denominator = n as f64 + 0.5;

    (numerator / denominator + 1.0).ln()
}

/// 计算 BM25 评分
///
/// 公式: score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
///
/// # 参数
/// - `query_terms`: 查询词列表
/// - `doc_term_freqs`: 文档词频表 (term -> frequency)
/// - `doc_length`: 文档长度 (词数)
/// - `avgdl`: 平均文档长度
/// - `idf_values`: 词的 IDF 值表 (term -> IDF)
/// - `params`: BM25 参数
///
/// # 返回
/// BM25 评分
pub fn bm25_score(
    query_terms: &[String],
    doc_term_freqs: &std::collections::HashMap<String, usize>,
    doc_length: usize,
    avgdl: f64,
    idf_values: &std::collections::HashMap<String, f64>,
    params: &BM25Params,
) -> f64 {
    if avgdl == 0.0 || doc_length == 0 {
        return 0.0;
    }

    let mut score = 0.0;

    for term in query_terms {
        // 获取 IDF
        let idf = idf_values.get(term).copied().unwrap_or(0.0);
        if idf == 0.0 {
            continue;
        }

        // 获取词频
        let tf = doc_term_freqs.get(term).copied().unwrap_or(0);
        if tf == 0 {
            continue;
        }

        // 计算 TF 归一化部分
        let dl_ratio = doc_length as f64 / avgdl;
        let tf_component = (tf as f64 * (params.k1 + 1.0))
            / (tf as f64 + params.k1 * (1.0 - params.b + params.b * dl_ratio));

        score += idf * tf_component;
    }

    score
}

/// 计算单个词在文档中的 BM25 分量
///
/// 用于调试和分析
#[allow(dead_code)]
pub fn bm25_term_score(
    _term: &str,
    tf: usize,
    doc_length: usize,
    avgdl: f64,
    idf: f64,
    params: &BM25Params,
) -> f64 {
    if tf == 0 || avgdl == 0.0 || idf == 0.0 {
        return 0.0;
    }

    let dl_ratio = doc_length as f64 / avgdl;
    let tf_component = (tf as f64 * (params.k1 + 1.0))
        / (tf as f64 + params.k1 * (1.0 - params.b + params.b * dl_ratio));

    idf * tf_component
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_compute_idf() {
        // 常见词 IDF 较低
        let idf_common = compute_idf(100, 100); // 所有文档都有
        assert!(idf_common < 1.0);

        // 稀有词 IDF 较高
        let idf_rare = compute_idf(1, 100); // 只有1个文档有
        assert!(idf_rare > idf_common);

        // 不存在的词
        let idf_zero = compute_idf(0, 100);
        assert_eq!(idf_zero, 0.0);
    }

    #[test]
    fn test_bm25_score() {
        let params = BM25Params::default();

        let query_terms = vec!["rust".to_string(), "programming".to_string()];

        let mut doc_term_freqs = HashMap::new();
        doc_term_freqs.insert("rust".to_string(), 2);
        doc_term_freqs.insert("programming".to_string(), 1);

        let mut idf_values = HashMap::new();
        idf_values.insert("rust".to_string(), 2.0);
        idf_values.insert("programming".to_string(), 1.5);

        let score = bm25_score(
            &query_terms,
            &doc_term_freqs,
            10,   // doc_length
            15.0, // avgdl
            &idf_values,
            &params,
        );

        // 评分应该为正
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_params() {
        let default = BM25Params::default();
        assert_eq!(default.k1, 1.5);
        assert_eq!(default.b, 0.75);

        let custom = BM25Params::with_values(2.0, 0.5);
        assert_eq!(custom.k1, 2.0);
        assert_eq!(custom.b, 0.5);
    }

    #[test]
    fn test_bm25_high_tf_document() {
        // 高词频文档应该得更高分
        let params = BM25Params::default();

        let query = vec!["rust".to_string()];
        let idf = HashMap::from([("rust".to_string(), 2.0)]);

        // 低词频文档
        let low_tf = HashMap::from([("rust".to_string(), 1)]);
        let score_low = bm25_score(&query, &low_tf, 10, 15.0, &idf, &params);

        // 高词频文档
        let high_tf = HashMap::from([("rust".to_string(), 5)]);
        let score_high = bm25_score(&query, &high_tf, 10, 15.0, &idf, &params);

        assert!(score_high > score_low);
    }
}
