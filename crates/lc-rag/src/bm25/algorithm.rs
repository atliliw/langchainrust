// src/retrieval/bm25/algorithm.rs
//! Core BM25 algorithm implementation
//!
//! BM25 formula: score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

/// BM25 parameter configuration
#[derive(Debug, Clone)]
pub struct BM25Params {
    /// Term-frequency saturation parameter, controls the influence of high-frequency terms (default 1.5)
    pub k1: f64,

    /// Document-length normalization parameter, controls the long-document penalty (default 0.75)
    pub b: f64,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

impl BM25Params {
    /// Creates the default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates custom parameters
    pub fn with_values(k1: f64, b: f64) -> Self {
        Self { k1, b }
    }
}

/// Computes the IDF (inverse document frequency)
///
/// Formula: IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
///
/// # Arguments
/// - `n`: the number of documents containing the term
/// - `N`: the total number of documents
///
/// # Returns
/// The IDF value
pub fn compute_idf(n: usize, total_docs: usize) -> f64 {
    if n == 0 || total_docs == 0 {
        return 0.0;
    }

    let numerator = total_docs as f64 - n as f64 + 0.5;
    let denominator = n as f64 + 0.5;

    (numerator / denominator + 1.0).ln()
}

/// Computes the BM25 score
///
/// Formula: score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
///
/// # Arguments
/// - `query_terms`: the list of query terms
/// - `doc_term_freqs`: the document term-frequency table (term -> frequency)
/// - `doc_length`: the document length (in terms)
/// - `avgdl`: the average document length
/// - `idf_values`: the term IDF table (term -> IDF)
/// - `params`: the BM25 parameters
///
/// # Returns
/// The BM25 score
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
        // Look up the IDF
        let idf = idf_values.get(term).copied().unwrap_or(0.0);
        if idf == 0.0 {
            continue;
        }

        // Look up the term frequency
        let tf = doc_term_freqs.get(term).copied().unwrap_or(0);
        if tf == 0 {
            continue;
        }

        // Compute the TF-normalized component
        let dl_ratio = doc_length as f64 / avgdl;
        let tf_component = (tf as f64 * (params.k1 + 1.0))
            / (tf as f64 + params.k1 * (1.0 - params.b + params.b * dl_ratio));

        score += idf * tf_component;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_compute_idf() {
        // A common term has a low IDF
        let idf_common = compute_idf(100, 100); // present in every document
        assert!(idf_common < 1.0);

        // A rare term has a high IDF
        let idf_rare = compute_idf(1, 100); // present in only 1 document
        assert!(idf_rare > idf_common);

        // A nonexistent term
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

        // The score should be positive
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
        // A higher-term-frequency document should score higher
        let params = BM25Params::default();

        let query = vec!["rust".to_string()];
        let idf = HashMap::from([("rust".to_string(), 2.0)]);

        // Low term frequency
        let low_tf = HashMap::from([("rust".to_string(), 1)]);
        let score_low = bm25_score(&query, &low_tf, 10, 15.0, &idf, &params);

        // High term frequency
        let high_tf = HashMap::from([("rust".to_string(), 5)]);
        let score_high = bm25_score(&query, &high_tf, 10, 15.0, &idf, &params);

        assert!(score_high > score_low);
    }
}
