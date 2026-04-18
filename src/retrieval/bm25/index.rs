// src/retrieval/bm25/index.rs
//! BM25 索引实现
//!
//! 存储文档并构建词频索引

use super::algorithm::{compute_idf, BM25Params};
use crate::vector_stores::Document;
use std::collections::HashMap;

/// BM25 索引
pub struct BM25Index {
    /// 文档集合
    documents: Vec<Document>,

    /// 文档词频表: term -> (doc_id, frequency)
    term_doc_freqs: HashMap<String, HashMap<usize, usize>>,

    /// 每个文档的词频表: doc_id -> (term -> frequency)
    doc_term_freqs: Vec<HashMap<String, usize>>,

    /// 文档长度
    doc_lengths: Vec<usize>,

    /// 平均文档长度
    avgdl: f64,

    /// 文档总数
    n_docs: usize,

    /// IDF 缓存: term -> IDF
    idf_cache: HashMap<String, f64>,

    /// BM25 参数
    params: BM25Params,
}

impl BM25Index {
    /// 创建新的 BM25 索引
    pub fn new() -> Self {
        Self::with_params(BM25Params::default())
    }

    /// 使用自定义参数创建索引
    pub fn with_params(params: BM25Params) -> Self {
        Self {
            documents: Vec::new(),
            term_doc_freqs: HashMap::new(),
            doc_term_freqs: Vec::new(),
            doc_lengths: Vec::new(),
            avgdl: 0.0,
            n_docs: 0,
            idf_cache: HashMap::new(),
            params,
        }
    }

    /// 添加文档并构建索引
    ///
    /// # 参数
    /// - `document`: 文档
    /// - `terms`: 文档的词列表（已分词）
    pub fn add_document(&mut self, document: Document, terms: Vec<String>) {
        let doc_id = self.n_docs;

        // 计算文档词频
        let mut term_freq = HashMap::new();
        for term in &terms {
            *term_freq.entry(term.clone()).or_insert(0) += 1;
        }

        // 更新全局词频表
        for (term, freq) in &term_freq {
            self.term_doc_freqs
                .entry(term.clone())
                .or_insert_with(HashMap::new)
                .insert(doc_id, *freq);
        }

        // 存储文档信息
        self.documents.push(document);
        self.doc_term_freqs.push(term_freq);
        self.doc_lengths.push(terms.len());
        self.n_docs += 1;

        // 更新平均文档长度
        self.update_avgdl();

        // 清除 IDF 缓存（需要重新计算）
        self.idf_cache.clear();
    }

    /// 批量添加文档
    pub fn add_documents(&mut self, documents: Vec<Document>, terms_list: Vec<Vec<String>>) {
        if documents.len() != terms_list.len() {
            return;
        }

        for (doc, terms) in documents.into_iter().zip(terms_list) {
            self.add_document(doc, terms);
        }
    }

    /// 更新平均文档长度
    fn update_avgdl(&mut self) {
        if self.n_docs == 0 {
            self.avgdl = 0.0;
        } else {
            let total_length: usize = self.doc_lengths.iter().sum();
            self.avgdl = total_length as f64 / self.n_docs as f64;
        }
    }

    /// 计算并缓存 IDF
    pub fn compute_idf_for_term(&mut self, term: &str) -> f64 {
        if let Some(idf) = self.idf_cache.get(term) {
            return *idf;
        }

        let n = self.term_doc_freqs.get(term).map(|m| m.len()).unwrap_or(0);

        let idf = compute_idf(n, self.n_docs);
        self.idf_cache.insert(term.to_string(), idf);

        idf
    }

    /// 批量计算 IDF
    pub fn compute_idf_for_terms(&mut self, terms: &[String]) -> HashMap<String, f64> {
        let mut idf_values = HashMap::new();
        for term in terms {
            idf_values.insert(term.clone(), self.compute_idf_for_term(term));
        }
        idf_values
    }

    /// 获取文档
    pub fn get_document(&self, doc_id: usize) -> Option<&Document> {
        self.documents.get(doc_id)
    }

    /// 获取所有文档
    pub fn get_documents(&self) -> &[Document] {
        &self.documents
    }

    /// 获取文档词频
    pub fn get_doc_term_freq(&self, doc_id: usize) -> Option<&HashMap<String, usize>> {
        self.doc_term_freqs.get(doc_id)
    }

    /// 获取文档长度
    pub fn get_doc_length(&self, doc_id: usize) -> usize {
        self.doc_lengths.get(doc_id).copied().unwrap_or(0)
    }

    /// 获取平均文档长度
    pub fn avgdl(&self) -> f64 {
        self.avgdl
    }

    /// 获取文档总数
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// 获取 BM25 参数
    pub fn params(&self) -> &BM25Params {
        &self.params
    }

    /// 清空索引
    pub fn clear(&mut self) {
        self.documents.clear();
        self.term_doc_freqs.clear();
        self.doc_term_freqs.clear();
        self.doc_lengths.clear();
        self.avgdl = 0.0;
        self.n_docs = 0;
        self.idf_cache.clear();
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_basic() {
        let mut index = BM25Index::new();

        let doc = Document::new("Rust programming");
        let terms = vec!["rust".to_string(), "programming".to_string()];

        index.add_document(doc, terms);

        assert_eq!(index.n_docs(), 1);
        assert_eq!(index.get_doc_length(0), 2);
    }

    #[test]
    fn test_index_idf() {
        let mut index = BM25Index::new();

        // 添加多个文档
        index.add_document(
            Document::new("Rust programming language"),
            vec![
                "rust".to_string(),
                "programming".to_string(),
                "language".to_string(),
            ],
        );

        index.add_document(
            Document::new("Python scripting language"),
            vec![
                "python".to_string(),
                "scripting".to_string(),
                "language".to_string(),
            ],
        );

        // "language" 出现在两个文档，IDF 应较低
        let idf_language = index.compute_idf_for_term("language");

        // "rust" 只出现在一个文档，IDF 应较高
        let idf_rust = index.compute_idf_for_term("rust");

        assert!(idf_rust > idf_language);
    }

    #[test]
    fn test_avgdl() {
        let mut index = BM25Index::new();

        index.add_document(Document::new("a"), vec!["a".to_string()]);
        index.add_document(
            Document::new("a b c"),
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );

        // 平均长度 = (1 + 3) / 2 = 2.0
        assert_eq!(index.avgdl(), 2.0);
    }
}
