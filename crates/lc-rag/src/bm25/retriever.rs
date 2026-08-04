use super::algorithm::{bm25_score, BM25Params};
use super::index::BM25Index;
use super::tokenizer::Tokenizer;
use lc_vector_stores::{Document, SearchResult};
use std::sync::Mutex;

pub struct BM25Retriever {
    index: Mutex<BM25Index>,
    tokenizer: Tokenizer,
}

impl BM25Retriever {
    pub fn new() -> Self {
        Self {
            index: Mutex::new(BM25Index::new()),
            tokenizer: Tokenizer::new(),
        }
    }

    pub fn with_params(k1: f64, b: f64) -> Self {
        Self {
            index: Mutex::new(BM25Index::with_params(BM25Params::with_values(k1, b))),
            tokenizer: Tokenizer::new(),
        }
    }

    pub fn with_tokenizer(tokenizer: Tokenizer) -> Self {
        Self {
            index: Mutex::new(BM25Index::new()),
            tokenizer,
        }
    }

    pub fn add_document(&self, document: Document) {
        let terms = self.tokenizer.tokenize(&document.content);
        let mut index = self.index.lock().unwrap_or_else(|e| e.into_inner());
        index.add_document(document, terms);
    }

    pub fn add_documents_sync(&self, documents: Vec<Document>) {
        for doc in documents {
            self.add_document(doc);
        }
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let mut index = self.index.lock().unwrap_or_else(|e| e.into_inner());

        if index.n_docs() == 0 {
            return Vec::new();
        }

        let query_terms = self.tokenizer.tokenize(query);

        if query_terms.is_empty() {
            return Vec::new();
        }

        let idf_values = index.compute_idf_for_terms(&query_terms);

        let mut scored_docs: Vec<(usize, f64)> = Vec::new();

        for doc_id in 0..index.n_docs() {
            let doc_term_freqs = index.get_doc_term_freq(doc_id);
            let doc_length = index.get_doc_length(doc_id);
            let avgdl = index.avgdl();
            let params = index.params();

            if let Some(term_freqs) = doc_term_freqs {
                let score = bm25_score(
                    &query_terms,
                    term_freqs,
                    doc_length,
                    avgdl,
                    &idf_values,
                    params,
                );

                if score > 0.0 {
                    scored_docs.push((doc_id, score));
                }
            }
        }

        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // M53: skip documents that are not found in the index instead of returning empty Document
        scored_docs
            .into_iter()
            .take(k)
            .filter_map(|(doc_id, score)| {
                let doc = index.get_document(doc_id).cloned()?;
                Some(SearchResult {
                    document: doc,
                    score: score as f32,
                })
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.index
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .n_docs()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.index.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    pub fn index(&self) -> std::sync::MutexGuard<'_, BM25Index> {
        self.index.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for BM25Retriever {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_retriever_basic() {
        let retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ]);

        assert_eq!(retriever.len(), 3);

        let results = retriever.search("programming language", 2);
        assert_eq!(results.len(), 2);

        assert!(results[0].document.content.contains("programming"));
    }

    #[test]
    fn test_bm25_retriever_chinese() {
        let retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust 是一门系统编程语言"),
            Document::new("Python 是脚本语言"),
            Document::new("JavaScript 用于网页开发"),
        ]);

        let results = retriever.search("编程语言", 2);
        assert!(!results.is_empty());

        assert!(results[0].document.content.contains("编程"));
    }

    #[test]
    fn test_bm25_retriever_empty() {
        let retriever = BM25Retriever::new();

        let results = retriever.search("test", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_retriever_params() {
        let retriever = BM25Retriever::with_params(2.0, 0.5);

        retriever.add_documents_sync(vec![
            Document::new("Rust programming"),
            Document::new("Python scripting"),
        ]);

        let results = retriever.search("programming", 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bm25_retriever_no_match() {
        let retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust programming language"),
            Document::new("Python scripting language"),
        ]);

        let results = retriever.search("javascript typescript", 5);
        assert!(results.is_empty());
    }
}
