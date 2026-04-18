use super::algorithm::{bm25_score, BM25Params};
use super::index::BM25Index;
use super::tokenizer::Tokenizer;
use crate::vector_stores::{Document, SearchResult};

pub struct BM25Retriever {
    index: BM25Index,
    tokenizer: Tokenizer,
}

impl BM25Retriever {
    pub fn new() -> Self {
        Self {
            index: BM25Index::new(),
            tokenizer: Tokenizer::new(),
        }
    }

    pub fn with_params(k1: f64, b: f64) -> Self {
        Self {
            index: BM25Index::with_params(BM25Params::with_values(k1, b)),
            tokenizer: Tokenizer::new(),
        }
    }

    pub fn with_tokenizer(tokenizer: Tokenizer) -> Self {
        Self {
            index: BM25Index::new(),
            tokenizer,
        }
    }

    pub fn add_document(&mut self, document: Document) {
        let terms = self.tokenizer.tokenize(&document.content);
        self.index.add_document(document, terms);
    }

    pub fn add_documents_sync(&mut self, documents: Vec<Document>) {
        for doc in documents {
            self.add_document(doc);
        }
    }

    pub fn search(&mut self, query: &str, k: usize) -> Vec<SearchResult> {
        if self.index.n_docs() == 0 {
            return Vec::new();
        }

        let query_terms = self.tokenizer.tokenize(query);

        if query_terms.is_empty() {
            return Vec::new();
        }

        let idf_values = self.index.compute_idf_for_terms(&query_terms);

        let mut scored_docs: Vec<(usize, f64)> = Vec::new();

        for doc_id in 0..self.index.n_docs() {
            let doc_term_freqs = self.index.get_doc_term_freq(doc_id);
            let doc_length = self.index.get_doc_length(doc_id);
            let avgdl = self.index.avgdl();
            let params = self.index.params();

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

        scored_docs
            .into_iter()
            .take(k)
            .map(|(doc_id, score)| {
                let doc = self
                    .index
                    .get_document(doc_id)
                    .cloned()
                    .unwrap_or(Document::new(""));
                SearchResult {
                    document: doc,
                    score: score as f32,
                }
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.index.n_docs()
    }

    pub fn is_empty(&self) -> bool {
        self.index.n_docs() == 0
    }

    pub fn clear(&mut self) {
        self.index.clear();
    }

    pub fn index(&self) -> &BM25Index {
        &self.index
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
        let mut retriever = BM25Retriever::new();

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
        let mut retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust 是一门系统编程语言"),
            Document::new("Python 是脚本语言"),
            Document::new("JavaScript 用于网页开发"),
        ]);

        let results = retriever.search("编程语言", 2);
        assert!(results.len() > 0);

        assert!(results[0].document.content.contains("编程"));
    }

    #[test]
    fn test_bm25_retriever_empty() {
        let mut retriever = BM25Retriever::new();

        let results = retriever.search("test", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_retriever_params() {
        let mut retriever = BM25Retriever::with_params(2.0, 0.5);

        retriever.add_documents_sync(vec![
            Document::new("Rust programming"),
            Document::new("Python scripting"),
        ]);

        let results = retriever.search("programming", 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bm25_retriever_no_match() {
        let mut retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust programming language"),
            Document::new("Python scripting language"),
        ]);

        let results = retriever.search("javascript typescript", 5);
        assert!(results.is_empty());
    }
}
