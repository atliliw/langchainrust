// lc-rag/src/bm25/retriever.rs
//! BM25 retriever
//!
//! Keyword-statistical retrieval (works for both Chinese and English, pure in-memory).
//! Since v0.15.0 this converges on `ChunkedBM25Index`: document text lives only in
//! lc-vector-stores' `ChunkedDocumentStoreTrait`, and retrieval results are aggregated by
//! parent (the old self-held `Vec<Document>` `BM25Index` is gone, eliminating the second
//! storage location).

use super::chunked::{ChunkedBM25Retriever, ChunkedSearchResult};
use crate::retriever::{RetrieverError, RetrieverTrait};
use async_trait::async_trait;
use lc_vector_stores::document_store::{ChunkedDocumentStore, ChunkedDocumentStoreTrait};
use lc_vector_stores::{Document, SearchResult, VectorStoreError};
use std::sync::{Arc, Mutex};

/// Keyword-statistical retriever.
///
/// P3-1: internally holds a `ChunkedBM25Retriever` (whose index is a `ChunkedBM25Index`),
/// sharing the same store storage location as `UnifiedHybridIndex` instead of holding the
/// document text itself.
pub struct BM25Retriever<S: ChunkedDocumentStoreTrait = ChunkedDocumentStore> {
    retriever: Mutex<ChunkedBM25Retriever<S>>,
}

impl BM25Retriever<ChunkedDocumentStore> {
    /// Creates a retriever (internally holds an in-memory store; no external arguments needed).
    pub fn new() -> Self {
        Self::with_store(Arc::new(ChunkedDocumentStore::new()))
    }

    /// Uses custom BM25 parameters (k1, b), internally holding an in-memory store.
    pub fn with_params(k1: f64, b: f64) -> Self {
        Self {
            retriever: Mutex::new(ChunkedBM25Retriever::with_params(
                Arc::new(ChunkedDocumentStore::new()),
                k1,
                b,
            )),
        }
    }
}

impl<S: ChunkedDocumentStoreTrait> BM25Retriever<S> {
    /// Creates with a shared store (sharing the document storage location with `UnifiedHybridIndex` and others).
    pub fn with_store(store: Arc<S>) -> Self {
        Self {
            retriever: Mutex::new(ChunkedBM25Retriever::new(store)),
        }
    }

    /// Adds a single document (chunks it, then indexes; the document text goes into the store).
    pub fn add_document(&self, document: Document) -> Result<(), VectorStoreError> {
        let mut retriever = self.retriever.lock().unwrap_or_else(|e| e.into_inner());
        retriever.add_document(document)
    }

    /// Adds documents in batch (sync; per-document failures are logged as errors and counted
    /// at the end, keeping the original `()` signature).
    pub fn add_documents_sync(&self, documents: Vec<Document>) {
        let total = documents.len();
        let mut failed = 0usize;
        for doc in documents {
            if let Err(e) = self.add_document(doc) {
                failed += 1;
                log::error!(
                    "BM25Retriever::add_documents_sync: failed to add document (it will be missing from retrieval results): {e}"
                );
            }
        }
        if failed > 0 {
            log::warn!(
                "BM25Retriever::add_documents_sync: failed to add {} of {} documents",
                failed,
                total
            );
        }
    }

    /// Keyword retrieval, returning parent-level results (the document id is the parent_id).
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let mut retriever = self.retriever.lock().unwrap_or_else(|e| e.into_inner());

        retriever
            .search(query, k)
            .into_iter()
            .map(|r: ChunkedSearchResult| SearchResult {
                document: Document::new(r.content()).with_id(r.parent_id),
                score: r.score,
            })
            .collect()
    }

    /// The number of indexed chunks (each document contributes at least one chunk).
    pub fn len(&self) -> usize {
        self.retriever
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Whether the retriever is empty (no documents indexed yet).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the BM25 index. With a shared store, the store's documents are cleared by the
    /// caller (`ChunkedDocumentStoreTrait::clear`), unrelated to this retriever.
    pub fn clear(&self) {
        self.retriever
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// P0-1: `BM25Retriever` implements `RetrieverTrait`, so it can be used uniformly with
/// other retrievers through `Arc<dyn RetrieverTrait>`.
#[async_trait]
impl<S: ChunkedDocumentStoreTrait> RetrieverTrait for BM25Retriever<S> {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        Ok(self
            .search(query, k)
            .into_iter()
            .map(|r| r.document)
            .collect())
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        Ok(self.search(query, k))
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        for doc in documents {
            self.add_document(doc).map_err(RetrieverError::StoreError)?;
        }
        Ok(())
    }
}

impl Default for BM25Retriever<ChunkedDocumentStore> {
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
