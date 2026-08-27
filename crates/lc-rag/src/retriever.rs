// lc-rag/src/retriever.rs
//! Retriever implementations
//!
//! Provides similarity-based document retrieval.

use async_trait::async_trait;
use lc_embeddings::Embeddings;
use lc_vector_stores::{Document, SearchResult, VectorStore, VectorStoreError};
use std::sync::Arc;

/// Retriever error type
#[derive(Debug)]
#[non_exhaustive]
pub enum RetrieverError {
    /// Vector store error
    StoreError(VectorStoreError),

    /// Embedding error
    EmbeddingError(String),

    /// LLM breakdown failure (call failed / output unparseable). Used by SelfQuery (S4).
    LlmError(String),

    /// The filter references a field outside the `allowed_attributes` whitelist (SelfQuery).
    /// Errors out explicitly, never silently falls back to unfiltered retrieval — that would
    /// return data that should have been filtered out (data-plane over-exposure).
    InvalidFilter(String),

    /// No results
    NoResults,
}

impl std::fmt::Display for RetrieverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrieverError::StoreError(e) => write!(f, "storage error: {}", e),
            RetrieverError::EmbeddingError(msg) => write!(f, "embedding error: {}", msg),
            RetrieverError::LlmError(msg) => write!(f, "LLM error: {}", msg),
            RetrieverError::InvalidFilter(msg) => write!(f, "invalid filter: {}", msg),
            RetrieverError::NoResults => write!(f, "no relevant documents found"),
        }
    }
}

impl std::error::Error for RetrieverError {}

impl From<VectorStoreError> for RetrieverError {
    fn from(e: VectorStoreError) -> Self {
        RetrieverError::StoreError(e)
    }
}

/// Retriever trait
#[async_trait]
pub trait RetrieverTrait: Send + Sync {
    /// Retrieves relevant documents
    ///
    /// # Arguments
    /// * `query` - the query text
    /// * `k` - the number of documents to return
    ///
    /// # Returns
    /// The list of relevant documents
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError>;

    /// Retrieves relevant documents (with scores)
    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError>;

    /// Adds documents
    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError>;
}

/// Similarity-based retriever
pub struct SimilarityRetriever {
    /// Vector store
    store: Arc<dyn VectorStore>,

    /// Embedding model
    embeddings: Arc<dyn Embeddings>,
}

impl SimilarityRetriever {
    /// Creates a new similarity retriever
    pub fn new(store: Arc<dyn VectorStore>, embeddings: Arc<dyn Embeddings>) -> Self {
        Self { store, embeddings }
    }
}

#[async_trait]
impl RetrieverTrait for SimilarityRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        let results = self.retrieve_with_scores(query, k).await?;
        Ok(results.into_iter().map(|r| r.document).collect())
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        // Generate the query vector
        let query_embedding = self
            .embeddings
            .embed_query(query)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;

        // Retrieve similar documents
        let results = self.store.similarity_search(&query_embedding, k).await?;

        Ok(results)
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        // Generate document embeddings
        let texts: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self
            .embeddings
            .embed_documents(&texts)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;

        // Add to storage
        self.store.add_documents(documents, embeddings).await?;

        Ok(())
    }
}

/// A simplified Retriever type alias (for quick use)
pub type Retriever = SimilarityRetriever;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::BM25Retriever;
    use crate::unified_hybrid::UnifiedHybridIndex;
    use lc_embeddings::MockEmbeddings;
    use lc_vector_stores::InMemoryVectorStore;

    /// P0-1: Verifies both BM25 / UnifiedHybrid work as
    /// `Arc<dyn RetrieverTrait>`, completing the full add + retrieve flow.
    #[tokio::test]
    async fn test_retriever_trait_object_hybrid_retrievers() {
        // BM25Retriever as a trait object
        let bm25: Arc<dyn RetrieverTrait> = Arc::new(BM25Retriever::new());
        bm25.add_documents(vec![Document::new(
            "Rust is a systems programming language",
        )])
        .await
        .unwrap();
        let results = bm25.retrieve("systems", 1).await.unwrap();
        assert!(!results.is_empty());

        // UnifiedHybridIndex as a trait object
        let embeddings = Arc::new(MockEmbeddings::new(128));
        let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        let unified: Arc<dyn RetrieverTrait> = Arc::new(UnifiedHybridIndex::new(
            embeddings.clone(),
            vector_store,
            128,
        ));
        unified
            .add_documents(vec![Document::new(
                "Rust is a systems programming language",
            )])
            .await
            .unwrap();
        let results = unified.retrieve("systems", 1).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_retriever() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(128));

        let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());

        // Add documents
        let docs = vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ];

        retriever.add_documents(docs).await.unwrap();
        assert_eq!(store.count().await, 3);

        // Retrieve documents
        let results = retriever.retrieve("programming language", 2).await.unwrap();
        assert!(
            !results.is_empty(),
            "expected at least 1 result, got {}",
            results.len()
        );
    }

    #[tokio::test]
    async fn test_retriever_with_scores() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(64));

        let retriever = SimilarityRetriever::new(store, embeddings);

        let docs = vec![Document::new("Document A"), Document::new("Document B")];

        retriever.add_documents(docs).await.unwrap();

        let results = retriever.retrieve_with_scores("query", 2).await.unwrap();
        assert_eq!(results.len(), 2);

        // Results should include scores
        assert!(results[0].score >= -1.0 && results[0].score <= 1.0);
    }
}
