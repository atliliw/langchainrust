// src/retrieval/unified_hybrid.rs
//! Unified Hybrid Index
//!
//! Manages BM25 + vector indexes together, auto-splitting documents and indexing into
//! both on a single add.

use lc_embeddings::Embeddings;
use lc_vector_stores::document_store::{ChunkedDocumentStore, ChunkedDocumentStoreTrait};
use lc_vector_stores::{Document, SearchResult, VectorStore, VectorStoreError};

use crate::bm25::{AutoMergingConfig, ChunkedBM25Retriever, ChunkedSearchResult};
use crate::hybrid::{reciprocal_rank_fusion, RetrievedDocument, RRF_K};
use crate::retriever::{RetrieverError, RetrieverTrait};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Unified hybrid index configuration
pub struct HybridIndexConfig {
    /// Document chunk size
    pub chunk_size: usize,
    /// Chunk overlap size
    pub chunk_overlap: usize,
    /// Number of BM25 retrieval results
    pub bm25_k: usize,
    /// Number of vector retrieval results
    pub vector_k: usize,
    /// RRF fusion parameter k
    pub rrf_k: usize,
    /// Threshold for merging leaf chunks into parent documents
    pub merge_threshold: f32,
    /// Minimum score threshold for vector retrieval (P1-2); default 0.0 keeps the old behavior.
    pub min_score: f32,
}

impl Default for HybridIndexConfig {
    fn default() -> Self {
        Self {
            chunk_size: 500,
            chunk_overlap: 50,
            bm25_k: 10,
            vector_k: 10,
            rrf_k: RRF_K,
            merge_threshold: 0.5,
            min_score: 0.0,
        }
    }
}

impl HybridIndexConfig {
    /// Creates a `HybridIndexConfig` with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the document chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Sets both the BM25 and vector retrieval result counts
    pub fn with_top_k(mut self, bm25_k: usize, vector_k: usize) -> Self {
        self.bm25_k = bm25_k;
        self.vector_k = vector_k;
        self
    }

    /// Sets the RRF fusion parameter k
    pub fn with_rrf_k(mut self, k: usize) -> Self {
        self.rrf_k = k;
        self
    }

    /// Sets the threshold for merging leaf chunks into parent documents
    pub fn with_merge_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    /// Sets the minimum score threshold for vector retrieval
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }
}

/// Hybrid search result (with detailed scores and rank information)
pub struct HybridSearchResult {
    /// The retrieved document
    pub document: Document,
    /// The RRF fusion score
    pub rrf_score: f64,
    /// The BM25 score (if present in the BM25 results)
    pub bm25_score: Option<f32>,
    /// The BM25 rank (if present in the BM25 results)
    pub bm25_rank: Option<usize>,
    /// The vector similarity score (if present in the vector results)
    pub vector_score: Option<f32>,
    /// The vector rank (if present in the vector results)
    pub vector_rank: Option<usize>,
    /// The ids of matched chunks
    pub matched_chunks: Vec<String>,
    /// The parent document id
    pub parent_id: Option<String>,
}

/// Unified hybrid index: manages BM25 + vector indexes together
pub struct UnifiedHybridIndex {
    document_store: Arc<ChunkedDocumentStore>,
    bm25_retriever: Arc<Mutex<ChunkedBM25Retriever>>,
    embeddings: Arc<dyn Embeddings>,
    /// P1-1: The vector index converges on `VectorStore` (the former self-held
    /// `Vec<VectorEntry>` brute-force scan is gone), reusing backends like
    /// InMemoryVectorStore / Qdrant.
    vector_store: Arc<dyn VectorStore>,
    /// Hybrid index configuration
    pub config: HybridIndexConfig,
}

impl UnifiedHybridIndex {
    /// Creates a new hybrid index with default configuration.
    ///
    /// `vector_store` is the vector-index backend (P1-1 converges on `VectorStore`, e.g.
    /// `InMemoryVectorStore` / `QdrantVectorStore`).
    /// `_vector_size` is retained for API compatibility (P1-7); the embedding
    /// dimension is derived from the `embeddings` backend itself, so it is no
    /// longer stored.
    pub fn new(
        embeddings: Arc<dyn Embeddings>,
        vector_store: Arc<dyn VectorStore>,
        _vector_size: usize,
    ) -> Self {
        Self::with_config(
            embeddings,
            vector_store,
            _vector_size,
            HybridIndexConfig::default(),
        )
    }

    /// Returns the underlying document store
    pub fn document_store(&self) -> Arc<ChunkedDocumentStore> {
        self.document_store.clone()
    }

    /// Creates a unified hybrid index with the given configuration
    pub fn with_config(
        embeddings: Arc<dyn Embeddings>,
        vector_store: Arc<dyn VectorStore>,
        _vector_size: usize,
        config: HybridIndexConfig,
    ) -> Self {
        let bm25_config = AutoMergingConfig::new()
            .with_leaf_size(config.chunk_size)
            .with_threshold(config.merge_threshold);

        let document_store = Arc::new(ChunkedDocumentStore::new());
        let bm25_retriever = ChunkedBM25Retriever::with_config(document_store.clone(), bm25_config);

        Self {
            document_store,
            bm25_retriever: Arc::new(Mutex::new(bm25_retriever)),
            embeddings,
            vector_store,
            config,
        }
    }

    /// Adds a single document: auto-chunks it and builds both the BM25 and vector indexes
    pub async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // P0-1: For a document without an id, attach the pre-allocated parent_id before
        // storing; otherwise the store generates a new uuid internally, making
        // get_chunks_for_parent look up the wrong key and return nothing.
        self.document_store
            .add_parent_document(
                document.clone().with_id(parent_id.clone()),
                self.config.chunk_size,
            )
            .await?;

        let chunks = self
            .document_store
            .get_chunks_for_parent(&parent_id)
            .await?;

        // P1-1: Build the BM25 index per chunk + vectorize, then write to vector_store in batch.
        // Chunks are stored by unique chunk_id (the InMemory backend overwrites by id,
        // avoiding id collisions among multiple chunks of the same parent).
        let mut chunk_docs = Vec::new();
        let mut chunk_embeddings = Vec::new();
        for chunk in &chunks {
            {
                let mut bm25 = self.bm25_retriever.lock().await;
                bm25.add_chunk_index(
                    chunk.chunk_id.clone(),
                    chunk.parent_id.clone(),
                    &chunk.content,
                );
            }

            let embedding = self
                .embeddings
                .embed_query(&chunk.content)
                .await
                .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

            chunk_docs.push(Document::new(chunk.content.clone()).with_id(chunk.chunk_id.clone()));
            chunk_embeddings.push(embedding);
        }

        if !chunk_docs.is_empty() {
            self.vector_store
                .add_documents(chunk_docs, chunk_embeddings)
                .await?;
        }

        Ok(parent_id)
    }

    /// Adds documents in batch, returning the id generated for each document
    pub async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError> {
        let mut ids = Vec::new();
        for doc in documents {
            let id = self.add_document(doc).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Hybrid retrieval: fuses BM25 and vector results, returning RRF-ranked documents
    pub async fn retrieve(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<RetrievedDocument>, VectorStoreError> {
        // H50: use config.bm25_k instead of hardcoded 10
        let bm25_k = self.config.bm25_k;
        let bm25_docs = {
            let mut bm25 = self.bm25_retriever.lock().await;
            bm25.search(query, bm25_k)
        };

        let bm25_docs: Vec<Document> = bm25_docs
            .into_iter()
            .map(|r: ChunkedSearchResult| Document::new(r.content()).with_id(r.parent_id))
            .collect();

        let vector_docs = self.vector_search(query).await?;

        let fused = reciprocal_rank_fusion(bm25_docs, vector_docs, self.config.rrf_k);

        Ok(fused.into_iter().take(k).collect())
    }

    /// Hybrid retrieval returning results with detailed scores and rank information
    pub async fn retrieve_with_details(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<HybridSearchResult>, VectorStoreError> {
        let bm25_k = self.config.bm25_k;
        let bm25_results = {
            let mut bm25 = self.bm25_retriever.lock().await;
            bm25.search(query, bm25_k)
        };

        let bm25_results: Vec<(Document, f32)> = bm25_results
            .into_iter()
            .map(|r| (Document::new(r.content()).with_id(r.parent_id), r.score))
            .collect();

        let vector_results = self.vector_search_with_scores(query).await?;

        let bm25_ranks: HashMap<String, usize> = bm25_results
            .iter()
            .enumerate()
            .map(|(rank, (doc, _))| (doc.id.clone().unwrap_or_default(), rank + 1))
            .collect();

        let vector_ranks: HashMap<String, usize> = vector_results
            .iter()
            .enumerate()
            .map(|(rank, (doc, _))| (doc.id.clone().unwrap_or_default(), rank + 1))
            .collect();

        let bm25_scores: HashMap<String, f32> = bm25_results
            .iter()
            .map(|(doc, score)| (doc.id.clone().unwrap_or_default(), *score))
            .collect();

        let vector_scores: HashMap<String, f32> = vector_results
            .iter()
            .map(|(doc, score)| (doc.id.clone().unwrap_or_default(), *score))
            .collect();

        let mut rrf_scores: HashMap<String, (f64, Document)> = HashMap::new();

        for (doc, _) in &bm25_results {
            let doc_id = doc.id.clone().unwrap_or_default();
            let rank = bm25_ranks.get(&doc_id).copied().unwrap_or(999);
            let contribution = 1.0 / (self.config.rrf_k as f64 + rank as f64);

            rrf_scores
                .entry(doc_id.clone())
                .and_modify(|(score, _)| *score += contribution)
                .or_insert((contribution, doc.clone()));
        }

        for (doc, _) in &vector_results {
            let doc_id = doc.id.clone().unwrap_or_default();
            let rank = vector_ranks.get(&doc_id).copied().unwrap_or(999);
            let contribution = 1.0 / (self.config.rrf_k as f64 + rank as f64);

            rrf_scores
                .entry(doc_id.clone())
                .and_modify(|(score, _)| *score += contribution)
                .or_insert((contribution, doc.clone()));
        }

        let mut results: Vec<(String, f64, Document)> = rrf_scores
            .into_iter()
            .map(|(id, (score, doc))| (id, score, doc))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let hybrid_results: Vec<HybridSearchResult> = results
            .into_iter()
            .take(k)
            .map(|(doc_id, rrf_score, document)| {
                HybridSearchResult {
                    document,
                    rrf_score,
                    bm25_score: bm25_scores.get(&doc_id).copied(),
                    bm25_rank: bm25_ranks.get(&doc_id).copied(),
                    vector_score: vector_scores.get(&doc_id).copied(),
                    vector_rank: vector_ranks.get(&doc_id).copied(),
                    matched_chunks: vec![doc_id.clone()],
                    // H49: use "::" separator consistent with chunk_id format
                    parent_id: Some(doc_id.split("::").next().unwrap_or_default().to_string()),
                }
            })
            .collect();

        Ok(hybrid_results)
    }

    async fn vector_search(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

        // P1-1: Delegates to vector_store.similarity_search_with_min_score — the
        // "filter by min_score first, then take top-k" semantics match the old
        // filter_by_score behavior of the self-held vector index. The vector backend
        // stores documents by chunk_id; look back into document_store for the parent_id
        // used in RRF aggregation.
        let results = self
            .vector_store
            .similarity_search_with_min_score(
                &query_embedding,
                self.config.vector_k,
                Some(self.config.min_score),
            )
            .await?;

        let mut docs = Vec::new();
        for r in results {
            let chunk_id = r.document.id.as_deref().unwrap_or_default();
            if let Some(chunk) = self.document_store.get_chunk(chunk_id).await? {
                docs.push(Document::new(chunk.content).with_id(chunk.parent_id));
            }
        }

        Ok(docs)
    }

    async fn vector_search_with_scores(
        &self,
        query: &str,
    ) -> Result<Vec<(Document, f32)>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

        // P1-1: Same as vector_search, delegates to vector_store and carries back f32 scores.
        let results = self
            .vector_store
            .similarity_search_with_min_score(
                &query_embedding,
                self.config.vector_k,
                Some(self.config.min_score),
            )
            .await?;

        let mut docs = Vec::new();
        for r in results {
            let chunk_id = r.document.id.as_deref().unwrap_or_default();
            if let Some(chunk) = self.document_store.get_chunk(chunk_id).await? {
                docs.push((
                    Document::new(chunk.content).with_id(chunk.parent_id),
                    r.score,
                ));
            }
        }

        Ok(docs)
    }

    /// Returns the number of indexed parent documents
    pub async fn document_count(&self) -> usize {
        self.document_store.parent_count().await
    }

    /// Returns the number of indexed chunks
    pub async fn chunk_count(&self) -> usize {
        self.document_store.chunk_count().await
    }

    /// Clears the BM25 index, vector index, and document store
    pub async fn clear(&self) -> Result<(), VectorStoreError> {
        ChunkedDocumentStoreTrait::clear(&*self.document_store).await?;

        {
            let mut bm25 = self.bm25_retriever.lock().await;
            bm25.clear();
        }

        self.vector_store.clear().await?;

        Ok(())
    }
}

/// P0-1: `UnifiedHybridIndex` implements `RetrieverTrait`.
///
/// The inherent `retrieve()` / `add_documents()` methods take precedence over the trait
/// methods during method resolution, so calling them directly does not recurse.
#[async_trait]
impl RetrieverTrait for UnifiedHybridIndex {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        let results = self.retrieve(query, k).await?;
        Ok(results.into_iter().map(|r| r.document).collect())
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        let results = self.retrieve(query, k).await?;
        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                document: r.document,
                // RetrievedDocument.score is f64, normalized to SearchResult's f32
                score: r.score as f32,
            })
            .collect())
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        self.add_documents(documents).await?;
        Ok(())
    }
}
