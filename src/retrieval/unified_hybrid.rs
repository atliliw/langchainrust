// src/retrieval/unified_hybrid.rs
//! Unified Hybrid Index - 统一混合索引
//!
//! 统一管理 BM25 + 向量索引，自动分割文档，一次添加双索引。

use crate::retrieval::bm25::{ChunkedBM25Retriever, AutoMergingConfig, ChunkedSearchResult};
use crate::retrieval::hybrid::{reciprocal_rank_fusion, RetrievedDocument, RRF_K};
use crate::vector_stores::document_store::{ChunkedDocumentStore, ChunkedDocumentStoreTrait};
use crate::vector_stores::{Document, VectorStoreError};
use crate::embeddings::Embeddings;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct HybridIndexConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub bm25_k: usize,
    pub vector_k: usize,
    pub rrf_k: usize,
    pub merge_threshold: f32,
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
        }
    }
}

impl HybridIndexConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_top_k(mut self, bm25_k: usize, vector_k: usize) -> Self {
        self.bm25_k = bm25_k;
        self.vector_k = vector_k;
        self
    }

    pub fn with_rrf_k(mut self, k: usize) -> Self {
        self.rrf_k = k;
        self
    }

    pub fn with_merge_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }
}

pub struct HybridSearchResult {
    pub document: Document,
    pub rrf_score: f64,
    pub bm25_score: Option<f32>,
    pub bm25_rank: Option<usize>,
    pub vector_score: Option<f32>,
    pub vector_rank: Option<usize>,
    pub matched_chunks: Vec<String>,
    pub parent_id: Option<String>,
}

/// 向量索引条目（只存索引信息，内容回表 ChunkedDocumentStore）
struct VectorEntry {
    chunk_id: String,
    embedding: Vec<f32>,
    parent_id: String,
}

pub struct UnifiedHybridIndex {
    document_store: Arc<ChunkedDocumentStore>,
    bm25_retriever: Arc<std::sync::Mutex<ChunkedBM25Retriever>>,
    embeddings: Arc<dyn Embeddings>,
    #[allow(dead_code)]
    vector_size: usize,
    pub config: HybridIndexConfig,
    vector_index: Arc<RwLock<Vec<VectorEntry>>>,
}

impl UnifiedHybridIndex {
    pub fn new(embeddings: Arc<dyn Embeddings>, vector_size: usize) -> Self {
        Self::with_config(embeddings, vector_size, HybridIndexConfig::default())
    }

    pub fn document_store(&self) -> Arc<ChunkedDocumentStore> {
        self.document_store.clone()
    }

    pub fn with_config(
        embeddings: Arc<dyn Embeddings>,
        vector_size: usize,
        config: HybridIndexConfig,
    ) -> Self {
        let bm25_config = AutoMergingConfig::new()
            .with_leaf_size(config.chunk_size)
            .with_threshold(config.merge_threshold);

        let document_store = Arc::new(ChunkedDocumentStore::new());
        let bm25_retriever = ChunkedBM25Retriever::with_config(document_store.clone(), bm25_config);

        Self {
            document_store,
            bm25_retriever: Arc::new(std::sync::Mutex::new(bm25_retriever)),
            embeddings,
            vector_size,
            config,
            vector_index: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let parent_id = document.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        self.document_store.add_parent_document(document.clone(), self.config.chunk_size).await?;
        
        let chunks = self.document_store.get_chunks_for_parent(&parent_id).await?;

        for chunk in &chunks {
            {
                let mut bm25 = self.bm25_retriever.lock().unwrap();
                bm25.add_chunk_index(
                    chunk.chunk_id.clone(),
                    chunk.parent_id.clone(),
                    &chunk.content
                );
            }

            let embedding = self.embeddings
                .embed_query(&chunk.content)
                .await
                .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

            {
                let mut vectors = self.vector_index.write().await;
                vectors.push(VectorEntry {
                    chunk_id: chunk.chunk_id.clone(),
                    embedding,
                    parent_id: chunk.parent_id.clone(),
                });
            }
        }

        Ok(parent_id)
    }

    pub async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>, VectorStoreError> {
        let mut ids = Vec::new();
        for doc in documents {
            let id = self.add_document(doc).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    pub async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievedDocument>, VectorStoreError> {
        let bm25_docs = tokio::task::spawn_blocking({
            let retriever = self.bm25_retriever.clone();
            let query = query.to_string();
            move || {
                let mut bm25 = retriever.lock().unwrap();
                bm25.search(&query, 10)
            }
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        let bm25_docs: Vec<Document> = bm25_docs.into_iter().map(|r: ChunkedSearchResult| Document::new(r.content()).with_id(r.parent_id)).collect();
        
        let vector_docs = self.vector_search(query).await?;

        let fused = reciprocal_rank_fusion(bm25_docs, vector_docs, self.config.rrf_k);

        Ok(fused.into_iter().take(k).collect())
    }

    pub async fn retrieve_with_details(&self, query: &str, k: usize) -> Result<Vec<HybridSearchResult>, VectorStoreError> {
        let bm25_results = tokio::task::spawn_blocking({
            let retriever = self.bm25_retriever.clone();
            let query = query.to_string();
            let bm25_k = self.config.bm25_k;
            move || {
                let mut bm25 = retriever.lock().unwrap();
                bm25.search(&query, bm25_k)
            }
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        let bm25_results: Vec<(Document, f32)> = bm25_results
            .into_iter()
            .map(|r| (Document::new(r.content()).with_id(r.parent_id), r.score))
            .collect();
        
        let vector_results = self.vector_search_with_scores(query).await?;

        let bm25_ranks: HashMap<String, usize> = bm25_results
            .iter()
            .enumerate()
            .map(|(rank, (doc, _))| {
                (doc.id.clone().unwrap_or_default(), rank + 1)
            })
            .collect();

        let vector_ranks: HashMap<String, usize> = vector_results
            .iter()
            .enumerate()
            .map(|(rank, (doc, _))| {
                (doc.id.clone().unwrap_or_default(), rank + 1)
            })
            .collect();

        let bm25_scores: HashMap<String, f32> = bm25_results
            .iter()
            .map(|(doc, score)| {
                (doc.id.clone().unwrap_or_default(), score.clone())
            })
            .collect();

        let vector_scores: HashMap<String, f32> = vector_results
            .iter()
            .map(|(doc, score)| {
                (doc.id.clone().unwrap_or_default(), score.clone())
            })
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
                    parent_id: Some(doc_id.split('_').next().unwrap_or_default().to_string()),
                }
            })
            .collect();

        Ok(hybrid_results)
    }

    #[allow(dead_code)]
    fn bm25_search(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
        let mut retriever = self.bm25_retriever.lock().unwrap();
        let results = retriever.search(query, self.config.bm25_k);

        let docs = results
            .into_iter()
            .map(|r| Document::new(r.content()).with_id(r.parent_id))
            .collect();

        Ok(docs)
    }

    #[allow(dead_code)]
    fn bm25_search_with_scores(&self, query: &str) -> Result<Vec<(Document, f32)>, VectorStoreError> {
        let mut retriever = self.bm25_retriever.lock().unwrap();
        let results = retriever.search(query, self.config.bm25_k);

        let docs = results
            .into_iter()
            .map(|r| (Document::new(r.content()).with_id(r.parent_id), r.score))
            .collect();

        Ok(docs)
    }

    async fn vector_search(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
        let query_embedding = self.embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

        let vectors = self.vector_index.read().await;

        let mut scored: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let score = Self::cosine_similarity(&query_embedding, &entry.embedding);
                (idx, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_indices: Vec<(usize, f32)> = scored.into_iter().take(self.config.vector_k).collect();

        let mut docs = Vec::new();
        for (idx, _score) in top_k_indices {
            let entry = &vectors[idx];
            if let Some(chunk) = self.document_store.get_chunk(&entry.chunk_id).await? {
                docs.push(Document::new(chunk.content).with_id(entry.parent_id.clone()));
            }
        }

        Ok(docs)
    }

    async fn vector_search_with_scores(&self, query: &str) -> Result<Vec<(Document, f32)>, VectorStoreError> {
        let query_embedding = self.embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

        let vectors = self.vector_index.read().await;

        let mut scored: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let score = Self::cosine_similarity(&query_embedding, &entry.embedding);
                (idx, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_indices: Vec<(usize, f32)> = scored.into_iter().take(self.config.vector_k).collect();

        let mut docs = Vec::new();
        for (idx, score) in top_k_indices {
            let entry = &vectors[idx];
            if let Some(chunk) = self.document_store.get_chunk(&entry.chunk_id).await? {
                docs.push((Document::new(chunk.content).with_id(entry.parent_id.clone()), score));
            }
        }

        Ok(docs)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    pub async fn document_count(&self) -> usize {
        self.document_store.parent_count().await
    }

    pub async fn chunk_count(&self) -> usize {
        self.document_store.chunk_count().await
    }

    pub async fn clear(&self) -> Result<(), VectorStoreError> {
        ChunkedDocumentStoreTrait::clear(&*self.document_store).await?;

        {
            let mut bm25 = self.bm25_retriever.lock().unwrap();
            bm25.clear();
        }

        {
            let mut vectors = self.vector_index.write().await;
            vectors.clear();
        }

        Ok(())
    }
}