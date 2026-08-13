// src/retrieval/chunked_hybrid.rs
//! Chunked Hybrid Retriever - BM25 + 向量混合检索器
//!
//! BM25 和向量检索共用同一个 DocumentStore，避免内容重复存储。

use lc_core::math::cosine_similarity;
use lc_embeddings::Embeddings;
use lc_vector_stores::document_store::{
    ChunkDocument, ChunkedDocumentStore, ChunkedDocumentStoreTrait,
};
use lc_vector_stores::{Document, SearchResult, VectorStoreError};

use crate::bm25::ChunkedBM25Retriever;
use crate::hybrid::{filter_by_score, reciprocal_rank_fusion, RetrievedDocument, RRF_K};
use crate::retriever::{RetrieverError, RetrieverTrait};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Parent-Child 分块 + BM25 + 向量混合检索器（已废弃）
///
/// 功能已被 [`UnifiedHybridIndex`](crate::unified_hybrid::UnifiedHybridIndex)
/// 覆盖：后者复用 `ChunkedBM25Retriever` 并实现 `RetrieverTrait`，
/// 支持以 trait object 参与 `RAGPipeline`。
#[deprecated(
    note = "Use UnifiedHybridIndex instead (see crate::unified_hybrid::UnifiedHybridIndex)"
)]
pub struct ChunkedHybridRetriever {
    bm25_retriever: Arc<Mutex<ChunkedBM25Retriever>>,
    document_store: Arc<ChunkedDocumentStore>,
    embeddings: Arc<dyn Embeddings>,
    bm25_k: usize,
    vector_k: usize,
    rrf_k: usize,
    /// 向量检索最小分数阈值(P1-2),默认 0.0 保持旧行为。
    min_score: f32,
    /// Cache: chunk_id -> embedding, to avoid re-embedding on every query (H25)
    embedding_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

#[allow(deprecated)] // 已弃用类型的内部实现仍需引用自身字段
impl ChunkedHybridRetriever {
    pub fn new(
        bm25_retriever: ChunkedBM25Retriever,
        document_store: Arc<ChunkedDocumentStore>,
        embeddings: Arc<dyn Embeddings>,
    ) -> Self {
        Self {
            bm25_retriever: Arc::new(Mutex::new(bm25_retriever)),
            document_store,
            embeddings,
            bm25_k: 10,
            vector_k: 10,
            rrf_k: RRF_K,
            min_score: 0.0,
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
        }
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

    /// 设置向量检索最小分数阈值(P1-2),默认 0.0 保持旧行为。
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    pub async fn retrieve(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<RetrievedDocument>, VectorStoreError> {
        let bm25_docs = self.bm25_search(query).await?;

        let vector_docs = self.vector_search(query).await?;

        let fused = reciprocal_rank_fusion(bm25_docs, vector_docs, self.rrf_k);

        Ok(fused.into_iter().take(k).collect())
    }

    async fn bm25_search(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
        let mut retriever = self.bm25_retriever.lock().await;
        let results = retriever.search(query, self.bm25_k);

        let docs: Vec<Document> = results
            .into_iter()
            .map(|r| {
                let content = r.content();
                Document::new(content).with_id(r.parent_id)
            })
            .collect();

        Ok(docs)
    }

    async fn vector_search(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

        let chunks: Vec<ChunkDocument> = self.document_store.get_all_chunks().await?;

        // Ensure all chunks have cached embeddings (H25: cache instead of re-embedding every query)
        {
            let mut cache = self.embedding_cache.lock().await;
            for chunk in &chunks {
                if !cache.contains_key(&chunk.chunk_id) {
                    let embedding = self
                        .embeddings
                        .embed_query(&chunk.content)
                        .await
                        .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
                    cache.insert(chunk.chunk_id.clone(), embedding);
                }
            }
        }

        // Score using cached embeddings
        let cache = self.embedding_cache.lock().await;
        let mut scored: Vec<(Document, f32)> = Vec::new();

        for chunk in chunks {
            if let Some(embedding) = cache.get(&chunk.chunk_id) {
                let score = cosine_similarity(&query_embedding, embedding).unwrap_or(0.0);
                scored.push((chunk.to_document(), score));
            }
        }

        let mut scored = filter_by_score(scored, self.min_score);

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(self.vector_k)
            .map(|(doc, _)| doc)
            .collect())
    }
}

/// P0-1: `ChunkedHybridRetriever` 实现 `RetrieverTrait`。
///
/// `add_documents` 委托给内部 `ChunkedBM25Retriever`(切分 + BM25 索引),
/// embedding_cache 为惰性填充,新加入的 chunk 会在下次向量检索时补全缓存。
#[allow(deprecated)] // 已弃用类型的 trait 实现仍需引用自身字段
#[async_trait]
impl RetrieverTrait for ChunkedHybridRetriever {
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
                // RetrievedDocument.score 为 f64,统一收敛到 SearchResult 的 f32
                score: r.score as f32,
            })
            .collect())
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        let mut bm25 = self.bm25_retriever.lock().await;
        bm25.add_documents_async(documents).await?;
        Ok(())
    }
}
