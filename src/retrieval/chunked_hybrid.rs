// src/retrieval/chunked_hybrid.rs
//! Chunked Hybrid Retriever - BM25 + 向量混合检索器
//!
//! BM25 和向量检索共用同一个 DocumentStore，避免内容重复存储。

use crate::retrieval::bm25::ChunkedBM25Retriever;
use crate::retrieval::hybrid::{reciprocal_rank_fusion, RetrievedDocument, RRF_K};
use crate::vector_stores::{Document, VectorStoreError};
use crate::vector_stores::document_store::{ChunkedDocumentStore, ChunkedDocumentStoreTrait, ChunkDocument};
use crate::embeddings::Embeddings;
use std::sync::Arc;

pub struct ChunkedHybridRetriever {
    bm25_retriever: Arc<std::sync::Mutex<ChunkedBM25Retriever>>,
    document_store: Arc<ChunkedDocumentStore>,
    embeddings: Arc<dyn Embeddings>,
    bm25_k: usize,
    vector_k: usize,
    rrf_k: usize,
}

impl ChunkedHybridRetriever {
    pub fn new(
        bm25_retriever: ChunkedBM25Retriever,
        document_store: Arc<ChunkedDocumentStore>,
        embeddings: Arc<dyn Embeddings>,
    ) -> Self {
        Self {
            bm25_retriever: Arc::new(std::sync::Mutex::new(bm25_retriever)),
            document_store,
            embeddings,
            bm25_k: 10,
            vector_k: 10,
            rrf_k: RRF_K,
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
    
    pub async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievedDocument>, VectorStoreError> {
        let bm25_docs = self.bm25_search(query)?;
        
        let vector_docs = self.vector_search(query).await?;
        
        let fused = reciprocal_rank_fusion(bm25_docs, vector_docs, self.rrf_k);
        
        Ok(fused.into_iter().take(k).collect())
    }
    
    fn bm25_search(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
        let mut retriever = self.bm25_retriever.lock().unwrap();
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
        let query_embedding = self.embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
        
        let chunks: Vec<ChunkDocument> = self.document_store.get_all_chunks().await?;
        
        let mut scored: Vec<(Document, f32)> = Vec::new();
        
        for chunk in chunks {
            let embedding = self.embeddings
                .embed_query(&chunk.content)
                .await
                .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
            
            let score = crate::embeddings::cosine_similarity(&query_embedding, &embedding);
            
            if score > 0.0 {
                scored.push((chunk.to_document(), score));
            }
        }
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(scored.into_iter().take(self.vector_k).map(|(doc, _)| doc).collect())
    }
}
