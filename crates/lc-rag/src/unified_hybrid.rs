// src/retrieval/unified_hybrid.rs
//! Unified Hybrid Index - 统一混合索引
//!
//! 统一管理 BM25 + 向量索引，自动分割文档，一次添加双索引。

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

/// 统一混合索引配置
pub struct HybridIndexConfig {
    /// 文档分块大小
    pub chunk_size: usize,
    /// 分块重叠大小
    pub chunk_overlap: usize,
    /// BM25 检索返回数量
    pub bm25_k: usize,
    /// 向量检索返回数量
    pub vector_k: usize,
    /// RRF 融合参数 k
    pub rrf_k: usize,
    /// 叶子块合并为父文档的阈值
    pub merge_threshold: f32,
    /// 向量检索最小分数阈值(P1-2),默认 0.0 保持旧行为。
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
    /// 创建使用默认配置的 `HybridIndexConfig`
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置文档分块大小
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// 同时设置 BM25 与向量检索返回数量
    pub fn with_top_k(mut self, bm25_k: usize, vector_k: usize) -> Self {
        self.bm25_k = bm25_k;
        self.vector_k = vector_k;
        self
    }

    /// 设置 RRF 融合参数 k
    pub fn with_rrf_k(mut self, k: usize) -> Self {
        self.rrf_k = k;
        self
    }

    /// 设置叶子块合并为父文档的阈值
    pub fn with_merge_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    /// 设置向量检索最小分数阈值
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }
}

/// 混合检索结果（带详细分数与排名信息）
pub struct HybridSearchResult {
    /// 检索到的文档
    pub document: Document,
    /// RRF 融合分数
    pub rrf_score: f64,
    /// BM25 分数（若在 BM25 结果中）
    pub bm25_score: Option<f32>,
    /// BM25 排名（若在 BM25 结果中）
    pub bm25_rank: Option<usize>,
    /// 向量相似度分数（若在向量结果中）
    pub vector_score: Option<f32>,
    /// 向量排名（若在向量结果中）
    pub vector_rank: Option<usize>,
    /// 命中的块 id 列表
    pub matched_chunks: Vec<String>,
    /// 所属父文档 id
    pub parent_id: Option<String>,
}

/// 统一混合索引：统一管理 BM25 + 向量索引
pub struct UnifiedHybridIndex {
    document_store: Arc<ChunkedDocumentStore>,
    bm25_retriever: Arc<Mutex<ChunkedBM25Retriever>>,
    embeddings: Arc<dyn Embeddings>,
    /// P1-1: 向量索引收敛到 `VectorStore`(原自持 `Vec<VectorEntry>` 暴力遍历
    /// 已删除),可复用 InMemoryVectorStore / Qdrant 等后端。
    vector_store: Arc<dyn VectorStore>,
    /// 混合索引配置
    pub config: HybridIndexConfig,
}

impl UnifiedHybridIndex {
    /// Creates a new hybrid index with default configuration.
    ///
    /// `vector_store` 是向量索引后端(P1-1 收敛到 `VectorStore`,如
    /// `InMemoryVectorStore` / `QdrantVectorStore`)。
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

    /// 获取底层的文档存储
    pub fn document_store(&self) -> Arc<ChunkedDocumentStore> {
        self.document_store.clone()
    }

    /// 使用指定配置创建统一混合索引
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

    /// 添加单个文档：自动分块并同时建立 BM25 与向量索引
    pub async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // P0-1: 无 id 的文档先把预分配的 parent_id 挂到文档上再入库,
        // 否则 store 内部会再生成一个新 uuid,导致 get_chunks_for_parent 用错 key 查空。
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

        // P1-1: 每块建 BM25 索引 + 向量化,批量写入 vector_store。
        // chunk 用唯一 chunk_id 入库(InMemory 后端按 id 覆盖,避免同 parent 多块撞 id)。
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

    /// 批量添加文档，返回每个文档生成的 id
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

    /// 混合检索：融合 BM25 与向量结果后返回 RRF 排序文档
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

    /// 混合检索并返回带详细分数与排名信息的结果
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

        // P1-1: 委托 vector_store.similarity_search_with_min_score —— "先按 min_score
        // 过滤、再取 top-k" 语义,与旧自持向量索引的 filter_by_score 行为一致。
        // 向量后端以 chunk_id 存文档,回表 document_store 取 parent_id 供 RRF 聚合。
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

        // P1-1: 同 vector_search,委托 vector_store 并带回 f32 分数。
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

    /// 返回已索引的父文档数量
    pub async fn document_count(&self) -> usize {
        self.document_store.parent_count().await
    }

    /// 返回已索引的块数量
    pub async fn chunk_count(&self) -> usize {
        self.document_store.chunk_count().await
    }

    /// 清空 BM25、向量索引与文档存储
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

/// P0-1: `UnifiedHybridIndex` 实现 `RetrieverTrait`。
///
/// 内部的 `retrieve()` / `add_documents()`(inherent 方法)在方法解析时优先于
/// trait 方法,故直接调用即可,不会产生递归。
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
                // RetrievedDocument.score 为 f64,统一收敛到 SearchResult 的 f32
                score: r.score as f32,
            })
            .collect())
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        self.add_documents(documents).await?;
        Ok(())
    }
}
