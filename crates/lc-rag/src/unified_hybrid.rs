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
use crate::mmr::mmr as select_mmr;
use crate::retriever::{RetrieverError, RetrieverTrait};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Hybrid 融合策略。
///
/// 默认 `Rrf` 只按排名融合,不读绝对分值;`Weighted` 把 BM25 与向量的
/// 原始分数各自 min-max 归一化到 \[0,1\] 后加权线性相加(RRF 保持默认)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMode {
    /// Reciprocal Rank Fusion:按排名合成,分数只作 tie-break+展示。
    Rrf,
    /// 加权线性融合:`w_bm25·norm_bm25 + w_vector·norm_vector`。
    Weighted {
        /// BM25 归一化分数权重
        bm25_weight: f32,
        /// 向量相似度归一化分数权重
        vector_weight: f32,
    },
}

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
    /// RRF fusion parameter k (Rrf mode only)
    pub rrf_k: usize,
    /// Threshold for merging leaf chunks into parent documents
    pub merge_threshold: f32,
    /// Minimum score threshold for vector retrieval (P1-2); default 0.0 keeps the old behavior.
    pub min_score: f32,
    /// 融合策略;默认 RRF,换 `Weighted` 需同时给权重。见 [`FusionMode`]。
    pub fusion: FusionMode,
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
            fusion: FusionMode::Rrf,
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

    /// Sets the fusion strategy. RRF is the default; switching to `Weighted`
    /// blends min-max-normalized BM25/vector scores with the given weights.
    pub fn with_fusion(mut self, fusion: FusionMode) -> Self {
        self.fusion = fusion;
        self
    }
}

/// 把一腿的原始分数(可正可负可零)线性归一化到 \[0,1\]。空表返回空;若
/// max == min(单元素或全同值)该腿所有出现项都记为 1.0——避免归一化把
/// 唯一一项压成 0 而让加权融合中这一腿彻底失声。
fn min_max_normalize(scores: &HashMap<String, f32>) -> HashMap<String, f64> {
    if scores.is_empty() {
        return HashMap::new();
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in scores.values() {
        min = min.min(*v);
        max = max.max(*v);
    }
    if max - min <= f32::EPSILON {
        return scores.keys().map(|k| (k.clone(), 1.0)).collect();
    }
    scores
        .iter()
        .map(|(k, v)| (k.clone(), ((v - min) / (max - min)) as f64))
        .collect()
}

/// Hybrid search result (with detailed scores and rank information)
#[derive(Debug, Clone)]
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
    ///
    /// 0.22.0 C5 fix: re-adding the same document id is **idempotent** — the
    /// stale chunk set is removed from the vector store before the fresh
    /// chunks are written (chunk ids are deterministic, and BM25 already
    /// overwrites by chunk id). Previously a duplicate ingest left parallel
    /// stale vectors that crowded out top-k.
    pub async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // C5: capture the stale chunk ids BEFORE the store replaces the
        // chunk set, then best-effort delete their vectors.
        let stale_chunk_ids = self
            .document_store
            .get_chunks_for_parent(&parent_id)
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.chunk_id)
            .collect::<Vec<_>>();

        // P0-1: For a document without an id, attach the pre-allocated parent_id before
        // storing; otherwise the store generates a new uuid internally, making
        // get_chunks_for_parent look up the wrong key and return nothing.
        self.document_store
            .add_parent_document(
                document.clone().with_id(parent_id.clone()),
                self.config.chunk_size,
            )
            .await?;

        // C5: remove the stale vectors (chunk ids are deterministic, so the
        // fresh upsert would otherwise leave the old duplicates in place on
        // vector-store backends that append rather than overwrite by id).
        for chunk_id in &stale_chunk_ids {
            let _ = self.vector_store.delete_document(chunk_id).await;
        }

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

            // Index documents with `embed_documents`, not `embed_query`: for
            // dual-encoder backends the query vector space and the document
            // vector space differ, so storing documents in the query space
            // silently breaks retrieval. (A6)
            let embedding = self
                .embeddings
                .embed_documents(&[chunk.content.as_str()])
                .await
                .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    VectorStoreError::EmbeddingError("embed_documents returned no vector".into())
                })?;

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

        // 融合分:按 config.fusion 选 RRF(默认,只排名)或加权线性(各腿分数
        // 各自 min-max 归一化后按权重相加)。
        let fused_scores: HashMap<String, (f64, Document)> = match self.config.fusion {
            FusionMode::Rrf => {
                let mut scores: HashMap<String, (f64, Document)> = HashMap::new();
                for (doc, _) in &bm25_results {
                    let doc_id = doc.id.clone().unwrap_or_default();
                    let rank = bm25_ranks.get(&doc_id).copied().unwrap_or(999);
                    let contribution = 1.0 / (self.config.rrf_k as f64 + rank as f64);
                    scores
                        .entry(doc_id.clone())
                        .and_modify(|(s, _)| *s += contribution)
                        .or_insert((contribution, doc.clone()));
                }
                for (doc, _) in &vector_results {
                    let doc_id = doc.id.clone().unwrap_or_default();
                    let rank = vector_ranks.get(&doc_id).copied().unwrap_or(999);
                    let contribution = 1.0 / (self.config.rrf_k as f64 + rank as f64);
                    scores
                        .entry(doc_id.clone())
                        .and_modify(|(s, _)| *s += contribution)
                        .or_insert((contribution, doc.clone()));
                }
                scores
            }
            FusionMode::Weighted {
                bm25_weight,
                vector_weight,
            } => {
                let norm_bm25 = min_max_normalize(&bm25_scores);
                let norm_vector = min_max_normalize(&vector_scores);
                let mut scores: HashMap<String, (f64, Document)> = HashMap::new();
                for (doc, _) in bm25_results.iter().chain(vector_results.iter()) {
                    let doc_id = doc.id.clone().unwrap_or_default();
                    // 并集权重:某腿没命中该 id 时该腿记 0(只由另一腿贡献)。
                    let combined = bm25_weight as f64
                        * norm_bm25.get(&doc_id).copied().unwrap_or(0.0)
                        + vector_weight as f64 * norm_vector.get(&doc_id).copied().unwrap_or(0.0);
                    scores
                        .entry(doc_id.clone())
                        .or_insert_with(|| (combined, doc.clone()));
                }
                scores
            }
        };

        let mut results: Vec<(String, f64, Document)> = fused_scores
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
                    // A7: `doc_id` is already the authoritative parent_id — both
                    // bm25 results (`with_id(r.parent_id)`) and vector results
                    // (`with_id(chunk.parent_id)`) set it from `document_store`.
                    // Slicing it on "::" would corrupt any parent_id itself
                    // containing the separator.
                    parent_id: Some(doc_id.clone()),
                }
            })
            .collect();

        Ok(hybrid_results)
    }

    /// MMR 多样性重排:先把 BM25+向量按当前融合策略融合出 `cand_k` 个候选,
    /// 再对候选内容用文档编码器取向量,按 `lambda` 在「相关(RRF/加权分)」
    /// 与「和已选集合最不相似」之间贪心重排,返回重排后至多 `k` 个结果。
    ///
    /// - `cand_k` 应大于 `k`,给去重留出候选池。
    /// - `lambda ∈ \[0,1\]`:1=纯相关(即退换为融合排序),0=纯去重。
    /// - 融合分在候选池内 min-max 归一化到 \[0,1\] 后再进 MMR:RRF 分是
    ///   ~0.01 量级的倒数排名和、Weighted 分已在 \[0,1\],不归一化 λ 的
    ///   「相关/多样」天平在两种融合策略下语义不一致(池内全等分时记 1.0)。
    /// - 这是显式后处理,会为 `cand_k` 个候选额外调用一次文档编码器。
    pub async fn retrieve_mmr(
        &self,
        query: &str,
        cand_k: usize,
        k: usize,
        lambda: f32,
    ) -> Result<Vec<HybridSearchResult>, VectorStoreError> {
        let pool = self.retrieve_with_details(query, cand_k).await?;
        if pool.is_empty() {
            return Ok(Vec::new());
        }

        // 只为候选池取向量:文档编码器批量一次,供 MMR 算两两余弦相似。
        let contents: Vec<&str> = pool.iter().map(|r| r.document.content.as_str()).collect();
        let vectors = self
            .embeddings
            .embed_documents(&contents)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;

        // 相关分池内归一化,与 mmr.rs 手算单测(余弦 ∈ [-1,1]、相关 ∈ [0,1])
        // 保持同一量纲。
        let mut rel_min = f64::INFINITY;
        let mut rel_max = f64::NEG_INFINITY;
        for r in &pool {
            rel_min = rel_min.min(r.rrf_score);
            rel_max = rel_max.max(r.rrf_score);
        }
        let rel_span = rel_max - rel_min;
        let normalize = |score: f64| {
            if rel_span <= f64::EPSILON {
                1.0
            } else {
                (score - rel_min) / rel_span
            }
        };

        let ranked: Vec<(String, f64, Vec<f32>)> = pool
            .iter()
            .zip(vectors)
            .map(|(r, v)| {
                (
                    r.document.id.clone().unwrap_or_default(),
                    normalize(r.rrf_score),
                    v,
                )
            })
            .collect();

        let order = select_mmr(&ranked, lambda, k);

        let by_id: HashMap<String, HybridSearchResult> = pool
            .into_iter()
            .map(|r| (r.document.id.clone().unwrap_or_default(), r))
            .collect();

        Ok(order
            .into_iter()
            .filter_map(|id| by_id.get(&id).cloned())
            .collect())
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

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::{l2_normalize, EmbeddingError};
    use lc_vector_stores::InMemoryVectorStore;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic word -> bucket hash for the toy dual encoder below.
    fn bucket_of(word: &str, dim: usize) -> usize {
        word.bytes().fold(0usize, |acc, b| {
            acc.wrapping_add((b as usize).wrapping_mul(31))
        }) % dim
    }

    /// Toy **dual encoder**: the document encoder and the query encoder are
    /// asymmetric, mirroring models whose query and document vector spaces
    /// differ in real deployment.
    ///
    /// - `embed_documents(T)` multi-hot-encodes *every* word of T (a document
    ///   representation);
    /// - `embed_query(T)` encodes only the *first* word of T (a query
    ///   representation).
    ///
    /// Both land in the same ambient space and share the word→bucket mapping,
    /// so a one-word query matches a document containing that word **only when
    /// the document was indexed through `embed_documents`**. If the index
    /// mistakenly embeds chunks through `embed_query` (the A6 bug), the stored
    /// vectors contain just each chunk's first word and the query vector has
    /// zero cosine similarity with them — vector retrieval silently returns
    /// nothing. Call counters additionally pin which method each path uses.
    struct DualEncoderMock {
        dim: usize,
        embed_document_calls: AtomicUsize,
        embed_query_calls: AtomicUsize,
    }

    impl DualEncoderMock {
        fn new(dim: usize) -> Arc<Self> {
            Arc::new(Self {
                dim,
                embed_document_calls: AtomicUsize::new(0),
                embed_query_calls: AtomicUsize::new(0),
            })
        }

        fn document_embedding(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0f32; self.dim];
            for word in text.split_whitespace() {
                v[bucket_of(word, self.dim)] = 1.0;
            }
            l2_normalize(&mut v);
            v
        }

        fn query_embedding(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0f32; self.dim];
            if let Some(first) = text.split_whitespace().next() {
                v[bucket_of(first, self.dim)] = 1.0;
            }
            l2_normalize(&mut v);
            v
        }
    }

    #[async_trait]
    impl Embeddings for DualEncoderMock {
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            if text.trim().is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }
            self.embed_query_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.query_embedding(text))
        }

        async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if texts.iter().any(|t| t.trim().is_empty()) {
                return Err(EmbeddingError::EmptyInput);
            }
            // One call per indexed chunk (the index batches one chunk per call).
            self.embed_document_calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts.iter().map(|t| self.document_embedding(t)).collect())
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn model_name(&self) -> &str {
            "dual-encoder-mock"
        }
    }

    fn small_index(embeddings: Arc<dyn Embeddings>) -> UnifiedHybridIndex {
        let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        let config = HybridIndexConfig::new()
            .with_chunk_size(80)
            .with_top_k(5, 5);
        UnifiedHybridIndex::with_config(embeddings, vector_store, 32, config)
    }

    /// A6: indexing goes through `embed_documents` (once per chunk) and
    /// retrieval through `embed_query`; with a genuinely asymmetric dual
    /// encoder the matching document is still returned by the *vector* path.
    #[tokio::test]
    async fn indexing_uses_embed_documents_and_retrieval_embed_query() {
        let mock = DualEncoderMock::new(32);
        let embeddings: Arc<dyn Embeddings> = mock.clone();
        let index = small_index(embeddings);

        // ~10 chunks at chunk_size 80, and every chunk contains "zebra".
        let doc_text = std::iter::repeat_n(
            "zebra rust is a systems programming language that runs blazingly fast",
            8,
        )
        .collect::<Vec<_>>()
        .join(" . ");
        let parent = index
            .add_document(Document::new(doc_text).with_id("doc-zebra"))
            .await
            .unwrap();
        assert_eq!(parent, "doc-zebra");

        let chunk_count = index.chunk_count().await;
        assert!(
            chunk_count >= 5,
            "expected several chunks, got {chunk_count}"
        );
        assert_eq!(
            mock.embed_document_calls.load(Ordering::SeqCst),
            chunk_count,
            "each chunk must be indexed via one embed_documents call"
        );
        assert_eq!(
            mock.embed_query_calls.load(Ordering::SeqCst),
            0,
            "indexing must never call embed_query"
        );

        // A distractor without the query word; RRF must rank the zebra doc on top.
        index
            .add_document(
                Document::new(
                    "python is a scripting language used for glue code and automation tasks",
                )
                .with_id("doc-python"),
            )
            .await
            .unwrap();

        let results = index.retrieve_with_details("zebra", 3).await.unwrap();
        assert_eq!(
            mock.embed_query_calls.load(Ordering::SeqCst),
            1,
            "retrieval must embed the query exactly once"
        );
        assert!(!results.is_empty(), "expected hybrid results");

        let top = &results[0];
        assert_eq!(top.document.id.as_deref(), Some("doc-zebra"));
        // The decisive A6 assertion: the vector leg contributed. Under the old
        // embed_query-indexing path the query vector was orthogonal to every
        // stored chunk vector, so vector_score/vector_rank would be None.
        assert!(
            top.vector_rank.is_some(),
            "vector retrieval must match the indexed document (vector_rank was None)"
        );
        assert!(
            !results
                .iter()
                .any(|r| r.document.id.as_deref() == Some("doc-python")),
            "distractor without the query word must not be retrieved"
        );
    }

    /// A7: a parent id containing the internal `::` separator must survive
    /// chunk id derivation (`{parent}::{segment}`) and come back verbatim from
    /// `retrieve_with_details` — the old `split("::")` reconstruction mangled
    /// it into the first segment ("ns").
    #[tokio::test]
    async fn parent_id_containing_separator_round_trips_intact() {
        let mock = DualEncoderMock::new(32);
        let embeddings: Arc<dyn Embeddings> = mock.clone();
        let index = small_index(embeddings);

        const PARENT_ID: &str = "ns::parent::id";
        let doc_text = std::iter::repeat_n(
            "zebra migration patterns follow seasonal rain across the savanna plains",
            8,
        )
        .collect::<Vec<_>>()
        .join(" . ");
        let returned = index
            .add_document(Document::new(doc_text).with_id(PARENT_ID))
            .await
            .unwrap();
        assert_eq!(returned, PARENT_ID);

        // Chunk metadata carries the full parent id and chunk ids are unique.
        let chunks = index
            .document_store()
            .get_chunks_for_parent(PARENT_ID)
            .await
            .unwrap();
        assert!(chunks.len() >= 2, "expected multiple chunks");
        let mut ids: Vec<&str> = chunks.iter().map(|c| c.chunk_id.as_str()).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "chunk ids must not collide");
        assert!(
            chunks.iter().all(|c| c.parent_id == PARENT_ID),
            "every chunk must point at the full parent id"
        );
        assert!(
            chunks
                .iter()
                .all(|c| c.chunk_id.starts_with(&format!("{PARENT_ID}::"))),
            "chunk ids keep the parent id as an exact prefix"
        );

        let results = index.retrieve_with_details("zebra", 5).await.unwrap();
        let hit = results
            .iter()
            .find(|r| r.parent_id.as_deref() == Some(PARENT_ID))
            .expect("result must carry the full '::'-containing parent_id");
        assert_eq!(hit.document.id.as_deref(), Some(PARENT_ID));
        assert!(!hit.matched_chunks.is_empty());

        // And no result must surface the mangled first-segment form.
        assert!(
            !results.iter().any(|r| r.parent_id.as_deref() == Some("ns")),
            "parent_id must not be reconstructed by splitting on '::'"
        );
    }

    /// Finds a toy-hash **collision token**: a different BM25 token that lands
    /// in the same embedding bucket as `word`. The vector leg then matches the
    /// collider while the lexical leg (different string) cannot — this is what
    /// lets the weighted-fusion tests isolate each leg deterministically.
    fn collision_token(word: &str, dim: usize) -> String {
        let target = bucket_of(word, dim);
        (0..10_000)
            .map(|i| format!("col{i}"))
            .find(|cand| cand != word && bucket_of(cand, dim) == target)
            .expect("a collision token exists within the scan range")
    }

    /// Picks `n` tokens whose toy-hash buckets are pairwise distinct and all
    /// different from `avoid`'s bucket, so the hand-computed cosine geometry in
    /// the MMR tests is exact (no accidental bucket overlaps).
    fn distinct_bucket_tokens(n: usize, avoid: &str, dim: usize) -> Vec<String> {
        let mut used = std::collections::HashSet::new();
        used.insert(bucket_of(avoid, dim));
        let mut out = Vec::new();
        let mut i = 0;
        while out.len() < n {
            let cand = format!("tk{i}");
            if used.insert(bucket_of(&cand, dim)) {
                out.push(cand);
            }
            i += 1;
        }
        out
    }

    /// T11: min-max normalization edges — empty map, a constant leg (every
    /// occurrence scores 1.0 so the weighted leg is not silenced), and the
    /// regular 0/1 mapping.
    #[test]
    fn min_max_normalize_edges() {
        assert!(min_max_normalize(&HashMap::new()).is_empty());

        let constant: HashMap<String, f32> = [("a".to_string(), 3.0), ("b".to_string(), 3.0)]
            .into_iter()
            .collect();
        let norm = min_max_normalize(&constant);
        assert_eq!(norm.get("a"), Some(&1.0));
        assert_eq!(norm.get("b"), Some(&1.0));

        let spread: HashMap<String, f32> = [
            ("a".to_string(), 0.0f32),
            ("b".to_string(), 2.0),
            ("c".to_string(), 1.0),
        ]
        .into_iter()
        .collect();
        let norm = min_max_normalize(&spread);
        assert_eq!(norm.get("a"), Some(&0.0));
        assert_eq!(norm.get("b"), Some(&1.0));
        assert_eq!(norm.get("c"), Some(&0.5));
    }

    /// Builds a single-chunk-per-doc index with the given fusion mode.
    async fn weighted_index(
        embeddings: Arc<dyn Embeddings>,
        fusion: FusionMode,
        docs: &[(&str, &str)],
    ) -> UnifiedHybridIndex {
        let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        let config = HybridIndexConfig::new()
            .with_chunk_size(80)
            .with_top_k(5, 5)
            .with_fusion(fusion);
        let index = UnifiedHybridIndex::with_config(embeddings, vector_store, 32, config);
        for (id, content) in docs {
            index
                .add_document(Document::new(*content).with_id(*id))
                .await
                .unwrap();
        }
        index
    }

    /// T11: weighted linear fusion — the two weights decide which leg wins.
    ///
    /// Geometry (dim 32):
    /// - `lex`: contains "zebra" → BM25 hit. Its vector carries zebra plus 7
    ///   distinct other buckets, so the query cosine is diluted to 1/√8.
    /// - `vec`: a single zebra-bucket **collision token** (different string) →
    ///   invisible to BM25, but its document vector is the zebra unit vector,
    ///   cosine 1.0.
    ///
    /// So (bm25=1, vector=0) must rank `lex` first; the reverse weighting must
    /// rank `vec` first. Default RRF is unaffected.
    #[tokio::test]
    async fn weighted_fusion_weights_control_which_leg_wins() {
        let mock = DualEncoderMock::new(32);
        let embeddings: Arc<dyn Embeddings> = mock.clone();
        let collider = collision_token("zebra", 32);
        let filler = distinct_bucket_tokens(7, "zebra", 32);
        let lex_content = std::iter::once("zebra".to_string())
            .chain(filler)
            .collect::<Vec<_>>()
            .join(" ");
        let docs = [("lex", lex_content.as_str()), ("vec", collider.as_str())];

        let lexical = weighted_index(
            embeddings.clone(),
            FusionMode::Weighted {
                bm25_weight: 1.0,
                vector_weight: 0.0,
            },
            &docs,
        )
        .await;
        let results = lexical.retrieve_with_details("zebra", 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id.as_deref(), Some("lex"));
        assert_eq!(results[0].bm25_rank, Some(1));

        let vectorial = weighted_index(
            embeddings,
            FusionMode::Weighted {
                bm25_weight: 0.0,
                vector_weight: 1.0,
            },
            &docs,
        )
        .await;
        let results = vectorial.retrieve_with_details("zebra", 2).await.unwrap();
        assert_eq!(results[0].document.id.as_deref(), Some("vec"));
        assert!(
            results[0].bm25_score.is_none(),
            "vec never hits the BM25 leg"
        );
    }

    /// T11 MMR geometry: d2 is a near-duplicate of d1 (4 shared filler tokens),
    /// d3 shares only "zebra" with them. d1 tops both legs (highest query cosine
    /// 1/√5 and shortest doc for BM25 length normalization).
    async fn mmr_index(embeddings: Arc<dyn Embeddings>, fusion: FusionMode) -> UnifiedHybridIndex {
        let toks = distinct_bucket_tokens(11, "zebra", 32);
        let d1 = std::iter::once("zebra".to_string())
            .chain(toks[0..4].iter().cloned())
            .collect::<Vec<_>>()
            .join(" ");
        let d2 = std::iter::once("zebra".to_string())
            .chain(toks[0..5].iter().cloned())
            .collect::<Vec<_>>()
            .join(" ");
        let d3 = std::iter::once("zebra".to_string())
            .chain(toks[5..11].iter().cloned())
            .collect::<Vec<_>>()
            .join(" ");
        weighted_index(
            embeddings,
            fusion,
            &[
                ("d1", d1.as_str()),
                ("d2", d2.as_str()),
                ("d3", d3.as_str()),
            ],
        )
        .await
    }

    /// T11: λ=1 means pure relevance — MMR must return the fusion ranking
    /// verbatim (pool-internal normalization is monotonic).
    #[tokio::test]
    async fn mmr_lambda_one_preserves_fusion_order() {
        let mock = DualEncoderMock::new(32);
        let embeddings: Arc<dyn Embeddings> = mock.clone();
        let index = mmr_index(embeddings, FusionMode::Rrf).await;

        let fused: Vec<String> = index
            .retrieve_with_details("zebra", 3)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.document.id.unwrap_or_default())
            .collect();

        let mmr_order: Vec<String> = index
            .retrieve_mmr("zebra", 3, 3, 1.0)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.document.id.unwrap_or_default())
            .collect();

        assert_eq!(mmr_order, fused);
        assert_eq!(
            mmr_order.iter().collect::<std::collections::HashSet<_>>(),
            std::collections::HashSet::from([
                &"d1".to_string(),
                &"d2".to_string(),
                &"d3".to_string()
            ])
        );
    }

    /// T11: λ=0 means pure diversity — after d1 the near-duplicate d2 must be
    /// skipped in favour of the dissimilar d3.
    #[tokio::test]
    async fn mmr_lambda_zero_jumps_to_dissimilar_candidate() {
        let mock = DualEncoderMock::new(32);
        let embeddings: Arc<dyn Embeddings> = mock.clone();
        let index = mmr_index(embeddings, FusionMode::Rrf).await;

        let order: Vec<String> = index
            .retrieve_mmr("zebra", 3, 3, 0.0)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.document.id.unwrap_or_default())
            .collect();

        assert_eq!(order.len(), 3);
        // The fused top result can be any of the three (the two legs can rank
        // the short/long docs oppositely and tie under RRF), but MMR's
        // invariant is layout-independent: the near-duplicate pair d1/d2 must
        // never occupy the top two slots together — the dissimilar d3 has to
        // break into the first two positions.
        // HashSet<&str>:contains 直接收 &str,免去临时 String
        //(clippy::unnecessary_to_owned;注意 HashSet<&String> 没有
        // Borrow<str> 实现,必须把集合元素类型本身改成 &str)。
        let top_two: std::collections::HashSet<&str> =
            order[0..2].iter().map(String::as_str).collect();
        assert!(
            top_two.contains("d3"),
            "diversity must surface the dissimilar d3 in the top two; got {top_two:?}"
        );
        assert!(
            !(top_two.contains("d1") && top_two.contains("d2")),
            "the near-duplicate pair must not own both top slots; got {top_two:?}"
        );
        // The extra embed_documents call is the single batched MMR re-embed
        // (one per indexed chunk happened during ingest).
        assert!(mock.embed_document_calls.load(Ordering::SeqCst) > 3);
    }

    /// T11: k truncates the MMR result; cand_k only sizes the candidate pool.
    #[tokio::test]
    async fn mmr_truncates_to_k() {
        let mock = DualEncoderMock::new(32);
        let embeddings: Arc<dyn Embeddings> = mock.clone();
        let index = mmr_index(embeddings, FusionMode::Rrf).await;

        let order = index.retrieve_mmr("zebra", 3, 2, 0.5).await.unwrap();
        assert_eq!(order.len(), 2);
    }

    /// T11: an empty pool stays empty (no spurious embed call, no panic).
    #[tokio::test]
    async fn mmr_empty_when_nothing_matches() {
        let mock = DualEncoderMock::new(32);
        let embeddings: Arc<dyn Embeddings> = mock.clone();
        let docs: [(&str, &str); 0] = [];
        let index = weighted_index(embeddings, FusionMode::Rrf, &docs).await;
        assert!(index
            .retrieve_mmr("zebra", 3, 3, 0.5)
            .await
            .unwrap()
            .is_empty());
    }
}
