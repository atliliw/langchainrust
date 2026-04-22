// src/retrieval/bm25/chunked.rs
//! BM25 Chunked Retriever - 支持 Parent-Child 文档结构的 BM25 检索器
//!
//! 基于 LlamaIndex AutoMerging 模式实现：
//! - 文档拆分为 Parent + Leaf 两层
//! - BM25 在 Leaf 层搜索
//! - AutoMerging 合并同一 Parent 的多个 Leaf
//! - 支持 Bincode 持久化

use super::algorithm::{bm25_score, compute_idf, BM25Params};
use super::tokenizer::Tokenizer;
use crate::vector_stores::document_store::{ChunkDocument, ChunkedDocumentStoreTrait};
use crate::vector_stores::Document;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// 数据结构定义
// ============================================================================

// ChunkDocument 现在在 document_store.rs 中定义，BM25 直接使用

/// AutoMerging 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMergingConfig {
    pub merge_threshold: f32,
    pub leaf_chunk_size: usize,
    pub parent_chunk_size: usize,
    pub leaves_per_parent: usize,
}

impl Default for AutoMergingConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.5,
            leaf_chunk_size: 400,
            parent_chunk_size: 2000,
            leaves_per_parent: 5,
        }
    }
}

impl AutoMergingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    pub fn with_leaf_size(mut self, size: usize) -> Self {
        self.leaf_chunk_size = size;
        self
    }

    pub fn with_parent_size(mut self, size: usize) -> Self {
        self.parent_chunk_size = size;
        self
    }
}

/// AutoMerging 搜索结果
#[derive(Debug, Clone)]
pub struct ChunkedSearchResult {
    pub merged_parent: Option<Document>,
    pub leaf_chunks: Vec<ChunkDocument>,
    pub score: f32,
    pub matched_terms: Vec<String>,
    pub parent_id: String,
}

impl ChunkedSearchResult {
    pub fn content(&self) -> String {
        if let Some(parent) = &self.merged_parent {
            parent.content.clone()
        } else {
            self.leaf_chunks
                .iter()
                .map(|c| c.content.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    pub fn is_merged(&self) -> bool {
        self.merged_parent.is_some()
    }
}

/// BM25 参数的可序列化版本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25ParamsData {
    pub k1: f64,
    pub b: f64,
}

impl From<BM25Params> for BM25ParamsData {
    fn from(params: BM25Params) -> Self {
        Self {
            k1: params.k1,
            b: params.b,
        }
    }
}

impl From<BM25ParamsData> for BM25Params {
    fn from(data: BM25ParamsData) -> Self {
        BM25Params::with_values(data.k1, data.b)
    }
}

/// 可序列化的索引数据（不含内容，内容在ChunkedDocumentStore中）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedIndexData {
    pub chunk_id_list: Vec<String>,
    pub chunk_term_freqs: Vec<HashMap<String, usize>>,
    pub term_index: HashMap<String, Vec<(usize, usize)>>,
    pub parent_to_leaves: HashMap<String, Vec<usize>>,
    pub doc_lengths: Vec<usize>,
    pub avgdl: f64,
    pub n_docs: usize,
    pub params: BM25ParamsData,
    pub config: AutoMergingConfig,
}

// ============================================================================
// ChunkedBM25Index 索引结构
// ============================================================================

pub struct ChunkedBM25Index<S: ChunkedDocumentStoreTrait = crate::vector_stores::ChunkedDocumentStore> {
    store: Arc<S>,
    chunk_id_list: Vec<String>,
    chunk_term_freqs: Vec<HashMap<String, usize>>,
    term_index: HashMap<String, Vec<(usize, usize)>>,
    parent_to_leaves: HashMap<String, Vec<usize>>,
    doc_lengths: Vec<usize>,
    avgdl: f64,
    n_docs: usize,
    idf_cache: HashMap<String, f64>,
    params: BM25Params,
    tokenizer: Tokenizer,
    config: AutoMergingConfig,
}

impl<S: ChunkedDocumentStoreTrait> ChunkedBM25Index<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self::with_config(store, AutoMergingConfig::default())
    }

    pub fn with_config(store: Arc<S>, config: AutoMergingConfig) -> Self {
        Self {
            store,
            chunk_id_list: Vec::new(),
            chunk_term_freqs: Vec::new(),
            term_index: HashMap::new(),
            parent_to_leaves: HashMap::new(),
            doc_lengths: Vec::new(),
            avgdl: 0.0,
            n_docs: 0,
            idf_cache: HashMap::new(),
            params: BM25Params::default(),
            tokenizer: Tokenizer::new(),
            config,
        }
    }

    pub fn with_params(store: Arc<S>, params: BM25Params) -> Self {
        let mut index = Self::new(store);
        index.params = params;
        index
    }

    /// 添加chunk索引（内容已在store中）
    pub fn add_chunk_index(&mut self, chunk_id: String, parent_id: String, content: &str) {
        let chunk_idx = self.n_docs;

        let terms = self.tokenizer.tokenize(content);
        let term_freq = self.compute_term_freq(&terms);

        // 更新倒排索引
        for (term, freq) in &term_freq {
            self.term_index
                .entry(term.clone())
                .or_insert_with(Vec::new)
                .push((chunk_idx, *freq));
        }

        // 更新parent到chunk的映射
        self.parent_to_leaves
            .entry(parent_id)
            .or_insert_with(Vec::new)
            .push(chunk_idx);

        // 存储chunk_id和词频（BM25计算需要）
        self.chunk_id_list.push(chunk_id);
        self.chunk_term_freqs.push(term_freq.clone());

        let doc_length: usize = term_freq.values().sum();
        self.doc_lengths.push(doc_length);
        self.n_docs += 1;
        self.update_avgdl();
        self.idf_cache.clear();
    }

    /// 批量添加chunk索引
    pub fn add_chunk_indexes(&mut self, chunks: Vec<(String, String, String)>) {
        for (chunk_id, parent_id, content) in chunks {
            self.add_chunk_index(chunk_id, parent_id, &content);
        }
    }

    fn compute_term_freq(&self, terms: &[String]) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        for term in terms {
            *freq.entry(term.clone()).or_insert(0) += 1;
        }
        freq
    }

    fn update_avgdl(&mut self) {
        if self.n_docs == 0 {
            self.avgdl = 0.0;
        } else {
            let total: usize = self.doc_lengths.iter().sum();
            self.avgdl = total as f64 / self.n_docs as f64;
        }
    }

    fn compute_idf_for_term(&mut self, term: &str) -> f64 {
        if let Some(idf) = self.idf_cache.get(term) {
            return *idf;
        }

        let n = self.term_index.get(term).map(|v| v.len()).unwrap_or(0);
        let idf = compute_idf(n, self.n_docs);
        self.idf_cache.insert(term.to_string(), idf);
        idf
    }

    pub fn get_chunk_id(&self, chunk_idx: usize) -> Option<&String> {
        self.chunk_id_list.get(chunk_idx)
    }

    pub fn get_chunk_ids_for_parent(&self, parent_id: &str) -> Vec<&String> {
        self.parent_to_leaves
            .get(parent_id)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|idx| self.chunk_id_list.get(*idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn config(&self) -> &AutoMergingConfig {
        &self.config
    }

    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    pub fn store(&self) -> &Arc<S> {
        &self.store
    }

    pub fn clear(&mut self) {
        self.chunk_id_list.clear();
        self.chunk_term_freqs.clear();
        self.term_index.clear();
        self.parent_to_leaves.clear();
        self.doc_lengths.clear();
        self.avgdl = 0.0;
        self.n_docs = 0;
        self.idf_cache.clear();
    }
}

impl Default for ChunkedBM25Index<crate::vector_stores::ChunkedDocumentStore> {
    fn default() -> Self {
        Self::new(Arc::new(crate::vector_stores::ChunkedDocumentStore::new()))
    }
}

// ============================================================================
// ChunkedBM25Retriever 检索器
// ============================================================================

pub struct ChunkedBM25Retriever<S: ChunkedDocumentStoreTrait = crate::vector_stores::ChunkedDocumentStore> {
    index: ChunkedBM25Index<S>,
}

impl<S: ChunkedDocumentStoreTrait> ChunkedBM25Retriever<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self {
            index: ChunkedBM25Index::new(store),
        }
    }

    pub fn with_config(store: Arc<S>, config: AutoMergingConfig) -> Self {
        Self {
            index: ChunkedBM25Index::with_config(store, config),
        }
    }

    pub fn with_params(store: Arc<S>, k1: f64, b: f64) -> Self {
        Self {
            index: ChunkedBM25Index::with_params(store, BM25Params::with_values(k1, b)),
        }
    }

    pub fn store(&self) -> &Arc<S> {
        self.index.store()
    }

    pub fn add_chunk_index(&mut self, chunk_id: String, parent_id: String, content: &str) {
        self.index.add_chunk_index(chunk_id, parent_id, content);
    }

    pub fn add_chunk_indexes(&mut self, chunks: Vec<(String, String, String)>) {
        self.index.add_chunk_indexes(chunks);
    }

    pub fn add_document(&mut self, document: Document) {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        self.index
            .store
            .add_parent_document_blocking(document, self.index.config.leaf_chunk_size)
            .ok();

        let chunks = self
            .index
            .store
            .blocking_get_chunks_for_parent(&parent_id)
            .ok()
            .unwrap_or_default();

        for chunk in chunks {
            self.add_chunk_index(
                chunk.chunk_id.clone(),
                chunk.parent_id.clone(),
                &chunk.content,
            );
        }
    }

    pub async fn add_document_async(&mut self, document: Document) {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        self.index
            .store
            .add_parent_document(document, self.index.config.leaf_chunk_size)
            .await
            .ok();

        let chunks = self
            .index
            .store
            .get_chunks_for_parent(&parent_id)
            .await
            .ok()
            .unwrap_or_default();

        for chunk in chunks {
            self.add_chunk_index(
                chunk.chunk_id.clone(),
                chunk.parent_id.clone(),
                &chunk.content,
            );
        }
    }

    pub fn add_documents(&mut self, documents: Vec<Document>) {
        for doc in documents {
            self.add_document(doc);
        }
    }

    pub async fn add_documents_async(&mut self, documents: Vec<Document>) {
        for doc in documents {
            self.add_document_async(doc).await;
        }
    }

    pub fn search(&mut self, query: &str, k: usize) -> Vec<ChunkedSearchResult> {
        if self.index.n_docs == 0 {
            return Vec::new();
        }

        let query_terms = self.index.tokenizer.tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let idf_values: HashMap<String, f64> = query_terms
            .iter()
            .map(|t| (t.clone(), self.index.compute_idf_for_term(t)))
            .collect();

        let scored_chunks = self.score_chunks(&query_terms, &idf_values);

        if scored_chunks.is_empty() {
            return Vec::new();
        }

        let top_chunks: Vec<(usize, f64)> = scored_chunks.into_iter().take(k * 2).collect();

        self.auto_merge_sync(top_chunks, k)
    }

    pub async fn search_async(&mut self, query: &str, k: usize) -> Vec<ChunkedSearchResult> {
        if self.index.n_docs == 0 {
            return Vec::new();
        }

        let query_terms = self.index.tokenizer.tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let idf_values: HashMap<String, f64> = query_terms
            .iter()
            .map(|t| (t.clone(), self.index.compute_idf_for_term(t)))
            .collect();

        let scored_chunks = self.score_chunks(&query_terms, &idf_values);

        if scored_chunks.is_empty() {
            return Vec::new();
        }

        let top_chunks: Vec<(usize, f64)> = scored_chunks.into_iter().take(k * 2).collect();

        self.auto_merge_async(top_chunks, k).await
    }

    fn auto_merge_sync(&self, scored_chunks: Vec<(usize, f64)>, k: usize) -> Vec<ChunkedSearchResult> {
        let threshold = self.index.config.merge_threshold;
        let leaves_per_parent = self.index.config.leaves_per_parent;

        let parent_stats = self.collect_parent_stats(&scored_chunks);

        let mut results: Vec<ChunkedSearchResult> = Vec::new();

        for (parent_id, matched_leaves) in parent_stats {
            let ratio = matched_leaves.len() as f32 / leaves_per_parent as f32;

            let avg_score =
                matched_leaves.iter().map(|(_, s)| s).sum::<f64>() / matched_leaves.len() as f64;

            let matched_terms = matched_leaves
                .iter()
                .filter_map(|(idx, _)| self.index.chunk_term_freqs.get(*idx))
                .flat_map(|tf| tf.keys().cloned())
                .collect::<Vec<_>>();

            if ratio >= threshold {
                let parent_doc = self
                    .index
                    .store()
                    .get_parent_document_blocking(&parent_id)
                    .ok()
                    .flatten();

                results.push(ChunkedSearchResult {
                    merged_parent: parent_doc,
                    leaf_chunks: Vec::new(),
                    score: avg_score as f32,
                    matched_terms,
                    parent_id,
                });
            } else {
                let leaf_chunks: Vec<ChunkDocument> = matched_leaves
                    .iter()
                    .filter_map(|(idx, _)| {
                        let chunk_id = self.index.get_chunk_id(*idx)?;
                        let chunk = self
                            .index
                            .store()
                            .get_chunk_blocking(&chunk_id)
                            .ok()
                            .flatten()?;
                        Some(chunk)
                    })
                    .collect();

                results.push(ChunkedSearchResult {
                    merged_parent: None,
                    leaf_chunks,
                    score: avg_score as f32,
                    matched_terms,
                    parent_id,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.into_iter().take(k).collect()
    }

    async fn auto_merge_async(&self, scored_chunks: Vec<(usize, f64)>, k: usize) -> Vec<ChunkedSearchResult> {
        use crate::vector_stores::document_store::ChunkedDocumentStoreTrait;
        
        let threshold = self.index.config.merge_threshold;
        let leaves_per_parent = self.index.config.leaves_per_parent;

        let parent_stats = self.collect_parent_stats(&scored_chunks);

        let mut results: Vec<ChunkedSearchResult> = Vec::new();

        for (parent_id, matched_leaves) in parent_stats {
            let ratio = matched_leaves.len() as f32 / leaves_per_parent as f32;

            let avg_score =
                matched_leaves.iter().map(|(_, s)| s).sum::<f64>() / matched_leaves.len() as f64;

            let matched_terms = matched_leaves
                .iter()
                .filter_map(|(idx, _)| self.index.chunk_term_freqs.get(*idx))
                .flat_map(|tf| tf.keys().cloned())
                .collect::<Vec<_>>();

            if ratio >= threshold {
                let parent_doc = self
                    .index
                    .store()
                    .get_parent_document(&parent_id)
                    .await
                    .ok()
                    .flatten();

                results.push(ChunkedSearchResult {
                    merged_parent: parent_doc,
                    leaf_chunks: Vec::new(),
                    score: avg_score as f32,
                    matched_terms,
                    parent_id,
                });
            } else {
                let mut leaf_chunks = Vec::new();
                for (idx, _) in matched_leaves {
                    if let Some(chunk_id) = self.index.get_chunk_id(idx) {
                        if let Some(chunk) = self.index.store().get_chunk(&chunk_id).await.ok().flatten() {
                            leaf_chunks.push(chunk);
                        }
                    }
                }

                results.push(ChunkedSearchResult {
                    merged_parent: None,
                    leaf_chunks,
                    score: avg_score as f32,
                    matched_terms,
                    parent_id,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.into_iter().take(k).collect()
    }

    fn score_chunks(
        &self,
        query_terms: &[String],
        idf_values: &HashMap<String, f64>,
    ) -> Vec<(usize, f64)> {
        let mut scored = Vec::new();

        for chunk_idx in 0..self.index.n_docs {
            if let Some(term_freqs) = self.index.chunk_term_freqs.get(chunk_idx) {
                let doc_length = *self.index.doc_lengths.get(chunk_idx).unwrap_or(&0);

                let score = bm25_score(
                    query_terms,
                    term_freqs,
                    doc_length,
                    self.index.avgdl,
                    idf_values,
                    &self.index.params,
                );

                if score > 0.0 {
                    scored.push((chunk_idx, score));
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    fn collect_parent_stats(
        &self,
        scored_chunks: &[(usize, f64)],
    ) -> HashMap<String, Vec<(usize, f64)>> {
        let mut stats: HashMap<String, Vec<(usize, f64)>> = HashMap::new();

        for (chunk_idx, score) in scored_chunks {
            if let Some(chunk_id) = self.index.chunk_id_list.get(*chunk_idx) {
                let parent_id = chunk_id.split('_').next().unwrap_or_default().to_string();
                stats
                    .entry(parent_id)
                    .or_insert_with(Vec::new)
                    .push((*chunk_idx, *score));
            }
        }

        stats
    }

    pub fn get_parent_document(&self, parent_id: &str) -> Option<Document> {
        self.index
            .store()
            .get_parent_document_blocking(parent_id)
            .ok()
            .flatten()
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

    pub fn config(&self) -> &AutoMergingConfig {
        self.index.config()
    }

    // 持久化方法
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let data = ChunkedIndexData {
            chunk_id_list: self.index.chunk_id_list.clone(),
            chunk_term_freqs: self.index.chunk_term_freqs.clone(),
            term_index: self.index.term_index.clone(),
            parent_to_leaves: self.index.parent_to_leaves.clone(),
            doc_lengths: self.index.doc_lengths.clone(),
            avgdl: self.index.avgdl,
            n_docs: self.index.n_docs,
            params: BM25ParamsData::from(self.index.params.clone()),
            config: self.index.config.clone(),
        };
        let encoded = bincode::serialize(&data)?;
        std::fs::write(path.as_ref(), encoded)?;
        Ok(())
    }
}

impl ChunkedBM25Retriever<crate::vector_stores::ChunkedDocumentStore> {
    pub fn load(
        store: Arc<crate::vector_stores::ChunkedDocumentStore>,
        path: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path.as_ref())?;
        let data: ChunkedIndexData = bincode::deserialize(&bytes)?;
        let params: BM25Params = data.params.into();

        Ok(Self {
            index: ChunkedBM25Index {
                store,
                chunk_id_list: data.chunk_id_list,
                chunk_term_freqs: data.chunk_term_freqs,
                term_index: data.term_index,
                parent_to_leaves: data.parent_to_leaves,
                doc_lengths: data.doc_lengths,
                avgdl: data.avgdl,
                n_docs: data.n_docs,
                idf_cache: HashMap::new(),
                params,
                tokenizer: Tokenizer::new(),
                config: data.config,
            },
        })
    }
}

impl Default for ChunkedBM25Retriever<crate::vector_stores::ChunkedDocumentStore> {
    fn default() -> Self {
        Self::new(Arc::new(crate::vector_stores::ChunkedDocumentStore::new()))
    }
}
