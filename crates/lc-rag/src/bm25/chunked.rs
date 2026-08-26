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
use lc_vector_stores::document_store::{ChunkDocument, ChunkedDocumentStoreTrait};
use lc_vector_stores::{Document, VectorStoreError};
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
    /// 合并阈值：同一 Parent 下命中 Leaf 占比达到该比例时合并为 Parent 文档
    pub merge_threshold: f32,
    /// Leaf chunk 的大小（字符数）
    pub leaf_chunk_size: usize,
    /// Parent chunk 的大小（字符数）
    pub parent_chunk_size: usize,
    /// 每个 Parent 下期望的 Leaf 数量
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
    /// 创建使用默认配置的 `AutoMergingConfig`
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置合并阈值
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    /// 设置 Leaf chunk 大小
    pub fn with_leaf_size(mut self, size: usize) -> Self {
        self.leaf_chunk_size = size;
        self
    }

    /// 设置 Parent chunk 大小
    pub fn with_parent_size(mut self, size: usize) -> Self {
        self.parent_chunk_size = size;
        self
    }
}

/// AutoMerging 搜索结果
#[derive(Debug, Clone)]
pub struct ChunkedSearchResult {
    /// 合并得到的 Parent 文档（若未触发合并则为 `None`）
    pub merged_parent: Option<Document>,
    /// 命中的 Leaf chunks
    pub leaf_chunks: Vec<ChunkDocument>,
    /// 该结果的 BM25 评分
    pub score: f32,
    /// 命中的查询词项
    pub matched_terms: Vec<String>,
    /// 所属 Parent 的 id
    pub parent_id: String,
}

impl ChunkedSearchResult {
    /// 返回合并结果的内容：优先返回 Parent 内容，否则拼接所有 Leaf 内容
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

    /// 是否触发了 AutoMerging 合并
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
    /// chunk 的 id 列表
    pub chunk_id_list: Vec<String>,
    /// 每个 chunk 的词频表
    pub chunk_term_freqs: Vec<HashMap<String, usize>>,
    /// 倒排索引：词项 -> (chunk 下标, 词频) 列表
    pub term_index: HashMap<String, Vec<(usize, usize)>>,
    /// Parent id -> 该 Parent 下 Leaf chunk 下标列表
    pub parent_to_leaves: HashMap<String, Vec<usize>>,
    /// 每个 chunk 的文档长度
    pub doc_lengths: Vec<usize>,
    /// 平均文档长度
    pub avgdl: f64,
    /// 文档数量
    pub n_docs: usize,
    /// BM25 参数
    pub params: BM25ParamsData,
    /// AutoMerging 配置
    pub config: AutoMergingConfig,
}

// ============================================================================
// ChunkedBM25Index 索引结构
// ============================================================================

/// 支持 Parent-Child 结构的 BM25 倒排索引
pub struct ChunkedBM25Index<S: ChunkedDocumentStoreTrait = lc_vector_stores::ChunkedDocumentStore> {
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
    /// 使用默认配置创建索引
    pub fn new(store: Arc<S>) -> Self {
        Self::with_config(store, AutoMergingConfig::default())
    }

    /// 使用指定配置创建索引
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

    /// 使用指定 BM25 参数创建索引
    pub fn with_params(store: Arc<S>, params: BM25Params) -> Self {
        let mut index = Self::new(store);
        index.params = params;
        index
    }

    /// 添加chunk索引（内容已在store中）
    pub fn add_chunk_index(
        &mut self,
        chunk_id: impl Into<String>,
        parent_id: impl Into<String>,
        content: &str,
    ) {
        let chunk_idx = self.n_docs;
        let chunk_id = chunk_id.into();
        let parent_id = parent_id.into();

        let terms = self.tokenizer.tokenize(content);
        let term_freq = self.compute_term_freq(&terms);

        // 更新倒排索引
        for (term, freq) in &term_freq {
            self.term_index
                .entry(term.clone())
                .or_default()
                .push((chunk_idx, *freq));
        }

        // 更新parent到chunk的映射
        self.parent_to_leaves
            .entry(parent_id)
            .or_default()
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

    /// 按 chunk 下标获取 chunk id
    pub fn get_chunk_id(&self, chunk_idx: usize) -> Option<&String> {
        self.chunk_id_list.get(chunk_idx)
    }

    /// 获取指定 Parent 下的所有 chunk id
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

    /// 返回 AutoMerging 配置
    pub fn config(&self) -> &AutoMergingConfig {
        &self.config
    }

    /// 返回已索引的文档数量
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// 返回底层文档存储
    pub fn store(&self) -> &Arc<S> {
        &self.store
    }

    /// 清空索引数据
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

impl Default for ChunkedBM25Index<lc_vector_stores::ChunkedDocumentStore> {
    fn default() -> Self {
        Self::new(Arc::new(lc_vector_stores::ChunkedDocumentStore::new()))
    }
}

// ============================================================================
// ChunkedBM25Retriever 检索器
// ============================================================================

/// 基于 AutoMerging 的 BM25 检索器
pub struct ChunkedBM25Retriever<
    S: ChunkedDocumentStoreTrait = lc_vector_stores::ChunkedDocumentStore,
> {
    index: ChunkedBM25Index<S>,
}

impl<S: ChunkedDocumentStoreTrait> ChunkedBM25Retriever<S> {
    /// 使用默认配置创建检索器
    pub fn new(store: Arc<S>) -> Self {
        Self {
            index: ChunkedBM25Index::new(store),
        }
    }

    /// 使用指定配置创建检索器
    pub fn with_config(store: Arc<S>, config: AutoMergingConfig) -> Self {
        Self {
            index: ChunkedBM25Index::with_config(store, config),
        }
    }

    /// 使用指定的 k1、b 参数创建检索器
    pub fn with_params(store: Arc<S>, k1: f64, b: f64) -> Self {
        Self {
            index: ChunkedBM25Index::with_params(store, BM25Params::with_values(k1, b)),
        }
    }

    /// 返回底层文档存储
    pub fn store(&self) -> &Arc<S> {
        self.index.store()
    }

    /// 添加单个 chunk 索引（内容已存储在 store 中）
    pub fn add_chunk_index(
        &mut self,
        chunk_id: impl Into<String>,
        parent_id: impl Into<String>,
        content: &str,
    ) {
        self.index.add_chunk_index(chunk_id, parent_id, content);
    }

    /// 批量添加 chunk 索引
    pub fn add_chunk_indexes(&mut self, chunks: Vec<(String, String, String)>) {
        self.index.add_chunk_indexes(chunks);
    }

    /// 以同步方式添加文档：自动拆分 Parent/Leaf 并建立索引
    pub fn add_document(&mut self, document: Document) -> Result<(), VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // P0-1: 无 id 的文档先把预分配的 parent_id 挂到文档上再入库,
        // 否则 store 内部会再生成一个新 uuid,导致 get_chunks_for_parent 用错 key 查空。
        self.index.store.add_parent_document_blocking(
            document.clone().with_id(parent_id.clone()),
            self.index.config.leaf_chunk_size,
        )?;

        let chunks = self
            .index
            .store
            .blocking_get_chunks_for_parent(&parent_id)?;

        for chunk in chunks {
            self.add_chunk_index(
                chunk.chunk_id.clone(),
                chunk.parent_id.clone(),
                &chunk.content,
            );
        }

        Ok(())
    }

    /// 以异步方式添加文档：自动拆分 Parent/Leaf 并建立索引
    pub async fn add_document_async(&mut self, document: Document) -> Result<(), VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        self.index
            .store
            .add_parent_document(
                document.clone().with_id(parent_id.clone()),
                self.index.config.leaf_chunk_size,
            )
            .await?;

        let chunks = self.index.store.get_chunks_for_parent(&parent_id).await?;

        for chunk in chunks {
            self.add_chunk_index(
                chunk.chunk_id.clone(),
                chunk.parent_id.clone(),
                &chunk.content,
            );
        }

        Ok(())
    }

    /// 批量以同步方式添加文档
    pub fn add_documents(&mut self, documents: Vec<Document>) -> Result<(), VectorStoreError> {
        for doc in documents {
            self.add_document(doc)?;
        }
        Ok(())
    }

    /// 批量以异步方式添加文档
    pub async fn add_documents_async(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<(), VectorStoreError> {
        for doc in documents {
            self.add_document_async(doc).await?;
        }
        Ok(())
    }

    /// 同步执行 BM25 检索，返回前 k 个 AutoMerging 结果
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

    /// 异步执行 BM25 检索，返回前 k 个 AutoMerging 结果
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

    /// 只读 BM25 检索:返回命中的 parent id 列表(去重),按最佳 chunk 分排序。
    ///
    /// 与 [`search`](Self::search)/[`search_async`](Self::search_async) 不同:
    /// 这里**不做** AutoMerging 比例门控 —— 任何 chunk 命中即让该 parent 入围,
    /// 语义即 [`ParentDocumentRetriever`](crate::parent_document::ParentDocumentRetriever)
    /// 需要的"命中子块 → 返回整篇父文档"。全 `&self` 只读(idf 不缓存),可安全并发。
    pub fn search_matched_parents(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if self.index.n_docs == 0 {
            return Vec::new();
        }

        let query_terms = self.index.tokenizer.tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        // 只读 idf:不写 idf_cache,避免 `&mut self`。
        let idf_values: HashMap<String, f64> = query_terms
            .iter()
            .map(|t| {
                let n = self.index.term_index.get(t).map(|v| v.len()).unwrap_or(0);
                (t.clone(), compute_idf(n, self.index.n_docs))
            })
            .collect();

        let scored_chunks = self.score_chunks(&query_terms, &idf_values);
        if scored_chunks.is_empty() {
            return Vec::new();
        }

        let top_chunks: Vec<(usize, f64)> = scored_chunks.into_iter().take(k * 2).collect();
        let parent_stats = self.collect_parent_stats(&top_chunks);

        let mut ranked: Vec<(String, f32)> = parent_stats
            .into_iter()
            .map(|(parent_id, leaves)| {
                let best = leaves.iter().map(|(_, s)| *s as f32).fold(0.0f32, f32::max);
                (parent_id, best)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().take(k).collect()
    }

    fn auto_merge_sync(
        &self,
        scored_chunks: Vec<(usize, f64)>,
        k: usize,
    ) -> Vec<ChunkedSearchResult> {
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
                            .get_chunk_blocking(chunk_id)
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

    async fn auto_merge_async(
        &self,
        scored_chunks: Vec<(usize, f64)>,
        k: usize,
    ) -> Vec<ChunkedSearchResult> {
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
                        match self.index.store().get_chunk(chunk_id).await {
                            Ok(Some(chunk)) => leaf_chunks.push(chunk),
                            Ok(None) => {}
                            Err(e) => {
                                // 不再静默吞错:读失败记日志,该 chunk 从结果中缺失
                                log::error!(
                                    "failed to read chunk `{}` during retrieval (chunk missing from results): {}",
                                    chunk_id,
                                    e
                                );
                            }
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
                let parent_id = chunk_id.split("::").next().unwrap_or_default().to_string();
                stats
                    .entry(parent_id)
                    .or_default()
                    .push((*chunk_idx, *score));
            }
        }

        stats
    }

    /// 按 Parent id 获取父文档
    pub fn get_parent_document(&self, parent_id: &str) -> Option<Document> {
        self.index
            .store()
            .get_parent_document_blocking(parent_id)
            .ok()
            .flatten()
    }

    /// 返回索引中的文档数量
    pub fn len(&self) -> usize {
        self.index.n_docs()
    }

    /// 索引是否为空
    pub fn is_empty(&self) -> bool {
        self.index.n_docs() == 0
    }

    /// 清空索引
    pub fn clear(&mut self) {
        self.index.clear();
    }

    /// 返回 AutoMerging 配置
    pub fn config(&self) -> &AutoMergingConfig {
        self.index.config()
    }

    // 持久化方法
    /// 将索引数据序列化为 Bincode 并保存到指定路径
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

impl ChunkedBM25Retriever<lc_vector_stores::ChunkedDocumentStore> {
    /// 从指定路径加载 Bincode 序列化的索引数据
    pub fn load(
        store: Arc<lc_vector_stores::ChunkedDocumentStore>,
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

impl Default for ChunkedBM25Retriever<lc_vector_stores::ChunkedDocumentStore> {
    fn default() -> Self {
        Self::new(Arc::new(lc_vector_stores::ChunkedDocumentStore::new()))
    }
}
