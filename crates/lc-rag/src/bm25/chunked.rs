// src/retrieval/bm25/chunked.rs
//! BM25 Chunked Retriever - a BM25 retriever supporting the Parent-Child document structure
//!
//! Implements the LlamaIndex AutoMerging pattern:
//! - Documents are split into Parent + Leaf layers
//! - BM25 searches at the Leaf layer
//! - AutoMerging merges multiple Leaves under the same Parent
//! - Supports Bincode persistence

use super::algorithm::{bm25_score, compute_idf, BM25Params};
use super::tokenizer::Tokenizer;
use lc_vector_stores::document_store::{ChunkDocument, ChunkedDocumentStoreTrait};
use lc_vector_stores::{Document, VectorStoreError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// Data structure definitions
// ============================================================================

// ChunkDocument is now defined in document_store.rs and used directly by BM25

/// AutoMerging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMergingConfig {
    /// Merge threshold: when the ratio of hit Leaves under the same Parent reaches this
    /// value, they are merged into a Parent document
    pub merge_threshold: f32,
    /// The size of a Leaf chunk (in characters)
    pub leaf_chunk_size: usize,
    /// The size of a Parent chunk (in characters)
    pub parent_chunk_size: usize,
    /// The expected number of Leaves per Parent
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
    /// Creates an `AutoMergingConfig` with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the merge threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    /// Sets the Leaf chunk size
    pub fn with_leaf_size(mut self, size: usize) -> Self {
        self.leaf_chunk_size = size;
        self
    }

    /// Sets the Parent chunk size
    pub fn with_parent_size(mut self, size: usize) -> Self {
        self.parent_chunk_size = size;
        self
    }
}

/// AutoMerging search result
#[derive(Debug, Clone)]
pub struct ChunkedSearchResult {
    /// The merged Parent document (`None` when merging was not triggered)
    pub merged_parent: Option<Document>,
    /// The hit Leaf chunks
    pub leaf_chunks: Vec<ChunkDocument>,
    /// The BM25 score of this result
    pub score: f32,
    /// The matched query terms
    pub matched_terms: Vec<String>,
    /// The id of the owning Parent
    pub parent_id: String,
}

impl ChunkedSearchResult {
    /// Returns the merged result's content: prefers the Parent content, otherwise
    /// concatenates all Leaf content
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

    /// Whether AutoMerging merging was triggered
    pub fn is_merged(&self) -> bool {
        self.merged_parent.is_some()
    }
}

/// Serializable version of the BM25 parameters
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

/// Serializable index data (no content; content lives in ChunkedDocumentStore)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedIndexData {
    /// The list of chunk ids
    pub chunk_id_list: Vec<String>,
    /// Each chunk's term-frequency table
    pub chunk_term_freqs: Vec<HashMap<String, usize>>,
    /// Inverted index: term -> list of (chunk index, term frequency)
    pub term_index: HashMap<String, Vec<(usize, usize)>>,
    /// Parent id -> the list of Leaf chunk indices under that Parent
    pub parent_to_leaves: HashMap<String, Vec<usize>>,
    /// Each chunk's document length
    pub doc_lengths: Vec<usize>,
    /// The average document length
    pub avgdl: f64,
    /// The number of documents
    pub n_docs: usize,
    /// BM25 parameters
    pub params: BM25ParamsData,
    /// AutoMerging configuration
    pub config: AutoMergingConfig,
}

// ============================================================================
// ChunkedBM25Index index structure
// ============================================================================

/// A BM25 inverted index supporting the Parent-Child structure
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
    /// Creates an index with default settings
    pub fn new(store: Arc<S>) -> Self {
        Self::with_config(store, AutoMergingConfig::default())
    }

    /// Creates an index with the given settings
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

    /// Creates an index with the given BM25 parameters
    pub fn with_params(store: Arc<S>, params: BM25Params) -> Self {
        let mut index = Self::new(store);
        index.params = params;
        index
    }

    /// Adds a chunk index (the content is already in the store)
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

        // Update the inverted index
        for (term, freq) in &term_freq {
            self.term_index
                .entry(term.clone())
                .or_default()
                .push((chunk_idx, *freq));
        }

        // Update the parent-to-chunk mapping
        self.parent_to_leaves
            .entry(parent_id)
            .or_default()
            .push(chunk_idx);

        // Store the chunk_id and term frequencies (needed for BM25 scoring)
        self.chunk_id_list.push(chunk_id);
        self.chunk_term_freqs.push(term_freq.clone());

        let doc_length: usize = term_freq.values().sum();
        self.doc_lengths.push(doc_length);
        self.n_docs += 1;
        self.update_avgdl();
        self.idf_cache.clear();
    }

    /// Adds chunk indexes in batch
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

    /// Gets the chunk id by chunk index
    pub fn get_chunk_id(&self, chunk_idx: usize) -> Option<&String> {
        self.chunk_id_list.get(chunk_idx)
    }

    /// Gets all chunk ids under the given Parent
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

    /// Returns the AutoMerging configuration
    pub fn config(&self) -> &AutoMergingConfig {
        &self.config
    }

    /// Returns the number of indexed documents
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// Returns the underlying document store
    pub fn store(&self) -> &Arc<S> {
        &self.store
    }

    /// Clears the index data
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
// ChunkedBM25Retriever
// ============================================================================

/// A BM25 retriever based on AutoMerging
pub struct ChunkedBM25Retriever<
    S: ChunkedDocumentStoreTrait = lc_vector_stores::ChunkedDocumentStore,
> {
    index: ChunkedBM25Index<S>,
}

impl<S: ChunkedDocumentStoreTrait> ChunkedBM25Retriever<S> {
    /// Creates a retriever with default settings
    pub fn new(store: Arc<S>) -> Self {
        Self {
            index: ChunkedBM25Index::new(store),
        }
    }

    /// Creates a retriever with the given settings
    pub fn with_config(store: Arc<S>, config: AutoMergingConfig) -> Self {
        Self {
            index: ChunkedBM25Index::with_config(store, config),
        }
    }

    /// Creates a retriever with the given k1 and b parameters
    pub fn with_params(store: Arc<S>, k1: f64, b: f64) -> Self {
        Self {
            index: ChunkedBM25Index::with_params(store, BM25Params::with_values(k1, b)),
        }
    }

    /// Returns the underlying document store
    pub fn store(&self) -> &Arc<S> {
        self.index.store()
    }

    /// Adds a single chunk index (the content is already stored in the store)
    pub fn add_chunk_index(
        &mut self,
        chunk_id: impl Into<String>,
        parent_id: impl Into<String>,
        content: &str,
    ) {
        self.index.add_chunk_index(chunk_id, parent_id, content);
    }

    /// Adds chunk indexes in batch
    pub fn add_chunk_indexes(&mut self, chunks: Vec<(String, String, String)>) {
        self.index.add_chunk_indexes(chunks);
    }

    /// Adds a document synchronously: automatically splits Parent/Leaf and builds the index
    pub fn add_document(&mut self, document: Document) -> Result<(), VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // P0-1: a document without an id has the pre-allocated parent_id attached before
        // storing; otherwise the store generates a fresh uuid, and get_chunks_for_parent
        // would look up with the wrong key and find nothing.
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

    /// Adds a document asynchronously: automatically splits Parent/Leaf and builds the index
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

    /// Adds documents in batch synchronously
    pub fn add_documents(&mut self, documents: Vec<Document>) -> Result<(), VectorStoreError> {
        for doc in documents {
            self.add_document(doc)?;
        }
        Ok(())
    }

    /// Adds documents in batch asynchronously
    pub async fn add_documents_async(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<(), VectorStoreError> {
        for doc in documents {
            self.add_document_async(doc).await?;
        }
        Ok(())
    }

    /// Runs BM25 retrieval synchronously, returning the top k AutoMerging results
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

    /// Runs BM25 retrieval asynchronously, returning the top k AutoMerging results
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

    /// Read-only BM25 retrieval: returns the list of matched parent ids (deduplicated),
    /// sorted by the best chunk score.
    ///
    /// Unlike [`search`](Self::search)/[`search_async`](Self::search_async):
    /// no AutoMerging ratio gating is applied here — any chunk hit lets its parent through,
    /// matching the "hit child chunk -> return the whole parent document" semantics that
    /// [`ParentDocumentRetriever`](crate::parent_document::ParentDocumentRetriever) needs.
    /// Fully `&self` read-only (idf is not cached), safe to call concurrently.
    pub fn search_matched_parents(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if self.index.n_docs == 0 {
            return Vec::new();
        }

        let query_terms = self.index.tokenizer.tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Read-only idf: does not write idf_cache, avoiding `&mut self`.
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
                                // No longer swallow errors silently: a failed read is logged,
                                // and the chunk is missing from the results
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

    /// Gets the parent document by Parent id
    pub fn get_parent_document(&self, parent_id: &str) -> Option<Document> {
        self.index
            .store()
            .get_parent_document_blocking(parent_id)
            .ok()
            .flatten()
    }

    /// Returns the number of documents in the index
    pub fn len(&self) -> usize {
        self.index.n_docs()
    }

    /// Whether the index is empty
    pub fn is_empty(&self) -> bool {
        self.index.n_docs() == 0
    }

    /// Clears the index
    pub fn clear(&mut self) {
        self.index.clear();
    }

    /// Returns the AutoMerging configuration
    pub fn config(&self) -> &AutoMergingConfig {
        self.index.config()
    }

    // Persistence methods
    /// Serializes the index data to Bincode and saves it to the given path
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
    /// Loads Bincode-serialized index data from the given path
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
