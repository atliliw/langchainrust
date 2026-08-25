// lc-vector-stores/src/document_store/types.rs
//! Core types and traits for document storage.
//!
//! Defines `DocumentStore` trait and `ChunkedDocumentStoreTrait` trait
//! used by all storage backends. `ChunkDocument` and `Document` are
//! re-exported from `lc-shared`.

use crate::VectorStoreError;
use async_trait::async_trait;
use lc_shared::document::Document;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export ChunkDocument from lc-shared for backward compatibility
// (used by sibling modules via `super::types::ChunkDocument`)
pub use lc_shared::document::ChunkDocument;

// ============================================================================
// DocumentStore Trait
// ============================================================================

/// 文档存储 trait
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// 添加文档
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError>;

    /// 批量添加文档
    async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError>;

    /// 获取文档
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError>;

    /// 删除文档
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError>;

    /// 获取文档数量
    async fn count(&self) -> usize;

    /// 清空存储
    async fn clear(&self) -> Result<(), VectorStoreError>;
}

// ============================================================================
// ChunkedDocumentStore Trait（抽象接口，支持多种存储后端）
// ============================================================================

/// 支持 Parent-Child 结构的文档存储 trait
///
/// 此 trait 定义了回表存储的统一接口，支持：
/// - MongoDB（生产环境）
/// - InMemory（开发/测试）
/// - Redis（缓存层，预留）
/// - SQLite（本地存储，预留）
#[async_trait]
pub trait ChunkedDocumentStoreTrait: Send + Sync {
    /// 添加 Parent 文档并自动分割为 chunks
    ///
    /// # 参数
    /// - `document`: 原始文档
    /// - `chunk_size`: 每个 chunk 的字符大小
    ///
    /// # 返回
    /// - `(parent_id, chunk_ids)`: Parent ID 和生成的 Chunk ID 列表
    async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError>;

    /// 批量添加 Parent 文档
    async fn add_parent_documents(
        &self,
        documents: Vec<Document>,
        chunk_size: usize,
    ) -> Result<Vec<(String, Vec<String>)>, VectorStoreError>;

    /// 获取 Parent 文档
    async fn get_parent_document(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError>;

    /// 获取单个 Chunk
    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError>;

    /// 获取单个 Chunk（转为 Document）
    async fn get_chunk_document(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Document>, VectorStoreError>;

    /// 获取 Parent 的所有 Chunks
    async fn get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError>;

    /// 获取 Parent 的所有 Chunks（转为 Document）
    async fn get_chunk_documents_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<Document>, VectorStoreError>;

    /// 删除 Parent 文档及其所有 Chunks
    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError>;

    /// 获取 Parent 文档数量
    async fn parent_count(&self) -> usize;

    /// 获取 Chunk 文档数量
    async fn chunk_count(&self) -> usize;

    /// 获取所有 Chunks
    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError>;

    /// 清空所有数据
    async fn clear(&self) -> Result<(), VectorStoreError>;

    // ========================================================================
    // Blocking 方法（同步版本，用于 BM25 等同步检索场景）
    // ========================================================================

    /// 添加 Parent 文档（阻塞版本）
    fn add_parent_document_blocking(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError>;

    /// 获取 Parent 文档（阻塞版本）
    fn get_parent_document_blocking(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError>;

    /// 获取单个 Chunk（阻塞版本）
    fn get_chunk_blocking(&self, chunk_id: &str)
        -> Result<Option<ChunkDocument>, VectorStoreError>;

    /// 获取 Parent 的所有 Chunks（阻塞版本）
    fn blocking_get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError>;
}

// ============================================================================
// Internal serialization helper
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChunkedStoreData {
    pub parent_docs: HashMap<String, Document>,
    pub chunks: HashMap<String, ChunkDocument>,
    pub parent_to_chunks: HashMap<String, Vec<String>>,
}
