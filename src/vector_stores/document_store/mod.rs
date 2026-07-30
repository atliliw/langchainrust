// src/vector_stores/document_store/mod.rs
//! 文档存储模块
//!
//! 单独存储文档内容，供 BM25 和向量检索共用。
//! 支持原始文档和分割后的 chunk 文档。

pub mod chunked;
pub mod store;
pub mod types;

#[cfg(test)]
mod tests;

pub use chunked::ChunkedDocumentStore;
pub use chunked::InMemoryChunkedDocumentStore;
pub use store::InMemoryDocumentStore;
pub use types::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
