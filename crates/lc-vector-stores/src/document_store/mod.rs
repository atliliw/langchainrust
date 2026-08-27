// lc-vector-stores/src/document_store/mod.rs
//! Document store module
//!
//! Stores document content separately, shared by BM25 and vector retrieval.
//! Supports both raw documents and split chunk documents.

pub mod chunked;
pub mod store;
pub mod types;

#[cfg(test)]
mod tests;

pub use chunked::ChunkedDocumentStore;
pub use chunked::InMemoryChunkedDocumentStore;
pub use store::InMemoryDocumentStore;
pub use types::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
