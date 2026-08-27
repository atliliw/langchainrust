// lc-vector-stores/src/document_store/tests.rs
//! Tests for document store implementations.
//!
//! Included by `document_store/mod.rs` via `#[cfg(test)] mod tests;`,
//! so this file IS the body of the `document_store::tests` module — do not wrap it in another
//! `mod tests` (avoiding module_inception).

use crate::document_store::chunked::ChunkedDocumentStore;
use crate::document_store::store::InMemoryDocumentStore;
use crate::document_store::types::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
use lc_shared::document::Document;

#[tokio::test]
async fn test_in_memory_document_store() {
    let store = InMemoryDocumentStore::new();

    // add a document
    let doc = Document::new("测试内容").with_id("doc_001");
    let id = store.add_document(doc).await.unwrap();

    assert_eq!(id, "doc_001");
    assert_eq!(store.count().await, 1);

    // get the document
    let retrieved = store.get_document("doc_001").await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, "测试内容");

    // delete the document
    store.delete_document("doc_001").await.unwrap();
    assert_eq!(store.count().await, 0);
}

#[tokio::test]
async fn test_chunked_document_store() {
    let store = ChunkedDocumentStore::new();

    // add a Parent document (chunk_size=20)
    let doc = Document::new("这是一段很长的测试文本，用于验证文档分割功能。").with_id("parent_001");

    let (parent_id, chunk_ids) = store.add_parent_document(doc, 20).await.unwrap();

    assert_eq!(parent_id, "parent_001");
    assert!(chunk_ids.len() > 1); // should be split into multiple chunks

    // get the Parent document
    let parent = store.get_parent_document("parent_001").await.unwrap();
    assert!(parent.is_some());

    // get all Chunks
    let chunks = store.get_chunks_for_parent("parent_001").await.unwrap();
    assert_eq!(chunks.len(), chunk_ids.len());

    // get a single Chunk
    let chunk = store.get_chunk(&chunk_ids[0]).await.unwrap();
    assert!(chunk.is_some());
    assert_eq!(chunk.unwrap().parent_id, "parent_001");

    // delete the Parent and all of its Chunks
    store.delete_parent_document("parent_001").await.unwrap();
    assert_eq!(store.parent_count().await, 0);
    assert_eq!(store.chunk_count().await, 0);
}

#[tokio::test]
async fn test_chunk_to_document() {
    let chunk = ChunkDocument::new(
        "chunk_001".to_string(),
        "parent_001".to_string(),
        "Chunk内容".to_string(),
        0,
    )
    .with_metadata("source", "test");

    let doc = chunk.to_document();

    assert_eq!(doc.id, Some("chunk_001".to_string()));
    assert_eq!(doc.content, "Chunk内容");
    assert_eq!(
        doc.metadata.get("source"),
        Some(&serde_json::Value::String("test".to_string()))
    );
}

#[tokio::test]
async fn test_persistence() {
    let store = ChunkedDocumentStore::new();

    // add a document
    let doc = Document::new("测试持久化功能的内容").with_id("parent_001");
    store.add_parent_document(doc, 10).await.unwrap();

    // save
    let temp_path = tempfile::NamedTempFile::new().unwrap();
    store.save(temp_path.path()).await.unwrap();

    // load
    let loaded = ChunkedDocumentStore::load(temp_path.path()).await.unwrap();

    assert_eq!(loaded.parent_count().await, store.parent_count().await);
    assert_eq!(loaded.chunk_count().await, store.chunk_count().await);

    let parent = loaded.get_parent_document("parent_001").await.unwrap();
    assert!(parent.is_some());
}
