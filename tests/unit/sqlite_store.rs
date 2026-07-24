//! SQLite 文档存储单元测试

use langchainrust::vector_stores::{ChunkedDocumentStoreTrait, DocumentStore};
use langchainrust::{Document, SQLiteDocumentStore, SQLiteStoreConfig};
use tempfile::tempdir;

fn create_test_store() -> (SQLiteDocumentStore, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db").to_string_lossy().to_string();
    let config = SQLiteStoreConfig::new(db_path);
    let store = SQLiteDocumentStore::new(config).unwrap();
    (store, dir)
}

#[test]
fn test_sqlite_config_default() {
    let config = SQLiteStoreConfig::default();
    assert_eq!(config.db_path, "langchainrust.db");
}

#[test]
fn test_sqlite_config_custom() {
    let config = SQLiteStoreConfig::new("/tmp/my_app.db");
    assert_eq!(config.db_path, "/tmp/my_app.db");
}

#[tokio::test]
async fn test_sqlite_add_and_get() {
    let (store, _dir) = create_test_store();

    let doc = Document::new("Test content").with_metadata("source", "test");
    let id = DocumentStore::add_document(&store, doc).await.unwrap();
    assert!(!id.is_empty());

    let retrieved = DocumentStore::get_document(&store, &id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retrieved.content, "Test content");
    assert_eq!(retrieved.metadata.get("source").unwrap(), "test");
}

#[tokio::test]
async fn test_sqlite_get_nonexistent() {
    let (store, _dir) = create_test_store();
    let result = DocumentStore::get_document(&store, "nonexistent-id")
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_sqlite_add_multiple() {
    let (store, _dir) = create_test_store();
    let docs = vec![
        Document::new("Doc 1"),
        Document::new("Doc 2"),
        Document::new("Doc 3"),
    ];
    let ids = DocumentStore::add_documents(&store, docs).await.unwrap();
    assert_eq!(ids.len(), 3);
    assert_eq!(DocumentStore::count(&store).await, 3);
}

#[tokio::test]
async fn test_sqlite_delete() {
    let (store, _dir) = create_test_store();
    let id = DocumentStore::add_document(&store, Document::new("To delete"))
        .await
        .unwrap();
    assert_eq!(DocumentStore::count(&store).await, 1);
    DocumentStore::delete_document(&store, &id).await.unwrap();
    assert_eq!(DocumentStore::count(&store).await, 0);
}

#[tokio::test]
async fn test_sqlite_clear() {
    let (store, _dir) = create_test_store();
    DocumentStore::add_document(&store, Document::new("A"))
        .await
        .unwrap();
    DocumentStore::add_document(&store, Document::new("B"))
        .await
        .unwrap();
    assert_eq!(DocumentStore::count(&store).await, 2);
    DocumentStore::clear(&store).await.unwrap();
    assert_eq!(DocumentStore::count(&store).await, 0);
}

#[tokio::test]
async fn test_sqlite_empty_count() {
    let (store, _dir) = create_test_store();
    assert_eq!(DocumentStore::count(&store).await, 0);
}

#[tokio::test]
async fn test_sqlite_chunked_storage() {
    let (store, _dir) = create_test_store();
    let long_content = "A".repeat(200);
    let doc = Document::new(long_content.clone());

    let (parent_id, chunk_ids) = ChunkedDocumentStoreTrait::add_parent_document(&store, doc, 50)
        .await
        .unwrap();
    assert!(!parent_id.is_empty());
    assert!(chunk_ids.len() >= 3);

    let parent = ChunkedDocumentStoreTrait::get_parent_document(&store, &parent_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(parent.content, long_content);

    let chunks = ChunkedDocumentStoreTrait::get_chunks_for_parent(&store, &parent_id)
        .await
        .unwrap();
    assert_eq!(chunks.len(), chunk_ids.len());
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.segment, i);
    }

    ChunkedDocumentStoreTrait::delete_parent_document(&store, &parent_id)
        .await
        .unwrap();
    assert_eq!(ChunkedDocumentStoreTrait::chunk_count(&store).await, 0);
}

#[tokio::test]
async fn test_sqlite_custom_id() {
    let (store, _dir) = create_test_store();
    let doc = Document::new("Custom ID doc").with_id("my-custom-id");
    let id = DocumentStore::add_document(&store, doc).await.unwrap();
    assert_eq!(id, "my-custom-id");
    let retrieved = DocumentStore::get_document(&store, "my-custom-id")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retrieved.content, "Custom ID doc");
}
