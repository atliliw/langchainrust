// src/vector_stores/document_store/tests.rs
//! Tests for document store implementations.

#[cfg(test)]
mod tests {
    use super::super::super::Document;
    use super::super::chunked::ChunkedDocumentStore;
    use super::super::store::InMemoryDocumentStore;
    use super::super::types::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};

    #[tokio::test]
    async fn test_in_memory_document_store() {
        let store = InMemoryDocumentStore::new();

        // 添加文档
        let doc = Document::new("测试内容").with_id("doc_001");
        let id = store.add_document(doc).await.unwrap();

        assert_eq!(id, "doc_001");
        assert_eq!(store.count().await, 1);

        // 获取文档
        let retrieved = store.get_document("doc_001").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "测试内容");

        // 删除文档
        store.delete_document("doc_001").await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_chunked_document_store() {
        let store = ChunkedDocumentStore::new();

        // 添加 Parent 文档（chunk_size=20）
        let doc =
            Document::new("这是一段很长的测试文本，用于验证文档分割功能。").with_id("parent_001");

        let (parent_id, chunk_ids) = store.add_parent_document(doc, 20).await.unwrap();

        assert_eq!(parent_id, "parent_001");
        assert!(chunk_ids.len() > 1); // 应该分割成多个 chunk

        // 获取 Parent 文档
        let parent = store.get_parent_document("parent_001").await.unwrap();
        assert!(parent.is_some());

        // 获取所有 Chunk
        let chunks = store.get_chunks_for_parent("parent_001").await.unwrap();
        assert_eq!(chunks.len(), chunk_ids.len());

        // 获取单个 Chunk
        let chunk = store.get_chunk(&chunk_ids[0]).await.unwrap();
        assert!(chunk.is_some());
        assert_eq!(chunk.unwrap().parent_id, "parent_001");

        // 删除 Parent 及所有 Chunk
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
        assert_eq!(doc.metadata.get("source"), Some(&"test".to_string()));
    }

    #[tokio::test]
    async fn test_persistence() {
        let store = ChunkedDocumentStore::new();

        // 添加文档
        let doc = Document::new("测试持久化功能的内容").with_id("parent_001");
        store.add_parent_document(doc, 10).await.unwrap();

        // 保存
        let temp_path = tempfile::NamedTempFile::new().unwrap();
        store.save(temp_path.path()).await.unwrap();

        // 加载
        let loaded = ChunkedDocumentStore::load(temp_path.path()).await.unwrap();

        assert_eq!(loaded.parent_count().await, store.parent_count().await);
        assert_eq!(loaded.chunk_count().await, store.chunk_count().await);

        let parent = loaded.get_parent_document("parent_001").await.unwrap();
        assert!(parent.is_some());
    }
}
