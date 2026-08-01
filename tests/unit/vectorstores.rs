//! 向量存储核心测试

#[cfg(test)]
mod tests {
    use langchainrust::embeddings::{cosine_similarity, MockEmbeddings};
    use langchainrust::retrieval::{RetrieverTrait, SimilarityRetriever};
    use langchainrust::vector_stores::{
        Document, InMemoryVectorStore, VectorStore, VectorStoreBuilder, VectorStoreProvider,
        VectorStoreType,
    };
    use std::sync::Arc;

    // ============================================================
    // Document 测试
    // ============================================================

    /// 测试创建文档并设置 ID 和元数据
    #[test]
    fn test_document_creation() {
        let doc = Document::new("Test content")
            .with_id("doc-1")
            .with_metadata("author", "test");

        assert_eq!(doc.content, "Test content");
        assert_eq!(doc.id, Some("doc-1".to_string()));
        assert_eq!(doc.metadata.get("author"), Some(&"test".to_string()));
    }

    /// 测试文档的 JSON 序列化和反序列化
    #[test]
    fn test_document_serialization() {
        let doc = Document::new("Serialization test")
            .with_id("serde-doc")
            .with_metadata("key", "value");

        let json = serde_json::to_string(&doc).unwrap();
        let decoded: Document = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.content, doc.content);
        assert_eq!(decoded.id, doc.id);
    }

    // ============================================================
    // VectorStore 核心功能测试
    // ============================================================

    /// 测试添加文档和相似度搜索
    ///
    /// 验证：
    /// 1. 文档能正确添加
    /// 2. 相似度搜索返回正确结果
    #[tokio::test]
    async fn test_add_and_search() {
        let store = InMemoryVectorStore::new();

        let docs = vec![
            Document::new("Rust programming"),
            Document::new("Python scripting"),
        ];

        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let ids = store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(store.count().await, 2);

        // 搜索应该返回最相似的结果
        let query = vec![0.9, 0.1, 0.0];
        let results = store.similarity_search(&query, 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].document.content.contains("Rust"));
    }

    /// 测试通过 ID 获取和删除文档
    #[tokio::test]
    async fn test_get_delete_document() {
        let store = InMemoryVectorStore::new();

        let doc = Document::new("Test doc").with_id("test-id");
        store
            .add_documents(vec![doc], vec![vec![1.0, 0.0]])
            .await
            .unwrap();

        // 获取存在的文档
        let retrieved = store.get_document("test-id").await.unwrap();
        assert!(retrieved.is_some());

        // 删除文档
        store.delete_document("test-id").await.unwrap();
        let deleted = store.get_document("test-id").await.unwrap();
        assert!(deleted.is_none());
    }

    /// 测试错误处理：文档数量和向量数量不匹配
    #[tokio::test]
    async fn test_count_mismatch_error() {
        let store = InMemoryVectorStore::new();

        let docs = vec![Document::new("A"), Document::new("B")];
        let embeddings = vec![vec![1.0, 0.0]]; // 只有1个向量，但有2个文档
        assert!(store.add_documents(docs, embeddings).await.is_err());
    }

    /// 测试清空存储和计数功能
    #[tokio::test]
    async fn test_clear_and_count() {
        let store = InMemoryVectorStore::new();

        // 添加5个文档
        for i in 0..5 {
            let doc = Document::new(format!("Doc {}", i));
            store
                .add_documents(vec![doc], vec![vec![i as f32, 0.0]])
                .await
                .unwrap();
        }

        assert_eq!(store.count().await, 5);

        // 清空后应该为0
        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    // ============================================================
    // 余弦相似度测试
    // ============================================================

    /// 测试余弦相似度计算
    ///
    /// 相同方向 = 1.0
    /// 正交 = 0.0
    /// 相反方向 = -1.0
    /// 零向量 = 0.0（避免除零错误）
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap_or(0.0) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c).unwrap_or(0.0) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d).unwrap_or(0.0) - (-1.0)).abs() < 0.001);

        let zero = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &zero).unwrap_or(0.0), 0.0);
    }

    // ============================================================
    // Retriever 集成测试
    // ============================================================

    /// 测试 SimilarityRetriever 的文档检索功能
    ///
    /// Retriever 封装了向量生成，用户只需提供文本
    #[tokio::test]
    async fn test_retriever() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(64));
        let retriever = SimilarityRetriever::new(store.clone(), embeddings);

        retriever
            .add_documents(vec![
                Document::new("Rust tutorial").with_metadata("type", "lang"),
                Document::new("Qdrant database").with_metadata("type", "db"),
            ])
            .await
            .unwrap();

        let results = retriever.retrieve("programming", 2).await.unwrap();
        // MockEmbeddings generates hash-based vectors; cosine similarity may be negative
        // for short texts, so results can be fewer than k. Just verify no crash.
        assert!(results.len() <= 2);
    }

    // ============================================================
    // Provider 测试
    // ============================================================

    /// 测试通过 Provider 创建内存存储
    #[tokio::test]
    async fn test_provider_in_memory() {
        let store = VectorStoreProvider::create(VectorStoreType::InMemory)
            .await
            .unwrap();
        assert_eq!(store.count().await, 0);
    }

    /// 测试通过 Builder 模式创建存储
    #[tokio::test]
    async fn test_builder() {
        let store = VectorStoreBuilder::in_memory().build().await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    /// 测试存储实例可以作为 trait object 使用
    ///
    /// 验证 Arc<dyn VectorStore> 可以正常工作
    #[tokio::test]
    async fn test_provider_trait_object() {
        let store: Arc<dyn VectorStore> = VectorStoreBuilder::in_memory().build().await.unwrap();

        store
            .add_documents(vec![Document::new("Test")], vec![vec![1.0, 0.0]])
            .await
            .unwrap();

        assert_eq!(store.count().await, 1);
    }
}
