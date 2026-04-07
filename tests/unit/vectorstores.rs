#[cfg(test)]
mod test_vectorstore_provider {
    use crate::vector_stores::{VectorStoreProvider, VectorStoreType};

    #[tokio::test]
    async fn test_create_in_memory() {
        let result = VectorStoreProvider::create(VectorStoreType::InMemory).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_builder_in_memory() {
        use crate::vector_stores::VectorStoreBuilder;
        let builder = VectorStoreBuilder::in_memory();
        let store = builder.build().await;
        assert!(store.is_ok());
    }
    
    #[tokio::test]
    async fn test_builder_file_backed() {
        use crate::vector_stores::VectorStoreBuilder;
        let builder = VectorStoreBuilder::file_backed();
        let store = builder.build().await;
        // 文件后端暂时回退到内存存储
        assert!(store.is_ok());
    }
    
    #[tokio::test]
    #[ignore = "Requires real Qdrant service"]
    async fn test_builder_qdrant() {
        use crate::vector_stores::VectorStoreBuilder;
        let builder = VectorStoreBuilder::qdrant("http://localhost:6334", "test_collection");
        let store = builder.build().await;
        // 要在实际的 Qdrant 服务运行时才应工作
        // 珿前的实现会回退到内存存储
        assert!(store.is_ok());
    }
}