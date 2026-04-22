// src/vector_stores/provider.rs
//! 向量存储提供者
//!
//! 提供多种向量存储引擎的选择：内存、持久化、Qdrant等

use crate::vector_stores::{VectorStore, VectorStoreError};
use std::sync::Arc;

/// 向量存储类型枚举
#[derive(Debug, Clone)]
pub enum VectorStoreType {
    /// 内存存储，适用于测试和小型应用
    InMemory,
    
    /// 文件持久化存储，适用于个人知识库
    FileBacked,
    
    /// Qdrant 向量数据库，适用于生产环境
    Qdrant {
        url: String,
        collection: String,
    },
}

/// 向量存储提供者
pub struct VectorStoreProvider;

impl VectorStoreProvider {
    /// 创建向量存储实例
    pub async fn create(store_type: VectorStoreType) -> Result<Arc<dyn VectorStore>, VectorStoreError> {
        match store_type {
            VectorStoreType::InMemory => {
                use crate::vector_stores::InMemoryVectorStore;
                Ok(Arc::new(InMemoryVectorStore::new()))
            }
            VectorStoreType::FileBacked => {
                // 暂时返回内存存储，等待实现文件存储
                use crate::vector_stores::InMemoryVectorStore;
                Ok(Arc::new(InMemoryVectorStore::new()))
            }
            VectorStoreType::Qdrant { url, collection } => {
                Self::create_qdrant_store(url, collection).await
            }
        }
    }

    /// 创建 Qdrant 向量存储
    async fn create_qdrant_store(url: String, collection: String) -> Result<Arc<dyn VectorStore>, VectorStoreError> {
        #[cfg(feature = "qdrant-integration")]
        {
            use crate::vector_stores::{QdrantVectorStore, QdrantConfig};
            let config = QdrantConfig::new(url, collection);
            let store = QdrantVectorStore::new(config).await?;
            Ok(Arc::new(store))
        }
        
        #[cfg(not(feature = "qdrant-integration"))]
        {
            eprintln!("Warning: Qdrant requested but feature 'qdrant-integration' not enabled. Falling back to InMemory store.");
            use crate::vector_stores::InMemoryVectorStore;
            Ok(Arc::new(InMemoryVectorStore::new()))
        }
    }
}

/// 向量存储构建器，提供便利的创建方法
pub struct VectorStoreBuilder {
    store_type: VectorStoreType,
}

impl VectorStoreBuilder {
    pub fn new() -> Self {
        Self {
            store_type: VectorStoreType::InMemory,
        }
    }
    
    pub fn in_memory() -> Self {
        Self {
            store_type: VectorStoreType::InMemory,
        }
    }
    
    pub fn file_backed() -> Self {
        Self {
            store_type: VectorStoreType::FileBacked,
        }
    }
    
    pub fn qdrant(url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            store_type: VectorStoreType::Qdrant {
                url: url.into(),
                collection: collection.into(),
            },
        }
    }
    
    pub async fn build(self) -> Result<Arc<dyn VectorStore>, VectorStoreError> {
        VectorStoreProvider::create(self.store_type).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_in_memory() {
        let result = VectorStoreProvider::create(VectorStoreType::InMemory).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_builder_in_memory() {
        let builder = VectorStoreBuilder::in_memory();
        let store = builder.build().await;
        assert!(store.is_ok());
    }
    
    #[tokio::test]
    async fn test_builder_qdrant_fallback() {
        // 没有 feature 时，应该回退到内存存储
        let builder = VectorStoreBuilder::qdrant("http://localhost:6334", "test_collection");
        let store = builder.build().await;
        assert!(store.is_ok());
    }
}