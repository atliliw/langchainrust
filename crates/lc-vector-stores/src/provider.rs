// lc-vector-stores/src/provider.rs
//! 向量存储提供者
//!
//! 提供多种向量存储引擎的选择：内存、持久化、Qdrant等

use crate::{VectorStore, VectorStoreError};
use std::sync::Arc;

/// 向量存储类型枚举
#[derive(Debug, Clone)]
pub enum VectorStoreType {
    /// 内存存储，适用于测试和小型应用
    InMemory,

    /// 文件持久化存储，适用于个人知识库
    FileBacked {
        /// 存储文件路径
        path: String,
        /// 向量维度
        dimension: usize,
    },

    /// Qdrant 向量数据库，适用于生产环境
    Qdrant {
        /// Qdrant 服务地址
        url: String,
        /// 集合名称
        collection: String,
    },
}

/// 向量存储提供者
pub struct VectorStoreProvider;

impl VectorStoreProvider {
    /// 创建向量存储实例
    pub async fn create(
        store_type: VectorStoreType,
    ) -> Result<Arc<dyn VectorStore>, VectorStoreError> {
        match store_type {
            VectorStoreType::InMemory => {
                use crate::InMemoryVectorStore;
                Ok(Arc::new(InMemoryVectorStore::new()))
            }
            VectorStoreType::FileBacked { path, dimension } => {
                use crate::FileVectorStore;
                let store = FileVectorStore::new(std::path::PathBuf::from(path), dimension)
                    .await
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                Ok(Arc::new(store))
            }
            VectorStoreType::Qdrant { url, collection } => {
                Self::create_qdrant_store(url, collection).await
            }
        }
    }

    /// 创建 Qdrant 向量存储
    async fn create_qdrant_store(
        url: String,
        collection: String,
    ) -> Result<Arc<dyn VectorStore>, VectorStoreError> {
        #[cfg(feature = "qdrant-integration")]
        {
            use crate::{QdrantConfig, QdrantVectorStore};
            let config = QdrantConfig::new(url, collection);
            let store = QdrantVectorStore::new(config).await?;
            Ok(Arc::new(store))
        }

        #[cfg(not(feature = "qdrant-integration"))]
        {
            // Q3: 未启用 feature 时显式报错,拒绝静默降级到内存存储 ——
            // 生产代码若以为在写 Qdrant 实际写进内存,进程重启数据即丢。
            Err(VectorStoreError::ConnectionError(format!(
                "Qdrant store requires the 'qdrant-integration' feature (url={url}, collection={collection}); refusing to silently fall back to InMemory, enable the feature in Cargo.toml"
            )))
        }
    }
}

/// 向量存储构建器，提供便利的创建方法
pub struct VectorStoreBuilder {
    store_type: VectorStoreType,
}

impl Default for VectorStoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStoreBuilder {
    /// 创建默认的内存存储构建器
    pub fn new() -> Self {
        Self {
            store_type: VectorStoreType::InMemory,
        }
    }

    /// 创建内存存储构建器
    pub fn in_memory() -> Self {
        Self {
            store_type: VectorStoreType::InMemory,
        }
    }

    /// 创建文件持久化存储构建器
    pub fn file_backed(path: impl Into<String>, dimension: usize) -> Self {
        Self {
            store_type: VectorStoreType::FileBacked {
                path: path.into(),
                dimension,
            },
        }
    }

    /// 创建 Qdrant 存储构建器
    pub fn qdrant(url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            store_type: VectorStoreType::Qdrant {
                url: url.into(),
                collection: collection.into(),
            },
        }
    }

    /// 构建向量存储实例
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

    #[cfg(not(feature = "qdrant-integration"))]
    #[tokio::test]
    async fn test_builder_qdrant_errors_when_feature_disabled() {
        // Q3: 未启用 feature 时必须显式报错,不能静默降级到内存存储。
        let builder = VectorStoreBuilder::qdrant("http://localhost:6334", "test_collection");
        let store = builder.build().await;
        assert!(store.is_err());
    }
}
