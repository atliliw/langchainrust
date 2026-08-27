// lc-vector-stores/src/provider.rs
//! Vector store provider
//!
//! Provides a choice of multiple vector store engines: in-memory, file-persistent, Qdrant, etc.

use crate::{VectorStore, VectorStoreError};
use std::sync::Arc;

/// Vector store type enum
#[derive(Debug, Clone)]
pub enum VectorStoreType {
    /// In-memory storage, suitable for tests and small applications
    InMemory,

    /// File-persistent storage, suitable for personal knowledge bases
    FileBacked {
        /// Storage file path
        path: String,
        /// Vector dimension
        dimension: usize,
    },

    /// Qdrant vector database, suitable for production
    Qdrant {
        /// Qdrant service URL
        url: String,
        /// Collection name
        collection: String,
    },
}

/// Vector store provider
pub struct VectorStoreProvider;

impl VectorStoreProvider {
    /// Creates a vector store instance
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

    /// Creates a Qdrant vector store
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
            // Q3: error explicitly when the feature is disabled, refusing to silently fall back
            // to in-memory storage — production code that believes it is writing to Qdrant while
            // actually writing to memory would lose all data on process restart.
            Err(VectorStoreError::ConnectionError(format!(
                "Qdrant store requires the 'qdrant-integration' feature (url={url}, collection={collection}); refusing to silently fall back to InMemory, enable the feature in Cargo.toml"
            )))
        }
    }
}

/// Vector store builder providing convenient creation methods
pub struct VectorStoreBuilder {
    store_type: VectorStoreType,
}

impl Default for VectorStoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStoreBuilder {
    /// Creates the default in-memory store builder
    pub fn new() -> Self {
        Self {
            store_type: VectorStoreType::InMemory,
        }
    }

    /// Creates an in-memory store builder
    pub fn in_memory() -> Self {
        Self {
            store_type: VectorStoreType::InMemory,
        }
    }

    /// Creates a file-persistent store builder
    pub fn file_backed(path: impl Into<String>, dimension: usize) -> Self {
        Self {
            store_type: VectorStoreType::FileBacked {
                path: path.into(),
                dimension,
            },
        }
    }

    /// Creates a Qdrant store builder
    pub fn qdrant(url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            store_type: VectorStoreType::Qdrant {
                url: url.into(),
                collection: collection.into(),
            },
        }
    }

    /// Builds a vector store instance
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
        // Q3: when the feature is disabled, it must error explicitly, never silently fall back to in-memory storage.
        let builder = VectorStoreBuilder::qdrant("http://localhost:6334", "test_collection");
        let store = builder.build().await;
        assert!(store.is_err());
    }
}
