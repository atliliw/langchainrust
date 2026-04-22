// src/vector_stores/qdrant_impl.rs
//! Qdrant 向量存储实现

use super::{Document, SearchResult, VectorStore, VectorStoreError};
use async_trait::async_trait;
use std::sync::Arc;
use uuid::Uuid;

#[cfg(feature = "qdrant-integration")]
use qdrant_client::Qdrant;
#[cfg(feature = "qdrant-integration")]
use qdrant_client::qdrant::{
    CreateCollection, CollectionConfig, VectorsConfig, VectorParams,
    PointStruct, UpsertPointsBuilder, SearchPointsBuilder, WithPayloadSelector,
    DeletePointsBuilder, PointsIdsList,
};

/// Qdrant 配置
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    pub url: String,
    pub collection_name: String,
    pub vector_size: usize,
}

impl QdrantConfig {
    pub fn new(url: impl Into<String>, collection: impl Into<String>, vector_size: usize) -> Self {
        Self {
            url: url.into(),
            collection_name: collection.into(),
            vector_size,
        }
    }

    pub fn from_env(vector_size: usize) -> Self {
        let url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        let collection = std::env::var("QDRANT_COLLECTION")
            .unwrap_or_else(|_| "langchainrust".to_string());
        
        Self::new(url, collection, vector_size)
    }
}

/// Qdrant 向量存储
pub struct QdrantVectorStore {
    config: QdrantConfig,
    #[cfg(feature = "qdrant-integration")]
    client: Arc<Qdrant>,
    #[cfg(not(feature = "qdrant-integration"))]
    _client_placeholder: (),
}

impl QdrantVectorStore {
    #[cfg(feature = "qdrant-integration")]
    pub async fn new(config: QdrantConfig) -> Result<Self, VectorStoreError> {
        let client = Qdrant::from_url(&config.url).build();
        
        let collection_exists = client
            .collection_exists(&config.collection_name)
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !collection_exists.result.unwrap_or(false) {
            client
                .create_collection(&CreateCollection {
                    collection_name: config.collection_name.clone(),
                    vectors_config: Some(VectorsConfig::Single(VectorParams {
                        size: config.vector_size as u64,
                        distance: qdrant_client::qdrant::Distance::Cosine,
                        ..Default::default()
                    })),
                    ..Default::default()
                })
                .await
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        }

        Ok(Self {
            config,
            client: Arc::new(client),
        })
    }

    #[cfg(not(feature = "qdrant-integration"))]
    pub async fn new(config: QdrantConfig) -> Result<Self, VectorStoreError> {
        if !config.url.contains("://") {
            return Err(VectorStoreError::ConnectionError(
                "Qdrant URL 格式无效，应为 http://host:port 形式".to_string()
            ));
        }
        
        Ok(Self {
            config,
            _client_placeholder: (),
        })
    }

    pub async fn from_env(vector_size: usize) -> Result<Self, VectorStoreError> {
        Self::new(QdrantConfig::from_env(vector_size)).await
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    #[cfg(feature = "qdrant-integration")]
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "文档数量和嵌入向量数量不匹配".to_string()
            ));
        }

        for embedding in &embeddings {
            if embedding.len() != self.config.vector_size {
                return Err(VectorStoreError::StorageError(format!(
                    "向量维度不匹配: 期望 {}, 实际 {}",
                    self.config.vector_size,
                    embedding.len()
                )));
            }
        }

        let ids: Vec<String> = documents.iter()
            .map(|_| Uuid::new_v4().to_string())
            .collect();

        let points: Vec<PointStruct> = documents
            .iter()
            .zip(embeddings.iter())
            .zip(ids.iter())
            .map(|((doc, embedding), id)| {
                let payload = serde_json::to_value(&doc.metadata)
                    .unwrap_or(serde_json::Value::Null);
                
                PointStruct {
                    id: Some(qdrant_client::qdrant::PointId::Uuid(id.clone())),
                    vectors: Some(qdrant_client::qdrant::Vectors::Single(embedding.clone())),
                    payload: Some(payload.as_object().unwrap().clone()),
                }
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                &self.config.collection_name,
                points,
            ).wait(true))
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(ids)
    }

    #[cfg(not(feature = "qdrant-integration"))]
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "文档数量和嵌入向量数量不匹配".to_string()
            ));
        }

        for embedding in &embeddings {
            if embedding.len() != self.config.vector_size {
                return Err(VectorStoreError::StorageError(format!(
                    "向量维度不匹配: 期望 {}, 实际 {}",
                    self.config.vector_size,
                    embedding.len()
                )));
            }
        }

        let ids: Vec<String> = documents.iter()
            .map(|_| Uuid::new_v4().to_string())
            .collect();

        Ok(ids)
    }

    #[cfg(feature = "qdrant-integration")]
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.config.vector_size {
            return Err(VectorStoreError::StorageError(format!(
                "查询向量维度不匹配: 期望 {}, 实际 {}",
                self.config.vector_size,
                query_embedding.len()
            )));
        }

        let result = self.client
            .search_points(SearchPointsBuilder::new(
                &self.config.collection_name,
                query_embedding.to_vec(),
                k as u64,
            ).with_payload(WithPayloadSelector::Enable(true)))
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let results: Vec<SearchResult> = result
            .result
            .unwrap_or_default()
            .iter()
            .map(|point| {
                let content = point.payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                
                SearchResult {
                    document: Document {
                        content,
                        metadata: point.payload.clone(),
                    },
                    score: 1.0 - point.score.unwrap_or(0.0),
                }
            })
            .collect();

        Ok(results)
    }

    #[cfg(not(feature = "qdrant-integration"))]
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.config.vector_size {
            return Err(VectorStoreError::StorageError(format!(
                "查询向量维度不匹配: 期望 {}, 实际 {}",
                self.config.vector_size,
                query_embedding.len()
            )));
        }
        Ok(vec![])
    }

    #[cfg(feature = "qdrant-integration")]
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let result = self.client
            .get_point(
                &self.config.collection_name,
                qdrant_client::qdrant::PointId::Uuid(id.to_string()),
                true,
                None,
            )
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let point = result.result;
        if let Some(point) = point {
            let content = point.payload
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            
            Ok(Some(Document {
                content,
                metadata: point.payload.clone(),
            }))
        } else {
            Ok(None)
        }
    }

    #[cfg(not(feature = "qdrant-integration"))]
    async fn get_document(&self, _id: &str) -> Result<Option<Document>, VectorStoreError> {
        Ok(None)
    }

    #[cfg(feature = "qdrant-integration")]
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        self.client
            .delete_points(DeletePointsBuilder::new(
                &self.config.collection_name,
                PointsIdsList {
                    points_ids: vec![qdrant_client::qdrant::PointId::Uuid(id.to_string())],
                },
            ).wait(true))
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(())
    }

    #[cfg(not(feature = "qdrant-integration"))]
    async fn delete_document(&self, _id: &str) -> Result<(), VectorStoreError> {
        Ok(())
    }

    #[cfg(feature = "qdrant-integration")]
    async fn count(&self) -> usize {
        self.client
            .collection_info(&self.config.collection_name)
            .await
            .ok()
            .and_then(|info| info.result)
            .and_then(|result| result.points_count)
            .unwrap_or(0)
    }

    #[cfg(not(feature = "qdrant-integration"))]
    async fn count(&self) -> usize {
        0
    }

    #[cfg(feature = "qdrant-integration")]
    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.client
            .delete_collection(&self.config.collection_name)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        self.client
            .create_collection(&CreateCollection {
                collection_name: self.config.collection_name.clone(),
                vectors_config: Some(VectorsConfig::Single(VectorParams {
                    size: self.config.vector_size as u64,
                    distance: qdrant_client::qdrant::Distance::Cosine,
                    ..Default::default()
                })),
                ..Default::default()
            })
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(())
    }

    #[cfg(not(feature = "qdrant-integration"))]
    async fn clear(&self) -> Result<(), VectorStoreError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_creation() {
        let config = QdrantConfig::new("http://localhost:6334", "my_collection", 128);
        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.collection_name, "my_collection");
        assert_eq!(config.vector_size, 128);
    }
    
    #[test]
    #[cfg(not(feature = "qdrant-integration"))]
    fn test_invalid_url_format() {
        let result = tokio_test::block_on(async {
            QdrantVectorStore::new(QdrantConfig::new("invalid-url", "test", 128)).await
        });
        assert!(result.is_err());
    }
}