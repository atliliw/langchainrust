// lc-vector-stores/src/qdrant.rs
//! Qdrant 向量存储实现

use crate::{Document, SearchResult, VectorStore, VectorStoreError};
use async_trait::async_trait;
use qdrant_client::{
    qdrant::{
        Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointId,
        PointStruct, QueryPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Qdrant 配置
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Qdrant 服务地址
    pub url: String,
    /// 集合名称
    pub collection_name: String,
    /// 向量维度
    pub vector_size: usize,
    /// 距离度量方式
    pub distance: QdrantDistance,
}

/// Qdrant 距离度量类型
#[derive(Debug, Clone, Copy)]
pub enum QdrantDistance {
    /// 余弦相似度
    Cosine,
    /// 欧几里得距离
    Euclid,
    /// 点积
    Dot,
}

impl From<QdrantDistance> for Distance {
    fn from(dist: QdrantDistance) -> Self {
        match dist {
            QdrantDistance::Cosine => Distance::Cosine,
            QdrantDistance::Euclid => Distance::Euclid,
            QdrantDistance::Dot => Distance::Dot,
        }
    }
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            collection_name: "langchainrust".to_string(),
            vector_size: 1536,
            distance: QdrantDistance::Cosine,
        }
    }
}

impl QdrantConfig {
    /// 使用服务地址和集合名创建配置,其余字段取默认值。
    pub fn new(url: impl Into<String>, collection_name: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            collection_name: collection_name.into(),
            ..Default::default()
        }
    }

    /// 设置向量维度。
    pub fn with_vector_size(mut self, size: usize) -> Self {
        self.vector_size = size;
        self
    }

    /// 设置距离度量方式。
    pub fn with_distance(mut self, distance: QdrantDistance) -> Self {
        self.distance = distance;
        self
    }
}

/// Qdrant 向量存储
pub struct QdrantVectorStore {
    client: Arc<Qdrant>,
    config: QdrantConfig,
}

impl QdrantVectorStore {
    /// 根据配置连接 Qdrant,若集合不存在则自动创建。
    pub async fn new(config: QdrantConfig) -> Result<Self, VectorStoreError> {
        let client = Qdrant::from_url(&config.url).build().map_err(|e| {
            VectorStoreError::ConnectionError(format!("failed to connect to Qdrant: {}", e))
        })?;

        let client = Arc::new(client);

        let exists = client
            .collection_exists(&config.collection_name)
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to check collection: {}", e))
            })?;

        if !exists {
            client
                .create_collection(
                    CreateCollectionBuilder::new(&config.collection_name).vectors_config(
                        VectorParamsBuilder::new(
                            config.vector_size as u64,
                            Distance::from(config.distance),
                        ),
                    ),
                )
                .await
                .map_err(|e| {
                    VectorStoreError::StorageError(format!("failed to create collection: {}", e))
                })?;
        }

        Ok(Self { client, config })
    }

    /// 从环境变量 `QDRANT_URL` 和 `QDRANT_COLLECTION` 读取配置创建存储。
    pub async fn from_env() -> Result<Self, VectorStoreError> {
        let url =
            std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
        let collection_name =
            std::env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "langchainrust".to_string());

        Self::new(QdrantConfig::new(url, collection_name)).await
    }

    /// 按 metadata 键值匹配删除点,返回实际删除数量。
    pub async fn delete_by_metadata(
        &self,
        key: &str,
        value: &str,
    ) -> Result<usize, VectorStoreError> {
        let filter = Filter::must([Condition::matches(key, value.to_string())]);

        // Q4: 先按 metadata 过滤统计匹配点,再删除,返回真实删除数。
        // 旧实现删完直接 Ok(0) —— 无论删除是否生效,上层都误以为"没删任何数据"。
        let total = self.count().await as u64;
        let matched = self
            .client
            .query(
                QueryPointsBuilder::new(&self.config.collection_name)
                    .query(vec![0.0; self.config.vector_size])
                    .filter(filter.clone())
                    .limit(total.max(1))
                    .with_payload(false),
            )
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!(
                    "failed to count matching points by metadata: {}",
                    e
                ))
            })?;

        let deleted = matched.result.len();

        if deleted > 0 {
            self.client
                .delete_points(
                    DeletePointsBuilder::new(&self.config.collection_name).points(filter),
                )
                .await
                .map_err(|e| {
                    VectorStoreError::StorageError(format!(
                        "failed to delete points by metadata: {}",
                        e
                    ))
                })?;
        }

        Ok(deleted)
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "document count and embedding count mismatch".to_string(),
            ));
        }

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        for embedding in &embeddings {
            if embedding.len() != self.config.vector_size {
                return Err(VectorStoreError::StorageError(format!(
                    "vector dimension mismatch: expected {}, got {}",
                    self.config.vector_size,
                    embedding.len()
                )));
            }
        }

        let mut ids = Vec::new();
        let mut points = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings) {
            let user_id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

            // Qdrant PointId 只接受 UUID 或数字，所以生成内部 UUID
            let internal_uuid = Uuid::new_v4();
            let point_id = PointId::from(internal_uuid.to_string());

            let mut payload = Payload::new();
            payload.insert("content", doc.content.clone());
            payload.insert("doc_id", user_id.clone()); // 用户 ID 存在 payload 中

            for (key, value) in &doc.metadata {
                payload.insert(key.clone(), value.clone());
            }

            let point = PointStruct::new(point_id, embedding, payload);
            points.push(point);
            ids.push(user_id);
        }

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                &self.config.collection_name,
                points,
            ))
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to insert documents: {}", e))
            })?;

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.config.vector_size {
            return Err(VectorStoreError::StorageError(format!(
                "query vector dimension mismatch: expected {}, got {}",
                self.config.vector_size,
                query_embedding.len()
            )));
        }

        let search_result = self
            .client
            .query(
                QueryPointsBuilder::new(&self.config.collection_name)
                    .query(query_embedding.to_vec())
                    .limit(k as u64)
                    .with_payload(true),
            )
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("search failed: {}", e)))?;

        let results: Vec<SearchResult> = search_result
            .result
            .into_iter()
            .map(|scored_point| {
                let payload = scored_point.payload;

                let content = payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();

                let id = payload
                    .get("doc_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let mut metadata = HashMap::new();
                for (key, value) in &payload {
                    if key != "content" && key != "doc_id" {
                        if let Some(s) = value.as_str() {
                            metadata.insert(key.clone(), s.clone().into());
                        }
                    }
                }

                SearchResult {
                    document: Document {
                        content,
                        metadata,
                        id,
                    },
                    score: scored_point.score,
                }
            })
            .collect();

        Ok(results)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let filter = Filter::must([Condition::matches("doc_id", id.to_string())]);

        let results = self
            .client
            .query(
                QueryPointsBuilder::new(&self.config.collection_name)
                    .query(vec![0.0; self.config.vector_size])
                    .filter(filter)
                    .limit(1)
                    .with_payload(true),
            )
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to get document: {}", e))
            })?;

        if let Some(point) = results.result.first() {
            let payload_map = point.payload.clone();

            let content = payload_map
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.as_str())
                .unwrap_or("")
                .to_string();

            let doc_id = payload_map
                .get("doc_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let mut metadata = HashMap::new();
            for (key, value) in &payload_map {
                if key != "content" && key != "doc_id" {
                    if let Some(s) = value.as_str() {
                        metadata.insert(key.clone(), s.clone().into());
                    }
                }
            }

            Ok(Some(Document {
                content,
                metadata,
                id: doc_id,
            }))
        } else {
            Ok(None)
        }
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let filter = Filter::must([Condition::matches("doc_id", id.to_string())]);

        let results = self
            .client
            .query(
                QueryPointsBuilder::new(&self.config.collection_name)
                    .query(vec![0.0; self.config.vector_size])
                    .filter(filter)
                    .limit(1)
                    .with_payload(true),
            )
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("failed to get vector: {}", e)))?;

        if let Some(point) = results.result.first() {
            if let Some(vectors) = &point.vectors {
                if let Some(qdrant_client::qdrant::vector_output::Vector::Dense(dense)) =
                    vectors.get_vector()
                {
                    return Ok(Some(dense.data.clone()));
                }
            }
        }
        Ok(None)
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let filter = Filter::must([Condition::matches("doc_id", id.to_string())]);

        self.client
            .delete_points(DeletePointsBuilder::new(&self.config.collection_name).points(filter))
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to delete document: {}", e))
            })?;

        Ok(())
    }

    async fn count(&self) -> usize {
        let info = self
            .client
            .collection_info(&self.config.collection_name)
            .await;

        info.map(|i| i.result.and_then(|r| r.points_count).unwrap_or(0) as usize)
            .unwrap_or(0)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let collection_name = self.config.collection_name.clone();

        self.client
            .delete_collection(&collection_name)
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to delete collection: {}", e))
            })?;

        self.client
            .create_collection(
                CreateCollectionBuilder::new(&collection_name).vectors_config(
                    VectorParamsBuilder::new(
                        self.config.vector_size as u64,
                        Distance::from(self.config.distance),
                    ),
                ),
            )
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to recreate collection: {}", e))
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QdrantConfig::default();
        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.collection_name, "langchainrust");
        assert_eq!(config.vector_size, 1536);
    }

    #[test]
    fn test_config_builder() {
        let config = QdrantConfig::new("http://custom:6334", "test_collection")
            .with_vector_size(3072)
            .with_distance(QdrantDistance::Euclid);

        assert_eq!(config.url, "http://custom:6334");
        assert_eq!(config.collection_name, "test_collection");
        assert_eq!(config.vector_size, 3072);
        assert!(matches!(config.distance, QdrantDistance::Euclid));
    }
}
