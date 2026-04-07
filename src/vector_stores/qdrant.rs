// src/vector_stores/qdrant.rs
//! Qdrant 向量存储实现
//!
//! 使用 Qdrant 向量数据库进行文档存储和检索，支持持久化。
#![cfg(feature = "qdrant-integration")]

use super::{Document, SearchResult, VectorStore, VectorStoreError};
use async_trait::async_trait;
use qdrant_client::{
    prelude::*,
    qdrant::{
        CreateCollection, Distance, PointStruct, SearchPoints, VectorParams, VectorsConfig,
        Payload, Value,
        vectors_config::Config as VectorConfig,
    },
};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Qdrant 配置
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Qdrant 服务 URL
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
    /// 创建新配置
    pub fn new(url: impl Into<String>, collection_name: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            collection_name: collection_name.into(),
            ..Default::default()
        }
    }

    /// 设置向量维度
    pub fn with_vector_size(mut self, size: usize) -> Self {
        self.vector_size = size;
        self
    }

    /// 设置距离度量
    pub fn with_distance(mut self, distance: QdrantDistance) -> Self {
        self.distance = distance;
        self
    }
}

/// Qdrant 向量存储
pub struct QdrantVectorStore {
    client: Arc<QdrantClient>,
    config: QdrantConfig,
}

impl QdrantVectorStore {
    /// 创建新的 Qdrant 向量存储
    ///
    /// 会自动创建集合（如果不存在）
    pub async fn new(config: QdrantConfig) -> Result<Self, VectorStoreError> {
        let client = QdrantClient::from_url(&config.url).build()
            .map_err(|e| VectorStoreError::ConnectionError(format!("连接 Qdrant 失败: {}", e)))?;

        let client = Arc::new(client);

        // 检查集合是否存在
        let exists = client.collection_exists(&config.collection_name).await
            .map_err(|e| VectorStoreError::StorageError(format!("检查集合失败: {}", e)))?;
        
        if !exists {
            // 使用较新 API 样式的创建集合
            client.create_collection(&CreateCollection {
                collection_name: config.collection_name.clone(),
                vectors_config: Some(qdrant_client::qdrant::VectorsConfig {
                    config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                        VectorParams {
                            size: config.vector_size as u64,
                            distance: Distance::from(config.distance),
                            ..Default::default()
                        })),
                }),
                ..Default::default()
            }).await
            .map_err(|e| VectorStoreError::StorageError(format!("创建集合失败: {}", e)))?;
        }

        Ok(Self { client, config })
    }

    /// 从环境变量创建
    pub async fn from_env() -> Result<Self, VectorStoreError> {
        let url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        let collection_name = std::env::var("QDRANT_COLLECTION")
            .unwrap_or_else(|_| "langchainrust".to_string());

        Self::new(QdrantConfig::new(url, collection_name)).await
    }

    /// 将文档转换为 payload
    fn document_to_payload(doc: &Document) -> Payload {
        let mut payload_map: HashMap<String, Value> = HashMap::new();
        
        payload_map.insert("content".to_string(), Value::from(doc.content.clone()));
        
        // 将 metadata 转换为 JSON value
        let metadata_json = serde_json::to_value(&doc.metadata)
            .unwrap_or(serde_json::Value::Null);
        payload_map.insert("metadata".to_string(), Value::from(metadata_json));
        
        if let Some(ref id) = doc.id {
            payload_map.insert("doc_id".to_string(), Value::from(id.clone()));
        }

        Payload::from(payload_map)
    }

    /// 从 payload 提取文档
    fn payload_to_document(payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>) -> Document {
        let content = payload.get("content")
            .and_then(|v| {
                if let Value::StringValue(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .unwrap_or_default()
            .to_string();

        let metadata: std::collections::HashMap<String, String> = payload.get("metadata")
            .and_then(|v| {
                // 尝试将其转换为 Value 类型，然后转换为 JSON
                if let Value::JsonValue(json_val) = v {
                    serde_json::from_value(json_val.clone().into()).ok()
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let id = payload.get("doc_id")
            .and_then(|v| {
                if let Value::StringValue(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .map(|s| s.to_string());

        Document {
            content,
            metadata,
            id,
        }
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
                "文档数量和嵌入向量数量不匹配".to_string()
            ));
        }

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // 验证向量维度
        for embedding in &embeddings {
            if embedding.len() != self.config.vector_size {
                return Err(VectorStoreError::StorageError(format!(
                    "向量维度不匹配: 期望 {}, 实际 {}",
                    self.config.vector_size,
                    embedding.len()
                )));
            }
        }

        let mut ids = Vec::new();
        let mut points = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings) {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

            let payload = Self::document_to_payload(&doc);

            let point = PointStruct::new(
                id.clone(),
                embedding,
                Some(payload),
            );

            points.push(point);
            ids.push(id);
        }

        // 批量插入
        self.client
            .upsert_points(&self.config.collection_name, points, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("插入文档失败: {}", e)))?;

        Ok(ids)
    }

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

        let search_result = self.client
            .search_points(&SearchPoints {
                collection_name: self.config.collection_name.clone(),
                vector: query_embedding.to_vec(),
                limit: k as u64,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("搜索失败: {}", e)))?;

        let results: Vec<SearchResult> = search_result.result.into_iter()
            .map(|scored_point| {
                let payload = scored_point.payload.unwrap_or_default();
                let document = Self::payload_to_document(&payload);

                SearchResult {
                    document,
                    score: scored_point.score,
                }
            })
            .collect();

        Ok(results)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let results = self.client
            .get_points(
                &self.config.collection_name,
                &[id.to_string()],
                Some(true.into()), // with_payload
                None, // with_vectors 
            )
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("获取文档失败: {}", e)))?;

        if let Some(point) = results.result.first() {
            let payload = point.payload.clone().unwrap_or_default();
            let document = Self::payload_to_document(&payload);
            Ok(Some(document))
        } else {
            Ok(None)
        }
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
        use qdrant_client::qdrant::{PointsSelector, PointsIdsList, PointId};

        let points_selector = PointsSelector {
            points_selector_one_of: Some(PointsSelectorOneOf::Points(PointsIdsList {
                ids: vec![PointId::from(id.to_string())],
            })),
        };

        self.client
            .delete_points(&self.config.collection_name, &points_selector, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("删除文档失败: {}", e)))?;

        Ok(())
    }

    async fn count(&self) -> usize {
        // 使用集合信息获取点数量
        let info = self.client
            .collection_info(&self.config.collection_name)
            .await;

        info.map(|i| i.result.and_then(|r| r.points_count).unwrap_or(0) as usize).unwrap_or(0)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // 删除并重建集合
        let collection_name = self.config.collection_name.clone();

        self.client
            .delete_collection(&collection_name)
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("删除集合失败: {}", e)))?;

        self.client
            .create_collection(&CreateCollection {
                collection_name,
                vectors_config: Some(qdrant_client::qdrant::VectorsConfig {
                    config: Some(VectorConfig::Params(VectorParams {
                        size: self.config.vector_size as u64,
                        distance: Distance::from(self.config.distance),
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("重建集合失败: {}", e)))?;

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

    #[tokio::test]
    #[ignore = "需要 Qdrant 服务运行"]
    async fn test_qdrant_integration() {
        let config = QdrantConfig::new("http://localhost:6334", "test_collection")
            .with_vector_size(3);

        let store = QdrantVectorStore::new(config).await.unwrap();

        let docs = vec![
            Document::new("Document 1"),
            Document::new("Document 2"),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];

        let ids = store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(ids.len(), 2);

        let results = store.similarity_search(&[0.9, 0.1, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);

        // 清理
        store.clear().await.unwrap();
    }
}