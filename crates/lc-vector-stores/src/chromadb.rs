// lc-vector-stores/src/chromadb.rs
//! ChromaDB 向量存储实现（HTTP API）
//!
//! 使用 ChromaDB 的 REST API 进行向量存储和检索。
//! 支持连接远程 ChromaDB 服务（docker run -p 8000:8000 chromadb/chroma）。

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

use crate::{Document, FilterOp, MetadataFilter, SearchResult, VectorStore, VectorStoreError};

/// ChromaDB 配置
#[derive(Debug, Clone)]
pub struct ChromaDBConfig {
    /// ChromaDB 服务地址，默认为 http://localhost:8000
    pub host: String,
    /// 集合名称
    pub collection_name: String,
    /// 向量维度
    pub vector_size: usize,
    /// 集合元数据（可选）
    pub metadata: Option<HashMap<String, String>>,
}

impl Default for ChromaDBConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:8000".to_string(),
            collection_name: "langchainrust".to_string(),
            vector_size: 1536,
            metadata: None,
        }
    }
}

impl ChromaDBConfig {
    /// 创建新的 ChromaDB 配置
    pub fn new(
        host: impl Into<String>,
        collection_name: impl Into<String>,
        vector_size: usize,
    ) -> Self {
        Self {
            host: host.into(),
            collection_name: collection_name.into(),
            vector_size,
            metadata: None,
        }
    }
}

/// ChromaDB 集合信息（从 API 返回解析）
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChromaCollection {
    id: String,
    name: String,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

/// ChromaDB add 请求体
#[derive(Debug, Serialize)]
struct ChromaAddRequest {
    ids: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadatas: Option<Vec<HashMap<String, serde_json::Value>>>,
}

/// ChromaDB query 请求体
#[derive(Debug, Serialize)]
struct ChromaQueryRequest {
    query_embeddings: Vec<Vec<f32>>,
    n_results: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    /// Chroma `where` 过滤字典(见 [`filter_to_chroma`])
    #[serde(rename = "where", skip_serializing_if = "Option::is_none")]
    where_filter: Option<serde_json::Value>,
}

/// ChromaDB query 响应
#[derive(Debug, Deserialize)]
struct ChromaQueryResponse {
    ids: Vec<Vec<String>>,
    distances: Vec<Vec<f64>>,
    documents: Vec<Vec<String>>,
    #[serde(default)]
    metadatas: Vec<Vec<Option<HashMap<String, serde_json::Value>>>>,
}

/// ChromaDB get 响应
#[derive(Debug, Deserialize)]
struct ChromaGetResponse {
    ids: Vec<String>,
    documents: Vec<Option<String>>,
    #[serde(default)]
    metadatas: Vec<Option<HashMap<String, serde_json::Value>>>,
    embeddings: Option<Vec<Vec<f32>>>,
}

/// ChromaDB 向量存储
///
/// 通过 HTTP API 连接 ChromaDB 服务。
///
/// # 示例
/// ```ignore
/// use lc_vector_stores::ChromaDBVectorStore;
///
/// let store = ChromaDBVectorStore::new(
///     ChromaDBConfig::new("http://localhost:8000", "my_collection", 384)
/// ).await?;
/// ```
pub struct ChromaDBVectorStore {
    config: ChromaDBConfig,
    client: reqwest::Client,
    collection_id: Option<String>,
}

impl ChromaDBVectorStore {
    /// 创建 ChromaDB 向量存储并自动初始化集合
    pub async fn new(config: ChromaDBConfig) -> Result<Self, VectorStoreError> {
        let client = reqwest::Client::new();
        let mut store = Self {
            config,
            client,
            collection_id: None,
        };
        store.init_collection().await?;
        Ok(store)
    }

    /// 初始化或获取集合
    async fn init_collection(&mut self) -> Result<(), VectorStoreError> {
        // 尝试获取已有集合
        let url = format!(
            "{}/api/v1/collections/{}",
            self.config.host, self.config.collection_name
        );
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if response.status().is_success() {
            let collection: ChromaCollection = response.json().await.map_err(|e| {
                VectorStoreError::StorageError(format!("failed to parse collection info: {}", e))
            })?;
            self.collection_id = Some(collection.id);
            return Ok(());
        }

        // 集合不存在，创建新集合
        let create_url = format!("{}/api/v1/collections", self.config.host);
        let mut body = json!({
            "name": self.config.collection_name,
        });

        if let Some(ref meta) = self.config.metadata {
            body["metadata"] = serde_json::to_value(meta).unwrap_or(json!({}));
        }

        let response = self
            .client
            .post(&create_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if response.status().is_success() {
            let collection: ChromaCollection = response.json().await.map_err(|e| {
                VectorStoreError::StorageError(format!(
                    "failed to parse new collection info: {}",
                    e
                ))
            })?;
            self.collection_id = Some(collection.id);
            Ok(())
        } else {
            let text = response.text().await.unwrap_or_default();
            Err(VectorStoreError::StorageError(format!(
                "failed to create collection: {}",
                text
            )))
        }
    }

    /// 获取集合 ID
    fn get_collection_id(&self) -> Result<&str, VectorStoreError> {
        self.collection_id.as_deref().ok_or_else(|| {
            VectorStoreError::StorageError("collection is not initialized".to_string())
        })
    }

    /// 构建集合 API 基础 URL
    fn collection_url(&self, endpoint: &str) -> Result<String, VectorStoreError> {
        let cid = self.get_collection_id()?;
        Ok(format!(
            "{}/api/v1/collections/{}/{}",
            self.config.host, cid, endpoint
        ))
    }

    /// 构造 Chroma query 请求体(pure function,便于测试)。
    fn query_request(
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> ChromaQueryRequest {
        ChromaQueryRequest {
            query_embeddings: vec![query_embedding.to_vec()],
            n_results: k,
            include: Some(vec![
                "documents".to_string(),
                "distances".to_string(),
                "metadatas".to_string(),
            ]),
            where_filter: filter.map(filter_to_chroma),
        }
    }

    /// POST `/query` 并解析结果(普通与过滤检索共用)。
    async fn query_impl(
        &self,
        request: ChromaQueryRequest,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = self.collection_url("query")?;
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "query failed: {}",
                text
            )));
        }

        let query_result: ChromaQueryResponse = response.json().await.map_err(|e| {
            VectorStoreError::StorageError(format!("failed to parse query results: {}", e))
        })?;

        let mut results = Vec::new();

        // ChromaDB 返回嵌套数组（每个 query 一个结果集）
        if let Some(doc_list) = query_result.documents.into_iter().next() {
            let dist_list = query_result
                .distances
                .into_iter()
                .next()
                .unwrap_or_default();
            let meta_list = query_result
                .metadatas
                .into_iter()
                .next()
                .unwrap_or_default();
            let id_list = query_result.ids.into_iter().next().unwrap_or_default();

            for (i, content) in doc_list.into_iter().enumerate() {
                let score = dist_list.get(i).copied().unwrap_or(0.0);
                // ChromaDB 返回的是 L2 距离，转换为相似度分数（1 / (1 + dist)）
                let similarity = 1.0 / (1.0 + score);
                let metadata = meta_list
                    .get(i)
                    .unwrap_or(&None)
                    .clone()
                    .unwrap_or_default();
                let doc_id = id_list.get(i).cloned();

                results.push(SearchResult {
                    document: Document {
                        content,
                        metadata,
                        id: doc_id,
                    },
                    score: similarity as f32,
                });
            }
        }

        // 按相似度降序排序
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }
}

#[async_trait]
impl VectorStore for ChromaDBVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let count = documents.len();
        let ids: Vec<String> = (0..count)
            .map(|i| {
                documents[i]
                    .id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
            })
            .collect();

        let contents: Vec<String> = documents.iter().map(|d| d.content.clone()).collect();
        let metadatas: Vec<HashMap<String, serde_json::Value>> =
            documents.iter().map(|d| d.metadata.clone()).collect();
        let has_metadata = metadatas.iter().any(|m| !m.is_empty());

        let request = ChromaAddRequest {
            ids: ids.clone(),
            embeddings,
            documents: contents,
            metadatas: if has_metadata { Some(metadatas) } else { None },
        };

        let url = self.collection_url("add")?;
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "failed to add documents: {}",
                text
            )));
        }

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let request = Self::query_request(query_embedding, k, None);
        self.query_impl(request).await
    }

    /// S3: 带元数据过滤的相似度检索 —— 过滤交给服务端(Chroma `where` 语法)。
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let request = Self::query_request(query_embedding, k, filter);
        self.query_impl(request).await
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let url = self.collection_url("get")?;
        let body = json!({
            "ids": [id],
            "include": ["documents", "metadatas"]
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let get_result: ChromaGetResponse = response.json().await.map_err(|e| {
            VectorStoreError::StorageError(format!("failed to parse document: {}", e))
        })?;

        if get_result.ids.is_empty() {
            return Ok(None);
        }

        let content = get_result
            .documents
            .into_iter()
            .next()
            .flatten()
            .unwrap_or_default();
        let metadata = get_result
            .metadatas
            .into_iter()
            .next()
            .flatten()
            .unwrap_or_default();

        Ok(Some(Document {
            content,
            metadata,
            id: Some(id.to_string()),
        }))
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let url = self.collection_url("get")?;
        let body = json!({
            "ids": [id],
            "include": ["embeddings"]
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let get_result: ChromaGetResponse = response.json().await.map_err(|e| {
            VectorStoreError::StorageError(format!("failed to parse document: {}", e))
        })?;

        if let Some(embeddings) = get_result.embeddings {
            Ok(embeddings.into_iter().next())
        } else {
            Ok(None)
        }
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let url = self.collection_url("delete")?;
        let body = json!({
            "ids": [id]
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "failed to delete document: {}",
                text
            )));
        }

        Ok(())
    }

    async fn count(&self) -> usize {
        let url = match self.collection_url("count") {
            Ok(u) => u,
            Err(e) => {
                log::warn!("ChromaDB count() failed to build URL: {}", e);
                return 0;
            }
        };

        let response = self.client.post(&url).send().await;
        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<usize>().await {
                        Ok(count) => count,
                        Err(e) => {
                            log::warn!("ChromaDB count() failed to parse response: {}", e);
                            0
                        }
                    }
                } else {
                    log::warn!("ChromaDB count() request failed with non-success status");
                    0
                }
            }
            Err(e) => {
                log::warn!("ChromaDB count() request error: {}", e);
                0
            }
        }
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // 获取所有文档 ID 后批量删除
        let get_url = self.collection_url("get")?;
        let body = json!({
            "include": []
        });

        let response = self
            .client
            .post(&get_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "failed to fetch document list: {}",
                text
            )));
        }

        let get_result: ChromaGetResponse = response.json().await.map_err(|e| {
            VectorStoreError::StorageError(format!("failed to parse document list: {}", e))
        })?;

        if get_result.ids.is_empty() {
            return Ok(());
        }

        // 批量删除
        let del_url = self.collection_url("delete")?;
        let del_body = json!({
            "ids": get_result.ids
        });

        let response = self
            .client
            .post(&del_url)
            .json(&del_body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "failed to clear collection: {}",
                text
            )));
        }

        Ok(())
    }
}

/// S3: [`MetadataFilter`] → Chroma `where` 过滤字典。
///
/// 单字段条件翻译成 `{ key: { "$op": value } }`(Chroma v2 支持
/// `$eq $ne $gt $gte $lt $lte $in $nin`),AND/OR 组合翻译成
/// `{ "$and": [...] }` / `{ "$or": [...] }`。与 Pinecone 的翻译同构,
/// 但按后端各自独立维护,语义完全交给服务端执行。
pub fn filter_to_chroma(filter: &MetadataFilter) -> serde_json::Value {
    fn op_str(op: FilterOp) -> &'static str {
        match op {
            FilterOp::Eq => "$eq",
            FilterOp::Ne => "$ne",
            FilterOp::Gt => "$gt",
            FilterOp::Gte => "$gte",
            FilterOp::Lt => "$lt",
            FilterOp::Lte => "$lte",
            FilterOp::In => "$in",
            FilterOp::Nin => "$nin",
        }
    }
    match filter {
        MetadataFilter::Field { key, op, value } => {
            serde_json::json!({ key.clone(): { op_str(*op): value.clone() } })
        }
        MetadataFilter::And(filters) => {
            let items: Vec<serde_json::Value> = filters.iter().map(filter_to_chroma).collect();
            serde_json::json!({ "$and": items })
        }
        MetadataFilter::Or(filters) => {
            let items: Vec<serde_json::Value> = filters.iter().map(filter_to_chroma).collect();
            serde_json::json!({ "$or": items })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S3: 单字段条件 → Chroma `where` 字典。
    #[test]
    fn test_filter_to_chroma_field() {
        assert_eq!(
            filter_to_chroma(&MetadataFilter::field("lang", FilterOp::Eq, "rust")),
            serde_json::json!({ "lang": { "$eq": "rust" } })
        );
        assert_eq!(
            filter_to_chroma(&MetadataFilter::field("year", FilterOp::Lt, 2020)),
            serde_json::json!({ "year": { "$lt": 2020 } })
        );
    }

    /// S3: AND/OR 组合 → `$and`/`$or` 嵌套。
    #[test]
    fn test_filter_to_chroma_and_or() {
        let f = MetadataFilter::or(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "python"),
            MetadataFilter::and(vec![
                MetadataFilter::field("lang", FilterOp::Eq, "rust"),
                MetadataFilter::field("tags", FilterOp::In, vec!["ml"]),
            ]),
        ]);
        assert_eq!(
            filter_to_chroma(&f),
            serde_json::json!({
                "$or": [
                    { "lang": { "$eq": "python" } },
                    { "$and": [
                        { "lang": { "$eq": "rust" } },
                        { "tags": { "$in": ["ml"] } }
                    ]}
                ]
            })
        );
    }

    /// S3: 无过滤时 `where_filter` 为 None,不序列化 `where` 字段。
    #[test]
    fn test_query_request_no_filter() {
        let req = ChromaDBVectorStore::query_request(&[1.0, 2.0], 3, None);
        assert!(req.where_filter.is_none());
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("where").is_none());
        assert_eq!(v["n_results"], 3);
    }

    /// S3: 有过滤时 `where` 字段序列化为 Chroma 字典。
    #[test]
    fn test_query_request_with_filter() {
        let f = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let req = ChromaDBVectorStore::query_request(&[1.0, 2.0], 3, Some(&f));
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["where"], serde_json::json!({ "lang": { "$eq": "rust" } }));
    }
}
