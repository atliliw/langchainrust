//! Pinecone vector store (HTTP API)

use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;

use crate::{Document, Embeddings, SearchResult, VectorStore, VectorStoreError};

/// Pinecone vector store client
pub struct PineconeStore {
    api_key: String,
    host: String,
    client: reqwest::Client,
}

impl PineconeStore {
    /// Create a Pinecone client.
    ///
    /// `host` format: `https://{index-name}.svc.{environment}.pinecone.io`
    pub fn new(api_key: impl Into<String>, host: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            host: host.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Build upsert request body (pure function, convenient for testing).
    pub fn build_upsert_body(docs: &[Document], vectors: &[Vec<f32>]) -> serde_json::Value {
        let vectors_json: Vec<serde_json::Value> = docs
            .iter()
            .zip(vectors.iter())
            .map(|(doc, vec)| {
                serde_json::json!({
                    "id": doc.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    "values": vec,
                    "metadata": doc.metadata,
                })
            })
            .collect();
        serde_json::json!({ "vectors": vectors_json })
    }

    /// Build query request body (pure function, convenient for testing).
    pub fn build_query_body(query_vec: &[f32], top_k: usize) -> serde_json::Value {
        serde_json::json!({
            "vector": query_vec,
            "topK": top_k,
            "includeMetadata": true,
        })
    }

    /// Upsert documents (auto-embed).
    pub async fn upsert(
        &self,
        docs: &[Document],
        embeddings: &dyn Embeddings,
    ) -> Result<(), String> {
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let vectors = embeddings
            .embed_documents(&texts)
            .await
            .map_err(|e| e.to_string())?;
        let body = Self::build_upsert_body(docs, &vectors);
        let url = format!("{}/vectors/upsert", self.host);
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Pinecone upsert error: {}", resp.status()));
        }
        Ok(())
    }

    /// Query similar documents.
    pub async fn query(&self, query_vec: Vec<f32>, top_k: usize) -> Result<Vec<Document>, String> {
        let body = Self::build_query_body(&query_vec, top_k);
        let url = format!("{}/query", self.host);
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Pinecone query error: {}", resp.status()));
        }
        let query_resp: QueryResponse = resp.json().await.map_err(|e| e.to_string())?;
        let result = query_resp
            .matches
            .into_iter()
            .map(|m| {
                let content = m
                    .metadata
                    .as_ref()
                    .and_then(|md| md.get("content").cloned())
                    .unwrap_or_default();
                Document {
                    content,
                    metadata: m.metadata.unwrap_or_default(),
                    id: Some(m.id),
                }
            })
            .collect();
        Ok(result)
    }

    /// 读取索引统计(真实 count 的唯一可靠来源)。
    ///
    /// Pinecone REST 提供 `describe_index_stats`,返回 `totalVectorCount`。
    pub async fn describe_index_stats(&self) -> Result<PineconeIndexStats, String> {
        let url = format!("{}/describe_index_stats", self.host);
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!(
                "Pinecone describe_index_stats error: {}",
                resp.status()
            ));
        }
        resp.json().await.map_err(|e| e.to_string())
    }

    /// Delete by IDs.
    pub async fn delete(&self, ids: &[String]) -> Result<(), String> {
        let url = format!("{}/vectors/delete", self.host);
        let body = serde_json::json!({ "ids": ids });
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Pinecone delete error: {}", resp.status()));
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for PineconeStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        let ids: Vec<String> = documents
            .iter()
            .map(|d| {
                d.id.clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
            })
            .collect();

        // Build upsert body with pre-computed embeddings
        let body = Self::build_upsert_body(&documents, &embeddings);
        let url = format!("{}/vectors/upsert", self.host);
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("Pinecone upsert failed: {}", e))
            })?;

        if !resp.status().is_success() {
            return Err(VectorStoreError::StorageError(format!(
                "Pinecone upsert HTTP error: {}",
                resp.status()
            )));
        }

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let body = Self::build_query_body(query_embedding, k);
        let url = format!("{}/query", self.host);
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("Pinecone query failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(VectorStoreError::StorageError(format!(
                "Pinecone query HTTP error: {}",
                resp.status()
            )));
        }

        let query_resp: QueryResponse = resp.json().await.map_err(|e| {
            VectorStoreError::StorageError(format!("Pinecone query parse error: {}", e))
        })?;

        let results = query_resp
            .matches
            .into_iter()
            .map(|m| {
                let content = m
                    .metadata
                    .as_ref()
                    .and_then(|md| md.get("content").cloned())
                    .unwrap_or_default();
                let doc = Document {
                    content,
                    metadata: m.metadata.unwrap_or_default(),
                    id: Some(m.id.clone()),
                };
                SearchResult {
                    document: doc,
                    score: m.score as f32,
                }
            })
            .collect();

        Ok(results)
    }

    async fn get_document(&self, _id: &str) -> Result<Option<Document>, VectorStoreError> {
        // Pinecone HTTP API doesn't support direct fetch by ID in the basic plan.
        // Use similarity_search with the ID as metadata filter instead.
        Err(VectorStoreError::StorageError(
            "Pinecone does not support direct document fetch by ID via HTTP API".to_string(),
        ))
    }

    async fn get_embedding(&self, _id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        Err(VectorStoreError::StorageError(
            "Pinecone does not support direct embedding fetch by ID via HTTP API".to_string(),
        ))
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        self.delete(&[id.to_string()])
            .await
            .map_err(VectorStoreError::StorageError)
    }

    async fn count(&self) -> usize {
        // Q4: 真实现 —— 通过 describe_index_stats 读取 totalVectorCount,不再写死 0。
        // trait 签名返回 usize,网络失败时退化为 0 并记录日志(不抛错)。
        match self.describe_index_stats().await {
            Ok(stats) => stats.total_vector_count,
            Err(e) => {
                log::warn!("Pinecone count 失败,按 0 处理: {}", e);
                0
            }
        }
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        Err(VectorStoreError::StorageError(
            "Pinecone does not support clearing all vectors via HTTP API. Delete by namespace or IDs instead.".to_string()
        ))
    }
}

#[derive(Deserialize)]
struct QueryResponse {
    matches: Vec<QueryMatch>,
}

#[derive(Deserialize)]
struct QueryMatch {
    id: String,
    score: f64,
    metadata: Option<HashMap<String, String>>,
}

/// Pinecone `describe_index_stats` 响应(只需我们关心的字段)。
///
/// Q4: 提供给 [`PineconeStore::describe_index_stats`],供调用方读取真实向量总数,
/// 也是 [`VectorStore::count`](crate::VectorStore::count) 的数据来源。
#[derive(Deserialize)]
pub struct PineconeIndexStats {
    #[serde(default)]
    pub total_vector_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, content: &str) -> Document {
        Document {
            content: content.to_string(),
            metadata: HashMap::new(),
            id: Some(id.to_string()),
        }
    }

    #[test]
    fn test_build_upsert_body() {
        let docs = vec![doc("1", "hello"), doc("2", "world")];
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let body = PineconeStore::build_upsert_body(&docs, &vectors);
        let vectors_arr = body.get("vectors").unwrap().as_array().unwrap();
        assert_eq!(vectors_arr.len(), 2);
        assert_eq!(vectors_arr[0]["id"], "1");
        assert_eq!(vectors_arr[0]["values"][0], 1.0);
    }

    #[test]
    fn test_build_upsert_body_generates_id_if_missing() {
        let mut d = doc("", "x");
        d.id = None;
        let body = PineconeStore::build_upsert_body(&[d], &[vec![0.1]]);
        let id = body["vectors"][0]["id"].as_str().unwrap();
        assert!(!id.is_empty());
    }

    #[test]
    fn test_build_query_body() {
        let body = PineconeStore::build_query_body(&[1.0, 2.0, 3.0], 5);
        assert_eq!(body["topK"], 5);
        assert_eq!(body["includeMetadata"], true);
        assert_eq!(body["vector"][2], 3.0);
    }

    #[test]
    fn test_new() {
        let store = PineconeStore::new("key", "https://index.svc.env.pinecone.io");
        assert_eq!(store.host, "https://index.svc.env.pinecone.io");
    }
}
