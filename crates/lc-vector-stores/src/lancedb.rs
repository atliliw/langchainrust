// lc-vector-stores/src/lancedb.rs
//! LanceDB vector store implementation.
//!
//! LanceDB is a serverless, low-latency vector database for AI applications.
//! This implementation uses the LanceDB HTTP API for remote/server mode.
//! For embedded/local mode, use the `lancedb` crate directly.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_vector_stores::lancedb::{LanceDBVectorStore, LanceDBConfig};
//!
//! let config = LanceDBConfig::new("http://localhost:1337", "my_table");
//! let store = LanceDBVectorStore::new(config);
//! store.add_documents(docs, embeddings).await?;
//! let results = store.similarity_search(&query_embedding, 5).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{Document, SearchResult, VectorStore, VectorStoreError};

/// LanceDB configuration.
#[derive(Debug, Clone)]
pub struct LanceDBConfig {
    /// LanceDB server URI (e.g., "http://localhost:1337" or "db://my-db").
    pub uri: String,
    /// Table name.
    pub table_name: String,
    /// API key (optional, for LanceDB Cloud).
    pub api_key: Option<String>,
    /// Region (optional, for LanceDB Cloud).
    pub region: Option<String>,
}

impl LanceDBConfig {
    /// Creates a new LanceDBConfig.
    pub fn new(uri: impl Into<String>, table_name: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            table_name: table_name.into(),
            api_key: None,
            region: None,
        }
    }

    /// Creates config from environment variables.
    pub fn from_env_result() -> Result<Self, String> {
        let uri = std::env::var("LANCEDB_URI")
            .map_err(|_| "LANCEDB_URI environment variable not set".to_string())?;
        let table_name = std::env::var("LANCEDB_TABLE_NAME")
            .map_err(|_| "LANCEDB_TABLE_NAME environment variable not set".to_string())?;
        let api_key = std::env::var("LANCEDB_API_KEY").ok();
        let region = std::env::var("LANCEDB_REGION").ok();
        Ok(Self {
            uri,
            table_name,
            api_key,
            region,
        })
    }

    /// Sets the API key for LanceDB Cloud.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the region for LanceDB Cloud.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }
}

/// LanceDB vector store.
///
/// Uses HTTP API to communicate with LanceDB server.
pub struct LanceDBVectorStore {
    config: LanceDBConfig,
    client: reqwest::Client,
}

impl std::fmt::Debug for LanceDBVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceDBVectorStore")
            .field("table", &self.config.table_name)
            .finish()
    }
}

impl LanceDBVectorStore {
    /// Creates a new LanceDBVectorStore with the given configuration.
    pub fn new(config: LanceDBConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates from environment variables.
    pub fn from_env_result() -> Result<Self, String> {
        Ok(Self::new(LanceDBConfig::from_env_result()?))
    }

    /// Builds the base URL for the table API.
    fn table_url(&self) -> String {
        format!(
            "{}/v1/table/{}",
            self.config.uri.trim_end_matches('/'),
            self.config.table_name
        )
    }

    /// Adds authorization headers to a request builder.
    fn add_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut req = req;
        if let Some(ref api_key) = self.config.api_key {
            req = req.header("x-api-key", api_key);
        }
        if let Some(ref region) = self.config.region {
            req = req.header("x-region", region);
        }
        req
    }
}

/// Internal document representation for LanceDB.
#[derive(Debug, Serialize, Deserialize)]
struct LanceDBDocument {
    id: String,
    vector: Vec<f32>,
    content: String,
    #[serde(default, skip_serializing_if = "hash_map_is_empty")]
    metadata: std::collections::HashMap<String, String>,
}

fn hash_map_is_empty(map: &std::collections::HashMap<String, String>) -> bool {
    map.is_empty()
}

/// LanceDB search response.
#[derive(Debug, Deserialize)]
struct LanceDBSearchResponse {
    data: Vec<LanceDBSearchItem>,
}

#[derive(Debug, Deserialize)]
struct LanceDBSearchItem {
    id: String,
    vector: Vec<f32>,
    content: String,
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
    #[serde(default)]
    score: Option<f32>,
}

#[async_trait]
impl VectorStore for LanceDBVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::EmbeddingError(
                "Number of documents and embeddings must match".to_string(),
            ));
        }

        let lancedb_docs: Vec<LanceDBDocument> = documents
            .into_iter()
            .zip(embeddings)
            .map(|(doc, vec)| {
                let id = doc
                    .id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                LanceDBDocument {
                    id: id.clone(),
                    vector: vec,
                    content: doc.content,
                    metadata: doc.metadata,
                }
            })
            .collect();

        let ids: Vec<String> = lancedb_docs.iter().map(|d| d.id.clone()).collect();

        let url = format!("{}/insert", self.table_url());
        let body = json!({
            "data": lancedb_docs,
        });

        let req = self.client.post(&url);
        let req = self.add_auth(req);
        let response = req
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = format!("{}/search", self.table_url());
        let body = json!({
            "vector": query_embedding,
            "k": k,
        });

        let req = self.client.post(&url);
        let req = self.add_auth(req);
        let response = req
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let search_response: LanceDBSearchResponse = response
            .json()
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(search_response
            .data
            .into_iter()
            .map(|item| {
                let mut doc = Document::new(item.content).with_id(item.id);
                for (key, value) in item.metadata {
                    doc = doc.with_metadata(key, value);
                }
                SearchResult {
                    document: doc,
                    score: item.score.unwrap_or(0.0),
                }
            })
            .collect())
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let url = format!("{}/get/{}", self.table_url(), id);

        let req = self.client.get(&url);
        let req = self.add_auth(req);
        let response = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let item: LanceDBSearchItem = response
            .json()
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let mut doc = Document::new(item.content).with_id(item.id);
        for (key, value) in item.metadata {
            doc = doc.with_metadata(key, value);
        }
        Ok(Some(doc))
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let url = format!("{}/get/{}", self.table_url(), id);

        let req = self.client.get(&url);
        let req = self.add_auth(req);
        let response = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let item: LanceDBSearchItem = response
            .json()
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(Some(item.vector))
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let url = format!("{}/delete/{}", self.table_url(), id);

        let req = self.client.delete(&url);
        let req = self.add_auth(req);
        let response = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        Ok(())
    }

    async fn count(&self) -> usize {
        let url = format!("{}/count", self.table_url());

        let req = self.client.get(&url);
        let req = self.add_auth(req);
        let result = req.send().await;

        match result {
            Ok(response) if response.status().is_success() => {
                let body: serde_json::Value = response.json().await.unwrap_or_default();
                body["count"].as_u64().unwrap_or(0) as usize
            }
            _ => 0,
        }
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let url = format!("{}/clear", self.table_url());

        let req = self.client.post(&url);
        let req = self.add_auth(req);
        let response = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::StorageError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = LanceDBConfig::new("http://localhost:1337", "my_table");
        assert_eq!(config.uri, "http://localhost:1337");
        assert_eq!(config.table_name, "my_table");
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = LanceDBConfig::new("http://localhost:1337", "test")
            .with_api_key("secret")
            .with_region("us-east-1");
        assert_eq!(config.api_key, Some("secret".to_string()));
        assert_eq!(config.region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_table_url() {
        let config = LanceDBConfig::new("http://localhost:1337", "my_table");
        let store = LanceDBVectorStore::new(config);
        assert_eq!(store.table_url(), "http://localhost:1337/v1/table/my_table");
    }

    #[test]
    fn test_table_url_trailing_slash() {
        let config = LanceDBConfig::new("http://localhost:1337/", "my_table");
        let store = LanceDBVectorStore::new(config);
        assert_eq!(store.table_url(), "http://localhost:1337/v1/table/my_table");
    }

    #[test]
    fn test_store_new() {
        let config = LanceDBConfig::new("http://localhost:1337", "test");
        let _store = LanceDBVectorStore::new(config);
    }

    #[test]
    fn test_lancedb_document_serialization() {
        let doc = LanceDBDocument {
            id: "test-1".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            content: "hello world".to_string(),
            metadata: std::collections::HashMap::new(),
        };
        let json = serde_json::to_value(&doc).unwrap();
        assert_eq!(json["id"], "test-1");
        assert!(json["vector"].is_array());
        assert_eq!(json["content"], "hello world");
    }
}
