// lc-vector-stores/src/neo4j.rs
//! Neo4j vector store implementation.
//!
//! Uses Neo4j's vector index feature (available since Neo4j 5.11) for
//! similarity search via the Cypher API over HTTP.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_vector_stores::neo4j::{Neo4jVectorStore, Neo4jConfig};
//!
//! let config = Neo4jConfig::new("bolt://localhost:7687", "neo4j", "password", "my_index");
//! let store = Neo4jVectorStore::new(config);
//! store.add_documents(docs, embeddings).await?;
//! let results = store.similarity_search(&query_embedding, 5).await?;
//! ```

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::{Document, SearchResult, VectorStore, VectorStoreError};

/// Neo4j vector store configuration.
#[derive(Debug, Clone)]
pub struct Neo4jConfig {
    /// Neo4j URI (e.g., "bolt://localhost:7687" or "neo4j://localhost:7687").
    pub uri: String,
    /// Username.
    pub username: String,
    /// Password.
    pub password: String,
    /// Database name (default: "neo4j").
    pub database: String,
    /// Node label for vector documents (default: "Document").
    pub node_label: String,
    /// Vector index name.
    pub index_name: String,
    /// Embedding property name on the node (default: "embedding").
    pub embedding_property: String,
    /// Content property name on the node (default: "content").
    pub content_property: String,
    /// Metadata property name on the node (default: "metadata").
    pub metadata_property: String,
    /// ID property name on the node (default: "id").
    pub id_property: String,
}

impl Neo4jConfig {
    /// Creates a new Neo4jConfig.
    pub fn new(
        uri: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
        index_name: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            username: username.into(),
            password: password.into(),
            database: "neo4j".to_string(),
            node_label: "Document".to_string(),
            index_name: index_name.into(),
            embedding_property: "embedding".to_string(),
            content_property: "content".to_string(),
            metadata_property: "metadata".to_string(),
            id_property: "id".to_string(),
        }
    }

    /// Creates config from environment variables.
    pub fn from_env_result() -> Result<Self, VectorStoreError> {
        let uri = std::env::var("NEO4J_URI").map_err(|_| {
            VectorStoreError::ConfigError("NEO4J_URI environment variable not set".to_string())
        })?;
        let username = std::env::var("NEO4J_USERNAME").map_err(|_| {
            VectorStoreError::ConfigError("NEO4J_USERNAME environment variable not set".to_string())
        })?;
        let password = std::env::var("NEO4J_PASSWORD").map_err(|_| {
            VectorStoreError::ConfigError("NEO4J_PASSWORD environment variable not set".to_string())
        })?;
        let index_name = std::env::var("NEO4J_VECTOR_INDEX_NAME").map_err(|_| {
            VectorStoreError::ConfigError(
                "NEO4J_VECTOR_INDEX_NAME environment variable not set".to_string(),
            )
        })?;
        let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());
        Ok(Self {
            uri,
            username,
            password,
            database,
            index_name,
            ..Default::default()
        })
    }

    /// Sets the database name.
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    /// Sets the node label.
    pub fn with_node_label(mut self, label: impl Into<String>) -> Self {
        self.node_label = label.into();
        self
    }

    /// Sets the embedding property name.
    pub fn with_embedding_property(mut self, prop: impl Into<String>) -> Self {
        self.embedding_property = prop.into();
        self
    }

    /// Sets the content property name.
    pub fn with_content_property(mut self, prop: impl Into<String>) -> Self {
        self.content_property = prop.into();
        self
    }
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: String::new(),
            database: "neo4j".to_string(),
            node_label: "Document".to_string(),
            index_name: "vector_index".to_string(),
            embedding_property: "embedding".to_string(),
            content_property: "content".to_string(),
            metadata_property: "metadata".to_string(),
            id_property: "id".to_string(),
        }
    }
}

/// Neo4j vector store.
///
/// Communicates with Neo4j via the HTTP transaction API.
pub struct Neo4jVectorStore {
    config: Neo4jConfig,
    client: reqwest::Client,
}

impl std::fmt::Debug for Neo4jVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neo4jVectorStore")
            .field("uri", &self.config.uri)
            .field("index", &self.config.index_name)
            .finish()
    }
}

impl Neo4jVectorStore {
    /// Creates a new Neo4jVectorStore with the given configuration.
    pub fn new(config: Neo4jConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates from environment variables.
    pub fn from_env_result() -> Result<Self, VectorStoreError> {
        Ok(Self::new(Neo4jConfig::from_env_result()?))
    }

    /// Builds the HTTP API URL for the transaction endpoint.
    fn tx_url(&self) -> String {
        // Convert bolt:// or neo4j:// to http:// for the REST API
        let http_uri = self
            .config
            .uri
            .replace("bolt://", "http://")
            .replace("neo4j://", "http://")
            .replace("bolt+s://", "https://")
            .replace("neo4j+s://", "https://");
        format!(
            "{}/db/{}/tx/commit",
            http_uri.trim_end_matches('/'),
            self.config.database
        )
    }

    /// Executes a Cypher query via the HTTP transaction API.
    async fn run_query(
        &self,
        query: &str,
        params: serde_json::Value,
    ) -> Result<Neo4jResponse, VectorStoreError> {
        let body = json!({
            "statements": [{
                "statement": query,
                "parameters": params,
            }]
        });

        let response = self
            .client
            .post(self.tx_url())
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Basic {}",
                    base64_encode(format!("{}:{}", self.config.username, self.config.password))
                ),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VectorStoreError::ConnectionError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let neo4j_response: Neo4jResponse = response
            .json()
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        // Check for Neo4j-level errors
        if let Some(errors) = &neo4j_response.errors {
            if !errors.is_empty() {
                let msg = errors
                    .iter()
                    .map(|e| e.message.clone())
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(VectorStoreError::StorageError(msg));
            }
        }

        Ok(neo4j_response)
    }
}

/// Base64 encoding helper (no external dependency needed).
fn base64_encode(input: String) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };

        result.push(CHARSET[((b0 >> 2) & 0x3F) as usize] as char);
        result.push(CHARSET[(((b0 << 4) | (b1 >> 4)) & 0x3F) as usize] as char);
        result.push(if i + 1 < bytes.len() {
            CHARSET[(((b1 << 2) | (b2 >> 6)) & 0x3F) as usize] as char
        } else {
            '='
        });
        result.push(if i + 2 < bytes.len() {
            CHARSET[(b2 & 0x3F) as usize] as char
        } else {
            '='
        });

        i += 3;
    }
    result
}

// ---------------------------------------------------------------------------
// Neo4j HTTP API response types
// ---------------------------------------------------------------------------

/// Neo4j transaction commit response.
#[derive(Debug, Deserialize)]
struct Neo4jResponse {
    results: Vec<Neo4jResult>,
    errors: Option<Vec<Neo4jError>>,
}

#[derive(Debug, Deserialize)]
struct Neo4jResult {
    data: Vec<Neo4jRow>,
}

#[derive(Debug, Deserialize)]
struct Neo4jRow {
    row: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct Neo4jError {
    message: String,
}

#[async_trait]
impl VectorStore for Neo4jVectorStore {
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

        let ids: Vec<String> = documents
            .iter()
            .map(|doc| {
                doc.id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
            })
            .collect();

        // Build UNWIND Cypher for batch insert
        let rows: Vec<serde_json::Value> = documents
            .into_iter()
            .zip(embeddings)
            .zip(ids.iter())
            .map(|((doc, vec), id)| {
                let metadata: serde_json::Value = doc
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect();
                json!({
                    "id": id,
                    "content": doc.content,
                    "embedding": vec,
                    "metadata": metadata,
                })
            })
            .collect();

        let query = format!(
            "UNWIND $rows AS row \
             MERGE (n:{label} {{{id_prop}: row.id}}) \
             SET n.{content_prop} = row.content, \
                 n.{embedding_prop} = row.embedding, \
                 n.{metadata_prop} = row.metadata",
            label = self.config.node_label,
            id_prop = self.config.id_property,
            content_prop = self.config.content_property,
            embedding_prop = self.config.embedding_property,
            metadata_prop = self.config.metadata_property,
        );

        self.run_query(&query, json!({ "rows": rows })).await?;

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Use Neo4j's db.index.vector.queryNodes procedure
        let query = format!(
            "CALL db.index.vector.queryNodes($index_name, $k, $query_vector) \
             YIELD node, score \
             RETURN node.{id_prop} AS id, \
                    node.{content_prop} AS content, \
                    node.{metadata_prop} AS metadata, \
                    score \
             ORDER BY score DESC",
            id_prop = self.config.id_property,
            content_prop = self.config.content_property,
            metadata_prop = self.config.metadata_property,
        );

        let params = json!({
            "index_name": self.config.index_name,
            "k": k,
            "query_vector": query_embedding,
        });

        let response = self.run_query(&query, params).await?;

        let result = response.results.first();
        let Some(neo4j_result) = result else {
            return Ok(Vec::new());
        };

        let mut search_results = Vec::new();
        for row in &neo4j_result.data {
            if row.row.len() >= 4 {
                let id = row.row[0].as_str().unwrap_or_default().to_string();
                let content = row.row[1].as_str().unwrap_or_default().to_string();
                let score = row.row[3].as_f64().unwrap_or(0.0) as f32;

                let mut doc = Document::new(content).with_id(id);

                // Parse metadata from JSON object
                if let Some(meta_obj) = row.row[2].as_object() {
                    for (key, value) in meta_obj {
                        if let Some(s) = value.as_str() {
                            doc = doc.with_metadata(key, s);
                        } else {
                            doc = doc.with_metadata(key, value.to_string());
                        }
                    }
                }

                search_results.push(SearchResult {
                    document: doc,
                    score,
                });
            }
        }

        Ok(search_results)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let query = format!(
            "MATCH (n:{label} {{{id_prop}: $id}}) \
             RETURN n.{content_prop} AS content, n.{metadata_prop} AS metadata",
            label = self.config.node_label,
            id_prop = self.config.id_property,
            content_prop = self.config.content_property,
            metadata_prop = self.config.metadata_property,
        );

        let response = self.run_query(&query, json!({ "id": id })).await?;

        let result = response.results.first();
        let Some(neo4j_result) = result else {
            return Ok(None);
        };

        let row = neo4j_result.data.first();
        let Some(row) = row else {
            return Ok(None);
        };

        if row.row.is_empty() {
            return Ok(None);
        }

        let content = row.row[0].as_str().unwrap_or_default().to_string();
        let mut doc = Document::new(content).with_id(id);

        if row.row.len() > 1 {
            if let Some(meta_obj) = row.row[1].as_object() {
                for (key, value) in meta_obj {
                    if let Some(s) = value.as_str() {
                        doc = doc.with_metadata(key, s);
                    } else {
                        doc = doc.with_metadata(key, value.to_string());
                    }
                }
            }
        }

        Ok(Some(doc))
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let query = format!(
            "MATCH (n:{label} {{{id_prop}: $id}}) \
             RETURN n.{embedding_prop} AS embedding",
            label = self.config.node_label,
            id_prop = self.config.id_property,
            embedding_prop = self.config.embedding_property,
        );

        let response = self.run_query(&query, json!({ "id": id })).await?;

        let result = response.results.first();
        let Some(neo4j_result) = result else {
            return Ok(None);
        };

        let row = neo4j_result.data.first();
        let Some(row) = row else {
            return Ok(None);
        };

        if row.row.is_empty() {
            return Ok(None);
        }

        let embedding: Vec<f32> = row.row[0]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        if embedding.is_empty() {
            Ok(None)
        } else {
            Ok(Some(embedding))
        }
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let query = format!(
            "MATCH (n:{label} {{{id_prop}: $id}}) \
             DETACH DELETE n",
            label = self.config.node_label,
            id_prop = self.config.id_property,
        );

        self.run_query(&query, json!({ "id": id })).await?;
        Ok(())
    }

    async fn count(&self) -> usize {
        let query = format!(
            "MATCH (n:{label}) RETURN count(n) AS cnt",
            label = self.config.node_label,
        );

        let result = self.run_query(&query, json!({})).await;
        match result {
            Ok(response) => {
                if let Some(neo4j_result) = response.results.first() {
                    if let Some(row) = neo4j_result.data.first() {
                        if let Some(cnt) = row.row.first() {
                            return cnt.as_u64().unwrap_or(0) as usize;
                        }
                    }
                }
                0
            }
            Err(_) => 0,
        }
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let query = format!(
            "MATCH (n:{label}) \
             DETACH DELETE n",
            label = self.config.node_label,
        );

        self.run_query(&query, json!({})).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = Neo4jConfig::new("bolt://localhost:7687", "neo4j", "pass", "my_index");
        assert_eq!(config.uri, "bolt://localhost:7687");
        assert_eq!(config.username, "neo4j");
        assert_eq!(config.password, "pass");
        assert_eq!(config.index_name, "my_index");
        assert_eq!(config.database, "neo4j");
    }

    #[test]
    fn test_config_builder() {
        let config = Neo4jConfig::new("bolt://localhost:7687", "neo4j", "pass", "idx")
            .with_database("mydb")
            .with_node_label("Chunk")
            .with_embedding_property("vec")
            .with_content_property("text");
        assert_eq!(config.database, "mydb");
        assert_eq!(config.node_label, "Chunk");
        assert_eq!(config.embedding_property, "vec");
        assert_eq!(config.content_property, "text");
    }

    #[test]
    fn test_config_default() {
        let config = Neo4jConfig::default();
        assert_eq!(config.uri, "bolt://localhost:7687");
        assert_eq!(config.node_label, "Document");
        assert_eq!(config.embedding_property, "embedding");
    }

    #[test]
    fn test_tx_url_bolt() {
        let config = Neo4jConfig::new("bolt://localhost:7687", "neo4j", "pass", "idx");
        let store = Neo4jVectorStore::new(config);
        assert_eq!(store.tx_url(), "http://localhost:7687/db/neo4j/tx/commit");
    }

    #[test]
    fn test_tx_url_neo4j_scheme() {
        let config = Neo4jConfig::new("neo4j://host:7687", "neo4j", "pass", "idx");
        let store = Neo4jVectorStore::new(config);
        assert_eq!(store.tx_url(), "http://host:7687/db/neo4j/tx/commit");
    }

    #[test]
    fn test_tx_url_bolt_s() {
        let config = Neo4jConfig::new("bolt+s://host:7687", "neo4j", "pass", "idx");
        let store = Neo4jVectorStore::new(config);
        assert_eq!(store.tx_url(), "https://host:7687/db/neo4j/tx/commit");
    }

    #[test]
    fn test_tx_url_custom_database() {
        let config =
            Neo4jConfig::new("bolt://localhost:7687", "neo4j", "pass", "idx").with_database("mydb");
        let store = Neo4jVectorStore::new(config);
        assert_eq!(store.tx_url(), "http://localhost:7687/db/mydb/tx/commit");
    }

    #[test]
    fn test_base64_encode() {
        // "neo4j:password" in base64
        let encoded = base64_encode("neo4j:password".to_string());
        assert_eq!(encoded, "bmVvNGo6cGFzc3dvcmQ=");
    }

    #[test]
    fn test_base64_encode_empty() {
        let encoded = base64_encode(String::new());
        assert_eq!(encoded, "");
    }

    #[test]
    fn test_store_new() {
        let config = Neo4jConfig::new("bolt://localhost:7687", "neo4j", "pass", "idx");
        let _store = Neo4jVectorStore::new(config);
    }

    #[test]
    fn test_store_debug() {
        let config = Neo4jConfig::new("bolt://localhost:7687", "neo4j", "pass", "idx");
        let store = Neo4jVectorStore::new(config);
        let debug_str = format!("{:?}", store);
        assert!(debug_str.contains("Neo4jVectorStore"));
        assert!(debug_str.contains("idx"));
    }
}
