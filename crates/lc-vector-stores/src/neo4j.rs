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

use crate::{Document, FilterOp, MetadataFilter, SearchResult, VectorStore, VectorStoreError};

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

    /// `db.index.vector.queryNodes` 前缀(不含 WHERE/RETURN)。
    fn search_query_prefix(&self) -> String {
        "CALL db.index.vector.queryNodes($index_name, $k, $query_vector) \
         YIELD node, score"
            .to_string()
    }

    /// RETURN 子句与排序(普通与过滤检索共用)。
    fn search_query_suffix(&self) -> String {
        format!(
            " RETURN node.{id_prop} AS id, \
                    node.{content_prop} AS content, \
                    node.{metadata_prop} AS metadata, \
                    score \
             ORDER BY score DESC",
            id_prop = self.config.id_property,
            content_prop = self.config.content_property,
            metadata_prop = self.config.metadata_property,
        )
    }

    /// 把 queryNodes 的返回行解析成 [`SearchResult`](普通与过滤检索共用)。
    fn parse_search_results(&self, response: Neo4jResponse) -> Vec<SearchResult> {
        let Some(neo4j_result) = response.results.first() else {
            return Vec::new();
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

        search_results
    }
}

/// S3: [`MetadataFilter`] → Cypher `WHERE` 表达式 + 参数表。
///
/// - 每个字段条件生成 `node.{metadata_prop}[$fNk] <op> $fNv`,key 与 value 全部
///   参数化(防 Cypher 注入);参数名单调递增。
/// - `In`/`Nin` 要求值是数组,生成 `... IN $fNv` / `NOT ... IN $fNv`。
/// - 标量操作(`Eq/Ne/Gt/Gte/Lt/Lte`)只接受字符串/数字/布尔值,其余类型返回
///   [`VectorStoreError::UnsupportedFilter`](不静默忽略)。
pub fn filter_to_cypher(
    filter: &MetadataFilter,
    metadata_prop: &str,
) -> Result<(String, serde_json::Value), VectorStoreError> {
    let mut state = CypherState { next: 0 };
    let mut params = serde_json::Map::new();
    let expr = state.expr(filter, metadata_prop, &mut params)?;
    Ok((expr, serde_json::Value::Object(params)))
}

/// 递归翻译的计数器(保证参数名唯一)。
struct CypherState {
    next: usize,
}

impl CypherState {
    fn expr(
        &mut self,
        filter: &MetadataFilter,
        metadata_prop: &str,
        params: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, VectorStoreError> {
        match filter {
            MetadataFilter::Field { key, op, value } => {
                let kp = format!("f{}k", self.next);
                let vp = format!("f{}v", self.next);
                self.next += 1;

                params.insert(kp.clone(), serde_json::Value::String(key.clone()));
                // 节点上的 metadata 属性是 map,用 `node.metadata[$k]` 按下标取键值
                let target = format!("node.{}[${}]", metadata_prop, kp);

                match op {
                    FilterOp::In | FilterOp::Nin => {
                        let arr = value.as_array().ok_or_else(|| {
                            VectorStoreError::UnsupportedFilter(format!(
                                "IN/NIN requires an array value for field `{}`, got {}",
                                key,
                                value_type_name(value)
                            ))
                        })?;
                        params.insert(vp.clone(), serde_json::Value::Array(arr.clone()));
                        if matches!(op, FilterOp::In) {
                            Ok(format!("{} IN ${}", target, vp))
                        } else {
                            Ok(format!("NOT {} IN ${}", target, vp))
                        }
                    }
                    _ => {
                        if !matches!(
                            value,
                            serde_json::Value::String(_)
                                | serde_json::Value::Number(_)
                                | serde_json::Value::Bool(_)
                        ) {
                            return Err(VectorStoreError::UnsupportedFilter(format!(
                                "cannot translate value of type {} to a Cypher comparison for field `{}`",
                                value_type_name(value),
                                key
                            )));
                        }
                        params.insert(vp.clone(), value.clone());
                        Ok(format!("{} {} ${}", target, cypher_op(op), vp))
                    }
                }
            }
            MetadataFilter::And(filters) => self.join(filters, metadata_prop, params, "AND"),
            MetadataFilter::Or(filters) => self.join(filters, metadata_prop, params, "OR"),
        }
    }

    fn join(
        &mut self,
        filters: &[MetadataFilter],
        metadata_prop: &str,
        params: &mut serde_json::Map<String, serde_json::Value>,
        keyword: &str,
    ) -> Result<String, VectorStoreError> {
        let parts = filters
            .iter()
            .map(|f| self.expr(f, metadata_prop, params))
            .collect::<Result<Vec<String>, _>>()?;
        if parts.is_empty() {
            return Ok("TRUE".to_string());
        }
        Ok(parts
            .iter()
            .map(|p| format!("({})", p))
            .collect::<Vec<_>>()
            .join(&format!(" {} ", keyword)))
    }
}

fn cypher_op(op: &FilterOp) -> &'static str {
    match op {
        FilterOp::Eq => "=",
        FilterOp::Ne => "<>",
        FilterOp::Gt => ">",
        FilterOp::Gte => ">=",
        FilterOp::Lt => "<",
        FilterOp::Lte => "<=",
        FilterOp::In | FilterOp::Nin => unreachable!("handled by filter_to_cypher"),
    }
}

fn value_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
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
        let query = format!(
            "{}{}",
            self.search_query_prefix(),
            self.search_query_suffix()
        );

        let params = json!({
            "index_name": self.config.index_name,
            "k": k,
            "query_vector": query_embedding,
        });

        let response = self.run_query(&query, params).await?;
        Ok(self.parse_search_results(response))
    }

    /// S3: 带元数据过滤的相似度检索 —— 用 Cypher `WHERE` 在服务端过滤
    /// `db.index.vector.queryNodes` 返回的结果(参数化 key/value,防注入)。
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut query = self.search_query_prefix();
        let mut params = json!({
            "index_name": self.config.index_name,
            "k": k,
            "query_vector": query_embedding,
        });
        if let Some(f) = filter {
            let (where_clause, extra_params) = filter_to_cypher(f, &self.config.metadata_property)?;
            query.push_str(&format!(" WHERE {}", where_clause));
            if let (Some(base), Some(extra)) = (params.as_object_mut(), extra_params.as_object()) {
                base.extend(extra.clone());
            }
        }
        query.push_str(&self.search_query_suffix());

        let response = self.run_query(&query, params).await?;
        Ok(self.parse_search_results(response))
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

    /// S3: 单字段条件 → 参数化 Cypher 表达式 + 参数表。
    #[test]
    fn test_filter_to_cypher_field() {
        let f = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let (expr, params) = filter_to_cypher(&f, "metadata").unwrap();
        assert_eq!(expr, "node.metadata[$f0k] = $f0v");
        assert_eq!(params["f0k"], "lang");
        assert_eq!(params["f0v"], "rust");

        let (expr, params) = filter_to_cypher(
            &MetadataFilter::field("year", FilterOp::Gte, 2020),
            "metadata",
        )
        .unwrap();
        assert_eq!(expr, "node.metadata[$f0k] >= $f0v");
        assert_eq!(params["f0v"], 2020);

        // Ne → <>(Cypher 的不等于)
        let (expr, _) = filter_to_cypher(
            &MetadataFilter::field("lang", FilterOp::Ne, "rust"),
            "metadata",
        )
        .unwrap();
        assert_eq!(expr, "node.metadata[$f0k] <> $f0v");
    }

    /// S3: IN/NOT IN → 数组参数。
    #[test]
    fn test_filter_to_cypher_in_nin() {
        let (expr, params) = filter_to_cypher(
            &MetadataFilter::field("tags", FilterOp::In, vec!["a", "b"]),
            "meta",
        )
        .unwrap();
        assert_eq!(expr, "node.meta[$f0k] IN $f0v");
        assert_eq!(params["f0v"], serde_json::json!(["a", "b"]));

        let (expr, _) = filter_to_cypher(
            &MetadataFilter::field("tags", FilterOp::Nin, vec!["x"]),
            "meta",
        )
        .unwrap();
        assert_eq!(expr, "NOT node.meta[$f0k] IN $f0v");

        // IN 值非数组 → 显式报错
        let err = filter_to_cypher(&MetadataFilter::field("tags", FilterOp::In, "oops"), "meta");
        assert!(matches!(err, Err(VectorStoreError::UnsupportedFilter(_))));
    }

    /// S3: AND/OR 组合 → 括号 + 参数名继续递增不冲突。
    #[test]
    fn test_filter_to_cypher_and_or() {
        let f = MetadataFilter::or(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "python"),
            MetadataFilter::and(vec![
                MetadataFilter::field("lang", FilterOp::Eq, "rust"),
                MetadataFilter::field("year", FilterOp::Gt, 2020),
            ]),
        ]);
        let (expr, params) = filter_to_cypher(&f, "metadata").unwrap();
        assert_eq!(
            expr,
            "(node.metadata[$f0k] = $f0v) OR ((node.metadata[$f1k] = $f1v) AND (node.metadata[$f2k] > $f2v))"
        );
        assert_eq!(params["f0v"], "python");
        assert_eq!(params["f1v"], "rust");
        assert_eq!(params["f2v"], 2020);
    }

    /// S3: 不可表达的标量比较(对象值)显式报错。
    #[test]
    fn test_filter_to_cypher_unsupported_value() {
        let f = MetadataFilter::field("nested", FilterOp::Eq, serde_json::json!({ "a": 1 }));
        assert!(matches!(
            filter_to_cypher(&f, "metadata"),
            Err(VectorStoreError::UnsupportedFilter(_))
        ));
    }
}
