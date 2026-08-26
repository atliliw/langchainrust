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

use crate::{Document, FilterOp, MetadataFilter, SearchResult, VectorStore, VectorStoreError};

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
    pub fn from_env_result() -> Result<Self, VectorStoreError> {
        let uri = std::env::var("LANCEDB_URI").map_err(|_| {
            VectorStoreError::ConfigError("LANCEDB_URI environment variable not set".to_string())
        })?;
        let table_name = std::env::var("LANCEDB_TABLE_NAME").map_err(|_| {
            VectorStoreError::ConfigError(
                "LANCEDB_TABLE_NAME environment variable not set".to_string(),
            )
        })?;
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
    pub fn from_env_result() -> Result<Self, VectorStoreError> {
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

    /// POST `/search` 并解析结果(普通与过滤检索共用)。
    async fn search_impl(
        &self,
        body: serde_json::Value,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = format!("{}/search", self.table_url());

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
}

/// S3: [`MetadataFilter`] → LanceDB SQL `where` 子句。
///
/// - 字符串值加单引号并转义内部单引号;数字/布尔原样输出。
/// - `In`/`Nin` 要求值是数组,生成 `key IN (...)` / `key NOT IN (...)`。
/// - 无法表达的值类型(对象/嵌套数组用于比较、`In` 的非数组值)返回
///   [`VectorStoreError::UnsupportedFilter`],不静默忽略。
pub fn filter_to_sql(filter: &MetadataFilter) -> Result<String, VectorStoreError> {
    match filter {
        MetadataFilter::Field { key, op, value } => {
            let ident = quote_ident(key);
            match op {
                FilterOp::In | FilterOp::Nin => {
                    let arr = value.as_array().ok_or_else(|| {
                        VectorStoreError::UnsupportedFilter(format!(
                            "IN/NIN requires an array value for field `{}`, got {}",
                            key,
                            value_type(value)
                        ))
                    })?;
                    let items = arr
                        .iter()
                        .map(sql_literal)
                        .collect::<Result<Vec<String>, _>>()?;
                    let keyword = if matches!(op, FilterOp::In) {
                        "IN"
                    } else {
                        "NOT IN"
                    };
                    Ok(format!("{} {} ({})", ident, keyword, items.join(", ")))
                }
                _ => Ok(format!("{} {} {}", ident, sql_op(op), sql_literal(value)?)),
            }
        }
        MetadataFilter::And(filters) => join_sql(filters, "AND"),
        MetadataFilter::Or(filters) => join_sql(filters, "OR"),
    }
}

/// AND/OR 组合:每个子过滤加括号后连接。
fn join_sql(filters: &[MetadataFilter], keyword: &str) -> Result<String, VectorStoreError> {
    if filters.is_empty() {
        return Ok("TRUE".to_string());
    }
    let parts = filters
        .iter()
        .map(filter_to_sql)
        .collect::<Result<Vec<String>, _>>()?;
    Ok(parts
        .iter()
        .map(|p| format!("({})", p))
        .collect::<Vec<_>>()
        .join(&format!(" {} ", keyword)))
}

/// 标识符(字段名)用双引号包裹并转义内部双引号,防注入。
fn quote_ident(key: &str) -> String {
    format!("\"{}\"", key.replace('"', "\"\""))
}

/// 标量值 → SQL 字面量;无法表达的类型返回 [`VectorStoreError::UnsupportedFilter`]。
fn sql_literal(value: &serde_json::Value) -> Result<String, VectorStoreError> {
    match value {
        serde_json::Value::String(s) => Ok(format!("'{}'", s.replace('\'', "''"))),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        _ => Err(VectorStoreError::UnsupportedFilter(format!(
            "cannot translate value of type {} to a SQL literal",
            value_type(value)
        ))),
    }
}

fn value_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn sql_op(op: &FilterOp) -> &'static str {
    match op {
        FilterOp::Eq => "=",
        FilterOp::Ne => "!=",
        FilterOp::Gt => ">",
        FilterOp::Gte => ">=",
        FilterOp::Lt => "<",
        FilterOp::Lte => "<=",
        FilterOp::In | FilterOp::Nin => unreachable!("handled by filter_to_sql"),
    }
}

/// Internal document representation for LanceDB.
#[derive(Debug, Serialize, Deserialize)]
struct LanceDBDocument {
    id: String,
    vector: Vec<f32>,
    content: String,
    #[serde(default, skip_serializing_if = "hash_map_is_empty")]
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

fn hash_map_is_empty(map: &std::collections::HashMap<String, serde_json::Value>) -> bool {
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
    metadata: std::collections::HashMap<String, serde_json::Value>,
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
        let body = json!({ "vector": query_embedding, "k": k });
        self.search_impl(body).await
    }

    /// S3: 带元数据过滤的相似度检索 —— 过滤交给服务端(LanceDB SQL `where` 子句)。
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut body = json!({ "vector": query_embedding, "k": k });
        if let Some(f) = filter {
            // 翻译失败(如 IN 的非数组值/嵌套对象)显式报错,不静默忽略过滤。
            body["where"] = serde_json::Value::String(filter_to_sql(f)?);
        }
        self.search_impl(body).await
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

    /// S3: 单字段条件 → SQL 表达式(字符串加引号,数字原样)。
    #[test]
    fn test_filter_to_sql_field() {
        assert_eq!(
            filter_to_sql(&MetadataFilter::field("lang", FilterOp::Eq, "rust")).unwrap(),
            r#""lang" = 'rust'"#
        );
        assert_eq!(
            filter_to_sql(&MetadataFilter::field("year", FilterOp::Gte, 2020)).unwrap(),
            r#""year" >= 2020"#
        );
        assert_eq!(
            filter_to_sql(&MetadataFilter::field("active", FilterOp::Ne, true)).unwrap(),
            r#""active" != true"#
        );
        // 单引号转义
        assert_eq!(
            filter_to_sql(&MetadataFilter::field("title", FilterOp::Eq, "it's")).unwrap(),
            r#""title" = 'it''s'"#
        );
    }

    /// S3: IN/NOT IN 需要数组值。
    #[test]
    fn test_filter_to_sql_in_nin() {
        assert_eq!(
            filter_to_sql(&MetadataFilter::field("tags", FilterOp::In, vec!["a", "b"])).unwrap(),
            r#""tags" IN ('a', 'b')"#
        );
        assert_eq!(
            filter_to_sql(&MetadataFilter::field("tags", FilterOp::Nin, vec!["x"])).unwrap(),
            r#""tags" NOT IN ('x')"#
        );
        // IN 值非数组 → 显式报错
        let err = filter_to_sql(&MetadataFilter::field("tags", FilterOp::In, "oops"));
        assert!(matches!(err, Err(VectorStoreError::UnsupportedFilter(_))));
    }

    /// S3: AND/OR 组合 → 括号包裹 + 连接词。
    #[test]
    fn test_filter_to_sql_and_or() {
        let f = MetadataFilter::or(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "python"),
            MetadataFilter::and(vec![
                MetadataFilter::field("lang", FilterOp::Eq, "rust"),
                MetadataFilter::field("year", FilterOp::Gt, 2020),
            ]),
        ]);
        assert_eq!(
            filter_to_sql(&f).unwrap(),
            r#"("lang" = 'python') OR (("lang" = 'rust') AND ("year" > 2020))"#
        );
    }

    /// S3: 不可表达的标量比较(对象值)显式报错。
    #[test]
    fn test_filter_to_sql_unsupported_value() {
        let f = MetadataFilter::field("nested", FilterOp::Eq, serde_json::json!({ "a": 1 }));
        assert!(matches!(
            filter_to_sql(&f),
            Err(VectorStoreError::UnsupportedFilter(_))
        ));
    }
}
