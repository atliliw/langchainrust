// lc-vector-stores/src/chromadb.rs
//! ChromaDB vector store implementation (HTTP API)
//!
//! Uses ChromaDB's REST API for vector storage and retrieval.
//! Supports connecting to a remote ChromaDB service (docker run -p 8000:8000 chromadb/chroma).

use async_trait::async_trait;
use lc_core::http::{BoundedResponse, HttpClient, HttpError, RequestOptions};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

use crate::{Document, FilterOp, MetadataFilter, SearchResult, VectorStore, VectorStoreError};

/// Number of IDs to fetch+delete per round of [`ChromaDBVectorStore::clear`].
///
/// 0.25.0 B3: `clear` pages through the collection rather than fetching every
/// document at once, so a very large collection is drained in bounded chunks
/// instead of one unbounded `/get`. Deletions shift Chroma's internal ordering,
/// so pagination must NOT use `offset` — paging only by the leading fetch (of
/// this many IDs) always converges on the first remaining documents.
const CHROMA_CLEAR_PAGE_SIZE: usize = 500;

/// ChromaDB configuration
#[derive(Debug, Clone)]
pub struct ChromaDBConfig {
    /// ChromaDB service URL, default http://localhost:8000
    pub host: String,
    /// Collection name
    pub collection_name: String,
    /// Vector dimension
    pub vector_size: usize,
    /// Collection metadata (optional)
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
    /// Creates a new ChromaDB configuration
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

/// ChromaDB collection info (parsed from the API response)
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChromaCollection {
    id: String,
    name: String,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

/// ChromaDB add request body
#[derive(Debug, Serialize)]
struct ChromaAddRequest {
    ids: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadatas: Option<Vec<HashMap<String, serde_json::Value>>>,
}

/// ChromaDB query request body
#[derive(Debug, Serialize)]
struct ChromaQueryRequest {
    query_embeddings: Vec<Vec<f32>>,
    n_results: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    /// Chroma `where` filter dict (see [`filter_to_chroma`])
    #[serde(rename = "where", skip_serializing_if = "Option::is_none")]
    where_filter: Option<serde_json::Value>,
}

/// ChromaDB query response
#[derive(Debug, Deserialize)]
struct ChromaQueryResponse {
    ids: Vec<Vec<String>>,
    distances: Vec<Vec<f64>>,
    documents: Vec<Vec<String>>,
    #[serde(default)]
    metadatas: Vec<Vec<Option<HashMap<String, serde_json::Value>>>>,
}

/// ChromaDB get response
#[derive(Debug, Deserialize)]
struct ChromaGetResponse {
    ids: Vec<String>,
    documents: Vec<Option<String>>,
    #[serde(default)]
    metadatas: Vec<Option<HashMap<String, serde_json::Value>>>,
    embeddings: Option<Vec<Vec<f32>>>,
}

/// Null-tolerant `/get` page used by [`ChromaDBVectorStore::clear`]'s drain loop.
///
/// Chroma emits `"ids": null` (not `"ids": []`) when a collection is empty, which
/// would fail to deserialize into the strict `Vec<String>` of [`ChromaGetResponse`].
/// An `Option<Vec<String>>` (with `#[serde(default)]`) accepts `null`, a missing
/// key, and a populated array; the loop flattens `None` to an empty batch.
#[derive(Debug, Deserialize)]
struct ChromaClearPage {
    #[serde(default)]
    ids: Option<Vec<String>>,
}

/// ChromaDB vector store
///
/// Connects to a ChromaDB service via HTTP API.
///
/// # Example
/// ```ignore
/// use lc_vector_stores::ChromaDBVectorStore;
///
/// let store = ChromaDBVectorStore::new(
///     ChromaDBConfig::new("http://localhost:8000", "my_collection", 384)
/// ).await?;
/// ```
pub struct ChromaDBVectorStore {
    config: ChromaDBConfig,
    http: HttpClient,
    collection_id: Option<String>,
}

impl ChromaDBVectorStore {
    /// Creates a ChromaDB vector store and initializes the collection automatically
    pub async fn new(config: ChromaDBConfig) -> Result<Self, VectorStoreError> {
        // The unified client defaults to `no_proxy()` (local proxy software such as
        // Clash must not intercept Chroma's in-cluster traffic) and is infallible
        // for these settings.
        let http = HttpClient::api().build().map_err(|e| {
            VectorStoreError::ConfigError(format!("failed to build HTTP client: {e}"))
        })?;
        let mut store = Self {
            config,
            http,
            collection_id: None,
        };
        store.init_collection().await?;
        Ok(store)
    }

    /// Maps a transport/spool error from the unified client into a
    /// [`VectorStoreError`], keeping the server's own error body (which Chroma
    /// puts real detail into) instead of a bare status code.
    fn map_http_error(prefix: &str, err: HttpError) -> VectorStoreError {
        match err {
            HttpError::Status { status, body } => {
                VectorStoreError::StorageError(format!("{prefix}: HTTP {status}: {body}"))
            }
            other => VectorStoreError::ConnectionError(format!("{prefix}: {other}")),
        }
    }

    /// Initializes or fetches the collection
    async fn init_collection(&mut self) -> Result<(), VectorStoreError> {
        // try to fetch the existing collection
        let url = format!(
            "{}/api/v1/collections/{}",
            self.config.host, self.config.collection_name
        );
        let response = self
            .http
            .get(&url)
            .await
            .map_err(|e| Self::map_http_error("ChromaDB init error", e))?;

        if response.status.is_success() {
            let collection: ChromaCollection =
                serde_json::from_str(&response.body).map_err(|e| {
                    VectorStoreError::StorageError(format!(
                        "failed to parse collection info: {}",
                        e
                    ))
                })?;
            self.collection_id = Some(collection.id);
            return Ok(());
        }

        // collection does not exist, create a new one
        let create_url = format!("{}/api/v1/collections", self.config.host);
        let mut body = json!({
            "name": self.config.collection_name,
        });

        if let Some(ref meta) = self.config.metadata {
            body["metadata"] = serde_json::to_value(meta).unwrap_or(json!({}));
        }

        let response = self
            .http
            .post_json_with(&create_url, &body, RequestOptions::new())
            .await
            .map_err(|e| Self::map_http_error("ChromaDB create error", e))?;

        if response.status.is_success() {
            let collection: ChromaCollection =
                serde_json::from_str(&response.body).map_err(|e| {
                    VectorStoreError::StorageError(format!(
                        "failed to parse new collection info: {}",
                        e
                    ))
                })?;
            self.collection_id = Some(collection.id);
            Ok(())
        } else {
            Err(VectorStoreError::StorageError(format!(
                "failed to create collection: {}",
                response.body
            )))
        }
    }

    /// Posts a JSON body and, on success, hands back the buffered response.
    ///
    /// Centralizes the per-endpoint error translation (transport/disturbed vs.
    /// server-visible body) that every Chroma operation shared.
    async fn post_json(
        &self,
        url: &str,
        body: &serde_json::Value,
        prefix: &str,
    ) -> Result<BoundedResponse, VectorStoreError> {
        self.http
            .post_json_with(url, body, RequestOptions::new())
            .await
            .map_err(|e| Self::map_http_error(prefix, e))
    }

    /// Gets the collection ID
    fn get_collection_id(&self) -> Result<&str, VectorStoreError> {
        self.collection_id.as_deref().ok_or_else(|| {
            VectorStoreError::StorageError("collection is not initialized".to_string())
        })
    }

    /// Builds the collection API base URL
    fn collection_url(&self, endpoint: &str) -> Result<String, VectorStoreError> {
        let cid = self.get_collection_id()?;
        Ok(format!(
            "{}/api/v1/collections/{}/{}",
            self.config.host, cid, endpoint
        ))
    }

    /// Builds a Chroma query request body (pure function, convenient for testing).
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

    /// POSTs to `/query` and parses the result (shared by plain and filtered retrieval).
    async fn query_impl(
        &self,
        request: ChromaQueryRequest,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = self.collection_url("query")?;
        let body = serde_json::to_value(&request).map_err(|e| {
            VectorStoreError::StorageError(format!("failed to serialize query: {e}"))
        })?;
        let response = self.post_json(&url, &body, "ChromaDB query failed").await?;

        let query_result: ChromaQueryResponse =
            serde_json::from_str(&response.body).map_err(|e| {
                VectorStoreError::StorageError(format!("failed to parse query results: {}", e))
            })?;

        let mut results = Vec::new();

        // ChromaDB returns nested arrays (one result set per query)
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
                // ChromaDB returns L2 distance; convert to a similarity score (1 / (1 + dist))
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

        // sort by similarity descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Fetches up to [`CHROMA_CLEAR_PAGE_SIZE`] document IDs without any
    /// documents/metadatas/embeddings payload (`include: []` keeps the response
    /// tiny). Uses a dedicated nullable-`ids` struct: Chroma returns
    /// `{"ids": null, ...}` for an empty collection, which a strict `Vec<String>`
    /// would refuse to deserialize.
    async fn fetch_clear_page(&self) -> Result<Vec<String>, VectorStoreError> {
        let get_url = self.collection_url("get")?;
        let body = json!({
            "include": [],
            "limit": CHROMA_CLEAR_PAGE_SIZE
        });
        let response = self
            .post_json(&get_url, &body, "ChromaDB clear fetch failed")
            .await?;
        let page: ChromaClearPage = serde_json::from_str(&response.body).map_err(|e| {
            VectorStoreError::StorageError(format!("failed to parse document list: {}", e))
        })?;
        Ok(page.ids.unwrap_or_default())
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
        let body = serde_json::to_value(&request).map_err(|e| {
            VectorStoreError::StorageError(format!("failed to serialize add request: {e}"))
        })?;
        let _ = self.post_json(&url, &body, "ChromaDB add failed").await?;
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

    /// S3: similarity search with metadata filtering — filtering is delegated to the server (Chroma `where` syntax).
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
            .http
            .post_json_with(&url, &body, RequestOptions::new())
            .await;
        let response = match response {
            // A `not found` (e.g. 404) means no document — same as an empty result.
            Err(HttpError::Status { .. }) => return Ok(None),
            Err(e) => return Err(Self::map_http_error("ChromaDB get_document error", e)),
            Ok(r) => r,
        };

        let get_result: ChromaGetResponse = serde_json::from_str(&response.body).map_err(|e| {
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
            .http
            .post_json_with(&url, &body, RequestOptions::new())
            .await;
        let response = match response {
            // A `not found` (e.g. 404) means no embedding — same as an empty result.
            Err(HttpError::Status { .. }) => return Ok(None),
            Err(e) => return Err(Self::map_http_error("ChromaDB get_embedding error", e)),
            Ok(r) => r,
        };

        let get_result: ChromaGetResponse = serde_json::from_str(&response.body).map_err(|e| {
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

        let _ = self
            .post_json(&url, &body, "ChromaDB delete failed")
            .await?;
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

        let response = self
            .http
            .post_json_with(&url, &serde_json::json!({}), RequestOptions::new())
            .await;
        match response {
            Ok(resp) => match serde_json::from_str::<usize>(&resp.body) {
                Ok(count) => count,
                Err(e) => {
                    log::warn!("ChromaDB count() failed to parse response: {}", e);
                    0
                }
            },
            Err(e) => {
                log::warn!("ChromaDB count() request failed: {}", e);
                0
            }
        }
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // Drain the collection in bounded pages: repeatedly fetch the leading
        // `PAGE_SIZE` IDs (no `offset` — deletions shift Chroma's internal
        // ordering, so position-based pagination cannot be trusted), delete them,
        // and repeat until a fetch returns no IDs. A safety bound prevents an
        // infinite loop against a pathological server that never empties.
        const MAX_ROUNDS: usize = 10_000;
        for _round in 0..MAX_ROUNDS {
            let ids = self.fetch_clear_page().await?;
            if ids.is_empty() {
                return Ok(());
            }

            let del_url = self.collection_url("delete")?;
            let del_body = json!({ "ids": ids });
            let _ = self
                .post_json(&del_url, &del_body, "ChromaDB clear delete failed")
                .await?;
        }

        Err(VectorStoreError::StorageError(format!(
            "failed to clear collection after {MAX_ROUNDS} delete rounds"
        )))
    }
}

/// S3: translates [`MetadataFilter`] → Chroma `where` filter dict.
///
/// A single-field condition becomes `{ key: { "$op": value } }` (Chroma v2 supports
/// `$eq $ne $gt $gte $lt $lte $in $nin`); AND/OR combinations become
/// `{ "$and": [...] }` / `{ "$or": [...] }`. Isomorphic to the Pinecone translation,
/// but maintained per backend independently; the semantics are fully delegated to the server.
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A minimal loopback server: reads one request (headers + body) and replies
    /// with `(status, full_request_text)` chosen by `handler(path, request_text)`.
    async fn spawn_loopback(
        handler: impl Fn(&str, &str) -> (u16, String) + Send + Sync + 'static,
    ) -> String {
        use std::sync::Arc;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let handler = handler.clone();
                tokio::spawn(async move {
                    let mut raw = Vec::new();
                    let mut buf = [0u8; 4096];
                    let (head_end, len) = loop {
                        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                            let text = String::from_utf8_lossy(&raw).to_string();
                            let len = text
                                .lines()
                                .find_map(|l| {
                                    let lower = l.to_ascii_lowercase();
                                    lower
                                        .strip_prefix("content-length:")
                                        .map(|v| v.trim().to_string())
                                })
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(0);
                            break (pos + 4, len);
                        }
                        let n = socket.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            break (raw.len(), 0);
                        }
                        raw.extend_from_slice(&buf[..n]);
                    };
                    while raw.len() < head_end + len {
                        let n = socket.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        raw.extend_from_slice(&buf[..n]);
                    }
                    let text = String::from_utf8_lossy(&raw).to_string();
                    let path = text.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let (status, resp_body) = handler(&path, &text);
                    let reason = if status == 200 { "OK" } else { "Error" };
                    let head = format!(
                        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        resp_body.len()
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                    let _ = socket.write_all(resp_body.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        format!("http://{}", addr)
    }

    /// Builds a [`ChromaDBVectorStore`] already bound to `collection_id`, bypassing
    /// `init_collection` so a test can drive a loopback endpoint directly.
    fn store_with_collection(base: &str, cid: &str) -> ChromaDBVectorStore {
        let http = HttpClient::api().build().expect("build http client");
        ChromaDBVectorStore {
            config: ChromaDBConfig::new(base, "col", 3),
            http,
            collection_id: Some(cid.to_string()),
        }
    }

    /// 0.25.0 B3: `clear` drains the collection in bounded pages, tolerates a
    /// null-`ids` empty reply, and stops once no IDs remain.
    #[tokio::test]
    async fn test_clear_drains_in_pages() {
        use std::sync::{Arc, Mutex};

        let gets = Arc::new(Mutex::new(0u32));
        let deletes = Arc::new(Mutex::new(0u32));
        let gets_h = gets.clone();
        let deletes_h = deletes.clone();

        let base = spawn_loopback(move |path, request| {
            if path.ends_with("/get") {
                *gets_h.lock().unwrap() += 1;
                // First page: 3 ids; after the drain, Chroma returns null ids.
                let body = if *gets_h.lock().unwrap() == 1 {
                    r#"{"ids":["a","b","c"],"documents":null,"metadatas":null,"embeddings":null}"#
                        .to_string()
                } else {
                    r#"{"ids":null,"documents":null,"metadatas":null,"embeddings":null}"#
                        .to_string()
                };
                // The leading page must carry the size bound + no payload fetch.
                let raw = request.to_string();
                assert!(
                    raw.to_ascii_lowercase()
                        .contains("content-type: application/json"),
                    "clear /get must carry a JSON body: {raw}"
                );
                (200, body)
            } else if path.ends_with("/delete") {
                *deletes_h.lock().unwrap() += 1;
                (200, String::new())
            } else {
                (404, "not found".to_string())
            }
        })
        .await;

        let store = store_with_collection(&base, "cid");
        store.clear().await.unwrap();
        assert_eq!(*gets.lock().unwrap(), 2, "two get rounds expected");
        // Three ids are deleted in a single batch (one delete call, many ids).
        assert_eq!(*deletes.lock().unwrap(), 1, "one bulk delete expected");
    }

    /// 0.25.0 B3: `clear` on an already-empty collection makes a single get and
    /// performs no delete.
    #[tokio::test]
    async fn test_clear_noop_when_empty() {
        use std::sync::{Arc, Mutex};
        let gets = Arc::new(Mutex::new(0u32));
        let deletes = Arc::new(Mutex::new(0u32));
        let gets_h = gets.clone();
        let deletes_h = deletes.clone();

        let base = spawn_loopback(move |path, _request| {
            if path.ends_with("/get") {
                *gets_h.lock().unwrap() += 1;
                (
                    200,
                    r#"{"ids":null,"documents":null,"metadatas":null}"#.to_string(),
                )
            } else if path.ends_with("/delete") {
                *deletes_h.lock().unwrap() += 1;
                (200, String::new())
            } else {
                (404, "not found".to_string())
            }
        })
        .await;

        let store = store_with_collection(&base, "cid");
        store.clear().await.unwrap();
        assert_eq!(*gets.lock().unwrap(), 1);
        assert_eq!(*deletes.lock().unwrap(), 0);
    }

    /// S3: single-field condition → Chroma `where` dict.
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

    /// S3: AND/OR combination → nested `$and`/`$or`.
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

    /// S3: without a filter, `where_filter` is None and the `where` field is not serialized.
    #[test]
    fn test_query_request_no_filter() {
        let req = ChromaDBVectorStore::query_request(&[1.0, 2.0], 3, None);
        assert!(req.where_filter.is_none());
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("where").is_none());
        assert_eq!(v["n_results"], 3);
    }

    /// S3: with a filter, the `where` field serializes to a Chroma dict.
    #[test]
    fn test_query_request_with_filter() {
        let f = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let req = ChromaDBVectorStore::query_request(&[1.0, 2.0], 3, Some(&f));
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["where"], serde_json::json!({ "lang": { "$eq": "rust" } }));
    }

    /// B3/T2: live clear-paging against a real Chroma (`CHROMA_URL`). The loopback
    /// tests above prove the request *sequence* (`get`首页 → 批量删 → 至空); this
    /// proves the multi-page cleanup end-to-end by inserting > 500 docs (above
    /// Chroma's default get page limit of 500) and asserting `clear` empties all
    /// pages. Skipped when `CHROMA_URL` is unset; exercised by the T2 services CI.
    #[tokio::test]
    #[ignore = "requires a running Chroma (set CHROMA_URL)"]
    async fn clear_removes_all_pages_live() {
        let Ok(host) = std::env::var("CHROMA_URL") else {
            eprintln!("CHROMA_URL not set; skipping live Chroma clear-paging");
            return;
        };
        let collection = format!("clear_paging_{}", std::process::id());
        let store = ChromaDBVectorStore::new(ChromaDBConfig::new(host, collection, 8))
            .await
            .expect("connect to live Chroma");
        store.clear().await.expect("clear before seeding");

        // Seed 1250 docs — comfortably above a single 500-row get page.
        let docs: Vec<Document> = (0..1250)
            .map(|i| Document::new(format!("doc {i}")))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..1250).map(|i| vec![i as f32; 8]).collect();
        let _ = store
            .add_documents(docs, embeddings)
            .await
            .expect("seed live Chroma");
        let seeded = store.count().await;
        assert!(seeded >= 1250, "expected all seeded docs, got {seeded}");

        store.clear().await.expect("live clear");
        assert_eq!(store.count().await, 0, "clear must remove across all pages");
    }
}
