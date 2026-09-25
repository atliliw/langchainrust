//! Pinecone vector store (HTTP API)

use std::collections::HashMap;

use async_trait::async_trait;
use lc_core::http::{HttpClient, RequestOptions};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    Document, Embeddings, FilterOp, MetadataFilter, SearchResult, VectorStore, VectorStoreError,
};

/// Upper bound on each vector's serialized metadata, per the Pinecone REST API
/// ("metadata": each vector's metadata is limited to 40 KiB).
const PINECONE_METADATA_MAX_BYTES: usize = 40 * 1024;

/// Pinecone vector store client
pub struct PineconeStore {
    api_key: String,
    host: String,
    http: HttpClient,
}

impl PineconeStore {
    /// Create a Pinecone client.
    ///
    /// `host` format: `https://{index-name}.svc.{environment}.pinecone.io`
    pub fn new(api_key: impl Into<String>, host: impl Into<String>) -> Self {
        let http = HttpClient::api()
            .build()
            .expect("build pinecone http client");
        Self {
            api_key: api_key.into(),
            host: host.into(),
            http,
        }
    }

    /// Per-request options: Pinecone authenticates via the `Api-Key` header.
    fn request_options(&self) -> RequestOptions {
        let mut opts = RequestOptions::new();
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&self.api_key) {
            opts = opts.header(reqwest::header::HeaderName::from_static("api-key"), v);
        }
        opts
    }

    /// Maps unified-layer errors onto [`VectorStoreError`], keeping a
    /// non-2xx status body readable for diagnostics.
    fn map_http_error(prefix: &str, err: lc_core::http::HttpError) -> VectorStoreError {
        let msg = match err {
            lc_core::http::HttpError::Status { status, body } => {
                format!("HTTP {status}: {body}")
            }
            other => other.to_string(),
        };
        VectorStoreError::ConnectionError(format!("{prefix}: {msg}"))
    }

    /// Build the upsert request body, injecting each document's text into
    /// `metadata.content` (only when the caller did not already set that key)
    /// and failing loudly when a vector's metadata exceeds the 40 KiB cap.
    ///
    /// 0.25.0 B3: read-back paths recover the document text from
    /// `metadata.content` ([`PineconeStore::query`]); without the injection the
    /// text is silently lost on the round trip. Oversized metadata is rejected
    /// here instead of surfacing as an opaque Pinecone 400 transport error.
    pub fn build_upsert_body(
        docs: &[Document],
        vectors: &[Vec<f32>],
        ids: &[String],
    ) -> Result<serde_json::Value, VectorStoreError> {
        if docs.len() != vectors.len() || docs.len() != ids.len() {
            return Err(VectorStoreError::StorageError(format!(
                "Pinecone upsert shape mismatch: {} docs, {} vectors, {} ids",
                docs.len(),
                vectors.len(),
                ids.len()
            )));
        }
        let mut vectors_json: Vec<serde_json::Value> = Vec::with_capacity(docs.len());
        for (i, (doc, id)) in docs.iter().zip(ids.iter()).enumerate() {
            // Inject the content text under the reserved key unless the caller
            // provided it (user-supplied wins).
            let mut metadata = doc.metadata.clone();
            if !metadata.contains_key("content") {
                metadata.insert("content".to_string(), Value::from(doc.content.clone()));
            }
            let metadata_bytes = serde_json::to_vec(&metadata)
                .map_err(|e| {
                    VectorStoreError::StorageError(format!(
                        "failed to serialize Pinecone metadata: {e}"
                    ))
                })?
                .len();
            if metadata_bytes > PINECONE_METADATA_MAX_BYTES {
                return Err(VectorStoreError::StorageError(format!(
                    "vector metadata exceeds the 40 KiB limit ({metadata_bytes} bytes); \
                     refusing to upsert id {id}"
                )));
            }
            vectors_json.push(serde_json::json!({
                "id": id,
                "values": &vectors[i],
                "metadata": metadata,
            }));
        }
        Ok(serde_json::json!({ "vectors": vectors_json }))
    }

    /// Build query request body (pure function, convenient for testing).
    pub fn build_query_body(query_vec: &[f32], top_k: usize) -> serde_json::Value {
        serde_json::json!({
            "vector": query_vec,
            "topK": top_k,
            "includeMetadata": true,
        })
    }

    /// Build query request body with a metadata filter (pure function, convenient for testing).
    ///
    /// S3: extends [`build_query_body`](Self::build_query_body) with a Pinecone `filter` field,
    /// where [`filter_to_pinecone`] translates [`MetadataFilter`] → Pinecone query syntax
    /// (field name → `$op` value, combinations via `$and`/`$or`).
    pub fn build_query_body_filtered(
        query_vec: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> serde_json::Value {
        let mut body = Self::build_query_body(query_vec, top_k);
        body["filter"] = filter_to_pinecone(filter);
        body
    }

    /// Upsert documents (auto-embed).
    pub async fn upsert(
        &self,
        docs: &[Document],
        embeddings: &dyn Embeddings,
    ) -> Result<(), VectorStoreError> {
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let vectors = embeddings
            .embed_documents(&texts)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
        let ids: Vec<String> = docs
            .iter()
            .map(|d| d.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()))
            .collect();
        let body = Self::build_upsert_body(docs, &vectors, &ids)?;
        let url = format!("{}/vectors/upsert", self.host);
        self.http
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(|e| Self::map_http_error("Pinecone upsert error", e))?;
        Ok(())
    }

    /// Query similar documents.
    pub async fn query(
        &self,
        query_vec: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<Document>, VectorStoreError> {
        let body = Self::build_query_body(&query_vec, top_k);
        let url = format!("{}/query", self.host);
        let resp = self
            .http
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(|e| Self::map_http_error("Pinecone query error", e))?;
        let query_resp: QueryResponse = serde_json::from_str(&resp.body).map_err(|e| {
            VectorStoreError::StorageError(format!("failed to parse Pinecone query: {e}"))
        })?;
        let result = query_resp
            .matches
            .into_iter()
            .map(|m| {
                let content = m
                    .metadata
                    .as_ref()
                    .and_then(|md| md.get("content").and_then(|v| v.as_str()))
                    .unwrap_or_default()
                    .to_string();
                Document {
                    content,
                    metadata: m.metadata.unwrap_or_default(),
                    id: Some(m.id),
                }
            })
            .collect();
        Ok(result)
    }

    /// Reads index statistics (the only reliable source for a true count).
    ///
    /// The Pinecone REST API provides `describe_index_stats`, returning `totalVectorCount`.
    pub async fn describe_index_stats(&self) -> Result<PineconeIndexStats, VectorStoreError> {
        let url = format!("{}/describe_index_stats", self.host);
        let resp = self
            .http
            .post_json_with(&url, &serde_json::json!({}), self.request_options())
            .await
            .map_err(|e| Self::map_http_error("Pinecone describe_index_stats error", e))?;
        serde_json::from_str(&resp.body).map_err(|e| {
            VectorStoreError::StorageError(format!(
                "failed to parse describe_index_stats response: {e}"
            ))
        })
    }

    /// Delete by IDs.
    pub async fn delete(&self, ids: &[String]) -> Result<(), VectorStoreError> {
        let url = format!("{}/vectors/delete", self.host);
        let body = serde_json::json!({ "ids": ids });
        self.http
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(|e| Self::map_http_error("Pinecone delete error", e))?;
        Ok(())
    }

    /// POSTs to `/query` and parses the result (shared by plain and filtered retrieval).
    async fn query_impl(
        &self,
        body: serde_json::Value,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = format!("{}/query", self.host);
        let resp = self
            .http
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(|e| Self::map_http_error("Pinecone query HTTP error", e))?;

        let query_resp: QueryResponse = serde_json::from_str(&resp.body).map_err(|e| {
            VectorStoreError::StorageError(format!("Pinecone query parse error: {e}"))
        })?;

        Ok(query_resp
            .matches
            .into_iter()
            .map(|m| {
                let content = m
                    .metadata
                    .as_ref()
                    .and_then(|md| md.get("content").and_then(|v| v.as_str()))
                    .unwrap_or_default()
                    .to_string();
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
            .collect())
    }
}

/// S3: translates [`MetadataFilter`] → Pinecone `filter` syntax.
///
/// A single-field condition becomes `{ key: { "$op": value } }` (Pinecone supports
/// `$eq $ne $gt $gte $lt $lte $in $nin`); AND/OR combinations become
/// `{ "$and": [...] }` / `{ "$or": [...] }`. The types map one-to-one with [`FilterOp`],
/// with no inexpressible construct.
pub fn filter_to_pinecone(filter: &MetadataFilter) -> serde_json::Value {
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
            let items: Vec<serde_json::Value> = filters.iter().map(filter_to_pinecone).collect();
            serde_json::json!({ "$and": items })
        }
        MetadataFilter::Or(filters) => {
            let items: Vec<serde_json::Value> = filters.iter().map(filter_to_pinecone).collect();
            serde_json::json!({ "$or": items })
        }
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

        // Build upsert body keyed by the SAME ids we return to the caller (B3/fix:
// previously `build_upsert_body` regenerated a fresh UUID for id-less docs, so
// the returned ids did not match the stored vectors and a later delete/get by
// the returned id hit nothing). The 40 KiB metadata cap is enforced here
// (content injection + serialization size check).
        let body = Self::build_upsert_body(&documents, &embeddings, &ids)?;
        let url = format!("{}/vectors/upsert", self.host);
        self.http
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(|e| Self::map_http_error("Pinecone upsert HTTP error", e))?;

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let body = Self::build_query_body(query_embedding, k);
        self.query_impl(body).await
    }

    /// S3: similarity search with metadata filtering — filtering is delegated to the server (native Pinecone filter syntax).
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let body = match filter {
            Some(f) => Self::build_query_body_filtered(query_embedding, k, f),
            None => Self::build_query_body(query_embedding, k),
        };
        self.query_impl(body).await
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
        self.delete(&[id.to_string()]).await
    }

    async fn count(&self) -> usize {
        // Q4: real implementation — reads totalVectorCount via describe_index_stats, no longer hardcoded to 0.
        // the trait signature returns usize; on network failure it degrades to 0 and logs (no error raised).
        match self.describe_index_stats().await {
            Ok(stats) => stats.total_vector_count,
            Err(e) => {
                log::warn!("Pinecone count failed, treating as 0: {}", e);
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
    metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Response of Pinecone `describe_index_stats` (only the fields we care about).
///
/// Q4: returned by [`PineconeStore::describe_index_stats`], letting callers read the true
/// total vector count; it is also the data source for [`VectorStore::count`].
#[derive(Deserialize)]
pub struct PineconeIndexStats {
    /// Total number of vectors in the index
    #[serde(default)]
    pub total_vector_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A minimal loopback server: reads one request (headers + body) and replies
    /// with `(status, body)` chosen by `handler(path, full_request_text)` where
    /// `full_request_text` is the raw HTTP request (request line + headers + JSON
    /// body), letting tests assert on headers like `Api-Key`.
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
        let ids = vec!["1".to_string(), "2".to_string()];
        let body = PineconeStore::build_upsert_body(&docs, &vectors, &ids).unwrap();
        let vectors_arr = body.get("vectors").unwrap().as_array().unwrap();
        assert_eq!(vectors_arr.len(), 2);
        assert_eq!(vectors_arr[0]["id"], "1");
        assert_eq!(vectors_arr[0]["values"][0], 1.0);
        // The document text is injected under metadata.content.
        assert_eq!(vectors_arr[0]["metadata"]["content"], "hello");
    }

    /// The stored ids must be the caller-supplied ids; the caller generates one
    /// up front for a document without an id (so the returned ids match the
    /// stored vectors — previously the builder re-generated a fresh UUID,
    /// silently orphaning the returned ids).
    #[test]
    fn test_build_upsert_body_uses_supplied_ids() {
        let mut d = doc("", "x");
        d.id = None;
        let ids = vec![uuid::Uuid::new_v4().to_string()];
        let body = PineconeStore::build_upsert_body(&[d], &[vec![0.1]], &ids).unwrap();
        assert_eq!(body["vectors"][0]["id"], ids[0]);
    }

    #[test]
    fn test_build_upsert_body_rejects_shape_mismatch() {
        let docs = vec![doc("1", "a")];
        let ids = vec!["1".to_string(), "2".to_string()]; // too long
        let err = PineconeStore::build_upsert_body(&docs, &[vec![0.1]], &ids).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"), "message: {err}");
    }

    /// 0.25.0 B3: a caller-supplied `content` metadata key must win over the
    /// injected document text.
    #[test]
    fn test_build_upsert_body_respects_caller_content_key() {
        let mut d = doc("1", "doc text");
        d.metadata
            .insert("content".to_string(), serde_json::json!("custom"));
        let body = PineconeStore::build_upsert_body(&[d], &[vec![1.0]], &["1".to_string()]).unwrap();
        assert_eq!(body["vectors"][0]["metadata"]["content"], "custom");
    }

    /// 0.25.0 B3: metadata that exceeds the 40 KiB per-vector cap is rejected
    /// loudly at build time, not surfaced as an opaque server 400.
    #[test]
    fn test_build_upsert_body_rejects_oversized_metadata() {
        let mut d = doc("1", "text");
        d.metadata.insert(
            "blob".to_string(),
            serde_json::Value::String("x".repeat(40 * 1024 + 1)),
        );
        let err =
            PineconeStore::build_upsert_body(&[d], &[vec![1.0]], &["1".to_string()]).unwrap_err();
        assert!(err.to_string().contains("40 KiB"), "message: {err}");
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

    /// S3: single-field condition → Pinecone `{ key: { "$op": value } }`.
    #[test]
    fn test_filter_to_pinecone_field_ops() {
        assert_eq!(
            filter_to_pinecone(&MetadataFilter::field("lang", FilterOp::Eq, "rust")),
            serde_json::json!({ "lang": { "$eq": "rust" } })
        );
        assert_eq!(
            filter_to_pinecone(&MetadataFilter::field("year", FilterOp::Gte, 2020)),
            serde_json::json!({ "year": { "$gte": 2020 } })
        );
        assert_eq!(
            filter_to_pinecone(&MetadataFilter::field("tags", FilterOp::Nin, vec!["blog"])),
            serde_json::json!({ "tags": { "$nin": ["blog"] } })
        );
    }

    /// S3: AND/OR combination → nested `$and`/`$or`.
    #[test]
    fn test_filter_to_pinecone_and_or() {
        let f = MetadataFilter::and(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "rust"),
            MetadataFilter::or(vec![
                MetadataFilter::field("year", FilterOp::Gte, 2020),
                MetadataFilter::field("tags", FilterOp::In, vec!["ml"]),
            ]),
        ]);
        assert_eq!(
            filter_to_pinecone(&f),
            serde_json::json!({
                "$and": [
                    { "lang": { "$eq": "rust" } },
                    { "$or": [
                        { "year": { "$gte": 2020 } },
                        { "tags": { "$in": ["ml"] } }
                    ]}
                ]
            })
        );
    }

    /// S3: the filtered query body = the plain query body + a filter field.
    #[test]
    fn test_build_query_body_filtered() {
        let f = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let body = PineconeStore::build_query_body_filtered(&[1.0, 2.0], 5, &f);
        assert_eq!(body["topK"], 5);
        assert_eq!(body["includeMetadata"], true);
        assert_eq!(
            body["filter"],
            serde_json::json!({ "lang": { "$eq": "rust" } })
        );
    }

    /// 0.25.0 B3: the `metadata.content` injected at upsert time must come back
    /// out of a real (loopback) `/query` call, so a document's text survives the
    /// round trip instead of being silently dropped.
    #[tokio::test]
    async fn test_query_recovers_metadata_content_roundtrip() {
        let base = spawn_loopback(|_path, _request| {
            (
                200,
                serde_json::json!({
                    "matches": [{
                        "id": "abc",
                        "score": 1.0,
                        "metadata": { "content": "hello world" }
                    }]
                })
                .to_string(),
            )
        })
        .await;
        let store = PineconeStore::new("key", &base);
        let docs = store.query(vec![1.0, 2.0], 3).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "hello world");
        assert_eq!(docs[0].id.as_deref(), Some("abc"));
    }

    /// 0.25.0 B3: every call must carry the `Api-Key` header (not Bearer), else
    /// Pinecone rejects the request with a 401.
    #[tokio::test]
    async fn test_http_sends_api_key_header() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let seen_h = seen.clone();
        let base = spawn_loopback(move |_path, request| {
            let auth: String = request
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("api-key:"))
                .unwrap_or("")
                .to_string();
            let mut guard = seen_h.lock().unwrap();
            *guard = auth;
            (200, serde_json::json!({ "matches": [] }).to_string())
        })
        .await;
        let store = PineconeStore::new("pinecone-key", &base);
        let _ = store.query(vec![1.0], 1).await.unwrap();
        let auth = &*seen.lock().unwrap();
        assert!(
            auth.to_ascii_lowercase().contains("api-key: pinecone-key"),
            "expected Api-Key header, got: {auth:?}"
        );
    }

    /// 0.25.0 B3: a non-2xx response must surface its body text (Pinecone's
    /// error messages live there), not a bare status code.
    #[tokio::test]
    async fn test_non_2xx_surfaces_body_text() {
        let base =
            spawn_loopback(|_path, _request| (400, r#"{"message":"quota exceeded"}"#.to_string()))
                .await;
        let store = PineconeStore::new("key", &base);
        let err = store.query(vec![1.0], 1).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("400"), "message: {msg}");
        assert!(msg.contains("quota exceeded"), "message: {msg}");
    }
}
