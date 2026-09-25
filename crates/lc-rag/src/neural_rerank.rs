// crates/lc-rag/src/neural_rerank.rs
//! Neural cross-encoder rerankers (Cohere / Jina).
//!
//! Unlike the local [`Reranker`](crate::reranking::Reranker) implementations
//! (keyword / BM25), a neural reranker posts the query *and each candidate
//! document* to a hosted cross-encoder which scores document relevance to the
//! query. That is an async HTTP round-trip, so these live behind an
//! [`AsyncReranker`] trait instead of the synchronous
//! [`Reranker`](crate::reranking::Reranker) trait.

use crate::reranking::RerankingError;
use async_trait::async_trait;
use lc_core::http::{HttpClient, RequestOptions};
use lc_vector_stores::{Document, SearchResult};
use serde::Deserialize;

/// Neural reranker over an async HTTP transport.
#[async_trait]
pub trait AsyncReranker: Send + Sync {
    /// Score each document against `query`, returning one score per document
    /// (index-aligned with `documents`, order preserved).
    async fn score_async(
        &self,
        query: &str,
        documents: &[Document],
    ) -> Result<Vec<f32>, RerankingError>;
}

/// Shared shape of the provider rerank responses we parse.
#[derive(Deserialize)]
struct RerankResponse {
    #[serde(default)]
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    #[serde(default)]
    relevance_score: f32,
}

/// Cohere rerank API client.
///
/// See <https://docs.cohere.com/reference/rerank>. Points at the default public
/// API when no `base_url` is provided.
pub struct CohereRerank {
    api_key: String,
    base_url: String,
    model: String,
    http: HttpClient,
}

impl CohereRerank {
    /// Create a client. An empty `api_key` falls back to `COHERE_API_KEY`.
    pub fn new(api_key: impl Into<String>) -> Self {
        let provided = api_key.into();
        let key = if provided.is_empty() {
            std::env::var("COHERE_API_KEY").unwrap_or_default()
        } else {
            provided
        };
        Self {
            api_key: key,
            base_url: "https://api.cohere.com".to_string(),
            model: "rerank-multilingual-v3.0".to_string(),
            // The unified client defaults to `no_proxy()` (build is infallible
            // for these settings and mirrors the historical `no_proxy()` client).
            http: HttpClient::api().build().expect("build rerank http client"),
        }
    }

    /// Override the API base URL (used by tests to point at a local mock).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Select the rerank model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl AsyncReranker for CohereRerank {
    async fn score_async(
        &self,
        query: &str,
        documents: &[Document],
    ) -> Result<Vec<f32>, RerankingError> {
        // 0.25.0 B3: request an explicit top_n = documents.len() (mirroring the
        // Jina path) so the parser can hold the response strictly to contract —
        // a short/unordered result set now surfaces as an error, not a silent
        // 0.0 "dissimilar" padding.
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents.iter().map(|d| d.content.as_str()).collect::<Vec<_>>(),
            "top_n": documents.len(),
            "return_documents": false,
        });
        let resp = self
            .http
            .post_json_with(
                &format!("{}/v1/rerank", self.base_url),
                &body,
                RequestOptions::new().bearer(self.api_key.clone()),
            )
            .await
            .map_err(|e| {
                RerankingError::ScoringError(format!("cohere rerank request failed: {e}"))
            })?;
        parse_relevance_str(&resp.body, documents.len())
    }
}

/// Jina rerank API client.
///
/// See <https://jina.ai/reranker/>. When no `base_url` is given, points at the
/// public `https://api.jina.ai` endpoint.
pub struct JinaRerank {
    api_key: String,
    base_url: String,
    model: String,
    http: HttpClient,
}

impl JinaRerank {
    /// Create a client. An empty `api_key` falls back to `JINA_API_KEY`.
    pub fn new(api_key: impl Into<String>) -> Self {
        let provided = api_key.into();
        let key = if provided.is_empty() {
            std::env::var("JINA_API_KEY").unwrap_or_default()
        } else {
            provided
        };
        Self {
            api_key: key,
            base_url: "https://api.jina.ai".to_string(),
            model: "jina-reranker-v2-base-multilingual".to_string(),
            // The unified client defaults to `no_proxy()` and ignores system
            // proxy env vars — an explicit `with_base_url` keeps tests hermetic.
            http: HttpClient::api().build().expect("build rerank http client"),
        }
    }

    /// Override the API base URL (used by tests to point at a local mock).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Select the rerank model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl AsyncReranker for JinaRerank {
    async fn score_async(
        &self,
        query: &str,
        documents: &[Document],
    ) -> Result<Vec<f32>, RerankingError> {
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents.iter().map(|d| d.content.as_str()).collect::<Vec<_>>(),
            "top_n": documents.len(),
        });
        let resp = self
            .http
            .post_json_with(
                &format!("{}/v1/rerank", self.base_url),
                &body,
                RequestOptions::new().bearer(self.api_key.clone()),
            )
            .await
            .map_err(|e| {
                RerankingError::ScoringError(format!("jina rerank request failed: {e}"))
            })?;
        parse_relevance_str(&resp.body, documents.len())
    }
}

/// Map a provider rerank response body back to per-index relevance scores.
///
/// Both providers are asked for `top_n == documents.len()` (one score per
/// candidate), so the response must cover every index exactly once. Since
/// results are not guaranteed to be index-ordered, each is placed at its own
/// `index`; a result whose index is out of range, or a repeated index (the
/// provider broke its top_n contract) is a hard [`RerankingError`] rather than
/// a silent 0.0 "dissimilar" row that would bury a genuinely relevant document.
fn parse_relevance_str(text: &str, n: usize) -> Result<Vec<f32>, RerankingError> {
    let parsed: RerankResponse = serde_json::from_str(text).map_err(|e| {
        RerankingError::ScoringError(format!("invalid rerank response: {e}: {text}"))
    })?;
    let mut scores: Vec<Option<f32>> = vec![None; n];
    for r in parsed.results {
        match r.index {
            // Must appear exactly once; a duplicate index means the provider
            // returned two rows for one document.
            i if i < n && scores[i].is_some() => {
                return Err(RerankingError::ScoringError(format!(
                    "rerank response repeats index {i}"
                )));
            }
            // Out of range — the response does not cover the requested set.
            i if i >= n => {
                return Err(RerankingError::ScoringError(format!(
                    "rerank response index {i} out of range for {n} documents"
                )));
            }
            i => scores[i] = Some(r.relevance_score),
        }
    }
    // A missing index silently buries that document at the bottom with a 0.0.
    if let Some((missing, _)) = scores.iter().enumerate().find(|(_, s)| s.is_none()) {
        return Err(RerankingError::ScoringError(format!(
            "rerank response missing index {missing}"
        )));
    }
    Ok(scores
        .into_iter()
        .map(|s| s.expect("all indices filled above"))
        .collect())
}

/// Re-rank `results` in place of the document order using an [`AsyncReranker`],
/// producing the same [`SearchResult`] order a synchronous
/// [`RerankingExecutor`](crate::reranking::RerankingExecutor) would.
pub async fn rerank_async(
    reranker: &dyn AsyncReranker,
    query: &str,
    results: Vec<SearchResult>,
    top_n: usize,
) -> Result<Vec<SearchResult>, RerankingError> {
    if results.is_empty() {
        return Ok(Vec::new());
    }
    let documents: Vec<Document> = results.iter().map(|r| r.document.clone()).collect();
    let scores = reranker.score_async(query, &documents).await?;
    let mut ranked: Vec<SearchResult> = results
        .into_iter()
        .zip(scores)
        .map(|(mut r, score)| {
            r.score = score;
            r
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_n);
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A minimal loopback HTTP/1.1 server: each accepted connection invokes
    /// `handler(path, body)` which returns `(status, response_body)`.
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
                    // Keep reading until the header terminator plus the declared
                    // Content-Length bytes have fully arrived (reqwest may split
                    // the request across writes).
                    let (head_end, content_length) = loop {
                        if raw.windows(4).any(|w| w == b"\r\n\r\n") {
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
                            break (
                                raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4,
                                len,
                            );
                        }
                        let n = socket.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            break (raw.len(), 0);
                        }
                        raw.extend_from_slice(&buf[..n]);
                    };
                    while raw.len() < head_end + content_length {
                        let n = socket.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        raw.extend_from_slice(&buf[..n]);
                    }
                    // Report the full HTTP text (headers + JSON body) to the
                    // handler; forwarding only the body here risks truncation.
                    let request = String::from_utf8_lossy(&raw).to_string();
                    let path = request.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let (status, resp_body) = handler(&path, &request);
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

    /// A deterministic fake reranker so `rerank_async`'s ordering can be tested
    /// without any network I/O (loopback connects are blocked in this sandbox).
    struct FakeReranker {
        scores: Vec<f32>,
    }

    #[async_trait]
    impl AsyncReranker for FakeReranker {
        async fn score_async(
            &self,
            _query: &str,
            documents: &[Document],
        ) -> Result<Vec<f32>, RerankingError> {
            assert_eq!(self.scores.len(), documents.len());
            Ok(self.scores.clone())
        }
    }

    #[test]
    fn parses_out_of_order_result_indices_back_to_position() {
        // Provider returns results out of order; the parser must place each at its
        // own index, giving [0.1, 0.5, 0.9] for the three documents.
        let body = json!({
            "results": [
                { "index": 2, "relevance_score": 0.9 },
                { "index": 0, "relevance_score": 0.1 },
                { "index": 1, "relevance_score": 0.5 },
            ]
        })
        .to_string();
        let scores = parse_relevance_str(&body, 3).unwrap();
        assert_eq!(scores, vec![0.1, 0.5, 0.9]);
    }

    #[test]
    fn out_of_range_index_is_a_hard_error() {
        // 0.25.0 B3: an out-of-range index is a provider contract break
        // (top_n = len() should cover every index), not a value to discard.
        let body = json!({
            "results": [
                { "index": 0, "relevance_score": 0.2 },
                { "index": 99, "relevance_score": 0.9 },
            ]
        })
        .to_string();
        let err = parse_relevance_str(&body, 1).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn missing_index_is_a_hard_error() {
        let body = json!({
            "results": [
                { "index": 1, "relevance_score": 0.5 },
            ]
        })
        .to_string();
        let err = parse_relevance_str(&body, 3).unwrap_err();
        assert!(err.to_string().contains("missing index"));
    }

    #[test]
    fn duplicate_index_is_a_hard_error() {
        let body = json!({
            "results": [
                { "index": 0, "relevance_score": 0.2 },
                { "index": 0, "relevance_score": 0.9 },
                { "index": 1, "relevance_score": 0.7 },
            ]
        })
        .to_string();
        let err = parse_relevance_str(&body, 2).unwrap_err();
        assert!(err.to_string().contains("repeats index 0"));
    }

    #[test]
    fn malformed_body_is_a_scoring_error() {
        assert!(parse_relevance_str("not json", 1).is_err());
    }

    #[tokio::test]
    async fn cohere_rerank_sends_top_n_and_auth() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new((String::new(), String::new())));
        let seen_h = seen.clone();
        let base = spawn_loopback(move |_path, request| {
            // `request` is the full HTTP text (headers + JSON body).
            let json_part = request.split("\r\n\r\n").nth(1).unwrap_or("");
            let body_value = serde_json::from_str::<serde_json::Value>(json_part).ok();
            let top_n = body_value
                .as_ref()
                .and_then(|v| v.get("top_n"))
                .and_then(|x| x.as_u64());
            let auth_header: String = request
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
                .unwrap_or("")
                .to_string();
            if top_n == Some(2) {
                let request_line = request.lines().next().unwrap_or("").to_string();
                let mut guard = seen_h.lock().unwrap();
                guard.0 = request_line;
                guard.1 = auth_header;
            }
            (
                200,
                r#"{"results":[{"index":0,"relevance_score":0.9},{"index":1,"relevance_score":0.1}]}"#
                    .to_string(),
            )
        })
        .await;

        let reranker = CohereRerank::new("cohere-key").with_base_url(base);
        let docs = vec![Document::new("a"), Document::new("b")];
        let scores = reranker.score_async("q", &docs).await.unwrap();
        assert_eq!(scores, vec![0.9, 0.1]);

        let (request_line, auth) = &*seen.lock().unwrap();
        assert!(
            request_line
                .to_ascii_lowercase()
                .starts_with("post /v1/rerank"),
            "{request_line}"
        );
        assert!(!auth.is_empty(), "should carry Bearer auth");
        assert!(auth.to_ascii_lowercase().contains("bearer cohere-key"));
    }

    #[tokio::test]
    async fn rerank_non_2xx_surfaces_error() {
        let base =
            spawn_loopback(|_path, _body| (401, r#"{"message":"bad key"}"#.to_string())).await;
        let reranker = JinaRerank::new("jina-key").with_base_url(base);
        let docs = vec![Document::new("a")];
        let err = reranker.score_async("q", &docs).await.unwrap_err();
        assert!(err.to_string().contains("401"));
        assert!(err.to_string().contains("bad key"));
    }

    #[tokio::test]
    async fn rerank_async_reorders_and_truncates() {
        let reranker = FakeReranker {
            scores: vec![0.1, 0.5, 0.9],
        };
        let results = vec![
            SearchResult {
                document: Document::new("doc A"),
                score: 0.0,
            },
            SearchResult {
                document: Document::new("doc B"),
                score: 0.0,
            },
            SearchResult {
                document: Document::new("doc C"),
                score: 0.0,
            },
        ];
        let reranked = rerank_async(&reranker, "q", results, 2).await.unwrap();
        let contents: Vec<&str> = reranked
            .iter()
            .map(|r| r.document.content.as_str())
            .collect();
        assert_eq!(contents, vec!["doc C", "doc B"]);

        // Empty input is a short-circuit, not an error.
        assert!(rerank_async(&reranker, "q", Vec::new(), 3)
            .await
            .unwrap()
            .is_empty());
    }
}
