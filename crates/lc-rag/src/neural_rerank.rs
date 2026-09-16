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
    client: reqwest::Client,
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
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("build rerank http client"),
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
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents.iter().map(|d| d.content.as_str()).collect::<Vec<_>>(),
            "return_documents": false,
        });
        let resp = self
            .client
            .post(format!("{}/v1/rerank", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                RerankingError::ScoringError(format!("cohere rerank request failed: {e}"))
            })?;
        parse_relevance(resp, documents.len()).await
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
    client: reqwest::Client,
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
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("build rerank http client"),
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
            .client
            .post(format!("{}/v1/rerank", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                RerankingError::ScoringError(format!("jina rerank request failed: {e}"))
            })?;
        parse_relevance(resp, documents.len()).await
    }
}

/// Parse a provider rerank response, mapping each result back to its
/// original index (responses are not guaranteed to be index-ordered).
async fn parse_relevance(resp: reqwest::Response, n: usize) -> Result<Vec<f32>, RerankingError> {
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| RerankingError::ScoringError(format!("read rerank response failed: {e}")))?;
    if !status.is_success() {
        return Err(RerankingError::ScoringError(format!(
            "rerank endpoint returned {status}: {text}"
        )));
    }
    parse_relevance_str(&text, n)
}

/// Map a provider rerank response body back to per-index relevance scores.
///
/// Provider responses are not guaranteed to be index-ordered, so each result
/// is placed at its own original `index`; any out-of-range index is ignored.
fn parse_relevance_str(text: &str, n: usize) -> Result<Vec<f32>, RerankingError> {
    let parsed: RerankResponse = serde_json::from_str(text).map_err(|e| {
        RerankingError::ScoringError(format!("invalid rerank response: {e}: {text}"))
    })?;
    let mut scores = vec![0.0_f32; n];
    for r in parsed.results {
        if r.index < n {
            scores[r.index] = r.relevance_score;
        }
    }
    Ok(scores)
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
    fn ignores_out_of_range_index_and_malformed_body() {
        let body = json!({
            "results": [
                { "index": 99, "relevance_score": 0.9 },
                { "index": 0, "relevance_score": 0.2 },
            ]
        })
        .to_string();
        // Index 99 is discarded; index 0 is kept.
        assert_eq!(parse_relevance_str(&body, 1).unwrap(), vec![0.2]);
        // Malformed JSON → scoring error, not a panic.
        assert!(parse_relevance_str("not json", 1).is_err());
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
