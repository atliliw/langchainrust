// crates/lc-rag/src/colbert.rs
//! ColBERT-style *late-interaction* token-level re-ranker.
//!
//! Rather than comparing a whole query against a whole document (early /
//! pooled interaction), late interaction embeds the query and each candidate
//! document at **token level** and scores relevance by **MaxSim**: for every
//! query token, take its maximum cosine similarity against any document token,
//! then sum those maxima. Partial matches on different tokens are rewarded
//! independently (a document that matches half the query tokens strongly beats
//! one that matches the whole query weakly), which is what distinguishes this
//! from a pooled cosine on the whole-document embedding.
//!
//! This is the **inference-side** integration: it re-ranks a candidate set you
//! already retrieved (typically the cheap first pass), exactly like the hosted
//! [`AsyncReranker`] implementations. It
//! does **not** store dense per-token document embeddings at index time — that
//! is the deferred (0.27) training/ingestion track.
//!
//! The token embedder comes from [`TokenLevelEmbeddings`], which is a
//! *non-object-safe* native-async trait (only the ONNX `LocalEmbeddings`
//! path implements it). So [`ColbertRerank`] is **generic** over the embedder
//! type and implements the object-safe [`AsyncReranker`]
//! trait, so it plugs straight into [`rerank_async`](crate::neural_rerank::rerank_async).
//!
//! ```ignore
//! use lc_embeddings::{LocalEmbeddings, TokenLevelEmbeddings};
//! use lc_rag::{ColbertRerank, rerank_async};
//!
//! let emb = LocalEmbeddings::from_file("model.onnx")
//!     .expect("local ONNX token-level embedder");
//! let reranker = ColbertRerank::new(emb)
//!     .with_max_query_tokens(32)
//!     .with_max_doc_tokens(256);
//! let ranked = rerank_async(&reranker, query, results, top_n).await?;
//! ```

use async_trait::async_trait;
use lc_embeddings::cosine_similarity;
use lc_embeddings::token_level::{TokenEmbedding, TokenLevelEmbeddings};
use lc_vector_stores::Document;

use crate::neural_rerank::AsyncReranker;
use crate::reranking::RerankingError;

/// Late-interaction (MaxSim) score between an embedded query and an embedded
/// document: for each query token, take the maximum cosine against any
/// document token and sum those maxima.
///
/// Returns `f32::NEG_INFINITY` when either side has no tokens (so an empty
/// document sinks to the bottom of the ranking rather than scoring 0).
fn maxsim(
    query: &[TokenEmbedding],
    doc: &[TokenEmbedding],
    max_query_tokens: Option<usize>,
    max_doc_tokens: Option<usize>,
) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return f32::NEG_INFINITY;
    }
    let query_lim = max_query_tokens.unwrap_or(query.len()).min(query.len());
    let doc_lim = max_doc_tokens.unwrap_or(doc.len()).min(doc.len());
    let mut total = 0.0f32;
    for qt in &query[..query_lim] {
        let mut best = f32::MIN;
        for dt in &doc[..doc_lim] {
            // Token vectors from the same embedder share a dimension; a length
            // mismatch can only arise from a misconfigured embedder — treat it
            // as zero similarity rather than erroring the whole batch.
            let sim = cosine_similarity(&qt.vector, &dt.vector).unwrap_or(0.0);
            if sim > best {
                best = sim;
            }
        }
        total += best;
    }
    total
}

/// Embed a single text into per-token vectors, mapping embedder errors onto
/// the reranking error type.
async fn embed<E>(embedder: &E, text: &str) -> Result<Vec<TokenEmbedding>, RerankingError>
where
    E: TokenLevelEmbeddings + Send + Sync,
{
    embedder
        .embed_tokens(text)
        .await
        .map_err(|e| RerankingError::ScoringError(format!("embed {text:?}: {e}")))
}

/// Score `documents` against `query` by late interaction. Returns one score
/// per document, index-aligned and order-preserved.
pub async fn score<E>(
    embedder: &E,
    query: &str,
    documents: &[Document],
) -> Result<Vec<f32>, RerankingError>
where
    E: TokenLevelEmbeddings + Send + Sync,
{
    let query_emb = embed(embedder, query).await?;
    let mut scores = Vec::with_capacity(documents.len());
    for doc in documents {
        // A blank candidate has no tokens to match; skip embedding (which the
        // token-level contract rejects) and sink it to the bottom instead of
        // failing the whole batch.
        if doc.page_content().trim().is_empty() {
            scores.push(f32::NEG_INFINITY);
            continue;
        }
        let doc_emb = embed(embedder, doc.page_content()).await?;
        scores.push(maxsim(&query_emb, &doc_emb, None, None));
    }
    Ok(scores)
}

/// ColBERT late-interaction re-ranker, generic over a [`TokenLevelEmbeddings`]
/// embedder. Implement [`AsyncReranker`] so it composes with the existing
/// [`rerank_async`](crate::neural_rerank::rerank_async) path.
#[derive(Clone)]
pub struct ColbertRerank<E> {
    embedder: E,
    /// Bound the number of query tokens scored (keeps cost bounded for long
    /// queries; `None` = all tokens).
    max_query_tokens: Option<usize>,
    /// Bound the number of document tokens scanned per query token (keeps cost
    /// bounded for long documents; `None` = all tokens).
    max_doc_tokens: Option<usize>,
}

impl<E> ColbertRerank<E> {
    /// Wrap a token-level embedder (e.g. the ONNX `LocalEmbeddings`).
    pub fn new(embedder: E) -> Self {
        Self {
            embedder,
            max_query_tokens: None,
            max_doc_tokens: None,
        }
    }

    /// Cap the number of query tokens considered, defaulting to all.
    pub fn with_max_query_tokens(mut self, n: usize) -> Self {
        self.max_query_tokens = Some(n);
        self
    }

    /// Cap the number of document tokens scanned per query token, defaulting
    /// to all.
    pub fn with_max_doc_tokens(mut self, n: usize) -> Self {
        self.max_doc_tokens = Some(n);
        self
    }
}

#[async_trait]
impl<E> AsyncReranker for ColbertRerank<E>
where
    E: TokenLevelEmbeddings + Send + Sync,
{
    async fn score_async(
        &self,
        query: &str,
        documents: &[Document],
    ) -> Result<Vec<f32>, RerankingError> {
        let query_emb = embed(&self.embedder, query).await?;
        let mut scores = Vec::with_capacity(documents.len());
        for doc in documents {
            if doc.page_content().trim().is_empty() {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let doc_emb = embed(&self.embedder, doc.page_content()).await?;
            scores.push(maxsim(
                &query_emb,
                &doc_emb,
                self.max_query_tokens,
                self.max_doc_tokens,
            ));
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::token_level::TokenSpan;
    use lc_embeddings::EmbeddingError;
    use lc_vector_stores::SearchResult;
    use std::collections::HashMap;

    /// Deterministic token-level embedder (offline, no model download).
    ///
    /// Each token maps to a fixed unit vector so the MaxSim math is exact, with
    /// crafted partial similarity: `dog`≈`dogs` (0.8) and `food`≈`feed` (0.5).
    struct MockTokenEmbedder {
        dim: usize,
        map: HashMap<String, Vec<f32>>,
    }

    impl MockTokenEmbedder {
        fn new() -> Self {
            let mut map = HashMap::new();
            map.insert("dog".into(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            map.insert("dogs".into(), vec![0.8, 0.6, 0.0, 0.0, 0.0, 0.0]);
            map.insert("cat".into(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
            map.insert("food".into(), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
            map.insert("feed".into(), vec![0.0, 0.0, 0.5, 0.866_025_4, 0.0, 0.0]);
            map.insert("walk".into(), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
            Self { dim: 6, map }
        }
    }

    impl TokenLevelEmbeddings for MockTokenEmbedder {
        async fn embed_tokens(&self, text: &str) -> Result<Vec<TokenEmbedding>, EmbeddingError> {
            if text.trim().is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }
            let mut out = Vec::new();
            let mut pos = 0usize;
            for tok in text.split_whitespace() {
                let start = pos + text[pos..].find(tok).unwrap();
                let end = start + tok.len();
                pos = end;
                let vector = self
                    .map
                    .get(tok)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.dim]);
                out.push(TokenEmbedding {
                    span: TokenSpan::new(start, end),
                    vector,
                });
            }
            Ok(out)
        }
    }

    fn embedder() -> MockTokenEmbedder {
        MockTokenEmbedder::new()
    }

    #[tokio::test]
    async fn maxsim_accumulates_partial_token_matches() {
        // Query "dog food"; expected MaxSim:
        //   "dog walk"   -> cos(dog,dog)=1.0 + cos(food,{dog:0,walk:0})=0.0  = 1.0
        //   "dogs eat X" -> cos(dog,dogs)=0.8 + cos(food,feed)=0.5           = 1.3
        //   "cat feed"   -> cos(dog,{cat:0,feed:0})=0 + cos(food,feed)=0.5   = 0.5
        let emb = embedder();
        let docs = vec![
            Document::new("dog walk"),
            Document::new("dogs eat feed"),
            Document::new("cat feed"),
        ];
        let scores = score(&emb, "dog food", &docs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < 1e-5, "got {}", scores[0]);
        assert!((scores[1] - 1.3).abs() < 1e-5, "got {}", scores[1]);
        assert!((scores[2] - 0.5).abs() < 1e-5, "got {}", scores[2]);
        // The token-level accumulation beats a naive pooled match: B > A because
        // two *partial* token matches outscore one exact + one disjoint token.
        assert!(scores[1] > scores[0] && scores[0] > scores[2]);
    }

    #[tokio::test]
    async fn rerank_async_sorts_desc_and_truncates() {
        let emb = embedder();
        let reranker = ColbertRerank::new(emb);
        let results = vec![
            SearchResult {
                document: Document::new("cat feed"),
                score: 0.1,
            },
            SearchResult {
                document: Document::new("dogs eat feed"),
                score: 0.2,
            },
            SearchResult {
                document: Document::new("dog walk"),
                score: 0.3,
            },
        ];
        let ranked = crate::neural_rerank::rerank_async(&reranker, "dog food", results, 2)
            .await
            .unwrap();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].document.page_content(), "dogs eat feed");
        assert_eq!(ranked[1].document.page_content(), "dog walk");
        assert!(ranked[0].score > ranked[1].score);
    }

    #[tokio::test]
    async fn empty_document_sinks_to_bottom() {
        let emb = embedder();
        let reranker = ColbertRerank::new(emb);
        let ranked = crate::neural_rerank::rerank_async(
            &reranker,
            "dog food",
            vec![
                SearchResult {
                    document: Document::new("dog walk"),
                    score: 5.0,
                },
                SearchResult {
                    document: Document::new(""),
                    score: 4.0,
                },
            ],
            2,
        )
        .await
        .unwrap();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].document.page_content(), "dog walk");
        assert_eq!(ranked[1].document.page_content(), "");
        assert_eq!(ranked[1].score, f32::NEG_INFINITY);
    }

    #[tokio::test]
    async fn empty_query_is_a_scoring_error() {
        let emb = embedder();
        let docs = vec![Document::new("dog walk")];
        let res = crate::colbert::score(&emb, "", &docs).await;
        assert!(matches!(res, Err(RerankingError::ScoringError(_))));
    }

    #[tokio::test]
    async fn max_query_tokens_cap_prunes_matching() {
        // With max_query_tokens=1 only the first query token ("dog") is scored:
        //   "dog walk" -> 1.0 ; "dogs eat feed" -> 0.8 ; "cat feed" -> 0.0
        let emb = embedder();
        let reranker = ColbertRerank::new(emb).with_max_query_tokens(1);
        let docs = vec![
            Document::new("dog walk"),
            Document::new("dogs eat feed"),
            Document::new("cat feed"),
        ];
        let scores = reranker.score_async("dog food", &docs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < 1e-5);
        assert!((scores[1] - 0.8).abs() < 1e-5);
        assert!(scores[2].abs() < 1e-5);
    }

    #[tokio::test]
    async fn dimension_mismatch_tolerated_per_token() {
        struct Mismatch;
        impl TokenLevelEmbeddings for Mismatch {
            async fn embed_tokens(
                &self,
                text: &str,
            ) -> Result<Vec<TokenEmbedding>, EmbeddingError> {
                if text.trim().is_empty() {
                    return Err(EmbeddingError::EmptyInput);
                }
                let mut out = Vec::new();
                let mut pos = 0usize;
                for tok in text.split_whitespace() {
                    let start = pos + text[pos..].find(tok).unwrap();
                    let end = start + tok.len();
                    pos = end;
                    // Query vs doc get different dimensions -> every cosine is a
                    // LengthMismatch, which must degrade to 0.0, not error out.
                    let dim = if text.starts_with("dog") { 2 } else { 3 };
                    out.push(TokenEmbedding {
                        span: TokenSpan::new(start, end),
                        vector: vec![1.0; dim],
                    });
                }
                Ok(out)
            }
        }
        let reranker = ColbertRerank::new(Mismatch);
        let docs = vec![Document::new("cat feed")];
        let scores = reranker.score_async("dog food", &docs).await.unwrap();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].is_finite(), "got {}", scores[0]);
    }
}
