// lc-rag/src/agentic_retrieval.rs
//! Iterative (agentic) retrieval — "query → judge → re-query"
//!
//! A single vector-store pass often cannot satisfy an ambiguous, multi-hop or
//! under-specified query in one shot. `IterativeRetriever` turns retrieval into a
//! small agentic loop: each round issues a query against an inner
//! [`RetrieverTrait`], **judges** whether the top hit is confident enough (score ≥
//! `min_score`), and — when it is not — **re-queries** with a refined query built
//! from the best hits seen so far (their title / first paragraph folded into the
//! query text as a retrieval hint). This is the "查→判→再查" design:
//!
//! - **查 (query)**: `retrieve_with_scores` per round, top `top_k` hits.
//! - **判 (judge)**: if any hit scores ≥ `min_score` the loop converges early; a
//!   confidence threshold avoids re-querying on already-good answers.
//! - **再查 (re-query)**: otherwise fold the top hit's snippet into the query via
//!   [`refine_query`] and run again, for at most `max_rounds` rounds.
//!
//! Results are merged across rounds and de-duplicated (keeping the best score per
//! document), so later rounds never throw away an early high-value hit. The loop
//! is fully off-network and deterministic (the refinement is a pure function), so
//! it composes cleanly as an agent tool and is easy to test.
//!
//! # Design trade-offs
//! - **Score-threshold convergence, not hard round counts**: `min_score` lets
//!   confident single-pass results short-circuit (the common case — cheap), while
//!   `max_rounds` bounds the worst case (no infinite loops).
//! - **Query-level refinement, not passage-level, per round**: re-issuing against
//!   the inner retriever keeps the loop retriever-agnostic (BM25 / vector / hybrid
//!   all work), at the cost of re-embedding per round for vector stores.
//! - **Return the accumulated best batch, not just the last round**: gives the
//!   caller a lossless superset to re-rank or pass to generation.

use lc_vector_stores::{Document, SearchResult};
use std::sync::Arc;

use crate::retriever::{RetrieverError, RetrieverTrait};

/// Outcome of an iterative retrieval: the de-duplicated best batch plus how many
/// rounds the loop actually ran (≥ 1). `rounds` lets a caller / test assert that
/// the "query → re-query" path really happened.
#[derive(Debug, Clone)]
pub struct IterativeRetrieval {
    /// De-duplicated best documents across all rounds, sorted by score descending.
    pub documents: Vec<Document>,
    /// Number of retrieval rounds executed (≤ `max_rounds`).
    pub rounds: usize,
}

/// Iterative, multi-round retriever wrapping any inner [`RetrieverTrait`].
///
/// Configuration is immutable and set at construction (`max_rounds`, `top_k`,
/// `min_score`); every round re-queries the inner retriever and refines the query
/// until a hit clears `min_score` or `max_rounds` is reached.
pub struct IterativeRetriever {
    inner: Arc<dyn RetrieverTrait>,
    /// Maximum retrieval rounds before giving up on refinement.
    max_rounds: usize,
    /// Documents fetched from the inner retriever per round.
    top_k: usize,
    /// Confidence threshold: a round with any hit scoring ≥ this converges early.
    min_score: f32,
}

impl IterativeRetriever {
    /// Creates an iterative retriever over `inner` with defaults
    /// (`max_rounds = 2`, `top_k = 4`, `min_score = 0.5`).
    pub fn new(inner: Arc<dyn RetrieverTrait>) -> Self {
        Self {
            inner,
            max_rounds: 2,
            top_k: 4,
            min_score: 0.5,
        }
    }

    /// Copies this retriever with a new round cap (leaves the original untouched).
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds.max(1);
        self
    }

    /// Copies this retriever with a new per-round fetch count.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    /// Copies this retriever with a new confidence threshold.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Runs the iterative retrieval loop.
    ///
    /// Returns the de-duplicated best `k` documents across all rounds and the
    /// number of rounds run. Convergence is early-exited when any accumulated hit
    /// scores ≥ `min_score`; otherwise the top hit is folded into the query and
    /// the loop re-runs, up to `max_rounds`.
    pub async fn retrieve(
        &self,
        query: &str,
        k: usize,
    ) -> Result<IterativeRetrieval, RetrieverError> {
        let mut round_query = query.trim().to_string();
        let mut accumulated: Vec<SearchResult> = Vec::new();
        let mut rounds = 0usize;

        for _ in 0..self.max_rounds {
            rounds += 1;
            let results = self.inner.retrieve_with_scores(&round_query, self.top_k).await?;
            merge_dedup(&mut accumulated, results);

            let best = accumulated.first();
            match best {
                // Judge: a confident hit converges immediately.
                Some(top) if top.score >= self.min_score => break,
                // No hits at all: nothing to refine against; stop.
                None => break,
                // Low confidence: fold the top hit into the query and re-query.
                Some(top) => {
                    round_query = refine_query(&round_query, std::slice::from_ref(&top.document));
                }
            }
        }

        let mut documents: Vec<Document> =
            accumulated.into_iter().map(|r| r.document).collect();
        documents.truncate(k);
        Ok(IterativeRetrieval { documents, rounds })
    }
}

/// Pure, deterministic query refinement: original query + the top hit's title /
/// first paragraph, joined into a richer query for the next round.
///
/// Determinism matters here: the refinement is a pure function of `(orig, hits)`,
/// so the same inputs always produce the same refined query (and the loop is
/// reproducible / unit-testable without a network).
pub fn refine_query(orig: &str, hits: &[Document]) -> String {
    let mut parts = vec![orig.trim().to_string()];
    if let Some(first) = hits.first() {
        let snippet = first_paragraph(&first.content);
        if !snippet.is_empty() {
            parts.push(snippet);
        }
    }
    parts.join(" ")
}

/// Extracts the first non-empty paragraph of a document's text (the "title /
/// snippet" hint fed back into refinement), trimmed and length-capped so the
/// query stays bounded.
fn first_paragraph(content: &str) -> String {
    for para in content.split_terminator(['\n', '\r']) {
        let trimmed = para.trim();
        if !trimmed.is_empty() {
            return trimmed.chars().take(200).collect();
        }
    }
    String::new()
}

/// Merges `incoming` results into `acc`, de-duplicating by document id (falling
/// back to content) and keeping the highest-scoring copy; `acc` is then sorted by
/// score descending. Immutable to the inputs beyond `acc`'s insertion.
fn merge_dedup(acc: &mut Vec<SearchResult>, incoming: Vec<SearchResult>) {
    for result in incoming {
        let key = doc_key(&result.document);
        let duplicate = acc.iter().position(|existing| doc_key(&existing.document) == key);
        match duplicate {
            Some(pos) => {
                if result.score > acc[pos].score {
                    acc[pos] = result;
                }
            }
            None => acc.push(result),
        }
    }
    acc.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Identity key used for de-duplication: document id when present, else its text.
fn doc_key(document: &Document) -> String {
    document
        .id
        .clone()
        .unwrap_or_else(|| document.content.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Handles one `retrieve_with_scores` invocation: maps `(query, k)` to hits.
    type Handler = Box<dyn Fn(&str, usize) -> Vec<SearchResult> + Send + Sync>;

    /// Scripted in-memory retriever: each `retrieve_with_scores` call runs
    /// `handler(query, k)` to decide the hits, so tests can program scores and
    /// branch on the refined query to prove "re-query" happened.
    struct FakeRetriever {
        handler: Handler,
    }

    impl FakeRetriever {
        fn new(handler: Handler) -> Self {
            Self { handler }
        }
    }

    #[async_trait]
    impl RetrieverTrait for FakeRetriever {
        async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
            Ok((self.handler)(query, k).into_iter().map(|r| r.document).collect())
        }

        async fn retrieve_with_scores(
            &self,
            query: &str,
            k: usize,
        ) -> Result<Vec<SearchResult>, RetrieverError> {
            Ok((self.handler)(query, k))
        }

        async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
            Ok(())
        }
    }

    fn doc(content: &str, id: &str) -> Document {
        Document::new(content).with_id(id)
    }

    fn hit(content: &str, id: &str, score: f32) -> SearchResult {
        SearchResult {
            document: doc(content, id),
            score,
        }
    }

    /// High-confidence single-pass hit → converges in exactly 1 round.
    #[tokio::test]
    async fn short_query_converges_in_one_round() {
        let retriever = IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(
            |_query, _k| vec![hit("Rust is systems programming", "r1", 0.9)],
        ))));
        let out = retriever.retrieve("what is rust", 4).await.unwrap();
        assert_eq!(out.rounds, 1, "confident hit should converge immediately");
        assert_eq!(out.documents.len(), 1);
        assert_eq!(out.documents[0].id.as_deref(), Some("r1"));
    }

    /// Round 1 scores too low (below `min_score`), refinement re-queries and the
    /// refined query hits a high-score doc in round 2.
    #[tokio::test]
    async fn low_score_triggers_requery_and_refined_hit() {
        let retriever = IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(
            |query, _k| {
                if query.contains("new development clone") {
                    // Refined query (original + round-1 snippet) finds the truly relevant doc.
                    vec![hit("Functional ownership summary", "f1", 0.95)]
                } else {
                    // First query returns a low-confidence doc.
                    vec![hit("The new development clone", "d1", 0.1)]
                }
            },
        ))));
        let out = retriever
            .retrieve("clone speeds up iteration", 4)
            .await
            .unwrap();
        assert_eq!(out.rounds, 2, "low score must trigger a re-query round");
        // High-score doc from the refined round is present; low-score round-1 doc is retained too.
        assert_eq!(out.documents[0].id.as_deref(), Some("f1"));
        assert!(
            out.documents.iter().any(|d| d.id.as_deref() == Some("f1")),
            "refined round's high-score hit must be returned"
        );
    }

    /// Low scores every round → the loop runs exactly `max_rounds` (2) then stops.
    #[tokio::test]
    async fn rounds_are_capped_at_max() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let retriever = IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(move |_, _| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            vec![hit("always low", "low", 0.05)]
        }))));
        let out = retriever.retrieve("query", 4).await.unwrap();
        assert_eq!(out.rounds, 2, "must not exceed max_rounds");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// The same document appearing across rounds is de-duplicated (best score kept).
    #[tokio::test]
    async fn hits_are_deduplicated_across_rounds() {
        let retriever = IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(
            |_query, _k| vec![hit("same doc content", "dup", 0.3)],
        ))));
        let out = retriever.retrieve("query", 4).await.unwrap();
        assert_eq!(out.rounds, 2, "low score keeps looping");
        assert_eq!(
            out.documents.len(),
            1,
            "duplicate doc across rounds must be merged"
        );
        assert_eq!(out.documents[0].id.as_deref(), Some("dup"));
    }

    /// No hits at all → empty result, stops after one round.
    #[tokio::test]
    async fn no_hits_returns_empty() {
        let retriever = IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(
            |_query, _k| Vec::new(),
        ))));
        let out = retriever.retrieve("query", 4).await.unwrap();
        assert!(out.documents.is_empty());
        assert_eq!(out.rounds, 1, "empty round must stop, not loop forever");
    }

    /// `refine_query` is deterministic and folds the original query + first hit
    /// snippet — the dependency the whole refinement test above relies on.
    #[test]
    fn refine_query_is_deterministic_and_includes_original_and_top_hit() {
        let hits = vec![doc("Iterate-first control flow", "h1")];
        let once = refine_query("what does rust borrow", &hits);
        let twice = refine_query("what does rust borrow", &hits);
        assert_eq!(once, twice, "refinement must be deterministic");
        assert!(
            once.contains("what does rust borrow"),
            "refined query must retain the original query"
        );
        assert!(
            once.contains("Iterate-first"),
            "refined query must fold in the first hit's title/snippet"
        );
    }

    /// Refinement with no hits returns just the trimmed original (defensive).
    #[test]
    fn refine_query_without_hits_keeps_original() {
        assert_eq!(refine_query("  lone query  ", &[]), "lone query");
    }
}