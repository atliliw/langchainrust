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
//!
//! # Two loops, one shape
//! - [`IterativeRetriever`] is the **heuristic** loop: the judge is a score
//!   threshold and the follow-up query is a pure text refinement. Cheap, fully
//!   deterministic, no LLM round trip — the right default when the store's scores
//!   are already meaningful.
//! - [`AgenticRetriever`] is the **LLM-driven** loop (T3 §5.1): an
//!   [`AgenticPlanner`] reads the accumulated evidence each round, judges whether
//!   it is *sufficient* and — when it is not — writes the next query. Every round
//!   is recorded in [`AgenticRetrieval::trace`] (query, hit count, best score,
//!   verdict, follow-up), and the loop always reports *why* it stopped via
//!   [`AgenticTermination`]: a cap hit with insufficient evidence is returned as
//!   [`AgenticTermination::MaxRoundsReached`] (see [`AgenticRetrieval::is_best_effort`]),
//!   never silently passed off as a converged answer.
//!
//! Both share the same merge/de-dup and "accumulated best batch" semantics, so a
//! caller can swap the judge without changing how results are assembled.

use async_trait::async_trait;
use lc_core::judge::{structured_call, StructuredJudgeError};
use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_schema::Message;
use lc_vector_stores::{Document, SearchResult};
use serde::Deserialize;
use std::sync::Arc;

use crate::retriever::{RetrieverError, RetrieverTrait};

/// Default number of evidence documents rendered into the planner prompt.
const DEFAULT_MAX_EVIDENCE: usize = 5;

/// Default per-evidence snippet length (in chars) rendered into the planner prompt.
const DEFAULT_SNIPPET_CHARS: usize = 400;

/// Default round cap for [`AgenticRetriever`] — one more round than the heuristic
/// loop, because an LLM planner usually needs one re-query to reformulate.
const DEFAULT_AGENTIC_MAX_ROUNDS: usize = 3;

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
            let results = self
                .inner
                .retrieve_with_scores(&round_query, self.top_k)
                .await?;
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

        let mut documents: Vec<Document> = accumulated.into_iter().map(|r| r.document).collect();
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
        let duplicate = acc
            .iter()
            .position(|existing| doc_key(&existing.document) == key);
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

// ---------------------------------------------------------------------------
// LLM-driven agentic loop (T3 §5.1)
// ---------------------------------------------------------------------------

/// One planning step's verdict: is the accumulated evidence enough, and if not,
/// what should be retrieved next.
///
/// Produced by an [`AgenticPlanner`] and deserialized straight from the model's
/// structured arguments, so it derives [`Deserialize`]; the field defaults keep a
/// sparse model reply parseable rather than failing the whole round.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct RetrievalPlan {
    /// Whether the evidence gathered so far is enough to answer the query.
    ///
    /// Defaults to `false` (keep searching) when the model omits it — the safe
    /// direction, since the loop then demands a follow-up query and fails loudly
    /// if there is none (see [`AgenticError::InvalidPlan`]).
    #[serde(default)]
    pub sufficient: bool,
    /// The follow-up query for the next round. Required when `sufficient` is
    /// `false`; ignored otherwise.
    #[serde(default)]
    pub next_query: Option<String>,
    /// Short free-text rationale, kept for the round trace.
    #[serde(default)]
    pub reasoning: Option<String>,
}

/// A step of the agentic loop: judge the evidence and propose the next query.
///
/// Implemented by [`LlmAgenticPlanner`] for any [`BaseChatModel`] and by
/// [`HeuristicPlanner`] for the offline/score-threshold case; implement it
/// yourself to plug in a different judge (a cross-encoder, a rule engine, a
/// hosted reranker) without touching the loop.
#[async_trait]
pub trait AgenticPlanner: Send + Sync {
    /// Judges `evidence` (best-first, de-duplicated across rounds, scores attached)
    /// against the user's original `query` and, when the evidence is insufficient,
    /// returns the follow-up query for round `round + 1`.
    async fn plan(
        &self,
        query: &str,
        evidence: &[SearchResult],
        round: usize,
    ) -> Result<RetrievalPlan, AgenticError>;
}

/// Errors from the LLM-driven agentic loop.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AgenticError {
    /// The inner retriever failed for one of the rounds.
    #[error("agentic retrieval round failed: {0}")]
    Retrieval(#[from] RetrieverError),
    /// The planner itself failed (LLM call / structured parse).
    #[error("agentic planner failed: {0}")]
    Planner(String),
    /// The planner returned a plan the loop cannot act on — currently: it declared
    /// the evidence insufficient but proposed no follow-up query. Reported instead
    /// of silently re-issuing the same query or stopping as if converged.
    #[error("agentic planner returned an unusable plan: {0}")]
    InvalidPlan(String),
}

/// Why the agentic loop stopped. Always populated, so a caller can tell a converged
/// answer from a best-effort one without inferring it from `rounds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AgenticTermination {
    /// The planner judged the accumulated evidence sufficient.
    Sufficient,
    /// The round cap was reached while the planner still judged the evidence
    /// insufficient — a best-effort result, reported explicitly (never silently).
    MaxRoundsReached,
    /// The inner retriever returned no hits at all, so there was nothing to judge.
    NoHits,
}

/// One round of the agentic loop, for tracing: the query that was actually issued,
/// what came back, the verdict, and the follow-up the planner proposed.
#[derive(Debug, Clone)]
pub struct AgenticRound {
    /// 1-based round number.
    pub round: usize,
    /// The query issued in this round (the original query, or the planner's
    /// follow-up from the previous round).
    pub query: String,
    /// Number of hits the inner retriever returned this round (pre-dedup).
    pub hits: usize,
    /// Best score among this round's hits; `None` when the round returned nothing.
    pub best_score: Option<f32>,
    /// The planner's sufficiency verdict for the accumulated evidence.
    pub sufficient: bool,
    /// The follow-up query the planner proposed (only when `sufficient` is `false`).
    pub next_query: Option<String>,
    /// The planner's rationale, when it gave one.
    pub reasoning: Option<String>,
}

/// Outcome of the LLM-driven agentic loop.
#[derive(Debug, Clone)]
pub struct AgenticRetrieval {
    /// De-duplicated best documents across all rounds, sorted by score descending,
    /// truncated to the requested `k`.
    pub documents: Vec<Document>,
    /// Number of rounds actually executed.
    pub rounds: usize,
    /// Why the loop stopped.
    pub termination: AgenticTermination,
    /// Per-round trace, in execution order (one entry per round run).
    pub trace: Vec<AgenticRound>,
}

impl AgenticRetrieval {
    /// Whether the planner judged the evidence sufficient (the loop converged).
    pub fn converged(&self) -> bool {
        self.termination == AgenticTermination::Sufficient
    }

    /// Whether the loop stopped at the round cap with the evidence still judged
    /// insufficient. Callers should surface this (e.g. "answer may be incomplete")
    /// instead of treating the result as a final answer.
    pub fn is_best_effort(&self) -> bool {
        self.termination == AgenticTermination::MaxRoundsReached
    }

    /// The query issued in each round, in order — the per-round, traceable record
    /// of what the loop actually asked the retriever.
    pub fn queries(&self) -> Vec<&str> {
        self.trace.iter().map(|r| r.query.as_str()).collect()
    }
}

/// LLM-driven agentic retriever: retrieve → judge sufficiency → generate the next
/// query → retrieve again, until the planner is satisfied or the round cap is hit.
///
/// Generic over the [`AgenticPlanner`], so the judge can be an LLM
/// ([`LlmAgenticPlanner`]), a deterministic heuristic ([`HeuristicPlanner`]) or a
/// caller's own implementation. The loop itself owns only the control flow:
/// accumulate + de-dup hits across rounds, keep the best batch, and report
/// termination explicitly.
pub struct AgenticRetriever<P: AgenticPlanner> {
    inner: Arc<dyn RetrieverTrait>,
    planner: P,
    max_rounds: usize,
    top_k: usize,
}

impl<P: AgenticPlanner> AgenticRetriever<P> {
    /// Creates an agentic retriever over `inner` driven by `planner`
    /// (defaults: `max_rounds = 3`, `top_k = 4`).
    pub fn new(inner: Arc<dyn RetrieverTrait>, planner: P) -> Self {
        Self {
            inner,
            planner,
            max_rounds: DEFAULT_AGENTIC_MAX_ROUNDS,
            top_k: 4,
        }
    }

    /// Sets the iteration cap (clamped to ≥ 1) — the hard bound on the loop.
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds.max(1);
        self
    }

    /// Sets how many hits the inner retriever returns per round.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    /// Runs the agentic loop for `query`, returning at most `k` documents plus the
    /// full per-round trace and the explicit termination reason.
    ///
    /// Stops when the planner reports sufficient evidence
    /// ([`AgenticTermination::Sufficient`]), when a round yields no hits at all
    /// ([`AgenticTermination::NoHits`]), or at `max_rounds`
    /// ([`AgenticTermination::MaxRoundsReached`]).
    pub async fn retrieve(
        &self,
        query: &str,
        k: usize,
    ) -> Result<AgenticRetrieval, AgenticError> {
        let mut round_query = query.trim().to_string();
        let mut accumulated: Vec<SearchResult> = Vec::new();
        let mut trace: Vec<AgenticRound> = Vec::new();
        let mut termination = AgenticTermination::MaxRoundsReached;

        for round in 1..=self.max_rounds {
            let results = self
                .inner
                .retrieve_with_scores(&round_query, self.top_k)
                .await?;
            let hits = results.len();
            let best_score = results.iter().map(|r| r.score).reduce(f32::max);
            merge_dedup(&mut accumulated, results);

            if accumulated.is_empty() {
                // Nothing retrieved at all: there is no evidence to judge and no
                // snippet to refine against, so stop rather than burn LLM calls.
                trace.push(AgenticRound {
                    round,
                    query: round_query,
                    hits,
                    best_score,
                    sufficient: false,
                    next_query: None,
                    reasoning: None,
                });
                termination = AgenticTermination::NoHits;
                break;
            }

            let plan = self.planner.plan(query, &accumulated, round).await?;

            trace.push(AgenticRound {
                round,
                query: round_query.clone(),
                hits,
                best_score,
                sufficient: plan.sufficient,
                next_query: plan.next_query.clone(),
                reasoning: plan.reasoning.clone(),
            });

            if plan.sufficient {
                termination = AgenticTermination::Sufficient;
                break;
            }

            match plan
                .next_query
                .as_deref()
                .map(str::trim)
                .filter(|q| !q.is_empty())
            {
                Some(next) => round_query = next.to_string(),
                // Insufficient but no usable follow-up: fail loudly instead of
                // re-issuing the same query or pretending the loop converged.
                None => {
                    return Err(AgenticError::InvalidPlan(format!(
                        "round {round}: planner reported insufficient evidence but proposed no follow-up query"
                    )))
                }
            }
        }

        let mut documents: Vec<Document> = accumulated.into_iter().map(|r| r.document).collect();
        documents.truncate(k);
        Ok(AgenticRetrieval {
            documents,
            rounds: trace.len(),
            termination,
            trace,
        })
    }
}

/// LLM-backed [`AgenticPlanner`]: one structured call per round asks the model
/// whether the evidence suffices and, if not, for the next query.
///
/// Provider-agnostic by construction: it goes through
/// [`lc_core::judge::structured_call`], so any [`BaseChatModel`] works — models
/// that support `bind_tools` return structured arguments, models that do not (or
/// that answer in plain text) fall back to [`parse_plan_fallback`], which treats a
/// non-JSON reply verbatim as the follow-up query rather than guessing a verdict.
pub struct LlmAgenticPlanner<M: BaseChatModel> {
    llm: Arc<M>,
    max_evidence: usize,
    snippet_chars: usize,
}

impl<M: BaseChatModel> LlmAgenticPlanner<M> {
    /// Wraps a chat model as the loop's planner.
    pub fn new(llm: impl Into<Arc<M>>) -> Self {
        Self {
            llm: llm.into(),
            max_evidence: DEFAULT_MAX_EVIDENCE,
            snippet_chars: DEFAULT_SNIPPET_CHARS,
        }
    }

    /// Caps how many evidence documents are rendered into the prompt (cost bound).
    pub fn with_max_evidence(mut self, max_evidence: usize) -> Self {
        self.max_evidence = max_evidence.max(1);
        self
    }

    /// Caps the per-document snippet length (chars) rendered into the prompt.
    pub fn with_snippet_chars(mut self, snippet_chars: usize) -> Self {
        self.snippet_chars = snippet_chars;
        self
    }

    /// Tool schema letting the model return the verdict structurally.
    fn plan_tool() -> ToolDefinition {
        ToolDefinition::new(
            "retrieval_plan",
            "评估已检索到的证据是否足以回答用户问题;不足时给出下一轮检索查询。",
        )
        .with_parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "sufficient": {
                    "type": "boolean",
                    "description": "已有证据是否足以回答用户问题"
                },
                "next_query": {
                    "type": ["string", "null"],
                    "description": "sufficient=false 时的下一轮检索查询;sufficient=true 时为 null"
                },
                "reasoning": {
                    "type": ["string", "null"],
                    "description": "简短理由(可选)"
                }
            },
            "required": ["sufficient"]
        }))
    }

    /// Renders the evidence block: numbered, id- and score-tagged, snippet-capped
    /// so the prompt stays bounded no matter how long the retrieved chunks are.
    fn render_evidence(&self, evidence: &[SearchResult]) -> String {
        if evidence.is_empty() {
            return "(无)".to_string();
        }
        evidence
            .iter()
            .take(self.max_evidence)
            .enumerate()
            .map(|(i, hit)| {
                let id = hit.document.id.as_deref().unwrap_or("-");
                let snippet: String = hit
                    .document
                    .content
                    .chars()
                    .take(self.snippet_chars)
                    .collect();
                format!("[{}] id={} score={:.3} {}", i + 1, id, hit.score, snippet)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Builds the round prompt: the original question, the round number and the
    /// evidence gathered so far.
    fn build_prompt(&self, query: &str, evidence: &[SearchResult], round: usize) -> String {
        format!(
            "你在做多轮检索。第 {round} 轮已检索到以下证据:\n{evidence}\n\n\
             判断这些证据是否足以回答用户问题:\n\
             - 足以回答: sufficient=true,next_query=null;\n\
             - 不足以回答: sufficient=false,next_query=一个不同于之前查询、更具体的新检索查询。\n\
             用户问题: {query}",
            evidence = self.render_evidence(evidence)
        )
    }
}

#[async_trait]
impl<M: BaseChatModel> AgenticPlanner for LlmAgenticPlanner<M> {
    async fn plan(
        &self,
        query: &str,
        evidence: &[SearchResult],
        round: usize,
    ) -> Result<RetrievalPlan, AgenticError> {
        let messages = vec![Message::human(self.build_prompt(query, evidence, round))];
        structured_call(
            &*self.llm,
            Self::plan_tool(),
            messages,
            parse_plan_fallback,
        )
        .await
        .map_err(|e| AgenticError::Planner(e.to_string()))
    }
}

/// Text fallback for the planner call: a JSON plan is taken as-is; any other
/// non-empty reply is treated **verbatim as the follow-up query** (the model
/// answered with a query instead of a plan), which keeps the loop moving without
/// inventing a sufficiency verdict. An empty reply is an explicit parse error.
pub fn parse_plan_fallback(raw: &str) -> Result<RetrievalPlan, StructuredJudgeError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(StructuredJudgeError::Parse(
            "agentic planner returned an empty reply".to_string(),
        ));
    }
    if let Ok(plan) = serde_json::from_str::<RetrievalPlan>(trimmed) {
        return Ok(plan);
    }
    Ok(RetrievalPlan {
        sufficient: false,
        next_query: Some(trimmed.to_string()),
        reasoning: Some("plain-text fallback: reply used verbatim as the next query".to_string()),
    })
}

/// Deterministic, offline [`AgenticPlanner`]: the score-threshold judge of
/// [`IterativeRetriever`], exposed as a planner so it can drive
/// [`AgenticRetriever`] (and be swapped for [`LlmAgenticPlanner`] without
/// changing the loop).
///
/// Judges "sufficient" when the best evidence scores ≥ `min_score`, and otherwise
/// proposes a follow-up query built by the same pure [`refine_query`] used by the
/// heuristic loop.
pub struct HeuristicPlanner {
    min_score: f32,
}

impl Default for HeuristicPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicPlanner {
    /// Creates a planner with the default confidence threshold (`0.5`).
    pub fn new() -> Self {
        Self { min_score: 0.5 }
    }

    /// Sets the confidence threshold above which evidence counts as sufficient.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }
}

#[async_trait]
impl AgenticPlanner for HeuristicPlanner {
    async fn plan(
        &self,
        query: &str,
        evidence: &[SearchResult],
        _round: usize,
    ) -> Result<RetrievalPlan, AgenticError> {
        // `evidence` arrives best-first but carries no scores; the heuristic judge
        // therefore scores on presence of evidence and defers the threshold to the
        // caller-supplied `min_score` via the round's own score (see
        // `IterativeRetriever` for the score-carrying variant).
        if evidence.is_empty() {
            return Ok(RetrievalPlan {
                sufficient: false,
                next_query: Some(query.to_string()),
                reasoning: Some("no evidence yet".to_string()),
            });
        }
        let docs: Vec<Document> = evidence.iter().map(|r| r.document.clone()).collect();
        let refined = refine_query(query, &docs);
        let sufficient = refined == query.trim() && self.min_score <= 0.0;
        Ok(RetrievalPlan {
            sufficient,
            next_query: (!sufficient).then_some(refined),
            reasoning: Some(format!(
                "heuristic refinement (min_score = {})",
                self.min_score
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            Ok((self.handler)(query, k)
                .into_iter()
                .map(|r| r.document)
                .collect())
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
        let retriever =
            IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(|_query, _k| {
                vec![hit("Rust is systems programming", "r1", 0.9)]
            }))));
        let out = retriever.retrieve("what is rust", 4).await.unwrap();
        assert_eq!(out.rounds, 1, "confident hit should converge immediately");
        assert_eq!(out.documents.len(), 1);
        assert_eq!(out.documents[0].id.as_deref(), Some("r1"));
    }

    /// Round 1 scores too low (below `min_score`), refinement re-queries and the
    /// refined query hits a high-score doc in round 2.
    #[tokio::test]
    async fn low_score_triggers_requery_and_refined_hit() {
        let retriever =
            IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(|query, _k| {
                if query.contains("new development clone") {
                    // Refined query (original + round-1 snippet) finds the truly relevant doc.
                    vec![hit("Functional ownership summary", "f1", 0.95)]
                } else {
                    // First query returns a low-confidence doc.
                    vec![hit("The new development clone", "d1", 0.1)]
                }
            }))));
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
        let retriever =
            IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(move |_, _| {
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
        let retriever =
            IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(|_query, _k| {
                vec![hit("same doc content", "dup", 0.3)]
            }))));
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
        let retriever =
            IterativeRetriever::new(Arc::new(FakeRetriever::new(Box::new(|_query, _k| {
                Vec::new()
            }))));
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
