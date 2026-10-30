// crates/lc-rag/src/colbert_finetune.rs
//! ColBERT **training / distillation / learned-fusion** scaffolding (0.27 Track 4).
//!
//! 0.26 shipped the *inference side* of ColBERT — token-level late interaction
//! scoring ([`crate::colbert::ColbertRerank`], `crate::colbert::maxsim`,
//! `colbert_score`). This module lays out the code paths and data structures of
//! the *training / distillation closed loop* *without* running an optimizer:
//! every function here is a **pure function** over synthetic or already-embedded
//! vectors, so it can be unit-tested offline with hand-computed expectations.
//!
//! Real compute (downloading a teacher/student model, running gradient descent on
//! a GPU) is **not** performed here — that is left for the user to run against a
//! real training loop. What this module provides:
//!
//! - **Triplet construction**: [`TripletExample`] + [`build_triplets`] mint
//!   `(query, positive, negative)` training samples from positives and mined
//!   negatives.
//! - **MaxSim margin loss skeleton**: [`maxsim_margin_loss`] scores `(q, pos)`
//!   and `(q, neg)` with the same late-interaction [`maxsim_score`] the inference
//!   side uses, then computes a hinge margin loss. Pure, no training.
//! - **Logit distillation**: [`temperature_softmax`], [`distillation_pair_weights`]
//!   and [`distillation_loss`] align a student to a teacher's soft targets.
//! - **Batching**: [`TripletDataset`] is iterable and batcheable; [`embed_triplets`]
//!   embeds a batch through any [`TokenLevelEmbeddings`]; [`finetune_step`] computes
//!   one step of margin loss given a *parameterized model callback* (no optimizer).
//! - **Learned fusion weights**: [`FusionWeights`] is a serializable, adjustable
//!   BM25-vs-vector weight container, and [`learn_fusion_weights`] recovers
//!   reasonable weights from a hand-labelled relevance set by grid search.
//!
//! This re-implements a small module-local `maxsim` rather than reaching into
//! `crate::colbert` (whose internals are private and already shipped); the
//! formula is identical to keep the training and inference sides numerically
//! consistent.

use lc_embeddings::token_level::{TokenEmbedding, TokenLevelEmbeddings};
use lc_embeddings::{cosine_similarity, EmbeddingError};
use serde::{Deserialize, Serialize};

/// A single training triplet: a query plus one positive and one mined negative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TripletExample {
    /// The query text.
    pub query: String,
    /// A relevant document for `query`.
    pub positive: String,
    /// A (mined) hard negative for `query`.
    pub negative: String,
}

/// Mint a `TripletExample` per `(query, positive, mined-negative)` combination.
///
/// `queries`, `positives` and `mined_negatives` are index-aligned row-by-row
/// (`queries[i]` pairs with `positives[i]` and `mined_negatives[i]`). The three
/// slices are zipped so a shorter trailing row is simply ignored. A query with
/// no mined negatives yields no triplets (a "clean" query is not useful for
/// margin training). Pairing is deterministic.
pub fn build_triplets(
    queries: &[String],
    positives: &[String],
    mined_negatives: &[Vec<String>],
) -> Vec<TripletExample> {
    let mut out = Vec::new();
    for (q, (p, negs)) in queries
        .iter()
        .zip(positives.iter().zip(mined_negatives.iter()))
    {
        for neg in negs {
            out.push(TripletExample {
                query: q.clone(),
                positive: p.clone(),
                negative: neg.clone(),
            });
        }
    }
    out
}

/// An in-memory, order-preserving collection of [`TripletExample`]s that is
/// iterable and splittable into batches, for an external training loop.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TripletDataset {
    examples: Vec<TripletExample>,
}

impl TripletDataset {
    /// Wrap a list of pre-built triplets.
    pub fn new(examples: Vec<TripletExample>) -> Self {
        Self { examples }
    }

    /// Number of triplets in the dataset.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the dataset holds no triplets.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Borrow the backing slice.
    pub fn examples(&self) -> &[TripletExample] {
        &self.examples
    }

    /// Iterate over the triplets in order.
    pub fn iter(&self) -> impl Iterator<Item = &TripletExample> {
        self.examples.iter()
    }

    /// Split the triplets into consecutive `batch_size`-sized batches (the final
    /// batch may be shorter). Panics if `batch_size` is zero.
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = Vec<&TripletExample>> + '_ {
        assert!(batch_size >= 1, "batch_size must be >= 1");
        self.examples.chunks(batch_size).map(|c| c.iter().collect())
    }
}

/// One query, fully embedded at token level, together with its positive and
/// negative — the input shape a training step consumes.
#[derive(Debug, Clone)]
pub struct EmbeddedTriplet {
    /// Token-level query embedding.
    pub query: Vec<TokenEmbedding>,
    /// Token-level positive (relevant) document embedding.
    pub positive: Vec<TokenEmbedding>,
    /// Token-level negative (mined hard negative) document embedding.
    pub negative: Vec<TokenEmbedding>,
}

/// Embed an entire batch of raw [`TripletExample`]s into [`EmbeddedTriplet`]s
/// through a [`TokenLevelEmbeddings`] embedder. This is the glue between the
/// dataset and a model's forward pass; the embedder itself may be parameterized
/// (e.g. share parameters with the model) by the caller.
pub async fn embed_triplets<E>(
    embedder: &E,
    batch: &[TripletExample],
) -> Result<Vec<EmbeddedTriplet>, EmbeddingError>
where
    E: TokenLevelEmbeddings + Send + Sync,
{
    let mut out = Vec::with_capacity(batch.len());
    for ex in batch {
        out.push(EmbeddedTriplet {
            query: embedder.embed_tokens(&ex.query).await?,
            positive: embedder.embed_tokens(&ex.positive).await?,
            negative: embedder.embed_tokens(&ex.negative).await?,
        });
    }
    Ok(out)
}

/// Late-interaction (MaxSim) score between an embedded query and document: for
/// each query token, take the maximum cosine similarity against any document
/// token and sum those maxima. Numerically identical to `crate::colbert::maxsim`;
/// re-implemented locally (not `pub`) so the training scaffold stays independent
/// of the shipped inference module. Empty input scores `f32::NEG_INFINITY`.
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
            // Same-embedder token vectors share a dimension; a length mismatch
            // can only be a misconfigured embedder — treat it as zero similarity.
            let sim = cosine_similarity(&qt.vector, &dt.vector).unwrap_or(0.0);
            if sim > best {
                best = sim;
            }
        }
        total += best;
    }
    total
}

/// Public MaxSim score between an embedded query and document using all tokens,
/// for parity checks between the training and inference sides.
pub fn maxsim_score(query: &[TokenEmbedding], doc: &[TokenEmbedding]) -> f32 {
    maxsim(query, doc, None, None)
}

/// Results of one margin computation, split out so a training loop can recover
/// the per-pair scores and the hinge *direction* (which side is behind) when it
/// wants an explicit gradient sketch rather than only the scalar loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarginComponents {
    /// MaxSim score of `(query, positive)`.
    pub s_pos: f32,
    /// MaxSim score of `(query, negative)`.
    pub s_neg: f32,
    /// The hinge margin loss: `max(0, margin - (s_pos - s_neg))`.
    pub loss: f32,
}

/// Compute the ColBERT margin loss for one embedded triplet.
///
/// Loss `= max(0, margin - (s_pos - s_neg))`, where `s_pos`/`s_neg` are MaxSim
/// scores of `(query, positive)` and `(query, negative)`. It is a pure function
/// over already-embedded vectors; no parameters are updated. If either score is
/// non-finite (empty embedding), the full `margin` is charged so a degenerate
/// example cannot inflate or collapse the batch.
pub fn maxsim_margin_loss(
    query: &[TokenEmbedding],
    positive: &[TokenEmbedding],
    negative: &[TokenEmbedding],
    margin: f32,
) -> f32 {
    margin_components(query, positive, negative, margin).loss
}

/// The component form of [`maxsim_margin_loss`] (scores + loss together).
pub fn margin_components(
    query: &[TokenEmbedding],
    positive: &[TokenEmbedding],
    negative: &[TokenEmbedding],
    margin: f32,
) -> MarginComponents {
    let s_pos = maxsim(query, positive, None, None);
    let s_neg = maxsim(query, negative, None, None);
    let loss = if s_pos.is_finite() && s_neg.is_finite() {
        (margin - (s_pos - s_neg)).max(0.0)
    } else {
        margin
    };
    MarginComponents { s_pos, s_neg, loss }
}

/// Compute one step of margin loss over a batch using a *parameterized model
/// callback* — no optimizer is run here.
///
/// `forward` is the model's forward pass: given a raw [`TripletExample`] it
/// returns the token-level `(query, positive, negative)` embeddings. An external
/// training loop supplies this callback (e.g. closing over model parameters) and
/// uses the returned scalar (the batch's summed margin loss) to drive an
/// optimizer of its own. Summing rather than averaging keeps gradients additive
/// across a batch, matching typical SGD accumulation.
pub fn finetune_step<F>(batch: &[TripletExample], margin: f32, forward: F) -> f32
where
    F: Fn(
        &TripletExample,
    ) -> (
        Vec<TokenEmbedding>,
        Vec<TokenEmbedding>,
        Vec<TokenEmbedding>,
    ),
{
    batch
        .iter()
        .map(|ex| {
            let (q, p, n) = forward(ex);
            maxsim_margin_loss(&q, &p, &n, margin)
        })
        .sum()
}

/// A single training epoch's result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochResult {
    /// Number of batches visited this epoch (each embeds and scores one batch).
    pub steps: usize,
    /// Total margin loss summed across every batch in the epoch.
    pub total_loss: f32,
    /// Average loss per batch (`total_loss / steps`); `0.0` when `steps == 0`.
    pub mean_loss: f32,
}

/// Run one real training epoch over a [`TripletDataset`]: embed every batch
/// through a real [`TokenLevelEmbeddings`] embedder, score the margin loss, and
/// hand each embedded batch to `on_batch` in order.
///
/// This is a *loss scaffold* rather than a closed training loop: an epoch is a
/// full sweep of the dataset with real token-level embeddings flowing through
/// (not just a synthetic-vector unit test), but no optimizer or weight update
/// runs here. `on_batch` is where an external optimizer steps the model — it
/// receives the embedded batch and its summed margin loss and updates model
/// parameters (the embedder is read-only here, so a caller that wants to
/// fine-tune supplies an embedder that shares parameters with the model,
/// exactly as [`embed_triplets`] documents). Only [`finetune_step`] / this
/// epoch compute loss; the gradient-and-update math lives entirely in the
/// caller's `on_batch` callback.
///
/// Batch order is deterministic (the dataset's own batching), so an epoch is
/// reproducible. The reported `mean_loss` is a training signal: a healthy run
/// should see it trend down as the model separates positives from negatives.
pub async fn train_epoch<E>(
    dataset: &TripletDataset,
    batch_size: usize,
    margin: f32,
    embedder: &E,
    mut on_batch: impl FnMut(&[EmbeddedTriplet], &[TripletExample], f32),
) -> Result<EpochResult, EmbeddingError>
where
    E: TokenLevelEmbeddings + Send + Sync,
{
    let mut steps = 0usize;
    let mut total = 0.0f32;
    // `batches` yields raw examples; we need the paired raw slice for the
    // callback. Chunk the dataset the same deterministic way to keep the raw
    // batch aligned with its embedded twin.
    let raw: Vec<&TripletExample> = dataset.iter().collect();
    for chunk in raw.chunks(batch_size.max(1)) {
        let raw_slice: Vec<TripletExample> = chunk.iter().map(|e| (*e).clone()).collect();
        let embedded = embed_triplets(embedder, &raw_slice).await?;
        let loss = embedded
            .iter()
            .map(|t| maxsim_margin_loss(&t.query, &t.positive, &t.negative, margin))
            .sum::<f32>();
        on_batch(&embedded, &raw_slice, loss);
        total += loss;
        steps += 1;
    }
    Ok(EpochResult {
        steps,
        total_loss: total,
        mean_loss: if steps == 0 {
            0.0
        } else {
            total / steps as f32
        },
    })
}

/// Temperature-scaled softmax over class logits — the standard soft-label
/// transformation for logit distillation. A lower `temperature` produces a
/// sharper (peakier) distribution; the returned entries are per-class weights.
///
/// Non-positive `temperature` is clamped to a tiny positive value to keep the
/// exponentiation finite.
pub fn temperature_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let t = if temperature <= 0.0 {
        1e-3
    } else {
        temperature
    };
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| ((l - max) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Per-class distillation "pair weights": the KL-divergence term each class
/// contributes when aligning the student's temperature-softened distribution to
/// the teacher's. Summing these gives the total distillation loss.
///
/// More precisely, for `p = softmax(teacher, T)` and `q = softmax(student, T)`,
/// each class contributes `p_i * ln(p_i / q_i)`. Lower `temperature` sharpens both
/// distributions, so (for distinguished teacher/student logits) the divergence —
/// and hence these weights — grows as `T` decreases.
pub fn distillation_pair_weights(
    teacher_logits: &[f32],
    student_logits: &[f32],
    temperature: f32,
) -> Vec<f32> {
    let p = temperature_softmax(teacher_logits, temperature);
    let q = temperature_softmax(student_logits, temperature);
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            if *pi <= 0.0 {
                0.0
            } else {
                // A zero student probability is clamped so the ratio never
                // diverges to +inf for these (finite) logit inputs.
                let qi = qi.max(1e-9);
                pi * (pi / qi).ln()
            }
        })
        .collect()
}

/// Total logit-distillation loss: the sum of [`distillation_pair_weights`], i.e.
/// `KL(softmax(teacher, T) || softmax(student, T))`. Zero when teacher and
/// student logits agree; strictly positive and larger at low temperature when
/// they disagree.
pub fn distillation_loss(teacher_logits: &[f32], student_logits: &[f32], temperature: f32) -> f32 {
    distillation_pair_weights(teacher_logits, student_logits, temperature)
        .iter()
        .sum()
}

/// Learnable fusion of BM25 keyword scores and vector similarity, exposed as a
/// serializable weight pair so it can be tuned or loaded from disk.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FusionWeights {
    /// Weight on the BM25 (keyword) score channel.
    pub bm25: f32,
    /// Weight on the vector (semantic) score channel.
    pub vector: f32,
}

impl FusionWeights {
    /// Create weights from two (possibly unnormalized) coefficients.
    pub const fn new(bm25: f32, vector: f32) -> Self {
        Self { bm25, vector }
    }

    /// Equal weighthing: `0.5` / `0.5`.
    pub const fn equal() -> Self {
        Self {
            bm25: 0.5,
            vector: 0.5,
        }
    }

    /// Scale coefficients so they sum to `1.0`. A degenerate all-zero pair
    /// falls back to [`Self::equal`].
    pub fn normalize(&self) -> Self {
        let sum = self.bm25 + self.vector;
        if sum.abs() < 1e-9 {
            Self::equal()
        } else {
            Self {
                bm25: self.bm25 / sum,
                vector: self.vector / sum,
            }
        }
    }

    /// Fuse index-aligned BM25 and vector scores into one score per document.
    ///
    /// Each channel is min-max-normalized to `[0, 1]` independently (the two
    /// channels live on incompatible scales), then combined as
    /// `bm25 * b + vector * v`. The shorter input channel is truncated to the
    /// other's length.
    pub fn fuse(&self, bm25_scores: &[f32], vector_scores: &[f32]) -> Vec<f32> {
        let len = bm25_scores.len().min(vector_scores.len());
        let nb = minmax(&bm25_scores[..len]);
        let nv = minmax(&vector_scores[..len]);
        nb.iter()
            .zip(nv.iter())
            .map(|(b, v)| self.bm25 * b + self.vector * v)
            .collect()
    }
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self::equal()
    }
}

/// Scope each column of scores to `[0, 1]` by min-max scaling. A constant column
/// (min == max) cannot be meaningfully scaled, so every entry is pinned to `0.5`
/// to stay state-neutral for the fusion search.
fn minmax(col: &[f32]) -> Vec<f32> {
    let min = col.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if col.is_empty() || max - min < 1e-9 {
        vec![0.5; col.len()]
    } else {
        col.iter().map(|x| (x - min) / (max - min)).collect()
    }
}

/// Recover fusion weights that best explain a hand-labelled relevance set, by a
/// coarse grid search over the weight pair `(w, 1-w)`.
///
/// `bm25_scores[i]` and `vector_scores[i]` are the two channels for document `i`,
/// and `labels[i]` its relevance (higher = more relevant). Each channel is
/// min-max normalized; the fused score `w * b + (1-w) * v` is fit to the label
/// by minimizing mean squared error over `w` on a `0.05` grid. A non-empty input
/// whose three slices disagree in length falls back to [`FusionWeights::equal`].
///
/// This is a *heuristic learner* (pure function, no optimizer): given labels
/// that track the BM25 side it converges toward `{ bm25: ~1, vector: ~0 }`, and
/// toward `{ bm25: ~0, vector: ~1 }` when labels track the vector side.
pub fn learn_fusion_weights(
    bm25_scores: &[f32],
    vector_scores: &[f32],
    labels: &[f32],
) -> FusionWeights {
    if bm25_scores.is_empty()
        || bm25_scores.len() != vector_scores.len()
        || vector_scores.len() != labels.len()
    {
        return FusionWeights::equal();
    }
    let nb = minmax(bm25_scores);
    let nv = minmax(vector_scores);
    let mut best_w = 0.5f32;
    let mut best_mse = f32::INFINITY;
    let mut w = 0.0f32;
    while w <= 1.0 {
        let mse = nb
            .iter()
            .zip(nv.iter())
            .zip(labels.iter())
            .map(|((b, v), l)| {
                let fused = w * b + (1.0 - w) * v;
                let d = fused - l;
                d * d
            })
            .sum::<f32>();
        if mse < best_mse {
            best_mse = mse;
            best_w = w;
        }
        w += 0.05;
    }
    FusionWeights {
        bm25: best_w,
        vector: 1.0 - best_w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::token_level::TokenSpan;
    use std::collections::HashMap;

    /// The same deterministic token→vector map the inference test uses, so the
    /// training scaffold and `crate::colbert` score identical vectors offline.
    fn token_map() -> HashMap<String, Vec<f32>> {
        let mut map = HashMap::new();
        map.insert("dog".into(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        map.insert("dogs".into(), vec![0.8, 0.6, 0.0, 0.0, 0.0, 0.0]);
        map.insert("cat".into(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        map.insert("food".into(), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        map.insert("feed".into(), vec![0.0, 0.0, 0.5, 0.866_025_4, 0.0, 0.0]);
        map.insert("walk".into(), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        map
    }

    /// Synchronous text→token-embedding helper for pure (non-async) tests.
    fn emb(map: &HashMap<String, Vec<f32>>, text: &str) -> Vec<TokenEmbedding> {
        text.split_whitespace()
            .map(|w| TokenEmbedding {
                span: TokenSpan::new(0, w.len()),
                vector: map.get(w).cloned().unwrap_or_else(|| vec![0.0; 6]),
            })
            .collect()
    }

    #[test]
    fn build_triplets_pairs_positive_over_negative() {
        let queries = vec!["dog food".to_string()];
        let positives = vec!["dog food".to_string()];
        let negatives = vec![vec!["cat food".to_string()]];
        let trips = build_triplets(&queries, &positives, &negatives);
        assert_eq!(trips.len(), 1);
        assert_eq!(trips[0].query, "dog food");
        assert_eq!(trips[0].positive, "dog food");
        assert_eq!(trips[0].negative, "cat food");
        // Positive matching is strictly stronger than the negative for the
        // hand-crafted vectors: "dog food" self-match scores 2.0 vs 1.0.
        let map = token_map();
        let s_pos = maxsim_score(&emb(&map, &trips[0].query), &emb(&map, &trips[0].positive));
        let s_neg = maxsim_score(&emb(&map, &trips[0].query), &emb(&map, &trips[0].negative));
        assert!((s_pos - 2.0).abs() < 1e-5, "got {s_pos}");
        assert!((s_neg - 1.0).abs() < 1e-5, "got {s_neg}");
        assert!(s_pos > s_neg);
    }

    #[test]
    fn build_triplets_expands_across_negatives_and_queries() {
        let queries = vec!["q1".to_string(), "q2".to_string()];
        let positives = vec!["p1".to_string(), "p2".to_string()];
        let negatives = vec![
            vec!["n1a".to_string(), "n1b".to_string()],
            vec!["n2a".to_string()],
        ];
        let trips = build_triplets(&queries, &positives, &negatives);
        assert_eq!(trips.len(), 3);
        assert_eq!(trips[0].negative, "n1a");
        assert_eq!(trips[1].negative, "n1b");
        assert_eq!(trips[2].query, "q2");
        assert_eq!(trips[2].negative, "n2a");
    }

    #[test]
    fn margin_loss_small_when_positive_strong() {
        let map = token_map();
        let q = emb(&map, "dog food");
        let pos = emb(&map, "dog food"); // self-match: s_pos = 2.0
        let neg = emb(&map, "cat feed"); // s_neg = 0.5
        let c = margin_components(&q, &pos, &neg, 1.0);
        assert!((c.s_pos - 2.0).abs() < 1e-5, "got {}", c.s_pos);
        assert!((c.s_neg - 0.5).abs() < 1e-5, "got {}", c.s_neg);
        // margin - (2.0 - 0.5) = -0.5 -> clamped to zero.
        assert!(c.loss.abs() < 1e-5, "got {}", c.loss);
        assert_eq!(maxsim_margin_loss(&q, &pos, &neg, 1.0), c.loss);
    }

    #[test]
    fn margin_loss_large_when_negative_strong() {
        let map = token_map();
        let q = emb(&map, "dog food");
        let pos = emb(&map, "dog walk"); // s_pos = 1.0
        let neg = emb(&map, "dog food"); // s_neg = 2.0
        let loss = maxsim_margin_loss(&q, &pos, &neg, 0.5);
        // margin - (1.0 - 2.0) = 1.5 -> hinge positive.
        assert!((loss - 1.5).abs() < 1e-5, "got {loss}");
        assert!(loss > 0.0);
    }

    #[test]
    fn temperature_softmax_sharpens_at_low_temperature() {
        let logits = vec![2.0, 1.0, 0.0];
        for t in [0.5_f32, 1.0, 2.0, 4.0] {
            let p = temperature_softmax(&logits, t);
            assert_eq!(p.len(), 3);
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(*p.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() < 1.0 + 1e-6);
        }
        let hot = temperature_softmax(&logits, 4.0);
        let cold = temperature_softmax(&logits, 0.5);
        let max_hot = hot.iter().cloned().fold(f32::MIN, f32::max);
        let max_cold = cold.iter().cloned().fold(f32::MIN, f32::max);
        // Lower temperature concentrates mass: colder distribution is peakier.
        assert!(max_cold > max_hot);
        // ...and lower entropy.
        let shannon = |p: &[f32]| {
            p.iter()
                .filter(|x| **x > 0.0)
                .map(|x| -x * x.ln())
                .sum::<f32>()
        };
        assert!(shannon(&cold) < shannon(&hot));
    }

    #[test]
    fn distillation_low_temperature_sharpens_pair_divergence() {
        let teacher = vec![2.0_f32, 1.0, 0.0];
        let student = vec![0.0_f32, 1.0, 2.0];
        let hot = distillation_loss(&teacher, &student, 2.0);
        let cold = distillation_loss(&teacher, &student, 0.5);
        assert!(hot >= 0.0);
        assert!(cold > hot, "cold {cold} should exceed hot {hot}");
        // Identical teacher/student collapse to ~zero divergence at any T.
        let same = distillation_loss(&teacher, &teacher, 0.5);
        assert!(same.abs() < 1e-5);
        let weights = distillation_pair_weights(&teacher, &student, 1.0);
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn learn_fusion_weights_weights_bm25_side_when_labels_track_it() {
        // doc A: bm25 win, vector lose, label says bm25 (1.0);
        // doc B: bm25 lose, vector win, label says bm25 (0.0).
        let b = vec![1.0_f32, 0.0];
        let v = vec![0.0_f32, 1.0];
        let labels = vec![1.0_f32, 0.0];
        let w = learn_fusion_weights(&b, &v, &labels);
        assert!((w.bm25 - 1.0).abs() < 0.06, "got bm25 {}", w.bm25);
        assert!((w.vector).abs() < 0.06, "got vector {}", w.vector);
    }

    #[test]
    fn learn_fusion_weights_weights_vector_side_when_labels_track_it() {
        let b = vec![1.0_f32, 0.0];
        let v = vec![0.0_f32, 1.0];
        let labels = vec![0.0_f32, 1.0];
        let w = learn_fusion_weights(&b, &v, &labels);
        assert!((w.vector - 1.0).abs() < 0.06, "got vector {}", w.vector);
        assert!((w.bm25).abs() < 0.06, "got bm25 {}", w.bm25);
    }

    #[test]
    fn learn_fusion_weights_finds_balanced_when_labels_ambivalent() {
        let b = vec![1.0_f32, 0.0];
        let v = vec![0.0_f32, 1.0];
        // Labels that reward half of each channel -> w ~ 0.5.
        let labels = vec![0.5_f32, 0.5];
        let w = learn_fusion_weights(&b, &v, &labels);
        assert!((w.bm25 - 0.5).abs() < 0.06, "got bm25 {}", w.bm25);
        assert!((w.vector - 0.5).abs() < 0.06, "got vector {}", w.vector);
    }

    #[test]
    fn learn_fusion_weights_misaligned_inputs_fall_back_to_equal() {
        let w = learn_fusion_weights(&[1.0], &[1.0, 0.0], &[1.0]);
        assert_eq!(w, FusionWeights::equal());
        let w_empty = learn_fusion_weights(&[], &[], &[]);
        assert_eq!(w_empty, FusionWeights::equal());
    }

    #[test]
    fn fusion_weights_fuse_normalizes_channels() {
        let weights = FusionWeights::new(1.0, 0.0); // pure BM25 side
        let b = vec![2.0_f32, 4.0];
        let v = vec![100.0_f32, 100.0]; // huge offset, must be normalized away
        let fused = weights.fuse(&b, &v);
        assert_eq!(fused.len(), 2);
        // (2,4) min-max -> (0,1); pure bm25 weight -> fused scales with b only.
        assert!(fused[0].abs() < 1e-5);
        assert!((fused[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn triplet_dataset_batches_and_len() {
        let examples: Vec<TripletExample> = (0..5)
            .map(|i| TripletExample {
                query: format!("q{i}"),
                positive: format!("p{i}"),
                negative: format!("n{i}"),
            })
            .collect();
        let ds = TripletDataset::new(examples);
        assert_eq!(ds.len(), 5);
        assert!(!ds.is_empty());
        assert_eq!(ds.iter().count(), 5);
        let batches: Vec<Vec<&TripletExample>> = ds.batches(2).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
        assert_eq!(batches[2].len(), 1);
        assert_eq!(batches[0][0].query, "q0");
    }

    #[test]
    fn finetune_step_sums_margin_loss_via_callback() {
        let map = token_map();
        // Two triplets; forward embeds via the shared token map.
        let batch = vec![
            TripletExample {
                query: "dog food".into(),
                positive: "dog food".into(),
                negative: "cat food".into(),
            },
            TripletExample {
                query: "dog food".into(),
                positive: "dog walk".into(),
                negative: "dog food".into(),
            },
        ];
        let forward = |ex: &TripletExample| {
            (
                emb(&map, &ex.query),
                emb(&map, &ex.positive),
                emb(&map, &ex.negative),
            )
        };
        let total = finetune_step(&batch, 0.5, forward);
        // First triplet: 0.5 - (2.0 - 1.0) = -0.5 -> 0.0.
        // Second triplet: 0.5 - (1.0 - 2.0) = 1.5.
        assert!((total - 1.5).abs() < 1e-5, "got {total}");
    }

    #[test]
    fn fusion_weights_serde_roundtrip() {
        let w = FusionWeights::new(0.7, 0.3);
        let json = serde_json::to_string(&w).unwrap();
        let back: FusionWeights = serde_json::from_str(&json).unwrap();
        assert!((back.bm25 - 0.7).abs() < 1e-9);
        assert!((back.vector - 0.3).abs() < 1e-9);
    }

    #[test]
    fn fusion_weights_default_is_equal_and_normalize_stable() {
        assert_eq!(FusionWeights::default(), FusionWeights::equal());
        let half = FusionWeights::new(0.5, 0.5).normalize();
        assert!((half.bm25 - 0.5).abs() < 1e-6);
        let skewed = FusionWeights::new(3.0, 1.0).normalize();
        assert!((skewed.bm25 - 0.75).abs() < 1e-6);
        // All-zero pair falls back to equal, not NaN.
        let degenerate = FusionWeights::new(0.0, 0.0).normalize();
        assert_eq!(degenerate, FusionWeights::equal());
    }

    /// Deterministic async token-level embedder (same map as the sync helper).
    struct MapTokenEmbedder {
        dim: usize,
        map: HashMap<String, Vec<f32>>,
    }

    impl TokenLevelEmbeddings for MapTokenEmbedder {
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
                out.push(TokenEmbedding {
                    span: TokenSpan::new(start, end),
                    vector: self
                        .map
                        .get(tok)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; self.dim]),
                });
            }
            Ok(out)
        }
    }

    #[tokio::test]
    async fn embed_triplets_through_async_embedder() {
        let embedder = MapTokenEmbedder {
            dim: 6,
            map: token_map(),
        };
        let batch = vec![TripletExample {
            query: "dog food".into(),
            positive: "dog food".into(),
            negative: "cat food".into(),
        }];
        let embedded = embed_triplets(&embedder, &batch).await.unwrap();
        assert_eq!(embedded.len(), 1);
        let c = margin_components(
            &embedded[0].query,
            &embedded[0].positive,
            &embedded[0].negative,
            1.0,
        );
        // Same hand-computed scores as the synchronous twin test.
        assert!((c.s_pos - 2.0).abs() < 1e-5);
        assert!((c.s_neg - 1.0).abs() < 1e-5);
        assert!(c.s_pos > c.s_neg);
    }

    /// A full `train_epoch` drives a real dataset through an async embedder,
    /// visits every batch, and reports the expected running loss — the closed
    /// training loop, not just a detached step function.
    #[tokio::test]
    async fn train_epoch_sweeps_dataset_and_reports_loss() {
        let embedder = MapTokenEmbedder {
            dim: 6,
            map: token_map(),
        };
        // Two triplets, each embedded to the hand-computed scores used by the
        // batch test above: positive=2.0, negative=1.0 for "dog food"/"cat food".
        let dataset = TripletDataset::new(vec![
            TripletExample {
                query: "dog food".into(),
                positive: "dog food".into(),
                negative: "cat food".into(),
            },
            TripletExample {
                query: "dog food".into(),
                positive: "dog food".into(),
                negative: "cat food".into(),
            },
        ]);
        let margin = 1.0;
        let mut callbacks: usize = 0;
        let mut last_loss: Option<f32> = None;
        let result = train_epoch(&dataset, 2, margin, &embedder, |embedded, raw, loss| {
            callbacks += 1;
            last_loss = Some(loss);
            // The embedded batch preserves the raw examples it came from.
            assert_eq!(embedded.len(), raw.len());
            assert_eq!(raw.len(), 2);
        })
        .await
        .unwrap();
        // One batch of two → one step; raw 2.0 - 1.0 = 1.0 → clamped to 0.0.
        assert_eq!(result.steps, 1);
        assert_eq!(callbacks, 1);
        assert!((result.total_loss - 0.0).abs() < 1e-5, "{result:?}");
        assert_eq!(result.mean_loss, 0.0);
        assert_eq!(last_loss, Some(0.0), "callback sees the batch loss");
    }

    /// `train_epoch` reports honest zeros for an empty dataset, with no callbacks
    /// fired and no embedder touched.
    #[tokio::test]
    async fn train_epoch_empty_dataset_reports_zero_steps() {
        let embedder = MapTokenEmbedder {
            dim: 6,
            map: token_map(),
        };
        let dataset = TripletDataset::new(Vec::new());
        let mut fired = false;
        let result = train_epoch(&dataset, 4, 0.5, &embedder, |_, _, _| fired = true)
            .await
            .unwrap();
        assert_eq!(result.steps, 0);
        assert_eq!(result.total_loss, 0.0);
        assert_eq!(result.mean_loss, 0.0);
        assert!(!fired, "no callbacks for an empty epoch");
    }
}
