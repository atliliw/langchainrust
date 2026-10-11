// lc-rag/src/context_editing.rs
//! Context editing: de-duplicate, de-redundify and budget-trim retrieval hits
//! before they are packed into the prompt (Track 5 / dynamic cropping).
//!
//! RAG feeds the model a bag of retrieved chunks, but that bag is rarely
//! clean: the same passage can be recalled across turns or from overlapping
//! sources ([`dedup_hits`]), two documents can carry byte-identical filler
//! paragraphs ([`trim_redundant`]), and the combined body often exceeds what
//! the context window has room for ([`fit_to_budget`]). This module turns that
//! raw bag into a tidy, budget-bounded list.
//!
//! Design:
//! - **pure and offline**: no LLM, no network, no clocks — every transform is a
//!   deterministic `&[Document] -> Vec<...>` function that is unit-testable for
//!   free and safe to run at query time;
//! - **no embeddings required**: all similarity is judged on raw text (Jaccard
//!   token overlap for de-dup, exact sentence overlap for redundancy), which is
//!   far cheaper than embedding every hit. (The [`lc_embeddings::cosine_similarity`]
//!   re-export is available if a caller wants to bring their own embeddings instead, but
//!   this module deliberately does not depend on them.);
//! - **keeps the first**: across all three phases the earliest/highest-relevance
//!   copy wins, so the "canonical" occurrence is always preserved;
//! - **traceable**: [`ContextEditor::edit`] returns one [`EditedDoc`] per input in
//!   input order, with `retained` and a [`DropReason`], so the caller can see
//!   exactly why a hit was dropped (and filter `retained == true` for the prompt).

use lc_vector_stores::Document;
use std::collections::HashSet;

/// Default Jaccard-overlap threshold above which two documents are considered
/// duplicates (0.0 = never de-dup, 1.0 = only byte-identical text).
const DEFAULT_DEDUP_THRESHOLD: f32 = 0.8;

/// Default minimum run of duplicated sentences that must be present before a
/// contiguously-repeated block is dropped as redundant.
const DEFAULT_TRIM_MIN_RUN: usize = 1;

/// Why a document was dropped from the final context by the editor.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DropReason {
    /// The document duplicated an earlier one and was removed by de-dup.
    Duplicate,
    /// The document was dropped because it did not fit the character budget.
    BudgetExceeded,
    /// The document was dropped because the maximum document count was reached.
    DocumentCap,
}

/// Result of applying [`ContextEditor`] to one input document.
///
/// The editor emits one `EditedDoc` per input (in input order) so the caller can
/// align outputs with inputs; a document that survived has `retained == true`,
/// and the prompt should be built from those only.
#[derive(Debug, Clone)]
pub struct EditedDoc {
    /// The (possibly trimmed) document content.
    pub document: Document,
    /// Whether the document survived editing and should enter the context.
    pub retained: bool,
    /// Why the document was dropped, if it was.
    pub drop_reason: Option<DropReason>,
    /// For a duplicate, the `id` of the earlier document it duplicated.
    pub duplicate_of: Option<String>,
    /// Number of characters removed from this document by redundancy trimming.
    pub trimmed_chars: usize,
    /// Character count of the original input document.
    pub original_chars: usize,
}

/// Configuration for [`ContextEditor`].
#[derive(Debug, Clone)]
pub struct ContextEditorConfig {
    /// Whether cross-source/cross-turn de-duplication is enabled.
    pub dedup_enabled: bool,
    /// Jaccard-token-overlap threshold for de-dup (see `DEFAULT_DEDUP_THRESHOLD`).
    pub dedup_threshold: f32,
    /// Whether inter-document redundancy trimming is enabled.
    pub trim_enabled: bool,
    /// Minimum run of duplicated sentences required to drop a block.
    pub trim_min_run: usize,
    /// Whether budget trimming of the tail is enabled.
    pub budget_enabled: bool,
    /// Maximum total characters accepted, from high to low relevance. `None` = unlimited.
    pub max_chars: Option<usize>,
    /// Maximum number of documents accepted. `None` = unlimited.
    pub max_documents: Option<usize>,
}

impl Default for ContextEditorConfig {
    fn default() -> Self {
        Self {
            dedup_enabled: true,
            dedup_threshold: DEFAULT_DEDUP_THRESHOLD,
            trim_enabled: true,
            trim_min_run: DEFAULT_TRIM_MIN_RUN,
            budget_enabled: true,
            max_chars: None,
            max_documents: None,
        }
    }
}

impl ContextEditorConfig {
    /// Creates the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether de-duplication runs.
    pub fn with_dedup(mut self, enabled: bool) -> Self {
        self.dedup_enabled = enabled;
        self
    }

    /// Sets the Jaccard-overlap threshold used for de-duplication.
    pub fn with_dedup_threshold(mut self, threshold: f32) -> Self {
        self.dedup_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets whether redundancy trimming runs.
    pub fn with_trim_redundant(mut self, enabled: bool) -> Self {
        self.trim_enabled = enabled;
        self
    }

    /// Sets the minimum duplicated-sentence run length before a block is dropped.
    pub fn with_trim_min_run(mut self, min_run: usize) -> Self {
        self.trim_min_run = min_run;
        self
    }

    /// Sets whether budget trimming runs.
    pub fn with_budget(mut self, enabled: bool) -> Self {
        self.budget_enabled = enabled;
        self
    }

    /// Sets the character budget (`max_chars`, an `Option`).
    pub fn with_max_chars(mut self, max_chars: Option<usize>) -> Self {
        self.max_chars = max_chars;
        self
    }

    /// Sets the maximum document count.
    pub fn with_max_documents(mut self, max_documents: Option<usize>) -> Self {
        self.max_documents = max_documents;
        self
    }
}

/// Builder-style editor that packs de-dup, redundancy trimming and budget
/// trimming into one pass over retrieval hits.
///
/// Documents are expected in relevance order (most relevant first) — this is
/// how retrievers return them — so when a budget applies the editor keeps the
/// higher-relevance (earlier) documents and drops the tail.
#[derive(Debug, Clone)]
pub struct ContextEditor {
    config: ContextEditorConfig,
}

impl Default for ContextEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextEditor {
    /// Creates an editor with the default configuration (all phases on, no budget).
    pub fn new() -> Self {
        Self {
            config: ContextEditorConfig::default(),
        }
    }

    /// Builds an editor from an explicit configuration.
    pub fn with_config(config: ContextEditorConfig) -> Self {
        Self { config }
    }

    /// Sets the full configuration.
    pub fn with_config_mut(mut self, config: ContextEditorConfig) -> Self {
        self.config = config;
        self
    }

    /// Toggles de-duplication.
    pub fn with_dedup(mut self, enabled: bool) -> Self {
        self.config.dedup_enabled = enabled;
        self
    }

    /// Sets the de-duplication similarity threshold.
    pub fn with_dedup_threshold(mut self, threshold: f32) -> Self {
        self.config.dedup_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Toggles redundancy trimming.
    pub fn with_trim_redundant(mut self, enabled: bool) -> Self {
        self.config.trim_enabled = enabled;
        self
    }

    /// Sets the minimum duplicated-sentence run length for redundancy trimming.
    pub fn with_trim_min_run(mut self, min_run: usize) -> Self {
        self.config.trim_min_run = min_run;
        self
    }

    /// Toggles budget trimming.
    pub fn with_budget(mut self, enabled: bool) -> Self {
        self.config.budget_enabled = enabled;
        self
    }

    /// Sets the character budget (an `Option<usize>`; `None` = unlimited).
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.config.max_chars = Some(max_chars);
        self
    }

    /// Sets the maximum document count.
    pub fn with_max_documents(mut self, max_documents: usize) -> Self {
        self.config.max_documents = Some(max_documents);
        self
    }

    /// Runs the configured pipeline (de-dup → redundancy trim → budget trim) over
    /// the hits and returns one [`EditedDoc`] per input, in input order.
    ///
    /// Callers building the actual context should filter to `retained == true`.
    pub fn edit(&self, documents: &[Document]) -> Vec<EditedDoc> {
        let mut work: Vec<Work> = documents
            .iter()
            .map(|d| Work {
                document: d.clone(),
                retained: true,
                drop_reason: None,
                duplicate_of: None,
                trimmed_chars: 0,
                original_chars: d.content.chars().count(),
            })
            .collect();

        self.dedup_phase(&mut work);
        self.trim_phase(&mut work);
        self.budget_phase(&mut work);

        work.into_iter()
            .map(|w| EditedDoc {
                document: w.document,
                retained: w.retained,
                drop_reason: w.drop_reason,
                duplicate_of: w.duplicate_of,
                trimmed_chars: w.trimmed_chars,
                original_chars: w.original_chars,
            })
            .collect()
    }

    /// Cross-source de-dup: drop a hit whose token overlap with any earlier kept
    /// hit reaches the threshold; the first copy wins.
    fn dedup_phase(&self, work: &mut [Work]) {
        if !self.config.dedup_enabled {
            return;
        }
        let mut kept: Vec<usize> = Vec::new();
        for i in 0..work.len() {
            if let Some(&pi) = kept.iter().find(|&&pi| {
                jaccard(&work[pi].document.content, &work[i].document.content)
                    >= self.config.dedup_threshold
            }) {
                work[i].retained = false;
                work[i].drop_reason = Some(DropReason::Duplicate);
                work[i].duplicate_of = work[pi].document.id.clone();
            } else {
                kept.push(i);
            }
        }
    }

    /// Redundancy trim: for each surviving hit, drop contiguous runs of sentences
    /// that were already emitted by an earlier document (kept once), reconstructing
    /// the remaining body faithfully.
    fn trim_phase(&self, work: &mut [Work]) {
        if !self.config.trim_enabled {
            return;
        }
        let min_run = self.config.trim_min_run.max(1);
        let mut bank: HashSet<String> = HashSet::new();
        for w in work.iter_mut() {
            if !w.retained {
                continue;
            }
            let original = std::mem::take(&mut w.document.content);
            let frames = split_sentences(&original);
            let keys: Vec<String> = frames.iter().map(|(t, _)| normalize_sentence(t)).collect();
            let dup: Vec<bool> = keys.iter().map(|k| bank.contains(k)).collect();
            let retain = compute_retain(&dup, min_run);
            let mut kept_text = String::with_capacity(original.len());
            for (idx, (text, sep)) in frames.iter().enumerate() {
                if retain[idx] {
                    kept_text.push_str(text);
                    kept_text.push_str(sep);
                    bank.insert(keys[idx].clone());
                }
            }
            w.document.content = kept_text;
            w.trimmed_chars = w.original_chars - w.document.content.chars().count();
        }
    }

    /// Budget trim: greedily keep hits in relevance order until the document count
    /// and character budget are both satisfied, then drop the rest of the tail.
    fn budget_phase(&self, work: &mut [Work]) {
        if !self.config.budget_enabled {
            return;
        }
        let mut included: Vec<usize> = Vec::new();
        let mut chars = 0usize;
        let mut stop: Option<DropReason> = None;
        for (i, w) in work.iter().enumerate() {
            if !w.retained {
                continue;
            }
            let len = w.document.content.chars().count();
            if let Some(k) = self.config.max_documents {
                if included.len() >= k {
                    stop = Some(DropReason::DocumentCap);
                    break;
                }
            }
            if let Some(m) = self.config.max_chars {
                if chars + len > m {
                    stop = Some(DropReason::BudgetExceeded);
                    break;
                }
            }
            included.push(i);
            chars += len;
        }
        let mut included_set = vec![false; work.len()];
        for i in included {
            included_set[i] = true;
        }
        for (i, w) in work.iter_mut().enumerate() {
            if w.retained && !included_set[i] {
                // Clone: `stop` is shared across every dropped document, and
                // `Option<DropReason>` is not Copy.
                w.retained = false;
                w.drop_reason = stop.clone();
            }
        }
    }
}

/// Intermediate per-document state threaded through the editor phases.
#[derive(Debug, Clone)]
struct Work {
    document: Document,
    retained: bool,
    drop_reason: Option<DropReason>,
    duplicate_of: Option<String>,
    trimmed_chars: usize,
    original_chars: usize,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// De-duplicates retrieval hits, keeping the first occurrence of each content
/// group and dropping later hits whose Jaccard token overlap with an earlier
/// kept hit reaches the default threshold.
///
/// Deterministic and offline: identical input in → identical output out.
pub fn dedup_hits(documents: &[Document]) -> Vec<Document> {
    dedup_hits_with_threshold(documents, DEFAULT_DEDUP_THRESHOLD)
}

/// De-duplicates hits with an explicit similarity threshold (0.0..=1.0).
pub fn dedup_hits_with_threshold(documents: &[Document], threshold: f32) -> Vec<Document> {
    let threshold = threshold.clamp(0.0, 1.0);
    let mut kept: Vec<&Document> = Vec::new();
    let mut out = Vec::with_capacity(documents.len());
    for doc in documents {
        if kept
            .iter()
            .any(|prev| jaccard(&prev.content, &doc.content) >= threshold)
        {
            continue;
        }
        out.push(doc.clone());
        kept.push(doc);
    }
    out
}

/// Removes inter-document redundancy by compressing duplicated sentence blocks.
///
/// A contiguous run of `trim_min_run`-or-more sentences that already appeared in
/// an earlier document is dropped from later documents (kept once), while every
/// non-duplicated sentence is preserved byte-for-byte, so no document's structure
/// or unique content is corrupted.
pub fn trim_redundant(documents: &[Document]) -> Vec<Document> {
    trim_redundant_with_min_run(documents, DEFAULT_TRIM_MIN_RUN)
}

/// As [`trim_redundant`], with an explicit minimum duplicated-run length.
pub fn trim_redundant_with_min_run(documents: &[Document], min_run: usize) -> Vec<Document> {
    let min_run = min_run.max(1);
    let mut bank: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(documents.len());
    for doc in documents {
        let frames = split_sentences(&doc.content);
        let keys: Vec<String> = frames.iter().map(|(t, _)| normalize_sentence(t)).collect();
        let dup: Vec<bool> = keys.iter().map(|k| bank.contains(k)).collect();
        let retain = compute_retain(&dup, min_run);
        let mut kept_text = String::with_capacity(doc.content.len());
        for (idx, (text, sep)) in frames.iter().enumerate() {
            if retain[idx] {
                kept_text.push_str(text);
                kept_text.push_str(sep);
                bank.insert(keys[idx].clone());
            }
        }
        out.push(remap_document(doc, kept_text));
    }
    out
}

/// Returns a copy of `source` whose content is replaced with `content`, keeping the
/// original `id` and all metadata (immutability-preserving rebuild).
fn remap_document(source: &Document, content: String) -> Document {
    let mut out = Document::new(content);
    if let Some(id) = &source.id {
        out = out.with_id(id.clone());
    }
    for (k, v) in &source.metadata {
        out.metadata.insert(k.clone(), v.clone());
    }
    out
}

/// Returns the subset of `documents` (in order) that fits within `max_chars`,
/// dropping the tail once the cumulative length would exceed the budget.
///
/// Pass `None` to disable the budget and keep every document.
pub fn fit_to_budget(documents: &[Document], max_chars: Option<usize>) -> Vec<&Document> {
    let mut out = Vec::new();
    let mut chars = 0usize;
    for doc in documents {
        let len = doc.content.chars().count();
        if let Some(m) = max_chars {
            if chars + len > m {
                break;
            }
            chars += len;
            out.push(doc);
        } else {
            out.push(doc);
        }
    }
    out
}

/// Owned variant of [`fit_to_budget`] that clones the surviving documents.
pub fn fit_to_budget_owned(documents: &[Document], max_chars: Option<usize>) -> Vec<Document> {
    fit_to_budget(documents, max_chars)
        .into_iter()
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Similarity & segmentation primitives
// ---------------------------------------------------------------------------

/// Jaccard token-overlap similarity in `0.0..=1.0`. Two empty texts score `1.0`
/// (identical); a non-empty vs empty pair scores `0.0`.
fn jaccard(a: &str, b: &str) -> f32 {
    let sa: HashSet<String> = tokens(a);
    let sb: HashSet<String> = tokens(b);
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as usize;
    let union = sa.union(&sb).count() as usize;
    inter as f32 / union as f32
}

/// Lower-cased, punctuation-stripped token set of `text`.
fn tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Splits `text` into `(sentence, separator)` frames such that concatenating all
/// `sentence + separator` frames reproduces the input exactly. A separator is the
/// sentence-ending punctuation (`.!?`) plus any immediately-following whitespace.
fn split_sentences(text: &str) -> Vec<(String, String)> {
    let mut frames = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?') {
            let mut sep = String::new();
            sep.push(c);
            cur.pop(); // remove the terminator from the sentence text
            while let Some(&n) = chars.peek() {
                if n.is_whitespace() {
                    sep.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            frames.push((std::mem::take(&mut cur), sep));
        }
    }
    if !cur.is_empty() {
        frames.push((cur, String::new()));
    }
    frames
}

/// Collapses whitespace and lower-cases a sentence so only semantic duplicates match.
fn normalize_sentence(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Computes per-sentence retention: every maximal run of duplicated sentences of
/// length `>= min_run` is dropped; shorter runs (and non-duplicated sentences) are
/// retained.
fn compute_retain(dup: &[bool], min_run: usize) -> Vec<bool> {
    let n = dup.len();
    let mut retain = vec![true; n];
    let mut i = 0;
    while i < n {
        if dup[i] {
            let mut run_end = i;
            while run_end + 1 < n && dup[run_end + 1] {
                run_end += 1;
            }
            let run_len = run_end - i + 1;
            if run_len >= min_run {
                retain[i..=run_end].fill(false);
            }
            i = run_end + 1;
        } else {
            i += 1;
        }
    }
    retain
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- de-dup ---

    #[test]
    fn dedup_keeps_first_of_exact_duplicates() {
        let a = Document::new("The quick brown fox jumps over the lazy dog.");
        let b = Document::new("The quick brown fox jumps over the lazy dog.");
        let c = Document::new("A completely different document about chlorophyll.");
        let out = dedup_hits(&[a.clone(), b.clone(), c.clone()]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content, a.content);
        assert_eq!(out[1].content, c.content, "distinct doc kept");
    }

    #[test]
    fn dedup_drops_near_duplicate_keeps_first() {
        let a = Document::new("Quantum computing uses qubits to process information.");
        let b = Document::new("Quantum computing uses qubits to process information quickly.");
        let out = dedup_hits(&[a.clone(), b.clone()]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, a.content, "first occurrence kept");
    }

    #[test]
    fn dedup_keeps_distinct_documents() {
        let d1 = Document::new("Alpha report covers fiscal policy and revenue for the year.");
        let d2 = Document::new("Beta discusses weather patterns and ocean currents extensively.");
        let out = dedup_hits(&[d1.clone(), d2.clone()]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content, d1.content);
        assert_eq!(out[1].content, d2.content);
    }

    // --- redundancy trim ---

    #[test]
    fn trim_removes_duplicate_block_preserves_structure() {
        let a = Document::new(
            "Revenue rose 3% last quarter. The board approved the new budget. Employees were notified.",
        );
        let b = Document::new(
            "Fresh finding here. The board approved the new budget. More details in the addendum.",
        );
        let out = trim_redundant(&[a.clone(), b.clone()]);
        assert_eq!(out[0].content, a.content, "first occurrence untouched");
        assert!(
            out[1].content.contains("Fresh finding here."),
            "unique kept"
        );
        assert!(
            !out[1].content.contains("The board approved the new budget"),
            "duplicated sentence removed from later doc"
        );
        assert!(
            out[1].content.contains("More details in the addendum."),
            "later unique kept"
        );
        // structure intact: concatenated kept frames equal output, order preserved
        assert!(out[1].content.starts_with("Fresh finding here."));
    }

    #[test]
    fn trim_leaves_disjoint_documents_unchanged() {
        let a = Document::new("One unique sentence about the sun. Another about the moon.");
        let b = Document::new("Unrelated sentence about rivers and their deltas worldwide.");
        let out = trim_redundant(&[a.clone(), b.clone()]);
        assert_eq!(out[0].content, a.content);
        assert_eq!(out[1].content, b.content);
    }

    #[test]
    fn trim_short_dup_run_kept_when_below_min_run() {
        let a = Document::new("Sentence alpha one. Sentence alpha two.");
        let b = Document::new("Sentence alpha one. Sentence beta distinct.");
        // min_run=2: the single duplicated sentence is kept, not dropped.
        let out = trim_redundant_with_min_run(&[a.clone(), b.clone()], 2);
        assert!(out[1].content.contains("Sentence alpha one."));
        // min_run=1: the single duplicated sentence is dropped.
        let out = trim_redundant(&[a.clone(), b.clone()]);
        assert!(!out[1].content.contains("Sentence alpha one."));
    }

    // --- budget trim ---

    #[test]
    fn fit_drops_tail_when_budget_exceeded() {
        let docs = vec![
            Document::new("aaaa"),
            Document::new("bbbb"),
            Document::new("cccc"),
        ];
        let kept = fit_to_budget_owned(&docs, Some(8)); // 4 + 4 fits, third overflows
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].content, "aaaa");
        assert_eq!(kept[1].content, "bbbb");
    }

    #[test]
    fn fit_keeps_all_when_budget_is_enough_or_unset() {
        let docs = vec![Document::new("aaaa"), Document::new("bbbb")];
        let kept = fit_to_budget_owned(&docs, Some(100));
        assert_eq!(kept.len(), 2);
        let kept = fit_to_budget_owned(&docs, None);
        assert_eq!(kept.len(), 2);
    }

    // --- editor pipeline ---

    #[test]
    fn editor_dedupes_and_reports_duplicate() {
        let editor = ContextEditor::new(); // no budget → dedup is the active phase
        let docs = vec![
            Document::new("Unique first document about helium balloons and air.").with_id("1"),
            Document::new("Unique first document about helium balloons and air.").with_id("2"),
            Document::new("A second document discussing tariffs and trade policy.").with_id("3"),
        ];
        let out = editor.edit(&docs);
        assert_eq!(out.len(), 3, "one EditedDoc per input");
        assert_eq!(out.iter().filter(|e| e.retained).count(), 2);
        assert!(out[0].retained);
        assert!(!out[1].retained);
        assert_eq!(out[1].drop_reason, Some(DropReason::Duplicate));
        assert_eq!(out[1].duplicate_of, Some("1".to_string()));
        assert!(out[2].retained);
    }

    #[test]
    fn editor_trims_redundancy_without_budget() {
        let editor = ContextEditor::new().with_budget(false);
        let docs = vec![
            Document::new(
                "Alpha intro sentence one. Shared summary text here. Alpha outro sentence two.",
            ),
            Document::new("Beta unique statement. Shared summary text here. Beta trailing note."),
        ];
        let out = editor.edit(&docs);
        assert_eq!(
            out[0].document.content, docs[0].content,
            "first doc untouched"
        );
        assert!(!out[1].document.content.contains("Shared summary text here"));
        assert!(out[1].document.content.contains("Beta unique statement."));
        assert!(out[1].document.content.contains("Beta trailing note."));
        assert!(out[1].trimmed_chars >= "Shared summary text here.".chars().count());
    }

    #[test]
    fn editor_drops_tail_on_budget() {
        let editor = ContextEditor::new().with_max_chars(10);
        let docs = vec![Document::new("aaaaaaaaaa"), Document::new("bbbbbbbbbb")];
        let out = editor.edit(&docs);
        assert!(out[0].retained);
        assert!(!out[1].retained);
        assert_eq!(out[1].drop_reason, Some(DropReason::BudgetExceeded));
    }

    #[test]
    fn editor_caps_document_count() {
        let editor = ContextEditor::new().with_max_documents(1);
        let docs = vec![Document::new("x content a"), Document::new("y content b")];
        let out = editor.edit(&docs);
        assert!(out[0].retained);
        assert!(!out[1].retained);
        assert_eq!(out[1].drop_reason, Some(DropReason::DocumentCap));
    }

    #[test]
    fn editor_all_disabled_passes_through() {
        let editor = ContextEditor::new()
            .with_dedup(false)
            .with_trim_redundant(false)
            .with_budget(false);
        let docs = vec![
            Document::new("Same exact sentence repeated here."),
            Document::new("Same exact sentence repeated here."),
        ];
        let out = editor.edit(&docs);
        assert!(out.iter().all(|e| e.retained));
        assert_eq!(
            out[0].document.content, out[1].document.content,
            "nothing trimmed when disabled"
        );
    }

    #[test]
    fn full_pipeline_dedup_then_redundancy_then_budget() {
        let editor = ContextEditor::new()
            .with_max_documents(1)
            .with_max_chars(200);
        let docs = vec![
            Document::new("Doc alpha unique content sentence one. Unique closing sentence there."),
            Document::new("Doc alpha unique content sentence one. Unique closing sentence there."), // dup of 0
            Document::new("Completely different document beta with its own phrasing here."),
        ];
        let out = editor.edit(&docs);
        // after dedup: docs 0 and 2 survive; after doc-cap(1): only doc 0 retained
        assert!(out[0].retained);
        assert!(!out[1].retained);
        assert_eq!(out[1].drop_reason, Some(DropReason::Duplicate));
        assert!(!out[2].retained);
        assert_eq!(out[2].drop_reason, Some(DropReason::DocumentCap));
    }
}
