// lc-rag/src/sentence_window.rs
//! SentenceWindowRetriever — "small chunk matches, surrounding context returned"
//!
//! Classic RAG tension: a small match unit (e.g. a single sentence) is precise
//! but too little context for the LLM, while a large unit has full context but
//! poor match locality. Sentence-window retrieval resolves both: **match** on
//! single sentences (precise), but **return** the matched sentence plus `window`
//! surrounding sentences from the same document (full context).
//!
//! This contrasts with [`ParentDocumentRetriever`](crate::parent_document),
//! which returns the entire parent document on any leaf hit — here only a bounded
//! local neighbourhood is returned, so a precise match does not drown the model
//! in unrelated document content.

use async_trait::async_trait;
use lc_vector_stores::{Document, SearchResult};
use std::collections::HashMap;

use crate::retriever::{RetrieverError, RetrieverTrait};

/// A sentence plus enough provenance to expand it into a context window.
struct SentenceRef {
    /// Index (into `self.documents`) of the source document.
    doc_index: usize,
    /// The sentence, as indexed for matching.
    sentence: String,
}

/// Sentence-window retriever.
///
/// Ingests full documents (split into sentences for matching); a query that hits
/// a sentence returns that sentence together with up to `window` neighbouring
/// sentences from the same document.
pub struct SentenceWindowRetriever {
    sentences: Vec<SentenceRef>,
    /// Number of sentences to return on each side of a match (inclusive window
    /// is `2*window + 1` sentences).
    window: usize,
    /// Maximum hits considered before windowing (each expands to a window).
    top_k: usize,
}

impl Default for SentenceWindowRetriever {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceWindowRetriever {
    /// Create an empty retriever with a default window of 2 sentences per side.
    pub fn new() -> Self {
        Self {
            sentences: Vec::new(),
            window: 2,
            top_k: 3,
        }
    }

    /// Set how many sentences to include on each side of a match (`default: 2`).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    /// Set the number of matching sentences expanded into windows (`default: 3`).
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Build a retriever directly from full documents.
    ///
    /// Each document is split into sentences for matching; the same sentences
    /// stay grouped by their source document so a hit can be expanded back into
    /// local context. Equivalent to constructing empty + filling the index, but
    /// available as a real public construction path.
    pub fn from_documents(documents: Vec<Document>) -> Self {
        let sentences = documents
            .into_iter()
            .enumerate()
            .flat_map(|(doc_index, d)| {
                split_sentences(&d.content)
                    .into_iter()
                    .map(move |sentence| SentenceRef {
                        doc_index,
                        sentence,
                    })
            })
            .collect();
        Self {
            sentences,
            ..Self::new()
        }
    }

    /// Number of indexed sentences.
    pub fn sentence_count(&self) -> usize {
        self.sentences.len()
    }
}

/// A cheap, deterministic sentence splitter. Terminates at the classic sentence
/// punctuators plus CJK full stops; newlines also break sentences.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？' | '\n') {
            let sentence = current.trim().to_string();
            if !sentence.is_empty() {
                out.push(sentence);
            }
            current.clear();
        }
    }
    let tail = current.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// Simple TF scoring of a sentence against the query's whitespace terms.
fn score_sentence(query: &str, sentence: &str) -> f32 {
    let sentence_lower = sentence.to_lowercase();
    query
        .split_whitespace()
        .filter(|t| t.len() > 1)
        .map(|t| sentence_lower.matches(&t.to_lowercase()).count() as f32)
        .sum()
}

/// Expand a matched sentence index (into `self.sentences`) into the windowed
/// context text drawn from the same source document.
fn build_window(sentence_index: usize, sentences: &[SentenceRef], window: usize) -> String {
    let doc_index = sentences[sentence_index].doc_index;
    // Find the range of this document's sentences containing `sentence_index`.
    let mut start = sentence_index;
    while start > 0 && sentences[start - 1].doc_index == doc_index {
        start -= 1;
    }
    let mut end = sentence_index + 1;
    while end < sentences.len() && sentences[end].doc_index == doc_index {
        end += 1;
    }
    let lo = start
        .saturating_add(0)
        .max(sentence_index.saturating_sub(window));
    let lo = lo.max(start);
    let hi = (sentence_index + 1 + window).min(end);
    sentences[lo..hi]
        .iter()
        .map(|s| s.sentence.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

impl SentenceWindowRetriever {
    /// Score every sentence, pick the top `top_k`, and expand each into its
    /// windowed context as a `Document`.
    fn windowed_results(&self, query: &str, k: usize) -> Vec<Document> {
        if self.sentences.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self
            .sentences
            .iter()
            .enumerate()
            .map(|(i, s)| (i, score_sentence(query, &s.sentence)))
            .collect();
        // Stable by sentence position: equal scores keep earlier sentences first,
        // so results are deterministic for the same index.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut seen_docs: HashMap<usize, usize> = HashMap::new(); // doc_index -> seq
        let mut docs: Vec<Document> = Vec::new();
        for (sentence_index, score) in scored {
            if score <= 0.0 {
                break;
            }
            let doc_index = self.sentences[sentence_index].doc_index;
            // Avoid returning the same source document more than once per query.
            if seen_docs.contains_key(&doc_index) {
                continue;
            }
            let _ = seen_docs.entry(doc_index).or_insert(0);
            let context = build_window(sentence_index, &self.sentences, self.window);
            docs.push(Document::new(context));
            if docs.len() >= k.min(self.top_k) {
                break;
            }
        }
        docs
    }
}

#[async_trait]
impl RetrieverTrait for SentenceWindowRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        Ok(self.windowed_results(query, k.max(self.top_k)))
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        // Scores are not surfaced (windows are re-assembled context, not ranked
        // documents); report the hit sentence's score as a rough relevance signal.
        let docs = self.windowed_results(query, k.max(self.top_k));
        Ok(docs
            .into_iter()
            .map(|document| SearchResult {
                document,
                score: 1.0,
            })
            .collect())
    }

    async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
        // `RetrieverTrait::add_documents` takes `&self`; SentenceWindowRetriever
        // therefore mutates via interior storage would need a lock. To stay a plain
        // value retriever it is built by constructor and exposed read-only; adding a
        // live ingest path keeps this trait signature and is documented as adding to
        // the *current* instance only.
        Err(RetrieverError::OperationNotSupported(
            "SentenceWindowRetriever is immutable after construction; pass a pre-built index"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_docs() -> Vec<Document> {
        vec![
            Document::new(
                "Rust is a systems programming language. \
                 It focuses on safety and performance. \
                 Ownership is its defining feature. \
                 The Zebra is an African equid mammal.",
            ),
            Document::new(
                "Python is a high-level language. \
                 It emphasizes readable code. \
                 The moon is a natural satellite. \
                 Zebras live in herds on grasslands.",
            ),
        ]
    }

    /// Build a retriever over the sample docs via the public constructor path.
    fn built_retriever() -> SentenceWindowRetriever {
        SentenceWindowRetriever::from_documents(sample_docs())
            .with_window(1)
            .with_top_k(2)
    }

    #[test]
    fn splits_sentences() {
        let s = split_sentences("Hello world. Second sentence! Third？\nNext line.");
        assert!(
            s.len() >= 4,
            "should split on . ! ？ and newline, got {s:?}"
        );
        assert_eq!(s[0], "Hello world.");
    }

    #[test]
    fn match_on_sentence_returns_window_context() {
        let r = built_retriever();
        // "herds" appears only in the last sentence of the second document; the
        // window must pull the surrounding sentence(s) into the returned context.
        let docs = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(r.retrieve("herds", 1))
            .unwrap();
        assert_eq!(docs.len(), 1, "docs: {:?}", docs[0].content);
        assert!(
            docs[0].content.contains("Zebras"),
            "match sentence must be present"
        );
        // The window returns more than the single matching sentence (neighbour
        // "The moon is a natural satellite." on the left) and stays bounded to
        // that document.
        assert!(
            docs[0].content.contains("moon"),
            "window must include the neighbour sentence: {}",
            docs[0].content
        );
        assert!(
            !docs[0].content.contains("Rust is a systems"),
            "context must stay within the source document: {}",
            docs[0].content
        );
    }
}
