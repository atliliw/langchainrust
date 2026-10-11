// lc-rag/src/context_injection.rs
//! Parent-title + neighbor-context injection into retrieval chunks (0.26 T6.3).
//!
//! A standalone chunk ("revenue grew 3%") loses the subject it refers to when it is
//! split out of its document and embedded on its own — so it is never recalled for a
//! query that mentions the actual subject. This module is the deterministic, zero-LLM
//! half of contextual retrieval: it re-wraps each chunk with the parent document's title
//! and a configurable window of *adjacent* chunks, so the text the embedder and the LLM
//! see carries the surrounding context:
//!
//! ```text
//! <document_chunk_context>
//!   <document_chunk_title>Quarterly Earnings</document_chunk_title>
//!   <!--Chunk 0 begins-->[prev]<!--Chunk 0 ends-->
//!   <!--Chunk 1 begins-->[this]<!--Chunk 1 ends-->
//!   <!--Chunk 2 begins-->[next]<!--Chunk 2 ends-->
//! </document_chunk_context>
//! ```
//!
//! It is the pre-LLM complement to the model-based [`crate::contextual::ContextualEnhancer`]
//! (whose prompt only receives `{chunk}`). Running the injector *first* gives the contextual
//! LLM a situating prompt, and even when the enhancement LLM is omitted the injected title +
//! neighbors measurably improve embedding recall on its own.
//!
//! Design:
//! - **pure and offline**: no network, no LLM, no clocks — fully unit-testable and safe
//!   to run at index time for free;
//! - **deterministic**: identical input → identical output, so re-indexing converges;
//! - **idempotent**: chunks already carrying the [`CONTEXT_INJECTED_KEY`] marker are passed
//!   through untouched, so re-running over an indexed corpus is a no-op;
//! - **bounded**: each neighbor is capped at [`ContextInjectionConfig::max_ctx_chars`], so
//!   injected context never blows up the embedding vector budget; the *current* chunk is
//!   never truncated;
//! - the current chunk is flagged in-place so a downstream consumer that re-splits can find
//!   it, and every chunk keeps its `id` and all existing metadata.

use lc_vector_stores::Document;

/// Metadata marker set on an injected chunk (used for idempotent skip).
pub const CONTEXT_INJECTED_KEY: &str = "context_injected";

/// Metadata key under which the resolved parent title is stored.
pub const PARENT_TITLE_KEY: &str = "parent_title";

/// Default metadata key read to resolve the parent document title.
pub const DEFAULT_TITLE_KEY: &str = "title";

/// Default number of neighbor chunks injected on each side.
pub const DEFAULT_WINDOW: usize = 1;

/// Default cap on each neighbor's injected character length.
pub const DEFAULT_MAX_CTX_CHARS: usize = 1500;

/// Default separator between the resolved title and the chunk body.
const DEFAULT_TITLE_SEP: &str = "\n\n";

/// Configuration for [`ContextInjector`].
#[derive(Clone, Debug)]
pub struct ContextInjectionConfig {
    /// Neighbor chunks injected on each side of the current chunk.
    pub window: usize,
    /// Metadata key whose string value is the parent document title.
    pub title_key: String,
    /// Cap on each *neighbor* chunk's injected characters (current chunk never cut).
    pub max_ctx_chars: usize,
    /// Separator placed between the title line and the chunk block.
    pub title_sep: String,
}

impl Default for ContextInjectionConfig {
    fn default() -> Self {
        Self {
            window: DEFAULT_WINDOW,
            title_key: DEFAULT_TITLE_KEY.to_string(),
            max_ctx_chars: DEFAULT_MAX_CTX_CHARS,
            title_sep: DEFAULT_TITLE_SEP.to_string(),
        }
    }
}

impl ContextInjectionConfig {
    /// Creates the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how many neighbors to inject on each side.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    /// Sets the metadata key from which the parent title is resolved.
    pub fn with_title_key(mut self, key: impl Into<String>) -> Self {
        self.title_key = key.into();
        self
    }

    /// Sets the per-neighbor character cap.
    pub fn with_max_ctx_chars(mut self, max: usize) -> Self {
        self.max_ctx_chars = max;
        self
    }

    /// Sets the separator between the title and the chunk block.
    pub fn with_title_sep(mut self, sep: impl Into<String>) -> Self {
        self.title_sep = sep.into();
        self
    }
}

/// Injects parent-title + neighbor context into retrieval chunks.
#[derive(Clone, Debug)]
pub struct ContextInjector {
    config: ContextInjectionConfig,
}

impl ContextInjector {
    /// Creates an injector with the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an injector from an explicit configuration.
    pub fn with_config(config: ContextInjectionConfig) -> Self {
        Self { config }
    }

    /// Whether a document is already considered injected.
    pub fn is_injected(doc: &Document) -> bool {
        doc.metadata.contains_key(CONTEXT_INJECTED_KEY)
    }

    /// Injects context into an ordered list of chunks with an explicit title.
    ///
    /// `chunks` must be in document order. Each returned chunk's content is re-wrapped with
    /// the title and its (up to `window`-sized) neighbors; already-injected chunks are passed
    /// through unchanged.
    pub fn inject_with_title(&self, title: &str, chunks: &[Document]) -> Vec<Document> {
        chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                if Self::is_injected(chunk) {
                    return chunk.clone(); // idempotent skip
                }
                let context = self.build_block(title, chunks, i);
                let mut out = Document::new(context);
                // Preserve id and carry the parent title + marker through.
                if let Some(id) = &chunk.id {
                    out = out.with_id(id.clone());
                }
                for (k, v) in &chunk.metadata {
                    out.metadata.insert(k.clone(), v.clone());
                }
                out.metadata.insert(
                    CONTEXT_INJECTED_KEY.to_string(),
                    serde_json::Value::Bool(true),
                );
                out.metadata.insert(
                    PARENT_TITLE_KEY.to_string(),
                    serde_json::Value::String(title.to_string()),
                );
                out
            })
            .collect()
    }

    /// Injects context, resolving the title from the first chunk's metadata
    /// ([`ContextInjectionConfig::title_key`]).
    ///
    /// If no chunk carries a title, an empty title is used (neighbors are still injected).
    pub fn inject(&self, chunks: &[Document]) -> Vec<Document> {
        let title = chunks
            .iter()
            .find_map(|c| c.metadata.get(&self.config.title_key))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        self.inject_with_title(title, chunks)
    }

    /// Builds the tagged block for the chunk at `idx`.
    ///
    /// Pure and deterministic: the title, each adjacent neighbor (capped at
    /// `max_ctx_chars`), and the full current chunk are wrapped in explicit
    /// "begin/end" markers so a downstream consumer can tell data from injection.
    fn build_block(&self, title: &str, chunks: &[Document], idx: usize) -> String {
        let window = self.config.window;
        let lo = idx.saturating_sub(window);
        let hi = (idx + 1 + window).min(chunks.len());

        let mut parts = Vec::new();
        parts.push("<document_chunk_context>".to_string());
        if !title.is_empty() {
            parts.push(format!(
                "<document_chunk_title>{}</document_chunk_title>{}",
                title, self.config.title_sep
            ));
        }
        for (i, chunk) in chunks.iter().enumerate().take(hi).skip(lo) {
            let is_current = i == idx;
            let body = if is_current {
                chunk.content.clone() // never truncate the current chunk
            } else {
                truncate_neighbor(&chunk.content, self.config.max_ctx_chars)
            };
            parts.push(format!(
                "<!--Chunk {} begins-->{body}<!--Chunk {} ends-->",
                i, i
            ));
        }
        parts.push("</document_chunk_context>".to_string());
        parts.join("")
    }
}

/// Truncates a neighbor chunk to `max` chars, appending an explicit ellipsis so the
/// truncation is unambiguous. A `max` of 0 elides the neighbor entirely.
fn truncate_neighbor(content: &str, max: usize) -> String {
    if max == 0 {
        return String::new(); // elide the neighbor entirely
    }
    if content.chars().count() <= max {
        return content.to_string();
    }
    let mut truncated: String = content.chars().take(max).collect();
    truncated.push_str("\n…[truncated]");
    truncated
}

impl Default for ContextInjector {
    fn default() -> Self {
        Self::with_config(ContextInjectionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str, idx: usize) -> Document {
        Document::new(text.to_string())
            .with_metadata("chunk", serde_json::Value::String(idx.to_string()))
    }

    fn doc(text: &str, id: &str, title: &str) -> Document {
        Document::new(text.to_string())
            .with_id(id.to_string())
            .with_metadata(
                DEFAULT_TITLE_KEY,
                serde_json::Value::String(title.to_string()),
            )
    }

    /// Title is present in every injected chunk, and the current chunk text survives intact.
    #[test]
    fn injects_title_into_each_chunk() {
        let injector = ContextInjector::new();
        let chunks = vec![chunk("revenue grew 3%", 0)];
        let out = injector.inject_with_title("Quarterly Earnings", &chunks);

        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("Quarterly Earnings"));
        assert!(out[0].content.contains("revenue grew 3%"));
        assert!(out[0].content.starts_with("<document_chunk_context>"));
        assert!(out[0].content.ends_with("</document_chunk_context>"));
    }

    /// Immediate neighbors appear around the current chunk; the current chunk is never cut.
    #[test]
    fn injects_immediate_neighbors() {
        let injector = ContextInjector::new();
        let chunks = vec![chunk("a", 0), chunk("b", 1), chunk("c", 2)];
        let out = injector.inject_with_title("T", &chunks);

        let middle = &out[1].content;
        assert!(middle.contains("<!--Chunk 0 begins-->a<!--Chunk 0 ends-->"));
        assert!(middle.contains("<!--Chunk 1 begins-->b<!--Chunk 1 ends-->"));
        assert!(middle.contains("<!--Chunk 2 begins-->c<!--Chunk 2 ends-->"));
    }

    /// First chunk has no preceding neighbor, last chunk no following neighbor.
    #[test]
    fn window_respects_boundaries() {
        let injector = ContextInjector::new();
        let chunks = vec![chunk("first", 0), chunk("mid", 1), chunk("last", 2)];
        let out = injector.inject_with_title("T", &chunks);

        let first = &out[0].content;
        assert!(!first.contains("<!--Chunk 0 begins--><!--Chunk 0 ends-->"));
        // There should be no chunk before 0; the first neighbor present is chunk 1.
        assert!(first.contains("<!--Chunk 1 begins-->"));
        assert!(!first.contains("This should not exist"));

        let last = &out[2].content;
        assert!(last.contains("<!--Chunk 1 begins-->"));
        // chunk 3 would be out of range → only up to 2 present.
        assert!(!last.contains("<!--Chunk 3 begins-->"));
    }

    /// window=0 injects only the title, no neighbors.
    #[test]
    fn window_zero_injects_only_title() {
        let config = ContextInjectionConfig::new().with_window(0);
        let injector = ContextInjector::with_config(config);
        let chunks = vec![chunk("a", 0), chunk("b", 1)];
        let out = injector.inject_with_title("T", &chunks);

        assert!(out[0].content.contains("T"));
        assert!(!out[0].content.contains("<!--Chunk 1 begins-->"));
        assert!(out[0].content.contains("<!--Chunk 0 begins-->a"));
    }

    /// Long neighbors are capped at max_ctx_chars; the current chunk is not truncated.
    #[test]
    fn truncates_long_neighbors() {
        let config = ContextInjectionConfig::new().with_max_ctx_chars(10);
        let injector = ContextInjector::with_config(config);
        let long = "x".repeat(100);
        // position 0 = neighbor, position 1 = current chunk
        let chunks = vec![chunk(long.as_str(), 0), chunk("current", 1)];
        let out = injector.inject_with_title("T", &chunks);

        assert!(out[1].content.contains("current"), "current chunk intact");
        // Neighbor (position 0, the long one) should be capped to 10 chars + ellipsis marker.
        let pos = out[1].content.find("<!--Chunk 0 begins-->").unwrap();
        let after = &out[1].content[pos..];
        let body_start = "<!--Chunk 0 begins-->".len();
        let body = &after[body_start..after.find("<!--Chunk 0 ends-->").unwrap()];
        assert!(body.len() <= 10 + "\n…[truncated]".len(), "neighbor capped");
        assert!(body.ends_with("[truncated]"));
        // The current chunk is emitted in full, never truncated.
        assert!(out[1].content.contains("<!--Chunk 1 begins-->current"));
    }

    /// A max_ctx_chars of 0 elides neighbors entirely (title-only + current chunk).
    #[test]
    fn zero_cap_elides_neighbors() {
        let config = ContextInjectionConfig::new().with_max_ctx_chars(0);
        let injector = ContextInjector::with_config(config);
        let chunks = vec![chunk("a", 0), chunk("b", 1)];
        let out = injector.inject_with_title("T", &chunks);

        assert!(!out[1].content.contains("<!--Chunk 0 begins-->a"));
        assert!(out[1].content.contains("b"));
    }

    /// id + all original metadata survive; the marker is added.
    #[test]
    fn preserves_id_and_metadata_and_marks() {
        let injector = ContextInjector::new();
        let chunks = vec![doc("payload", "id-123", "The Title")];
        let out = injector.inject_with_title("The Title", &chunks);

        assert_eq!(out[0].id.as_deref(), Some("id-123"));
        assert_eq!(
            out[0]
                .metadata
                .get(PARENT_TITLE_KEY)
                .and_then(|v| v.as_str()),
            Some("The Title")
        );
        assert!(out[0].metadata.contains_key(CONTEXT_INJECTED_KEY));
        assert_eq!(
            out[0].metadata.get("source_flag").and_then(|v| v.as_str()),
            None
        );
    }

    /// A second pass over injected chunks is a no-op (content byte-identical).
    #[test]
    fn idempotent_rerun_is_noop() {
        let injector = ContextInjector::new();
        let chunks = vec![chunk("a", 0), chunk("b", 1)];
        let once = injector.inject_with_title("T", &chunks);
        assert!(once[0].content.contains("<!--Chunk 1 begins-->b"));

        let twice = injector.inject_with_title("T", &once);
        for (first, second) in once.iter().zip(twice.iter()) {
            assert_eq!(first.content, second.content, "no double-wrap on rerun");
        }
    }

    /// When no title metadata is present, neighbors are still injected with an empty title.
    #[test]
    fn missing_title_still_injects_neighbors() {
        let injector = ContextInjector::new();
        let chunks = vec![chunk("a", 0), chunk("b", 1)];
        let out = injector.inject(&chunks);

        assert!(out[0].content.contains("<!--Chunk 1 begins-->b"));
        assert!(!out[0].content.contains("<document_chunk_title>"));
    }

    /// Adjacent context for window=2 includes two neighbors on each side where present.
    #[test]
    fn multi_window_neighbors() {
        let config = ContextInjectionConfig::new().with_window(2);
        let injector = ContextInjector::with_config(config);
        let chunks: Vec<Document> = (0..5).map(|i| chunk(&format!("c{i}"), i)).collect();
        let out = injector.inject_with_title("T", &chunks);

        // Middle chunk (idx 2) sees chunks 0..4.
        let mid = &out[2].content;
        for i in 0..5 {
            assert!(mid.contains(&format!("<!--Chunk {} begins-->c{}", i, i)));
        }
    }
}
