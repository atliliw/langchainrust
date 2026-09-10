// lc-rag/src/late_chunking.rs
//! Late chunking: embed the whole document token by token first, then pool
//! along chunk boundaries (0.21.0 S5.2).
//!
//! Early (classic) chunking embeds each chunk independently, so cross-chunk
//! references ("the company", "this plan") embed without their antecedent and
//! fail to match queries phrased against the antecedent. Late chunking keeps
//! the full document in one embedding pass (preserving cross-token attention)
//! and only *pools* per chunk afterwards 鈥?the query flow is unchanged.
//!
//! Requires a token-level embedder ([`TokenLevelEmbeddings`], e.g.
//! Qwen3-Embedding / BGE-M3 / Jina v3); pooled-only models cannot implement
//! it. Best paired with large chunks (one document per pass).

use lc_embeddings::token_level::{TokenEmbedding, TokenLevelEmbeddings};
use lc_embeddings::EmbeddingError;

/// Configuration for late chunking.
#[derive(Debug, Clone)]
pub struct LateChunkConfig {
    /// Chunk size in bytes (chunks are byte windows over the document).
    pub chunk_size: usize,
    /// Overlap between consecutive chunks in bytes.
    pub chunk_overlap: usize,
}

impl Default for LateChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            chunk_overlap: 128,
        }
    }
}

impl LateChunkConfig {
    /// Creates a config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the chunk size in bytes.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Sets the chunk overlap in bytes.
    pub fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.chunk_overlap = chunk_overlap;
        self
    }

    /// Validates the config: `chunk_overlap < chunk_size` (otherwise chunks
    /// would not advance).
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.chunk_size == 0 {
            return Err(EmbeddingError::Config(
                "late chunking: chunk_size must be > 0".to_string(),
            ));
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(EmbeddingError::Config(format!(
                "late chunking: chunk_overlap ({}) must be < chunk_size ({})",
                self.chunk_overlap, self.chunk_size
            )));
        }
        Ok(())
    }

    /// Computes the chunk byte ranges `[start, end)` over `text`.
    ///
    /// Ranges are monotonically increasing, non-empty, cover the whole
    /// document, and — 0.22.0 C6 fix — are always **snapped to UTF-8 char
    /// boundaries** (the window advances in bytes for stable sizing, then
    /// edges snap inward; slicing `text[start..end]` can no longer panic on
    /// CJK / emoji text whose character straddles a window edge).
    pub fn chunk_ranges(&self, text: &str) -> Vec<(usize, usize)> {
        let text_len = text.len();
        let mut ranges = Vec::new();
        let step = self.chunk_size - self.chunk_overlap;
        let mut start = 0usize;
        while start < text_len {
            let raw_end = (start + self.chunk_size).min(text_len);
            // Snap the window edges inward to char boundaries.
            let mut end = prev_char_boundary(text, raw_end);
            if end <= start {
                // Degenerate snap (e.g. start inside a wide char): push the
                // end forward instead so the range is never empty.
                end = next_char_boundary(text, raw_end).min(text_len);
            }
            if end <= start {
                break;
            }
            ranges.push((start, end));
            if end >= text_len {
                break;
            }
            let next = next_char_boundary(text, (start + step).min(text_len));
            if next <= start {
                break;
            }
            start = next;
        }
        ranges
    }
}

/// Next UTF-8 char boundary at or after `pos` (clamped to `text.len()`).
fn next_char_boundary(text: &str, pos: usize) -> usize {
    let mut p = pos.min(text.len());
    while p < text.len() && !text.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Previous UTF-8 char boundary at or before `pos`.
fn prev_char_boundary(text: &str, pos: usize) -> usize {
    let mut p = pos.min(text.len());
    while p > 0 && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

/// One late chunk: a byte range of the original document plus its pooled,
/// L2-normalized vector.
#[derive(Debug, Clone, PartialEq)]
pub struct LateChunk {
    /// Byte range `[start, end)` of this chunk in the original document.
    pub range: (usize, usize),
    /// The chunk text (sliced from the original document).
    pub text: String,
    /// Mean-pooled, L2-normalized chunk vector.
    pub vector: Vec<f32>,
}

/// Mean-pools the token vectors whose spans intersect `[range_start, range_end)`
/// into a single L2-normalized vector.
///
/// Pure helper so the pooling math is unit-testable without an embedder.
/// Tokens spanning a boundary contribute to both sides 鈥?that is the point of
/// late chunking (context leaks across chunk borders by design). Returns
/// `Err(EmptyInput)` when no token intersects (caller-side bug, not a valid
/// chunk).
pub fn pool_tokens(
    tokens: &[TokenEmbedding],
    range_start: usize,
    range_end: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    let dim = tokens.first().map(|t| t.vector.len()).unwrap_or(0);
    let mut pooled = vec![0.0f32; dim];
    let mut count = 0usize;
    for token in tokens {
        if token.span.intersects(range_start, range_end) {
            for (p, v) in pooled.iter_mut().zip(token.vector.iter()) {
                *p += v;
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err(EmbeddingError::EmptyInput);
    }
    for p in &mut pooled {
        *p /= count as f32;
    }
    lc_embeddings::l2_normalize(&mut pooled);
    Ok(pooled)
}

/// Runs late chunking over a whole document.
///
/// 1. one token-level embedding pass over the full text (single model call);
/// 2. slide a byte window ([`LateChunkConfig`]) over the token list;
/// 3. pool each window into a chunk vector.
///
/// Generic over the embedder (static dispatch 鈥?the [`TokenLevelEmbeddings`]
/// trait is RPITIT-based and not dyn-compatible by design; see its module docs).
pub async fn late_chunk<E: TokenLevelEmbeddings>(
    embedder: &E,
    text: &str,
    config: &LateChunkConfig,
) -> Result<Vec<LateChunk>, EmbeddingError> {
    config.validate()?;
    let tokens = embedder.embed_tokens(text).await?;
    let mut chunks = Vec::new();
    for (start, end) in config.chunk_ranges(text) {
        let vector = pool_tokens(&tokens, start, end)?;
        chunks.push(LateChunk {
            range: (start, end),
            text: text[start..end].to_string(),
            vector,
        });
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::token_level::{TokenEmbedding, TokenSpan};

    /// Whitespace tokenizer over fixed-size vectors 鈥?same shape as the
    /// `lc-embeddings` test mock; keeps pooling tests model-free.
    fn token_embeddings(text: &str) -> Vec<TokenEmbedding> {
        let mut out = Vec::new();
        let mut cursor = 0usize;
        for word in text.split_whitespace() {
            let start = text[cursor..]
                .find(word)
                .map(|p| cursor + p)
                .unwrap_or(cursor);
            let end = start + word.len();
            cursor = end;
            out.push(TokenEmbedding {
                span: TokenSpan::new(start, end),
                vector: vec![word.bytes().map(|b| b as f32).sum::<f32>(), 1.0],
            });
        }
        out
    }

    #[test]
    fn config_validates_overlap() {
        assert!(LateChunkConfig::new().validate().is_ok());
        let bad = LateChunkConfig {
            chunk_size: 10,
            chunk_overlap: 10,
        };
        assert!(bad.validate().is_err());
        let bad = LateChunkConfig {
            chunk_size: 0,
            chunk_overlap: 0,
        };
        assert!(bad.validate().is_err());
    }

    /// Sliding windows are monotonically increasing and cover the document.
    #[test]
    fn chunk_ranges_cover_document() {
        let config = LateChunkConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        // ASCII-only document of 25 bytes: boundaries are identity.
        let text_ascii = "x".repeat(25);
        let ranges = config.chunk_ranges(&text_ascii);
        // step 8: [0,10) [8,18) [16,25) — monotone, cover everything.
        assert_eq!(ranges, vec![(0, 10), (8, 18), (16, 25)]);
    }

    #[test]
    fn chunk_ranges_shorter_than_chunk_size() {
        let config = LateChunkConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        assert_eq!(config.chunk_ranges("xxxxx"), vec![(0, 5)]);
    }

    /// C6 fix: a CJK window edge inside a multi-byte character snaps to a
    /// char boundary — slicing `text[start..end]` never panics.
    #[test]
    fn chunk_ranges_snap_to_char_boundaries() {
        let config = LateChunkConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        // 6 CJK chars = 18 bytes; step 8 would slice at byte 8 (mid-char).
        let text = "你好世界天地".to_string(); // 3-byte chars, 18 bytes
        let ranges = config.chunk_ranges(&text);
        assert!(!ranges.is_empty());
        for (start, end) in &ranges {
            assert!(text.is_char_boundary(*start), "start {start} not a boundary");
            assert!(text.is_char_boundary(*end), "end {end} not a boundary");
            // Slicing is the regression: this would panic before the fix.
            let _ = &text[*start..*end];
        }
        // Coverage: the last range reaches the end of the document.
        assert_eq!(ranges.last().unwrap().1, 18);
    }

    /// Pooling averages intersecting tokens and L2-normalizes the result.
    #[test]
    fn pool_tokens_averages_and_normalizes() {
        let tokens = token_embeddings("alpha beta");
        // "alpha" = [98+108+112+104+97=519, 1], "beta" = [98+101+116+97=412, 1].
        let pooled = pool_tokens(&tokens, 0, 10).unwrap();
        let mean0: f32 = (519.0 + 412.0) / 2.0;
        let norm = (mean0 * mean0 + 1.0).sqrt();
        assert!((pooled[0] - mean0 / norm).abs() < 1e-5);
        assert!((pooled[1] - 1.0 / norm).abs() < 1e-5);
        let norm_sq: f32 = pooled.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5, "L2-normalized");
    }

    /// Tokens spanning a chunk boundary contribute to both chunks (late
    /// chunking's context-leak property).
    #[test]
    fn pool_tokens_boundary_leaks() {
        let tokens = token_embeddings("abcdef");
        // Chunk A covers only "abc", chunk B covers only "def", but both meet
        // at the "c"/"d" boundary; a token spanning [2, 5) intersects both.
        let spanning = vec![TokenEmbedding {
            span: TokenSpan::new(2, 5),
            vector: vec![1.0, 0.0],
        }];
        assert!(pool_tokens(&spanning, 0, 3).is_ok());
        assert!(pool_tokens(&spanning, 3, 6).is_ok());
        let _ = tokens; // whitespace tokenizer has no spanning token; boundary check above suffices
    }

    /// Pooling a range with no intersecting token is an explicit error.
    #[test]
    fn pool_tokens_empty_range_errors() {
        let tokens = token_embeddings("alpha");
        let err = pool_tokens(&tokens, 100, 200).unwrap_err();
        assert!(matches!(err, EmbeddingError::EmptyInput));
    }

    /// End-to-end with a generic (static-dispatch) embedder: one model pass,
    /// per-chunk pooled vectors, text sliced from the original document.
    #[tokio::test]
    async fn late_chunk_end_to_end() {
        struct MockTokenEmbeddings;

        impl lc_embeddings::token_level::TokenLevelEmbeddings for MockTokenEmbeddings {
            async fn embed_tokens(
                &self,
                text: &str,
            ) -> Result<Vec<TokenEmbedding>, EmbeddingError> {
                Ok(token_embeddings(text))
            }
        }

        let text = "alpha beta gamma delta epsilon";
        let config = LateChunkConfig {
            chunk_size: 16,
            chunk_overlap: 0,
        };
        let chunks = late_chunk(&MockTokenEmbeddings, text, &config)
            .await
            .unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "alpha beta gamma");
        assert_eq!(chunks[1].text, " delta epsilon");
        // Each chunk vector is L2-normalized.
        for chunk in &chunks {
            let norm_sq: f32 = chunk.vector.iter().map(|v| v * v).sum();
            assert!((norm_sq - 1.0).abs() < 1e-4);
        }
        // The second chunk is pooled from "delta"/"epsilon" only.
        let delta_sum = b"delta".iter().map(|b| *b as f32).sum::<f32>();
        let eps_sum: f32 = b"epsilon".iter().map(|b| *b as f32).sum();
        let mean = (delta_sum + eps_sum) / 2.0;
        let norm = (mean * mean + 1.0f32).sqrt();
        assert!((chunks[1].vector[0] - mean / norm).abs() < 1e-4);
    }

    /// Invalid config fails fast before any model call.
    #[tokio::test]
    async fn late_chunk_rejects_invalid_config() {
        struct MockTokenEmbeddings;

        impl lc_embeddings::token_level::TokenLevelEmbeddings for MockTokenEmbeddings {
            async fn embed_tokens(
                &self,
                _text: &str,
            ) -> Result<Vec<TokenEmbedding>, EmbeddingError> {
                Ok(Vec::new())
            }
        }
        let config = LateChunkConfig {
            chunk_size: 8,
            chunk_overlap: 8,
        };
        let err = late_chunk(&MockTokenEmbeddings, "text", &config)
            .await
            .unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }
}
