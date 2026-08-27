// src/retrieval/semantic_splitter.rs
//! Semantic chunking
//!
//! Splits text by semantic relevance: sentences are embedded first, and a chunk boundary is
//! cut where the cosine similarity between adjacent sentences drops sharply. Compared to
//! character-level splitting this preserves semantic integrity better and improves retrieval.
//!
//! Note: embedding is an async operation while the `TextSplitter` trait has a sync signature.
//! To keep the existing sync trait intact, this chunker exposes standalone async interfaces
//! `split_text` / `split_document` instead of implementing the sync `TextSplitter`.

use lc_embeddings::{cosine_similarity, EmbeddingError, Embeddings};
use lc_vector_stores::Document;

/// Semantic chunker
///
/// Cuts a chunk where the similarity between adjacent sentences drops below
/// `breakpoint_threshold`; forces a break when the accumulated length exceeds
/// `max_chunk_size`.
pub struct SemanticSplitter<E> {
    embeddings: E,
    /// Break a chunk when adjacent-sentence similarity drops below this threshold
    breakpoint_threshold: f32,
    /// Maximum characters per chunk; exceeding it forces a break
    max_chunk_size: usize,
}

impl<E: Embeddings> SemanticSplitter<E> {
    /// Creates a semantic chunker
    ///
    /// # Arguments
    /// * `embeddings` - the embedding model
    /// * `breakpoint_threshold` - the breakpoint threshold for adjacent-sentence similarity
    ///   (0.0–1.0; lower is harder to break)
    /// * `max_chunk_size` - the maximum characters per chunk
    pub fn new(embeddings: E, breakpoint_threshold: f32, max_chunk_size: usize) -> Self {
        Self {
            embeddings,
            breakpoint_threshold,
            max_chunk_size,
        }
    }

    /// Creates with default parameters (threshold=0.5, max=1000)
    pub fn with_defaults(embeddings: E) -> Self {
        Self::new(embeddings, 0.5, 1000)
    }

    /// Splits into sentences: supports Chinese and English sentence-final punctuation
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if matches!(ch, '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?') {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }
        sentences
    }

    /// Chunks text asynchronously
    pub async fn split_text(&self, text: &str) -> Result<Vec<String>, EmbeddingError> {
        let sentences = Self::split_sentences(text);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }
        if sentences.len() == 1 {
            return Ok(vec![sentences.into_iter().next().unwrap_or_default()]);
        }

        // Embed in batch
        let refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let embeddings = self.embeddings.embed_documents(&refs).await?;

        let mut chunks = Vec::new();
        let mut current = sentences[0].clone();

        for i in 1..sentences.len() {
            let sim = cosine_similarity(&embeddings[i - 1], &embeddings[i]).unwrap_or(0.0);
            let would_exceed = current.len() + sentences[i].len() + 1 > self.max_chunk_size;

            if sim < self.breakpoint_threshold || would_exceed {
                chunks.push(std::mem::take(&mut current));
                current = sentences[i].clone();
            } else {
                current.push('\n');
                current.push_str(&sentences[i]);
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        Ok(chunks)
    }

    /// Chunks a document asynchronously, preserving metadata (writes the chunk index)
    pub async fn split_document(
        &self,
        document: &Document,
    ) -> Result<Vec<Document>, EmbeddingError> {
        let chunks = self.split_text(&document.content).await?;
        Ok(chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut metadata = document.metadata.clone();
                metadata.insert("chunk".to_string(), i.to_string().into());
                Document {
                    content: chunk,
                    metadata,
                    id: None,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use lc_embeddings::{EmbeddingError, Embeddings, MockEmbeddings};

    /// An embedding that always fails, used to test the error path
    struct FailingEmbeddings;
    #[async_trait]
    impl Embeddings for FailingEmbeddings {
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            Err(EmbeddingError::ApiError("intentional failure".to_string()))
        }
        fn dimension(&self) -> usize {
            32
        }
        fn model_name(&self) -> &str {
            "failing"
        }
    }

    fn splitter(threshold: f32, max_chunk: usize) -> SemanticSplitter<MockEmbeddings> {
        SemanticSplitter::new(MockEmbeddings::new(32), threshold, max_chunk)
    }

    #[tokio::test]
    async fn test_empty_text() {
        let s = splitter(0.5, 1000);
        assert!(s.split_text("").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_single_sentence() {
        let s = splitter(0.5, 1000);
        let chunks = s.split_text("只有一句没有标点").await.unwrap();
        assert_eq!(chunks, vec!["只有一句没有标点".to_string()]);
    }

    #[tokio::test]
    async fn test_chunks_contain_all_sentences() {
        let s = splitter(0.5, 1000);
        let text = "苹果是一种水果。香蕉是黄色的。樱桃很小。";
        let chunks = s.split_text(text).await.unwrap();
        assert!(!chunks.is_empty());
        // However it splits, every sentence should appear in some chunk
        let joined = chunks.join("");
        assert!(joined.contains("苹果是一种水果"));
        assert!(joined.contains("香蕉是黄色的"));
        assert!(joined.contains("樱桃很小"));
    }

    #[tokio::test]
    async fn test_max_chunk_size_enforces_break() {
        // max_chunk is tiny, so multiple sentences are necessarily forced into several chunks
        let s = splitter(0.0, 5);
        let text = "AAAA。BBBB。CCCC。";
        let chunks = s.split_text(text).await.unwrap();
        assert!(
            chunks.len() >= 2,
            "max_chunk=5 应强制断块, 实际 {} 块",
            chunks.len()
        );
    }

    #[tokio::test]
    async fn test_split_document_metadata() {
        let s = splitter(0.0, 5);
        let doc = Document::new("AAAA。BBBB。CCCC。").with_metadata("source", "test");
        let chunks = s.split_document(&doc).await.unwrap();
        assert!(chunks.len() >= 2);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(
                c.metadata.get("chunk"),
                Some(&serde_json::Value::String(i.to_string()))
            );
            assert_eq!(
                c.metadata.get("source"),
                Some(&serde_json::Value::String("test".to_string()))
            );
            assert!(c.id.is_none());
        }
    }

    #[tokio::test]
    async fn test_embedding_failure_returns_error() {
        // M54: embedding failure now returns an error instead of fallback
        let s = SemanticSplitter::new(FailingEmbeddings, 0.5, 1000);
        let text = "句子一。句子二。句子三。";
        let result = s.split_text(text).await;
        assert!(result.is_err());
    }
}
