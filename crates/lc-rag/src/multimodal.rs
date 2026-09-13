// lc-rag/src/multimodal.rs
//! Multimodal (image + text) chunking and retrieval (B7, v0.22.4).
//!
//! A cross-modal embedding model ([`VisionEmbeddings`]) maps images **and**
//! text into one shared vector space. This module closes the RAG loop:
//!
//! 1. [`MultimodalChunker`] turns an ordered mix of text blocks and image
//!    assets into modality-tagged [`Document`]s;
//! 2. [`MultimodalRetriever`] embeds image documents with
//!    [`VisionEmbeddings::embed_image`] and text documents with
//!    [`VisionEmbeddings::embed_text`], stores them through any
//!    [`VectorStore`], and answers plain-text queries with a mixed
//!    image/text result set;
//! 3. modality filtering (`mm_kind` metadata) lets callers retrieve images
//!    only or text only; image queries are supported via
//!    [`MultimodalRetriever::retrieve_by_image`].
//!
//! # Metadata convention
//!
//! Every emitted document carries an `mm_kind` tag (`"image"` / `"text"`);
//! image documents additionally carry `mm_url`, `mm_caption`, and `mm_mime`.
//! Keys are exported as constants so downstream filters stay in sync.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lc_embeddings::{ImageInput, VisionEmbeddings};
use lc_vector_stores::{Document, FilterOp, MetadataFilter, SearchResult, VectorStore};
use serde_json::Value;

use crate::retriever::{RetrieverError, RetrieverTrait};

/// Metadata key marking the document modality (`"image"` / `"text"`).
pub const MM_KIND_KEY: &str = "mm_kind";
/// Metadata key holding the image reference (http(s) URL or data URI).
pub const MM_URL_KEY: &str = "mm_url";
/// Metadata key holding the image caption / alt text.
pub const MM_CAPTION_KEY: &str = "mm_caption";
/// Metadata key holding the image MIME type.
pub const MM_MIME_KEY: &str = "mm_mime";

/// `mm_kind` value for image documents.
pub const MM_KIND_IMAGE: &str = "image";
/// `mm_kind` value for text documents.
pub const MM_KIND_TEXT: &str = "text";

/// One ordered block of a multimodal source document.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MediaBlock {
    /// A text block (may be split into several chunks).
    Text(String),
    /// An image asset with an optional caption.
    Image(ImageAsset),
}

/// An image reference plus optional descriptive metadata.
#[derive(Debug, Clone, Default)]
pub struct ImageAsset {
    /// Image reference: an http(s) URL the embedding provider can fetch, or
    /// a complete `data:` URI carrying inline base64 bytes.
    pub url: String,
    /// Caption / alt text. Stored verbatim as the document content so that
    /// lexical tooling (BM25, keyword filters) stays functional.
    pub caption: Option<String>,
    /// Image MIME type (e.g. `image/png`). Informational for URL references,
    /// required when the reference is inline bytes on inline-only backends.
    pub mime_type: Option<String>,
}

impl ImageAsset {
    /// Creates an image asset from a reference (URL or data URI).
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            caption: None,
            mime_type: None,
        }
    }

    /// Attaches a caption.
    pub fn with_caption(mut self, caption: impl Into<String>) -> Self {
        self.caption = Some(caption.into());
        self
    }

    /// Attaches a MIME type.
    pub fn with_mime(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }
}

/// Configuration for [`MultimodalChunker`].
#[derive(Debug, Clone, Default)]
pub struct MultimodalChunkConfig {
    /// Maximum characters per text chunk. `None` keeps each text block as a
    /// single document. Splitting is UTF-8 boundary-safe (no mid-`char` cuts).
    pub text_chunk_chars: Option<usize>,
    /// Metadata inherited by every emitted document (source, page, …). The
    /// `mm_*` keys are reserved and overwrite inherited values.
    pub common_metadata: HashMap<String, Value>,
}

impl MultimodalChunkConfig {
    /// Creates config with a fixed maximum text-chunk length.
    pub fn with_chunk_chars(mut self, chunk_chars: usize) -> Self {
        self.text_chunk_chars = Some(chunk_chars.max(1));
        self
    }

    /// Adds one inherited metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.common_metadata.insert(key.into(), value.into());
        self
    }
}

/// Turns ordered [`MediaBlock`]s into modality-tagged [`Document`]s.
#[derive(Debug, Clone, Default)]
pub struct MultimodalChunker {
    config: MultimodalChunkConfig,
}

impl MultimodalChunker {
    /// Creates a chunker with default config (one document per text block).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a chunker with explicit config.
    pub fn with_config(config: MultimodalChunkConfig) -> Self {
        Self { config }
    }

    /// Chunks the blocks into [`Document`]s, preserving input order.
    ///
    /// - Empty/whitespace text blocks are skipped;
    /// - text blocks are optionally split into UTF-8-safe windows;
    /// - images always become one document each; an image without a caption
    ///   still embeds (its content is empty), but [`RetrieverError`] is never
    ///   raised at chunk time — validation happens at embedding time.
    pub fn chunk(&self, blocks: &[MediaBlock]) -> Vec<Document> {
        let mut documents = Vec::with_capacity(blocks.len());
        for block in blocks {
            match block {
                MediaBlock::Text(text) if text.trim().is_empty() => continue,
                MediaBlock::Text(text) => {
                    for piece in split_text(text, self.config.text_chunk_chars) {
                        documents.push(self.text_document(piece));
                    }
                }
                MediaBlock::Image(asset) => documents.push(self.image_document(asset)),
            }
        }
        documents
    }

    /// Builds a tagged text document directly.
    pub fn text_document(&self, content: impl Into<String>) -> Document {
        let mut doc = Document::new(content);
        for (key, value) in &self.config.common_metadata {
            doc.metadata.insert(key.clone(), value.clone());
        }
        doc.metadata
            .insert(MM_KIND_KEY.to_string(), Value::from(MM_KIND_TEXT));
        doc
    }

    /// Builds a tagged image document directly.
    pub fn image_document(&self, asset: &ImageAsset) -> Document {
        let mut doc = Document::new(asset.caption.clone().unwrap_or_default());
        for (key, value) in &self.config.common_metadata {
            doc.metadata.insert(key.clone(), value.clone());
        }
        doc.metadata
            .insert(MM_KIND_KEY.to_string(), Value::from(MM_KIND_IMAGE));
        doc.metadata
            .insert(MM_URL_KEY.to_string(), Value::from(asset.url.clone()));
        if let Some(caption) = &asset.caption {
            doc.metadata
                .insert(MM_CAPTION_KEY.to_string(), Value::from(caption.clone()));
        }
        if let Some(mime) = &asset.mime_type {
            doc.metadata
                .insert(MM_MIME_KEY.to_string(), Value::from(mime.clone()));
        }
        doc
    }
}

/// Which modality a multimodal retrieval call may return.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModalityFilter {
    /// Both images and text (default).
    #[default]
    Any,
    /// Image documents only.
    Images,
    /// Text documents only.
    Texts,
}

impl ModalityFilter {
    fn metadata_filter(self) -> Option<MetadataFilter> {
        match self {
            ModalityFilter::Any => None,
            ModalityFilter::Images => Some(MetadataFilter::field(
                MM_KIND_KEY,
                FilterOp::Eq,
                MM_KIND_IMAGE,
            )),
            ModalityFilter::Texts => Some(MetadataFilter::field(
                MM_KIND_KEY,
                FilterOp::Eq,
                MM_KIND_TEXT,
            )),
        }
    }
}

/// Retriever over a shared image/text vector space.
pub struct MultimodalRetriever {
    store: Arc<dyn VectorStore>,
    vision: Arc<dyn VisionEmbeddings>,
}

impl std::fmt::Debug for MultimodalRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultimodalRetriever")
            .field("model", &self.vision.model_name())
            .field("dimension", &self.vision.dimension())
            .finish()
    }
}

impl MultimodalRetriever {
    /// Creates a multimodal retriever over `store`, embedding both modalities
    /// with the shared-space `vision` model.
    pub fn new(store: Arc<dyn VectorStore>, vision: Arc<dyn VisionEmbeddings>) -> Self {
        Self { store, vision }
    }

    /// Retrieves with a text query restricted to `modality`.
    pub async fn retrieve_modality(
        &self,
        query: &str,
        k: usize,
        modality: ModalityFilter,
    ) -> Result<Vec<Document>, RetrieverError> {
        let results = self.search_text(query, k, modality).await?;
        Ok(results.into_iter().map(|r| r.document).collect())
    }

    /// Retrieves with a text query, returning scores and modality metadata.
    pub async fn retrieve_with_scores_modality(
        &self,
        query: &str,
        k: usize,
        modality: ModalityFilter,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        self.search_text(query, k, modality).await
    }

    /// Retrieves with an image query (image→image / image→text search).
    pub async fn retrieve_by_image(
        &self,
        image: &ImageInput,
        k: usize,
        modality: ModalityFilter,
    ) -> Result<Vec<Document>, RetrieverError> {
        let query_embedding = self
            .vision
            .embed_image(image)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;
        let results = self.search_vector(&query_embedding, k, modality).await?;
        Ok(results.into_iter().map(|r| r.document).collect())
    }

    async fn search_text(
        &self,
        query: &str,
        k: usize,
        modality: ModalityFilter,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        let query_embedding = self
            .vision
            .embed_text(query)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;
        self.search_vector(&query_embedding, k, modality).await
    }

    async fn search_vector(
        &self,
        query_embedding: &[f32],
        k: usize,
        modality: ModalityFilter,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        let results = match modality.metadata_filter() {
            None => self.store.similarity_search(query_embedding, k).await?,
            Some(filter) => {
                self.store
                    .similarity_search_with_filter(query_embedding, k, Some(&filter))
                    .await?
            }
        };

        // Belt-and-braces: the trait's default filtered method errors rather
        // than ignoring the filter, but a custom backend that silently ignores
        // it would otherwise leak the wrong modality into a narrowed query.
        Ok(results
            .into_iter()
            .filter(|r| match modality {
                ModalityFilter::Any => true,
                ModalityFilter::Images => {
                    r.document.metadata.get(MM_KIND_KEY).and_then(Value::as_str)
                        == Some(MM_KIND_IMAGE)
                }
                ModalityFilter::Texts => {
                    r.document.metadata.get(MM_KIND_KEY).and_then(Value::as_str)
                        == Some(MM_KIND_TEXT)
                }
            })
            .collect())
    }

    /// Embeds a mixed document batch: one batched image call, text one-by-one
    /// (the trait has no text batch primitive), then restores request order.
    async fn embed_documents_mixed(
        &self,
        documents: &[Document],
    ) -> Result<Vec<Vec<f32>>, RetrieverError> {
        let mut image_slots = Vec::new();
        let mut image_inputs = Vec::new();
        let mut text_slots = Vec::new();

        for (index, doc) in documents.iter().enumerate() {
            match doc.metadata.get(MM_KIND_KEY).and_then(Value::as_str) {
                Some(MM_KIND_IMAGE) => {
                    let reference = doc
                        .metadata
                        .get(MM_URL_KEY)
                        .and_then(Value::as_str)
                        .filter(|url| !url.trim().is_empty())
                        .ok_or_else(|| {
                            RetrieverError::InvalidDocument(format!(
                                "image document {index} is missing {MM_URL_KEY}"
                            ))
                        })?;
                    image_slots.push(index);
                    image_inputs.push(parse_image_reference(reference));
                }
                // Untagged documents are treated as text — the historical default.
                Some(MM_KIND_TEXT) | None => {
                    if doc.content.trim().is_empty() {
                        return Err(RetrieverError::EmbeddingError(format!(
                            "text document {index} has empty content"
                        )));
                    }
                    text_slots.push(index);
                }
                Some(other) => {
                    return Err(RetrieverError::InvalidDocument(format!(
                        "unknown {MM_KIND_KEY} value {other:?} on document {index}"
                    )));
                }
            }
        }

        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; documents.len()];

        if !image_inputs.is_empty() {
            let image_vectors = self
                .vision
                .embed_images(&image_inputs)
                .await
                .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;
            for (slot, vector) in image_slots.into_iter().zip(image_vectors) {
                vectors[slot] = Some(vector);
            }
        }

        for slot in text_slots {
            vectors[slot] = Some(
                self.vision
                    .embed_text(&documents[slot].content)
                    .await
                    .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?,
            );
        }

        // Every slot was filled or the function returned early above.
        Ok(vectors.into_iter().map(Option::unwrap).collect())
    }
}

#[async_trait]
impl RetrieverTrait for MultimodalRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        self.retrieve_modality(query, k, ModalityFilter::Any).await
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        self.retrieve_with_scores_modality(query, k, ModalityFilter::Any)
            .await
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        if documents.is_empty() {
            return Ok(());
        }
        let embeddings = self.embed_documents_mixed(&documents).await?;
        self.store.add_documents(documents, embeddings).await?;
        Ok(())
    }
}

/// Converts a stored reference (`mm_url`) into an [`ImageInput`]: data URIs are
/// marked inline, everything else passes through as a provider-fetched URL.
pub(crate) fn parse_image_reference(reference: &str) -> ImageInput {
    if reference.starts_with("data:") {
        ImageInput::from_data_uri(reference)
    } else {
        ImageInput::from_url(reference)
    }
}

/// UTF-8-safe fixed-width window splitter.
///
/// `None` or zero-sized windows return the text as one (trimmed, non-empty)
/// piece. Windows are cut on whitespace boundaries when one falls within the
/// last 20% of the window, otherwise at a safe char boundary.
fn split_text(text: &str, chunk_chars: Option<usize>) -> Vec<String> {
    let Some(chunk_chars) = chunk_chars.filter(|n| *n > 0) else {
        return vec![text.to_string()];
    };
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= chunk_chars {
        return vec![text.to_string()];
    }

    let mut pieces = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let mut end = (start + chunk_chars).min(chars.len());
        if end < chars.len() {
            // Prefer a whitespace boundary in the trailing part of the window.
            let look_back = start + (chunk_chars * 4 / 5);
            if let Some(space) = (look_back..end).rfind(|i| chars[*i].is_whitespace()) {
                end = space;
            } else if !chars[end].is_whitespace() {
                // Otherwise avoid leaving a dangling whitespace at the boundary.
                end = chars[..end]
                    .iter()
                    .rposition(|c| c.is_whitespace())
                    .filter(|p| *p > start)
                    .unwrap_or(end);
            }
        }
        let piece: String = chars[start..end].iter().collect();
        let trimmed = piece.trim();
        if !trimmed.is_empty() {
            pieces.push(trimmed.to_string());
        }
        if end <= start {
            // Defensive: guarantee forward progress on pathological input.
            end = start + 1;
        }
        start = end;
        while start < chars.len() && chars[start].is_whitespace() {
            start += 1;
        }
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::MockVisionEmbeddings;
    use lc_vector_stores::InMemoryVectorStore;

    fn sample_blocks() -> Vec<MediaBlock> {
        vec![
            MediaBlock::Text("a cat sat on the mat".into()),
            MediaBlock::Image(
                ImageAsset::new("https://cdn.example.com/cat.png")
                    .with_caption("a photo of a cat")
                    .with_mime("image/png"),
            ),
            MediaBlock::Text("the dog ran in the park".into()),
            MediaBlock::Image(
                ImageAsset::new("data:image/jpeg;base64,amVlZw").with_caption("a photo of a dog"),
            ),
        ]
    }

    #[test]
    fn chunker_tags_modality_and_preserves_order() {
        let config = MultimodalChunkConfig::default().with_metadata("source", "catalog");
        let chunker = MultimodalChunker::with_config(config);
        let docs = chunker.chunk(&sample_blocks());
        assert_eq!(docs.len(), 4);
        assert_eq!(kind(&docs[0]), Some(MM_KIND_TEXT));
        assert_eq!(docs[0].content, "a cat sat on the mat");
        assert_eq!(
            docs[0].metadata.get("source").and_then(Value::as_str),
            Some("catalog")
        );

        assert_eq!(kind(&docs[1]), Some(MM_KIND_IMAGE));
        assert_eq!(
            docs[1].metadata.get(MM_URL_KEY).and_then(Value::as_str),
            Some("https://cdn.example.com/cat.png")
        );
        assert_eq!(
            docs[1].metadata.get(MM_CAPTION_KEY).and_then(Value::as_str),
            Some("a photo of a cat")
        );
        // Caption is the document content, keeping lexical tooling usable.
        assert_eq!(docs[1].content, "a photo of a cat");

        assert_eq!(kind(&docs[2]), Some(MM_KIND_TEXT));
        assert_eq!(kind(&docs[3]), Some(MM_KIND_IMAGE));
        assert_eq!(
            docs[3].metadata.get(MM_URL_KEY).and_then(Value::as_str),
            Some("data:image/jpeg;base64,amVlZw")
        );
    }

    #[test]
    fn chunker_skips_blank_text_and_splits_long_blocks_safely() {
        let config = MultimodalChunkConfig::default().with_chunk_chars(10);
        let chunker = MultimodalChunker::with_config(config);
        let blocks = vec![
            MediaBlock::Text("   ".into()),
            MediaBlock::Text("abcdefghij klmnopqrst".into()),
        ];
        let docs = chunker.chunk(&blocks);
        assert!(docs.len() >= 2);
        assert!(docs.iter().all(|d| !d.content.trim().is_empty()));
        // No chunk exceeds the configured window (whitespace-trimmed).
        assert!(docs.iter().all(|d| d.content.chars().count() <= 10));
        // All pieces stay valid UTF-8 (they do by construction) and reorder back.
        let joined: String = docs
            .iter()
            .map(|d| d.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(joined.starts_with("abcdefghij"));
    }

    #[test]
    fn unicode_split_does_not_panic_or_split_scalar() {
        let chunker =
            MultimodalChunker::with_config(MultimodalChunkConfig::default().with_chunk_chars(3));
        let docs = chunker.chunk(&[MediaBlock::Text("猫🐶狗🦊兔".into())]);
        assert!(!docs.is_empty());
        let rejoined: String = docs.iter().map(|d| d.content.clone()).collect();
        assert_eq!(rejoined, "猫🐶狗🦊兔");
    }

    fn kind(doc: &Document) -> Option<&str> {
        doc.metadata.get(MM_KIND_KEY).and_then(Value::as_str)
    }

    /// Aligns mock vectors so that text "…cat…" and the cat image share one
    /// direction, and likewise for dog.
    fn aligned_vision() -> Arc<dyn VisionEmbeddings> {
        let vision = MockVisionEmbeddings::new(4);
        vision.with_text_vector("a cat sat on the mat", vec![1.0, 0.0, 0.0, 0.0]);
        vision.with_text_vector("the dog ran in the park", vec![0.0, 1.0, 0.0, 0.0]);
        vision.with_image_vector(
            &ImageInput::from_url("https://cdn.example.com/cat.png"),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        vision.with_image_vector(
            &ImageInput::from_data_uri("data:image/jpeg;base64,amVlZw"),
            vec![0.0, 1.0, 0.0, 0.0],
        );
        Arc::new(vision)
    }

    /// B7 图文混合检索连通: text query over a mixed image/text corpus returns
    /// cross-modal matches; modality filters and image queries work as well.
    #[tokio::test]
    async fn mixed_image_text_retrieval_is_connected() {
        let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        let vision = aligned_vision();
        let retriever = MultimodalRetriever::new(store.clone(), vision);

        let documents = MultimodalChunker::new().chunk(&sample_blocks());
        retriever.add_documents(documents).await.unwrap();
        assert_eq!(store.count().await, 4);

        // Text query "cat" shares its space with the cat TEXT and cat IMAGE.
        let results = retriever
            .retrieve_with_scores_modality("a cat sat on the mat", 2, ModalityFilter::Any)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        let kinds: Vec<&str> = results.iter().map(|r| kind(&r.document).unwrap()).collect();
        assert!(
            kinds.contains(&MM_KIND_IMAGE) && kinds.contains(&MM_KIND_TEXT),
            "expected one image + one text hit, got {kinds:?}"
        );
        assert!((results[0].score - 1.0).abs() < 1e-5);

        // Images-only filter: text docs never leak through.
        let images = retriever
            .retrieve_modality("a cat sat on the mat", 5, ModalityFilter::Images)
            .await
            .unwrap();
        assert_eq!(images.len(), 2);
        assert!(images.iter().all(|d| kind(d) == Some(MM_KIND_IMAGE)));
        assert_eq!(
            images[0].metadata.get(MM_URL_KEY).and_then(Value::as_str),
            Some("https://cdn.example.com/cat.png")
        );

        // Texts-only filter: both text docs are eligible, the matching one ranks first.
        let texts = retriever
            .retrieve_modality("the dog ran in the park", 5, ModalityFilter::Texts)
            .await
            .unwrap();
        assert_eq!(texts.len(), 2);
        assert!(texts.iter().all(|d| kind(d) == Some(MM_KIND_TEXT)));
        assert!(texts[0].content.contains("dog"));

        // Image→everything query with the dog picture hits the dog side.
        let dog_image = ImageInput::from_data_uri("data:image/jpeg;base64,amVlZw");
        let by_image = retriever
            .retrieve_by_image(&dog_image, 1, ModalityFilter::Any)
            .await
            .unwrap();
        assert_eq!(by_image.len(), 1);
        let hit = &by_image[0];
        assert!(
            hit.content.contains("dog")
                || hit
                    .metadata
                    .get(MM_URL_KEY)
                    .and_then(Value::as_str)
                    .is_some_and(|u| u.contains("amVlZw"))
        );
    }

    #[tokio::test]
    async fn missing_image_reference_is_an_explicit_error() {
        let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        let retriever = MultimodalRetriever::new(store, aligned_vision());

        let bad = Document::new("broken image")
            .with_metadata(MM_KIND_KEY, MM_KIND_IMAGE)
            .with_metadata(MM_URL_KEY, "  ");
        let err = retriever.add_documents(vec![bad]).await.unwrap_err();
        assert!(matches!(err, RetrieverError::InvalidDocument(_)));
    }

    #[tokio::test]
    async fn works_as_retriever_trait_object() {
        let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        let retriever: Arc<dyn RetrieverTrait> =
            Arc::new(MultimodalRetriever::new(store, aligned_vision()));

        let documents = MultimodalChunker::new().chunk(&sample_blocks());
        retriever.add_documents(documents).await.unwrap();
        let hits = retriever.retrieve("a cat sat on the mat", 1).await.unwrap();
        assert_eq!(hits.len(), 1);
    }
}
