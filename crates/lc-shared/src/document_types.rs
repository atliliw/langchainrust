// lc-shared/src/document_types.rs
//! Document types shared across crates.
//!
//! These types are needed by both `lc-vector-stores` and `lc-rag`,
//! so they live here to break the circular dependency.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Document structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document content.
    pub content: String,

    /// Document metadata.
    pub metadata: HashMap<String, Value>,

    /// Document ID (optional).
    pub id: Option<String>,
}

impl Document {
    /// Creates a new document.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
            id: None,
        }
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Sets ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Returns page content (alias).
    pub fn page_content(&self) -> &str {
        &self.content
    }
}

/// Vector document with embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    /// Document.
    pub document: Document,

    /// Embedding vector.
    pub embedding: Vec<f32>,
}

/// Search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document.
    pub document: Document,

    /// Similarity score.
    pub score: f32,
}

/// Chunk document (split document fragment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDocument {
    /// Chunk ID
    pub chunk_id: String,

    /// Original document ID (Parent ID)
    pub parent_id: String,

    /// Chunk content
    pub content: String,

    /// Chunk sequence number
    pub segment: usize,

    /// Chunk metadata
    pub metadata: HashMap<String, Value>,
}

impl ChunkDocument {
    /// Create a new chunk document
    pub fn new(
        chunk_id: impl Into<String>,
        parent_id: impl Into<String>,
        content: impl Into<String>,
        segment: usize,
    ) -> Self {
        Self {
            chunk_id: chunk_id.into(),
            parent_id: parent_id.into(),
            content: content.into(),
            segment,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// 整体替换 metadata(用于 chunk 继承父文档元数据等场景)。
    pub fn with_metadata_map(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Convert to Document
    pub fn to_document(&self) -> Document {
        Document {
            content: self.content.clone(),
            metadata: self.metadata.clone(),
            id: Some(self.chunk_id.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Hello, world!")
            .with_metadata("source", "test")
            .with_id("doc-1");

        assert_eq!(doc.content, "Hello, world!");
        assert_eq!(
            doc.metadata.get("source").and_then(|v| v.as_str()),
            Some("test")
        );
        assert_eq!(doc.id, Some("doc-1".to_string()));
    }

    #[test]
    fn test_document_page_content() {
        let doc = Document::new("Test content");
        assert_eq!(doc.page_content(), "Test content");
    }

    #[test]
    fn test_chunk_document_to_document() {
        let chunk = ChunkDocument::new("c1".to_string(), "p1".to_string(), "hello".to_string(), 0);
        let doc = chunk.to_document();
        assert_eq!(doc.content, "hello");
        assert_eq!(doc.id, Some("c1".to_string()));
    }
}
