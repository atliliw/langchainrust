// lc-shared/src/splitter_types.rs
//! Text splitter types shared across crates.
//!
//! `TextSplitter` trait and `RecursiveCharacterSplitter` are needed by
//! both `lc-vector-stores` and `lc-rag`, so they live here to break
//! the circular dependency.

use crate::document_types::Document;
use serde_json::Value;

/// Text splitter trait
pub trait TextSplitter: Send + Sync {
    /// Split text into chunks
    fn split_text(&self, text: &str) -> Vec<String>;

    /// Split a document into smaller documents
    fn split_document(&self, document: &Document) -> Vec<Document> {
        let chunks = self.split_text(&document.content);
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut metadata = document.metadata.clone();
                // Only insert the chunk index when the user hasn't already
                // provided a "chunk" key — never silently overwrite it.
                metadata
                    .entry("chunk".to_string())
                    .or_insert(Value::String(i.to_string()));

                Document {
                    content: chunk,
                    metadata,
                    id: None,
                }
            })
            .collect()
    }
}

/// Recursive character splitter
///
/// Splits text by separator priority, recursively trying smaller separators.
pub struct RecursiveCharacterSplitter {
    /// Chunk size (character count)
    chunk_size: usize,

    /// Chunk overlap (character count)
    chunk_overlap: usize,

    /// Separator list (by priority)
    separators: Vec<String>,
}

impl RecursiveCharacterSplitter {
    /// Create a new recursive character splitter
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators: vec![
                "\n\n".to_string(), // paragraph
                "\n".to_string(),   // line
                "。".to_string(),   // Chinese period
                ".".to_string(),    // English period
                " ".to_string(),    // space
                "".to_string(),     // character
            ],
        }
    }

    /// Create with default parameters (chunk_size=1000, chunk_overlap=200)
    pub fn with_defaults() -> Self {
        Self::new(1000, 200)
    }

    /// Set custom separators
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    /// Split text (internal recursive method)
    fn split_text_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        let mut chunks = Vec::new();

        if text.is_empty() {
            return chunks;
        }

        // If text is already small enough, return as-is
        if text.chars().count() <= self.chunk_size {
            chunks.push(text.to_string());
            return chunks;
        }

        // Find a suitable separator
        let separator = separators
            .iter()
            .find(|s| text.contains(s.as_str()))
            .cloned()
            .unwrap_or_default();

        // Split by separator
        let splits: Vec<String> = if separator.is_empty() {
            text.chars().map(|c| c.to_string()).collect()
        } else {
            text.split(&separator).map(|s| s.to_string()).collect()
        };

        // Merge splits into chunks
        let mut current_chunk = String::new();

        for split in splits {
            let split_with_sep = if separator.is_empty() {
                split.clone()
            } else if current_chunk.is_empty() {
                split
            } else {
                format!("{}{}", separator, split)
            };

            // If a single split exceeds chunk size, recurse
            if split_with_sep.chars().count() > self.chunk_size {
                // Save current chunk first
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }

                // Recurse with next separator
                let next_separators = if separators.len() > 1 {
                    &separators[1..]
                } else {
                    &[]
                };

                let sub_chunks = self.split_text_recursive(&split_with_sep, next_separators);
                chunks.extend(sub_chunks);
            } else if current_chunk.chars().count() + split_with_sep.chars().count()
                > self.chunk_size
            {
                // Current chunk is full, save and start new
                chunks.push(current_chunk.clone());
                current_chunk = split_with_sep;
            } else {
                current_chunk.push_str(&split_with_sep);
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }
}

impl TextSplitter for RecursiveCharacterSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut chunks = self.split_text_recursive(text, &self.separators);

        // Handle overlap
        if self.chunk_overlap > 0 && chunks.len() > 1 {
            let mut overlapped = Vec::new();

            for (i, chunk) in chunks.into_iter().enumerate() {
                if i == 0 {
                    overlapped.push(chunk);
                } else {
                    // Take overlap from end of previous chunk (using chars, not bytes)
                    let prev = &overlapped[i - 1];
                    let chars: Vec<char> = prev.chars().collect();
                    let overlap_chars = chars.len().saturating_sub(self.chunk_overlap);
                    let mut overlap: String = chars[overlap_chars..].iter().collect();

                    // `chunk_size` is a hard cap: the prepended overlap counts
                    // toward the quota, so trim the overlap from the back if
                    // pushing it in would exceed `chunk_size`. This keeps
                    // `chunk_size` a true upper bound on every emitted chunk.
                    let budget = self.chunk_size.saturating_sub(chunk.chars().count());
                    if overlap.chars().count() > budget {
                        let ov_chars: Vec<char> = overlap.chars().collect();
                        overlap = ov_chars[ov_chars.len().saturating_sub(budget)..]
                            .iter()
                            .collect();
                    }

                    overlapped.push(format!("{}{}", overlap, chunk));
                }
            }

            chunks = overlapped;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_splitter() {
        let splitter = RecursiveCharacterSplitter::new(50, 10);

        let text = "This is a sentence. This is another sentence. And a third one.";
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
        // chunk_size is a hard cap — even with overlap prepended
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 50);
        }
    }

    #[test]
    fn test_split_document() {
        let splitter = RecursiveCharacterSplitter::new(100, 20);

        let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
            .with_metadata("source", "test");

        let chunks = splitter.split_document(&doc);

        assert!(!chunks.is_empty());
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(chunk.metadata.contains_key("chunk"));
            assert_eq!(
                chunk.metadata.get("chunk"),
                Some(&Value::String(i.to_string()))
            );
            assert_eq!(
                chunk.metadata.get("source"),
                Some(&Value::String("test".to_string()))
            );
        }
    }

    #[test]
    fn test_split_document_preserves_user_chunk_key() {
        let splitter = RecursiveCharacterSplitter::new(20, 5);

        // User already numbered the chunks — split_document must not overwrite it
        let doc = Document::new("This is a longer paragraph that gets split into several chunks.")
            .with_metadata("chunk", "user-supplied");

        let chunks = splitter.split_document(&doc);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(
                chunk.metadata.get("chunk"),
                Some(&Value::String("user-supplied".to_string()))
            );
        }
    }

    #[test]
    fn test_empty_text() {
        let splitter = RecursiveCharacterSplitter::new(100, 20);
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_text() {
        let splitter = RecursiveCharacterSplitter::new(1000, 200);
        let text = "Short text";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text");
    }
}
