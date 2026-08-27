// src/retrieval/splitter.rs
//! Text splitter implementation
//!
//! Splits long documents into smaller chunks for easier processing and retrieval.
//! The `TextSplitter` trait and `RecursiveCharacterSplitter` have moved to the lc-shared crate;
//! a re-export is kept here for backward compatibility.

// Re-export shared splitter types from lc-shared
pub use lc_shared::splitter::{RecursiveCharacterSplitter, TextSplitter};

#[cfg(test)]
mod tests {
    use super::*;
    use lc_shared::document::Document;

    #[test]
    fn test_recursive_splitter() {
        let splitter = RecursiveCharacterSplitter::new(50, 10);

        let text = "This is a sentence. This is another sentence. And a third one.";
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() <= 60); // allow some slack
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
                Some(&serde_json::Value::String(i.to_string()))
            );
            assert_eq!(
                chunk.metadata.get("source"),
                Some(&serde_json::Value::String("test".to_string()))
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
