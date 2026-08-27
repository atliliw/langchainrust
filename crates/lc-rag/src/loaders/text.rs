// src/retrieval/loaders/text.rs
//! Text document loader implementation
//!
//! Provides loading content from plain-text files.

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;

/// Text document loader
///
/// Supports loading plain-text files (.txt), treating the entire file content as one document.
pub struct TextLoader {
    /// Text file path
    pub path: PathBuf,

    /// Whether to split by line (optional)
    /// If true, each line is returned as a separate document
    pub split_by_line: bool,
}

impl TextLoader {
    /// Creates a new Text loader
    ///
    /// # Arguments
    /// * `path` - the text file path
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            split_by_line: false,
        }
    }

    /// Creates a line-splitting Text loader
    ///
    /// Each line is returned as a separate document.
    ///
    /// # Arguments
    /// * `path` - the text file path
    pub fn new_with_line_split(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            split_by_line: true,
        }
    }

    /// Sets whether to split by line
    pub fn with_split_by_line(mut self, split: bool) -> Self {
        self.split_by_line = split;
        self
    }
}

#[async_trait]
impl DocumentLoader for TextLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        // Verify the file exists
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "text file does not exist: {}",
                self.path.display()
            )));
        }

        // Read the file content
        let content = std::fs::read_to_string(&self.path)?;

        if self.split_by_line {
            // Split by line
            let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
            let documents = lines
                .iter()
                .enumerate()
                .map(|(idx, line)| {
                    let mut doc = Document::new(line.to_string());
                    doc = doc.with_metadata("source".to_string(), self.path.display().to_string());
                    doc = doc.with_metadata("format".to_string(), "text".to_string());
                    doc = doc.with_metadata("line_number".to_string(), (idx + 1).to_string());
                    doc
                })
                .collect();

            Ok(documents)
        } else {
            // Treat the entire file as one document
            let mut document = Document::new(content);
            document =
                document.with_metadata("source".to_string(), self.path.display().to_string());
            document = document.with_metadata("format".to_string(), "text".to_string());

            Ok(vec![document])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_text_loader_nonexistent() {
        let loader = TextLoader::new("./nonexistent.txt");
        let result = loader.load().await;

        assert!(result.is_err());
        match result.unwrap_err() {
            LoaderError::Other(msg) => assert!(msg.contains("does not exist")),
            _ => panic!("Expected Other error"),
        }
    }

    #[tokio::test]
    async fn test_text_loader_single_document() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "Hello, World!\nThis is a test.").unwrap();

        let loader = TextLoader::new(temp_file.path());
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("Hello, World!"));
        assert_eq!(
            docs[0].metadata.get("format"),
            Some(&serde_json::Value::String("text".to_string()))
        );
    }

    #[tokio::test]
    async fn test_text_loader_split_by_line() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Line 1").unwrap();
        writeln!(temp_file, "Line 2").unwrap();
        writeln!(temp_file, "Line 3").unwrap();

        let loader = TextLoader::new_with_line_split(temp_file.path());
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].content, "Line 1");
        assert_eq!(
            docs[0].metadata.get("line_number"),
            Some(&serde_json::Value::String("1".to_string()))
        );
    }

    #[tokio::test]
    async fn test_text_loader_skip_empty_lines() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Line 1").unwrap();
        writeln!(temp_file).unwrap();
        writeln!(temp_file, "   ").unwrap();
        writeln!(temp_file, "Line 2").unwrap();

        let loader = TextLoader::new_with_line_split(temp_file.path());
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2); // empty lines are skipped
    }
}
