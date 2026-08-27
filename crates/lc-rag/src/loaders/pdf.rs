// src/retrieval/loaders/pdf.rs
//! PDF document loader implementation
//!
//! Provides text-content loading from PDF files.

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;

/// PDF document loader
pub struct PDFLoader {
    /// PDF file path
    pub path: PathBuf,
}

impl PDFLoader {
    /// Creates a new PDF loader
    ///
    /// # Arguments
    /// * `path` - the PDF file path
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl DocumentLoader for PDFLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        // Verify the file exists
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "PDF file does not exist: {}",
                self.path.display()
            )));
        }

        // Extract text using the pdf_extract library
        let text = pdf_extract::extract_text(&self.path)
            .map_err(|e| LoaderError::PdfError(format!("PDF parse failed: {}", e)))?;

        // Create the document object, including metadata
        let mut document = Document::new(text);
        document = document.with_metadata("source".to_string(), self.path.display().to_string());
        document = document.with_metadata("format".to_string(), "pdf".to_string());

        Ok(vec![document])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pdf_loader_nonexistent() {
        let loader = PDFLoader::new("./nonexistent.pdf");
        let result = loader.load().await;

        assert!(result.is_err());
        match result.unwrap_err() {
            LoaderError::Other(msg) => assert!(msg.contains("does not exist")),
            _ => panic!("Expected Other error"),
        }
    }
}
