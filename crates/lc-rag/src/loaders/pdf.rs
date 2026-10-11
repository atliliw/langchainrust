// src/retrieval/loaders/pdf.rs
//! PDF document loader implementation
//!
//! Provides text-content loading from PDF files.

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;

// Size budgets are the shared loader-wide constants in `super`: the on-disk file
// cap (`MAX_LOCAL_FILE_BYTES`, checked via a `metadata` stat before parsing) and
// the produced-text cap (`MAX_EXTRACTED_BYTES`, checked after extraction).
// `pdf_extract` has no size limit of its own, so both are enforced here (0.27.0 §4
// unified policy; the PDF loader previously kept its own 1 GiB on-disk budget).

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

    /// Fails a payload whose byte length exceeds `max_bytes` (decompression-bomb
    /// guard). Extracted as a helper so the bound is unit-testable without
    /// allocating a full 256 MiB string.
    fn bound_text_len(text: String, max_bytes: usize) -> Result<String, LoaderError> {
        if text.len() > max_bytes {
            return Err(LoaderError::Other(format!(
                "extracted PDF text exceeds {max_bytes} bytes"
            )));
        }
        Ok(text)
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

        // Pre-flight size guard (0.26.0 MEDIUM): a compressed PDF can expand
        // enormously inside `pdf_extract` (which has no size limit we can hand
        // it), so refuse to even begin parsing an absurdly large file. This
        // bounds the CPU + layout memory spent on a bomb; the post-extract
        // byte cap below still stops the produced payload from flowing on.
        let file_size = tokio::fs::metadata(&self.path)
            .await
            .map_err(|e| LoaderError::Other(format!("PDF stat failed: {e}")))?
            .len();
        if file_size > super::MAX_LOCAL_FILE_BYTES as u64 {
            return Err(LoaderError::Other(format!(
                "PDF file size {file_size} exceeds pre-parse limit of {} bytes",
                super::MAX_LOCAL_FILE_BYTES
            )));
        }

        // Extract text using the pdf_extract library. The extraction is CPU-bound and can be slow,
        // so run it on the blocking thread pool rather than stalling the async executor (L1).
        let path = self.path.clone();
        let text = tokio::task::spawn_blocking(move || pdf_extract::extract_text(&path))
            .await
            .map_err(|e| LoaderError::Other(format!("PDF extract task failed to complete: {e}")))?
            .map_err(|e| LoaderError::PdfError(format!("PDF parse failed: {e}")))?;

        // Bound the extracted text (decompression-bomb guard): `pdf_extract`
        // materializes the full string internally before we see it, so cap the
        // output and fail closed rather than let an oversized `Document` flow on.
        let text = Self::bound_text_len(text, super::MAX_EXTRACTED_BYTES)?;

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

    #[test]
    fn oversize_extracted_text_is_rejected() {
        let big = "A".repeat(4096);
        let err = PDFLoader::bound_text_len(big, 1024).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "got: {err}");
    }

    #[test]
    fn within_cap_text_is_accepted() {
        let ok = PDFLoader::bound_text_len("Hello".to_string(), 1024).unwrap();
        assert_eq!(ok, "Hello");
    }

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
