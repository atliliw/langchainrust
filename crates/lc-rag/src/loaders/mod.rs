// src/retrieval/loaders/mod.rs
//! Document loader implementations
//!
//! Provides document loading from files in various formats, including PDF, CSV, Text, JSON,
//! Markdown, HTML, etc. v0.4.1 added the WebScraper, Sitemap, and Docx loaders.

mod csv;
mod docx;
mod html;
mod json;
mod markdown;
mod pdf;
mod sitemap;
mod text;
mod web_scraper;

pub use csv::CSVLoader;
pub use docx::DocxLoader;
pub use html::HTMLLoader;
pub use json::JSONLoader;
pub use markdown::MarkdownLoader;
pub use pdf::PDFLoader;
pub use sitemap::SitemapLoader;
pub use text::TextLoader;
pub use web_scraper::WebScraperLoader;

use async_trait::async_trait;
use lc_vector_stores::Document;

/// Document loader error type
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoaderError {
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// CSV parse error
    #[error("CSV parse error: {0}")]
    CsvError(String),

    /// PDF parse error
    #[error("PDF parse error: {0}")]
    PdfError(String),

    /// JSON parse error
    #[error("JSON parse error: {0}")]
    JsonError(String),

    /// Unknown error
    #[error("unknown error: {0}")]
    Other(String),
}

impl From<pdf_extract::Error> for LoaderError {
    fn from(err: pdf_extract::Error) -> Self {
        LoaderError::PdfError(err.to_string())
    }
}

/// Document loader trait
///
/// Defines the common interface for loading documents from a source.
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    /// Loads documents from the source
    ///
    /// # Returns
    /// The loaded documents
    async fn load(&self) -> Result<Vec<Document>, LoaderError>;
}
