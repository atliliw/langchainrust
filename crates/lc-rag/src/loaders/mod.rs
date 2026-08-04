// src/retrieval/loaders/mod.rs
//! 文档加载器实现
//!
//! 提供从不同格式文件加载文档的功能，包括 PDF、CSV、Text、JSON、Markdown、HTML 等。
//! v0.4.1 新增: WebScraper、Sitemap、Docx 加载器。

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

/// 文档加载器错误类型
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    /// IO 错误
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    /// CSV 解析错误
    #[error("CSV 解析错误: {0}")]
    CsvError(String),

    /// PDF 解析错误
    #[error("PDF 解析错误: {0}")]
    PdfError(String),

    /// JSON 解析错误
    #[error("JSON 解析错误: {0}")]
    JsonError(String),

    /// 未知错误
    #[error("未知错误: {0}")]
    Other(String),
}

impl From<pdf_extract::Error> for LoaderError {
    fn from(err: pdf_extract::Error) -> Self {
        LoaderError::PdfError(err.to_string())
    }
}

/// 文档加载器 trait
///
/// 定义从源加载文档的通用接口。
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    /// 从源加载文档
    ///
    /// # 返回
    /// 加载的文档列表
    async fn load(&self) -> Result<Vec<Document>, LoaderError>;
}
