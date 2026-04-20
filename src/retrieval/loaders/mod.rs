// src/retrieval/loaders/mod.rs
//! 文档加载器实现
//!
//! 提供从不同格式文件加载文档的功能，包括 PDF、CSV、Text、JSON、Markdown 等。

mod pdf;
mod csv;
mod text;
mod json;
mod markdown;

pub use pdf::PDFLoader;
pub use csv::CSVLoader;
pub use text::TextLoader;
pub use json::JSONLoader;
pub use markdown::MarkdownLoader;

use crate::vector_stores::Document;
use async_trait::async_trait;
use std::error::Error;

/// 文档加载器错误类型
#[derive(Debug)]
pub enum LoaderError {
    /// IO 错误
    IoError(std::io::Error),
    
    /// CSV 解析错误
    CsvError(String),
    
    /// PDF 解析错误
    PdfError(String),
    
    /// JSON 解析错误
    JsonError(String),
    
    /// 未知错误
    Other(String),
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::IoError(e) => write!(f, "IO 错误: {}", e),
            LoaderError::CsvError(msg) => write!(f, "CSV 解析错误: {}", msg),
            LoaderError::PdfError(msg) => write!(f, "PDF 解析错误: {}", msg),
            LoaderError::JsonError(msg) => write!(f, "JSON 解析错误: {}", msg),
            LoaderError::Other(msg) => write!(f, "未知错误: {}", msg),
        }
    }
}

impl Error for LoaderError {}

impl From<std::io::Error> for LoaderError {
    fn from(e: std::io::Error) -> Self {
        LoaderError::IoError(e)
    }
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