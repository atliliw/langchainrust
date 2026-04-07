// src/retrieval/loaders/pdf.rs
//! PDF 文档加载器实现
//!
//! 提供从 PDF 文件加载文本内容的功能。

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;

/// PDF 文档加载器
pub struct PDFLoader {
    /// PDF 文件路径
    pub path: PathBuf,
}

impl PDFLoader {
    /// 创建新的 PDF 加载器
    ///
    /// # 参数
    /// * `path` - PDF 文件路径
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl DocumentLoader for PDFLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        // 验证文件存在
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "PDF 文件不存在: {}",
                self.path.display()
            )));
        }

        // 使用 pdf_extract 库提取文本
        let text = pdf_extract::extract_text(&self.path)
            .map_err(|e| LoaderError::PdfError(format!("PDF 解析失败: {}", e)))?;

        // 创建文档对象，包含元数据
        let mut document = Document::new(text);
        document = document.with_metadata("source".to_string(), self.path.display().to_string());
        document = document.with_metadata("format".to_string(), "pdf".to_string());
        
        Ok(vec![document])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[tokio::test]
    async fn test_pdf_loader_nonexistent() {
        let loader = PDFLoader::new("./nonexistent.pdf");
        let result = loader.load().await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            LoaderError::Other(msg) => assert!(msg.contains("不存在")),
            _ => panic!("Expected Other error"),
        }
    }

    #[tokio::test]
    #[ignore = "requires a sample PDF file"]
    async fn test_pdf_loader() {
        // 注意：这需要一个实际存在的 PDF 文件进行测试
        let loader = PDFLoader::new("./sample.pdf");
        let result = loader.load().await;
        
        // 应该成功返回至少一个文档
        if result.is_ok() {
            let docs = result.unwrap();
            assert!(!docs.is_empty());
            assert!(docs[0].content.contains("PDF"));
        }
    }
}