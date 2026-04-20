// src/retrieval/loaders/text.rs
//! Text 文档加载器实现
//!
//! 提供从纯文本文件加载内容的功能。

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;

/// Text 文档加载器
///
/// 支持加载纯文本文件（.txt），将整个文件内容作为一个文档。
pub struct TextLoader {
    /// 文本文件路径
    pub path: PathBuf,
    
    /// 是否按行分割（可选）
    /// 如果为 true，每行作为一个独立文档
    pub split_by_line: bool,
}

impl TextLoader {
    /// 创建新的 Text 加载器
    ///
    /// # 参数
    /// * `path` - 文本文件路径
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            split_by_line: false,
        }
    }
    
    /// 创建按行分割的 Text 加载器
    ///
    /// 每行文本将作为独立文档返回。
    ///
    /// # 参数
    /// * `path` - 文本文件路径
    pub fn new_with_line_split(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            split_by_line: true,
        }
    }
    
    /// 设置是否按行分割
    pub fn with_split_by_line(mut self, split: bool) -> Self {
        self.split_by_line = split;
        self
    }
}

#[async_trait]
impl DocumentLoader for TextLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        // 验证文件存在
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "文本文件不存在: {}",
                self.path.display()
            )));
        }

        // 读取文件内容
        let content = std::fs::read_to_string(&self.path)?;
        
        if self.split_by_line {
            // 按行分割
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
            // 整个文件作为一个文档
            let mut document = Document::new(content);
            document = document.with_metadata("source".to_string(), self.path.display().to_string());
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
            LoaderError::Other(msg) => assert!(msg.contains("不存在")),
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
        assert_eq!(docs[0].metadata.get("format"), Some(&"text".to_string()));
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
        assert_eq!(docs[0].metadata.get("line_number"), Some(&"1".to_string()));
    }

    #[tokio::test]
    async fn test_text_loader_skip_empty_lines() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Line 1").unwrap();
        writeln!(temp_file, "").unwrap();
        writeln!(temp_file, "   ").unwrap();
        writeln!(temp_file, "Line 2").unwrap();
        
        let loader = TextLoader::new_with_line_split(temp_file.path());
        let result = loader.load().await;
        
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2); // 空行被跳过
    }
}