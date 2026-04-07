// src/retrieval/loaders/csv.rs
//! CSV 文档加载器实现
//!
//! 提供给定列作为文档内容的方式来加载 CSV 文件。

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use csv::Reader;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

/// CSV 文档加载器
pub struct CSVLoader {
    /// CSV 文件路径
    pub path: PathBuf,
    
    /// 作为文档内容的列名
    pub content_column: String,
}

impl CSVLoader {
    /// 创建新的 CSV 加载器
    ///
    /// # 参数
    /// * `path` - CSV 文件路径
    /// * `content_column` - 作为文档内容的列名
    pub fn new(path: impl Into<PathBuf>, content_column: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content_column: content_column.into(),
        }
    }
}

#[async_trait]
impl DocumentLoader for CSVLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        // 验证文件存在
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "CSV 文件不存在: {}",
                self.path.display()
            )));
        }

        // 创建 CSV reader
        let file = File::open(&self.path)?;
        let buf_reader = BufReader::new(file);
        let mut reader = Reader::from_reader(buf_reader);

        let mut documents = Vec::new();
        
        // 直接处理可能的错误
        let headers_result = reader.headers();
        let headers = match headers_result {
            Ok(headers) => headers.clone(),
            Err(e) => return Err(LoaderError::CsvError(e.to_string())),
        };

        for result in reader.records() {
            let record = match result {
                Ok(record) => record,
                Err(e) => return Err(LoaderError::CsvError(e.to_string())),
            };
            
            // 查找内容列的索引
            let content_idx = headers.iter().position(|h| h == self.content_column.as_str());
            
            if let Some(idx) = content_idx {
                // 获取内容
                let content = record.get(idx).unwrap_or_default().to_string();
                
                // 如果内容為空則跳過此行
                if content.is_empty() {
                    continue;
                }

                // 创建文档内容，包含 CSV 行的所有列
                let mut document = Document::new(content);
                
                // 添加所有列的值作为元数据
                for (i, header) in headers.iter().enumerate() {
                    let value = record.get(i).unwrap_or_default().to_string();
                    document = document.with_metadata(header.to_string(), value);
                }
                
                // 添加文件源信息
                document = document.with_metadata("source".to_string(), self.path.display().to_string());
                document = document.with_metadata("format".to_string(), "csv".to_string());
                document = document.with_metadata("content_column".to_string(), self.content_column.clone());
                
                documents.push(document);
            } else {
                // 如果内容列不存在，返回错误
                return Err(LoaderError::CsvError(format!(
                    "内容列 '{}' 在 CSV 文件中不存在",
                    self.content_column
                )));
            }
        }
        
        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_csv_loader_nonexistent() {
        let loader = CSVLoader::new("./nonexistent.csv", "content");
        let result = loader.load().await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_csv_loader_content_column_not_found() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "col1,col2").unwrap();
        writeln!(temp_file, "val1,val2").unwrap();
        
        let loader = CSVLoader::new(temp_file.path(), "content");
        let result = loader.load().await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            LoaderError::CsvError(msg) => assert!(msg.contains("不存在")),
            _ => panic!("Expected CsvError"),
        }
    }
    
    #[tokio::test]
    async fn test_csv_loader_valid_data() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "title,content,author").unwrap();
        writeln!(temp_file, "Example Title,\"This is the content\",John Doe").unwrap();
        writeln!(temp_file, "Another Title,\"More content\",Jane Smith").unwrap();
        
        let loader = CSVLoader::new(temp_file.path(), "content");
        let result = loader.load().await;
        
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2);
        
        // 检查第一个文档的内容和元数据
        if !docs.is_empty() {
            let doc = &docs[0];
            assert!(doc.content.contains("This is the content"));
            assert_eq!(doc.metadata.get("title"), Some(&"Example Title".to_string()));
            assert_eq!(doc.metadata.get("author"), Some(&"John Doe".to_string()));
            assert_eq!(doc.metadata.get("content"), Some(&"This is the content".to_string()));
            assert_eq!(doc.metadata.get("format"), Some(&"csv".to_string()));
            assert_eq!(doc.metadata.get("content_column"), Some(&"content".to_string()));
        }
    }
}