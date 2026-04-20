// src/retrieval/loaders/markdown.rs
//! Markdown 文档加载器实现
//!
//! 提供从 Markdown 文件加载内容的功能，支持按标题分割。

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use regex::Regex;
use std::path::PathBuf;

/// Markdown 文档加载器
///
/// 支持加载 Markdown 文件，可按标题分割为多个文档。
pub struct MarkdownLoader {
    /// Markdown 文件路径
    pub path: PathBuf,
    
    /// 是否按标题分割
    /// 如果为 true，按 `#` 标题分割为多个文档
    pub split_by_heading: bool,
    
    /// 分割的标题级别（1-6）
    /// 例如 heading_level=2 表示按 `##` 分割
    pub heading_level: usize,
}

impl MarkdownLoader {
    /// 创建新的 Markdown 加载器
    ///
    /// # 参数
    /// * `path` - Markdown 文件路径
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            split_by_heading: false,
            heading_level: 1,
        }
    }
    
    /// 创建按标题分割的 Markdown 加载器
    ///
    /// # 参数
    /// * `path` - Markdown 文件路径
    /// * `heading_level` - 分割的标题级别（1-6）
    pub fn new_with_heading_split(path: impl Into<PathBuf>, heading_level: usize) -> Self {
        Self {
            path: path.into(),
            split_by_heading: true,
            heading_level: heading_level.clamp(1, 6),
        }
    }
    
    /// 设置是否按标题分割
    pub fn with_split_by_heading(mut self, split: bool) -> Self {
        self.split_by_heading = split;
        self
    }
    
    /// 设置标题级别
    pub fn with_heading_level(mut self, level: usize) -> Self {
        self.heading_level = level.clamp(1, 6);
        self
    }
}

#[async_trait]
impl DocumentLoader for MarkdownLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "Markdown 文件不存在: {}",
                self.path.display()
            )));
        }

        let content = std::fs::read_to_string(&self.path)?;

        if self.split_by_heading {
            self.split_by_headings(&content)
        } else {
            let mut doc = Document::new(content);
            doc = doc.with_metadata("source", self.path.display().to_string());
            doc = doc.with_metadata("format", "markdown".to_string());
            Ok(vec![doc])
        }
    }
}

impl MarkdownLoader {
    fn split_by_headings(&self, content: &str) -> Result<Vec<Document>, LoaderError> {
        let heading_prefix = "#".repeat(self.heading_level);
        let pattern = format!(r"^{}[ \t]+(.+)", heading_prefix);
        let heading_regex = Regex::new(&pattern)
            .map_err(|e| LoaderError::Other(format!("正则错误: {}", e)))?;

        let mut documents = Vec::new();
        let mut sections: Vec<(String, String)> = Vec::new();
        let mut current_title = "Untitled".to_string();
        let mut current_content = String::new();

        for line in content.lines() {
            if let Some(caps) = heading_regex.captures(line) {
                if !current_content.trim().is_empty() {
                    sections.push((current_title.clone(), current_content.trim().to_string()));
                }
                current_title = caps.get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "Untitled".to_string());
                current_content = String::new();
            } else {
                if !line.trim().is_empty() {
                    current_content.push_str(line);
                    current_content.push('\n');
                }
            }
        }

        if !current_content.trim().is_empty() {
            sections.push((current_title, current_content.trim().to_string()));
        }

        for (title, section_content) in sections {
            if section_content.trim().is_empty() {
                continue;
            }

            let mut doc = Document::new(section_content);
            doc = doc.with_metadata("source", self.path.display().to_string());
            doc = doc.with_metadata("format", "markdown".to_string());
            doc = doc.with_metadata("heading", title);
            doc = doc.with_metadata("heading_level", self.heading_level.to_string());

            documents.push(doc);
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
    async fn test_markdown_loader_nonexistent() {
        let loader = MarkdownLoader::new("./nonexistent.md");
        let result = loader.load().await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_markdown_loader_single_document() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "# Title\n\nContent here.").unwrap();
        
        let loader = MarkdownLoader::new(temp_file.path());
        let result = loader.load().await;
        
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("Title"));
        assert_eq!(docs[0].metadata.get("format"), Some(&"markdown".to_string()));
    }

    #[tokio::test]
    async fn test_markdown_loader_split_by_heading() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Section 1").unwrap();
        writeln!(temp_file, "Content for section 1.").unwrap();
        writeln!(temp_file, "").unwrap();
        writeln!(temp_file, "# Section 2").unwrap();
        writeln!(temp_file, "Content for section 2.").unwrap();
        
        let loader = MarkdownLoader::new_with_heading_split(temp_file.path(), 1);
        let result = loader.load().await;
        
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].metadata.get("heading"), Some(&"Section 1".to_string()));
        assert_eq!(docs[1].metadata.get("heading"), Some(&"Section 2".to_string()));
    }

    #[tokio::test]
    async fn test_markdown_loader_heading_level_2() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Main Title").unwrap();
        writeln!(temp_file, "Intro.").unwrap();
        writeln!(temp_file, "").unwrap();
        writeln!(temp_file, "## Subsection 1").unwrap();
        writeln!(temp_file, "Sub content 1.").unwrap();
        writeln!(temp_file, "").unwrap();
        writeln!(temp_file, "## Subsection 2").unwrap();
        writeln!(temp_file, "Sub content 2.").unwrap();
        
        let loader = MarkdownLoader::new_with_heading_split(temp_file.path(), 2);
        let result = loader.load().await;
        
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].content.contains("Main Title"));
    }
}