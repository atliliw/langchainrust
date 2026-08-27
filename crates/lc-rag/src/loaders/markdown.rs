// src/retrieval/loaders/markdown.rs
//! Markdown document loader implementation
//!
//! Loads content from Markdown files, supporting splitting by heading.

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;

/// L11: pre-compile heading regexes for all 6 levels using LazyLock
fn heading_regex(level: usize) -> &'static regex::Regex {
    use std::sync::LazyLock;
    static HEADING_RE: [LazyLock<regex::Regex>; 6] = [
        LazyLock::new(|| regex::Regex::new(r"^#[ \t]+(.+)").unwrap()),
        LazyLock::new(|| regex::Regex::new(r"^##[ \t]+(.+)").unwrap()),
        LazyLock::new(|| regex::Regex::new(r"^###[ \t]+(.+)").unwrap()),
        LazyLock::new(|| regex::Regex::new(r"^####[ \t]+(.+)").unwrap()),
        LazyLock::new(|| regex::Regex::new(r"^#####[ \t]+(.+)").unwrap()),
        LazyLock::new(|| regex::Regex::new(r"^######[ \t]+(.+)").unwrap()),
    ];
    &HEADING_RE[level.saturating_sub(1).min(5)]
}

/// Markdown document loader
///
/// Supports loading Markdown files, optionally splitting them into multiple documents by heading.
pub struct MarkdownLoader {
    /// Markdown file path
    pub path: PathBuf,

    /// Whether to split by heading
    /// If true, splits into multiple documents by `#` heading
    pub split_by_heading: bool,

    /// The heading level to split on (1-6)
    /// For example, heading_level=2 splits on `##`
    pub heading_level: usize,
}

impl MarkdownLoader {
    /// Creates a new Markdown loader
    ///
    /// # Arguments
    /// * `path` - the Markdown file path
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            split_by_heading: false,
            heading_level: 1,
        }
    }

    /// Creates a heading-splitting Markdown loader
    ///
    /// # Arguments
    /// * `path` - the Markdown file path
    /// * `heading_level` - the heading level to split on (1-6)
    pub fn new_with_heading_split(path: impl Into<PathBuf>, heading_level: usize) -> Self {
        Self {
            path: path.into(),
            split_by_heading: true,
            heading_level: heading_level.clamp(1, 6),
        }
    }

    /// Sets whether to split by heading
    pub fn with_split_by_heading(mut self, split: bool) -> Self {
        self.split_by_heading = split;
        self
    }

    /// Sets the heading level
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
                "markdown file does not exist: {}",
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
        let heading_regex = heading_regex(self.heading_level);

        let mut documents = Vec::new();
        let mut sections: Vec<(String, String)> = Vec::new();
        let mut current_title = "Untitled".to_string();
        let mut current_content = String::new();

        for line in content.lines() {
            if let Some(caps) = heading_regex.captures(line) {
                if !current_content.trim().is_empty() {
                    sections.push((current_title.clone(), current_content.trim().to_string()));
                }
                current_title = caps
                    .get(1)
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "Untitled".to_string());
                current_content = String::new();
            } else if !line.trim().is_empty() {
                current_content.push_str(line);
                current_content.push('\n');
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
        assert_eq!(
            docs[0].metadata.get("format"),
            Some(&serde_json::Value::String("markdown".to_string()))
        );
    }

    #[tokio::test]
    async fn test_markdown_loader_split_by_heading() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Section 1").unwrap();
        writeln!(temp_file, "Content for section 1.").unwrap();
        writeln!(temp_file).unwrap();
        writeln!(temp_file, "# Section 2").unwrap();
        writeln!(temp_file, "Content for section 2.").unwrap();

        let loader = MarkdownLoader::new_with_heading_split(temp_file.path(), 1);
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(
            docs[0].metadata.get("heading"),
            Some(&serde_json::Value::String("Section 1".to_string()))
        );
        assert_eq!(
            docs[1].metadata.get("heading"),
            Some(&serde_json::Value::String("Section 2".to_string()))
        );
    }

    #[tokio::test]
    async fn test_markdown_loader_heading_level_2() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Main Title").unwrap();
        writeln!(temp_file, "Intro.").unwrap();
        writeln!(temp_file).unwrap();
        writeln!(temp_file, "## Subsection 1").unwrap();
        writeln!(temp_file, "Sub content 1.").unwrap();
        writeln!(temp_file).unwrap();
        writeln!(temp_file, "## Subsection 2").unwrap();
        writeln!(temp_file, "Sub content 2.").unwrap();

        let loader = MarkdownLoader::new_with_heading_split(temp_file.path(), 2);
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 3);
        assert!(docs[0].content.contains("Main Title"));
    }
}
