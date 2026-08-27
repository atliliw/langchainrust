// src/retrieval/loaders/json.rs
//! JSON document loader implementation
//!
//! Loads content from JSON files, supporting a designated field as the document content.

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use serde_json::Value;
use std::path::PathBuf;

/// JSON document loader
///
/// Supports loading JSON files, optionally using a specific field as the document content.
/// - For a JSON array, each element becomes one document
/// - For a JSON object, the whole object becomes one document (or a specified field)
pub struct JSONLoader {
    /// JSON file path
    pub path: PathBuf,

    /// The field name used as the document content (optional)
    /// If specified, that field's value is extracted as the content
    pub content_key: Option<String>,

    /// Whether to keep the raw JSON as metadata
    pub preserve_raw: bool,
}

impl JSONLoader {
    /// Creates a new JSON loader
    ///
    /// # Arguments
    /// * `path` - the JSON file path
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            content_key: None,
            preserve_raw: false,
        }
    }

    /// Creates a JSON loader with a content field
    ///
    /// # Arguments
    /// * `path` - the JSON file path
    /// * `content_key` - the field name used as the document content
    pub fn new_with_content_key(path: impl Into<PathBuf>, content_key: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content_key: Some(content_key.into()),
            preserve_raw: false,
        }
    }

    /// Sets whether to keep the raw JSON
    pub fn with_preserve_raw(mut self, preserve: bool) -> Self {
        self.preserve_raw = preserve;
        self
    }
}

#[async_trait]
impl DocumentLoader for JSONLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "JSON file does not exist: {}",
                self.path.display()
            )));
        }

        let content = std::fs::read_to_string(&self.path)?;
        let json: Value =
            serde_json::from_str(&content).map_err(|e| LoaderError::JsonError(e.to_string()))?;

        let documents = match json {
            Value::Array(arr) => arr
                .iter()
                .filter_map(|item| self.json_value_to_document(item))
                .collect(),
            Value::Object(_) => match self.json_value_to_document(&json) {
                Some(doc) => vec![doc],
                None => vec![],
            },
            _ => {
                vec![Document::new(json.to_string())
                    .with_metadata("source", self.path.display().to_string())
                    .with_metadata("format", "json")]
            }
        };

        Ok(documents)
    }
}

impl JSONLoader {
    fn json_value_to_document(&self, value: &Value) -> Option<Document> {
        match value {
            Value::Object(obj) => {
                let content = if let Some(key) = &self.content_key {
                    obj.get(key)
                        .map(|v| self.extract_string_value(v))
                        .unwrap_or_else(|| value.to_string())
                } else {
                    value.to_string()
                };

                if content.is_empty() || content == "null" {
                    return None;
                }

                let mut doc = Document::new(content);
                doc = doc.with_metadata("source", self.path.display().to_string());
                doc = doc.with_metadata("format", "json".to_string());

                if let Some(key) = &self.content_key {
                    doc = doc.with_metadata("content_key", key.clone());
                }

                for (k, v) in obj {
                    if self.content_key.as_ref() != Some(k) {
                        doc = doc.with_metadata(k.clone(), self.extract_string_value(v));
                    }
                }

                if self.preserve_raw {
                    doc = doc.with_metadata("raw_json", value.to_string());
                }

                Some(doc)
            }
            Value::String(s) => {
                if s.is_empty() {
                    return None;
                }
                Some(
                    Document::new(s.clone())
                        .with_metadata("source", self.path.display().to_string())
                        .with_metadata("format", "json"),
                )
            }
            Value::Number(_) | Value::Bool(_) => Some(
                Document::new(value.to_string())
                    .with_metadata("source", self.path.display().to_string())
                    .with_metadata("format", "json"),
            ),
            Value::Null => None,
            Value::Array(_) => Some(
                Document::new(value.to_string())
                    .with_metadata("source", self.path.display().to_string())
                    .with_metadata("format", "json"),
            ),
        }
    }

    fn extract_string_value(&self, value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            _ => value.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_json_loader_nonexistent() {
        let loader = JSONLoader::new("./nonexistent.json");
        let result = loader.load().await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_loader_invalid_json() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{{ invalid json }}").unwrap();

        let loader = JSONLoader::new(temp_file.path());
        let result = loader.load().await;

        assert!(result.is_err());
        match result.unwrap_err() {
            LoaderError::JsonError(_) => {}
            _ => panic!("Expected JsonError"),
        }
    }

    #[tokio::test]
    async fn test_json_loader_single_object() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{{\"title\": \"Test\", \"content\": \"Hello\"}}").unwrap();

        let loader = JSONLoader::new_with_content_key(temp_file.path(), "content");
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("Hello"));
    }

    #[tokio::test]
    async fn test_json_loader_array() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "[{{\"title\": \"A\", \"content\": \"Content A\"}}, {{\"title\": \"B\", \"content\": \"Content B\"}}]").unwrap();

        let loader = JSONLoader::new_with_content_key(temp_file.path(), "content");
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].content.contains("Content A"));
        assert_eq!(
            docs[0].metadata.get("title"),
            Some(&serde_json::Value::String("A".to_string()))
        );
    }

    #[tokio::test]
    async fn test_json_loader_with_preserve_raw() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{{\"name\": \"test\", \"value\": 123}}").unwrap();

        let loader = JSONLoader::new(temp_file.path()).with_preserve_raw(true);
        let result = loader.load().await;

        assert!(result.is_ok());
        let docs = result.unwrap();
        assert!(docs[0].metadata.contains_key("raw_json"));
    }
}
