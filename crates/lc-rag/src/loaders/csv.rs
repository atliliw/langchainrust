// src/retrieval/loaders/csv.rs
//! CSV document loader implementation
//!
//! Loads CSV files using a given column as the document content.

use super::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use csv::Reader;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

/// CSV document loader
pub struct CSVLoader {
    /// CSV file path
    pub path: PathBuf,

    /// The column name used as the document content
    pub content_column: String,
}

impl CSVLoader {
    /// Creates a new CSV loader
    ///
    /// # Arguments
    /// * `path` - the CSV file path
    /// * `content_column` - the column name used as the document content
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
        // Verify the file exists
        if !self.path.exists() {
            return Err(LoaderError::Other(format!(
                "CSV file does not exist: {}",
                self.path.display()
            )));
        }

        // Create a CSV reader
        let file = File::open(&self.path)?;
        let buf_reader = BufReader::new(file);
        let mut reader = Reader::from_reader(buf_reader);

        let mut documents = Vec::new();

        // Handle possible errors directly
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

            // Find the index of the content column
            let content_idx = headers
                .iter()
                .position(|h| h == self.content_column.as_str());

            if let Some(idx) = content_idx {
                // Get the content
                let content = record.get(idx).unwrap_or_default().to_string();

                // Skip this row if the content is empty
                if content.is_empty() {
                    continue;
                }

                // Create the document content, including all columns of the CSV row
                let mut document = Document::new(content);

                // Add the values of all columns as metadata
                for (i, header) in headers.iter().enumerate() {
                    let value = record.get(i).unwrap_or_default().to_string();
                    document = document.with_metadata(header.to_string(), value);
                }

                // Add the file source information
                document =
                    document.with_metadata("source".to_string(), self.path.display().to_string());
                document = document.with_metadata("format".to_string(), "csv".to_string());
                document = document
                    .with_metadata("content_column".to_string(), self.content_column.clone());

                documents.push(document);
            } else {
                // Return an error if the content column does not exist
                return Err(LoaderError::CsvError(format!(
                    "content column '{}' does not exist in CSV file",
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
            LoaderError::CsvError(msg) => assert!(msg.contains("does not exist")),
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

        // Check the content and metadata of the first document
        if !docs.is_empty() {
            let doc = &docs[0];
            assert!(doc.content.contains("This is the content"));
            assert_eq!(
                doc.metadata.get("title"),
                Some(&serde_json::Value::String("Example Title".to_string()))
            );
            assert_eq!(
                doc.metadata.get("author"),
                Some(&serde_json::Value::String("John Doe".to_string()))
            );
            assert_eq!(
                doc.metadata.get("content"),
                Some(&serde_json::Value::String(
                    "This is the content".to_string()
                ))
            );
            assert_eq!(
                doc.metadata.get("format"),
                Some(&serde_json::Value::String("csv".to_string()))
            );
            assert_eq!(
                doc.metadata.get("content_column"),
                Some(&serde_json::Value::String("content".to_string()))
            );
        }
    }
}
