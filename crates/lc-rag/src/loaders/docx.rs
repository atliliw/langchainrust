//! DOCX document loader
//!
//! Loads document content from .docx files (plain-text extraction).
//! No external crate required: uses ZIP decompression + XML parsing to extract text.

use std::collections::HashMap;
use std::io::Read;

use async_trait::async_trait;
use regex::Regex;
use std::sync::LazyLock;

use super::{DocumentLoader, LoaderError};
use lc_vector_stores::Document;

/// M60: compile regexes once using LazyLock instead of on every call
static WT_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<w:t[^>]*>(.*?)</w:t>").unwrap());
static PARA_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"</w:p>").unwrap());

/// DOCX document loader
///
/// Loads documents from a .docx file path, extracting the body text.
/// A DOCX file is a ZIP package; the body lives in `word/document.xml`.
pub struct DocxLoader {
    /// File path
    path: String,
}

impl DocxLoader {
    /// Creates a loader from a file path
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// Extracts text from DOCX bytes
    ///
    /// A DOCX file is a ZIP package; the body lives in `word/document.xml`,
    /// with the text inside `<w:t>` tags.
    fn extract_text_from_bytes(data: &[u8]) -> Result<String, LoaderError> {
        let reader = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(reader)
            .map_err(|e| LoaderError::Other(format!("DOCX is not a valid ZIP: {}", e)))?;

        // Read word/document.xml
        let mut xml_content = String::new();
        let mut found = false;
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| LoaderError::Other(format!("failed to read ZIP entry: {}", e)))?;
            if file.name() == "word/document.xml" {
                file.read_to_string(&mut xml_content).map_err(|e| {
                    LoaderError::Other(format!("failed to read document.xml: {}", e))
                })?;
                found = true;
                break;
            }
        }

        if !found {
            return Err(LoaderError::Other(
                "word/document.xml not found in DOCX".to_string(),
            ));
        }

        // Extract the <w:t> tag content
        Self::extract_text_from_xml(&xml_content)
    }

    /// Extracts text from document.xml
    fn extract_text_from_xml(xml: &str) -> Result<String, LoaderError> {
        // M60: use pre-compiled regexes from LazyLock
        let mut result = String::new();
        let mut last_end = 0;

        for cap in WT_REGEX.captures_iter(xml) {
            let Some(m) = cap.get(1) else {
                continue;
            };
            // Check whether a </w:p> (new paragraph) precedes this <w:t>
            let before = &xml[last_end..m.start()];
            if PARA_REGEX.is_match(before) && !result.is_empty() {
                result.push('\n');
            } else if !result.is_empty() {
                // Consecutive text within the same paragraph
            }
            result.push_str(m.as_str());
            last_end = m.end();
        }

        Ok(result)
    }
}

// The zip crate is in dev-dependencies; normally we would handle it via features,
// but for simplicity it is added directly as a dependency.
// Note: only std::io + zip are used here as the minimal dependency set.

#[async_trait]
impl DocumentLoader for DocxLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let data = tokio::task::spawn_blocking({
            let path = self.path.clone();
            move || std::fs::read(&path)
        })
        .await
        .map_err(|e| LoaderError::Other(format!("failed to read file: {}", e)))?
        .map_err(LoaderError::IoError)?;

        let text = Self::extract_text_from_bytes(&data)?;

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "docx".to_string().into());
        metadata.insert("source".to_string(), self.path.clone().into());

        Ok(vec![Document {
            content: text,
            metadata,
            id: None,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_xml() {
        let xml = r#"<?xml version="1.0"?>
        <w:document>
            <w:body>
                <w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> World</w:t></w:r></w:p>
                <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
            </w:body>
        </w:document>"#;
        let text = DocxLoader::extract_text_from_xml(xml).unwrap();
        assert!(text.contains("Hello World"));
        assert!(text.contains("Second paragraph"));
    }

    #[test]
    fn test_extract_text_from_xml_empty() {
        let xml = r#"<?xml version="1.0"?><w:document><w:body></w:body></w:document>"#;
        let text = DocxLoader::extract_text_from_xml(xml).unwrap();
        assert!(text.is_empty());
    }

    #[test]
    fn test_extract_text_from_xml_with_xml_space() {
        // w:t may carry an xml:space="preserve" attribute
        let xml = r#"<w:p><w:r><w:t xml:space="preserve">  spaced  </w:t></w:r></w:p>"#;
        let text = DocxLoader::extract_text_from_xml(xml).unwrap();
        assert_eq!(text, "  spaced  ");
    }

    #[test]
    fn test_new() {
        let loader = DocxLoader::new("/path/to/file.docx");
        assert_eq!(loader.path, "/path/to/file.docx");
    }
}
