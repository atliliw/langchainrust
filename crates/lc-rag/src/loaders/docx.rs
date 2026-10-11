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
static WT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<w:t[^>]*>(.*?)</w:t>").expect("static regex literal must compile")
});
static PARA_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"</w:p>").expect("static regex literal must compile"));

// Screen before unzipping: the on-disk file cap is the shared `super::MAX_LOCAL_FILE_BYTES`
// (checked in `load` via a metadata stat), and the decompressed `word/document.xml` body is
// bounded by the shared `super::MAX_EXTRACTED_BYTES` — a DOS is a ZIP, so a crafted archive
// can compress a huge XML body to a tiny on-disk entry (decompression bomb); both come from
// the unified loader policy (0.27.0 §4, previously DOCX kept its own duplicated constant).

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

    /// Extracts text from DOCX bytes (default decompression-size cap).
    ///
    /// A DOCX file is a ZIP package; the body lives in `word/document.xml`,
    /// with the text inside `<w:t>` tags.
    fn extract_text_from_bytes(data: &[u8]) -> Result<String, LoaderError> {
        Self::extract_text_from_bytes_with_limit(data, super::MAX_EXTRACTED_BYTES)
    }

    /// Extracts text from DOCX bytes, bounding the decompressed size of
    /// `word/document.xml` to `max_decompressed_bytes`. A crafted small .docx
    /// (whole archive read into memory by the caller) can decompress a single
    /// entry to far beyond the on-disk size; without a cap this balloons memory.
    fn extract_text_from_bytes_with_limit(
        data: &[u8],
        max_decompressed_bytes: usize,
    ) -> Result<String, LoaderError> {
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
                // Bound the decompressed bytes: take cap+1 so we can tell "exactly
                // at cap" from "truncated", and fail closed rather than silently
                // accept an over-limit entry.
                (&mut file)
                    .take((max_decompressed_bytes + 1) as u64)
                    .read_to_string(&mut xml_content)
                    .map_err(|e| {
                        LoaderError::Other(format!("failed to read document.xml: {}", e))
                    })?;
                if xml_content.len() > max_decompressed_bytes {
                    return Err(LoaderError::Other(format!(
                        "decompressed document.xml exceeds {max_decompressed_bytes} bytes (decompression bomb)"
                    )));
                }
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
            // Check whether a </w:p> (new paragraph) precedes this <w:t>; the former
            // `else if !result.is_empty() { }` branch was a no-op and is dropped here.
            let before = &xml[last_end..m.start()];
            if PARA_REGEX.is_match(before) && !result.is_empty() {
                result.push('\n');
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
        // 0.27.0 §4: unified on-disk budget — reject an oversized .docx before reading it into
        // memory. A DOCX can decompress far beyond its on-disk size, and the unzipped
        // `word/document.xml` is separately capped by `MAX_EXTRACTED_BYTES` inside
        // `extract_text_from_bytes`; this stat check bounds the archive we're willing to read.
        let file_size = tokio::fs::metadata(&self.path).await?.len();
        if file_size > super::MAX_LOCAL_FILE_BYTES as u64 {
            return Err(LoaderError::Other(format!(
                "DOCX file size {file_size} exceeds pre-parse limit of {} bytes",
                super::MAX_LOCAL_FILE_BYTES
            )));
        }

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

    /// Build a minimal .docx (ZIP) whose `word/document.xml` is `entry_xml`.
    fn build_docx(entry_xml: &str) -> Vec<u8> {
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zw = zip::ZipWriter::new(&mut buf);
            zw.start_file(
                "word/document.xml",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
            zw.write_all(entry_xml.as_bytes()).unwrap();
            zw.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn decompression_bomb_is_rejected() {
        // Highly-compressible body (~500 KiB of repeated text) compresses to a
        // tiny on-disk entry but balloons on decompression. A small cap must make
        // extraction fail closed instead of materializing the whole entry.
        let bomb_xml = format!("<w:p>{}</w:p>", "A".repeat(500 * 1024));
        let docx = build_docx(&bomb_xml);
        assert!(docx.len() < bomb_xml.len(), "sanity: bomb has a good ratio");
        let err = DocxLoader::extract_text_from_bytes_with_limit(&docx, 1024).unwrap_err();
        assert!(err.to_string().contains("decompression bomb"), "got: {err}");
    }

    #[test]
    fn within_cap_extracts_ok() {
        let docx = build_docx("<w:p><w:t>Hello World</w:t></w:p>");
        let text = DocxLoader::extract_text_from_bytes_with_limit(&docx, 1 << 20).unwrap();
        assert!(text.contains("Hello World"));
    }
}
