//! DOCX 文档加载器
//!
//! 从 .docx 文件加载文档内容(纯文本提取)。
//! 不依赖外部 crate,使用 ZIP 解压 + XML 解析提取文本。

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

/// DOCX 文档加载器
///
/// 从 .docx 文件路径加载文档,提取正文文本。
/// DOCX 本质是 ZIP 包,正文在 `word/document.xml` 中。
pub struct DocxLoader {
    /// 文件路径
    path: String,
}

impl DocxLoader {
    /// 从文件路径创建加载器
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// 从 DOCX 字节流提取文本
    ///
    /// DOCX 是 ZIP 格式,正文在 `word/document.xml` 中,
    /// 文本在 `<w:t>` 标签内。
    fn extract_text_from_bytes(data: &[u8]) -> Result<String, LoaderError> {
        let reader = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(reader)
            .map_err(|e| LoaderError::Other(format!("DOCX is not a valid ZIP: {}", e)))?;

        // 读取 word/document.xml
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

        // 提取 <w:t> 标签内容
        Self::extract_text_from_xml(&xml_content)
    }

    /// 从 document.xml 提取文本
    fn extract_text_from_xml(xml: &str) -> Result<String, LoaderError> {
        // M60: use pre-compiled regexes from LazyLock
        let mut result = String::new();
        let mut last_end = 0;

        for cap in WT_REGEX.captures_iter(xml) {
            let Some(m) = cap.get(1) else {
                continue;
            };
            // 检查这个 <w:t> 之前是否有 </w:p>(新段落)
            let before = &xml[last_end..m.start()];
            if PARA_REGEX.is_match(before) && !result.is_empty() {
                result.push('\n');
            } else if !result.is_empty() {
                // 同一段落内的连续文本
            }
            result.push_str(m.as_str());
            last_end = m.end();
        }

        Ok(result)
    }
}

// zip crate 在 dev-dependencies 中,我们需要在 features 中处理
// 但为简单起见,直接添加为依赖
// 注:此处使用 std::io + zip 最小依赖

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
        // w:t 可能带 xml:space="preserve" 属性
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
