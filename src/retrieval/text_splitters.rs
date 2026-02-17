use crate::retrieval::document::{Document, DocumentChunk};
pub use crate::retrieval::traits::TextSplitter;
use regex::Regex;

/// 递归字符文本分割器
pub struct RecursiveCharacterSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
}

impl RecursiveCharacterSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
                "".to_string(),
            ],
        }
    }

    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    /// 使用给定的分隔符分割文本
    fn split_with_separator(&self, text: &str, separator: &str) -> Vec<String> {
        if separator.is_empty() {
            // 如果分隔符为空，按字符分割
            text.chars().map(|c| c.to_string()).collect()
        } else {
            text.split(separator)
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .join(" &")
                .split(" &")
                .map(|s| s.to_string())
                .collect()
        }
    }

    /// 递归分割文本
    fn recursive_split(&self, text: &str, separators: &[String]) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        let separator = &separators[0];
        let splits = self.split_with_separator(text, separator);

        // 如果分割后还是太大，使用更小的分隔符
        if splits.len() == 1 && separators.len() > 1 {
            return self.recursive_split(text, &separators[1..]);
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for split in splits {
            if current_chunk.len() + split.len() + separator.len() <= self.chunk_size {
                if !current_chunk.is_empty() {
                    current_chunk.push_str(separator);
                }
                current_chunk.push_str(&split);
            } else {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    // 处理重叠部分
                    let overlap_size = (self.chunk_overlap).min(current_chunk.len());
                    current_chunk = current_chunk
                        .chars()
                        .skip(current_chunk.len() - overlap_size)
                        .collect();
                }
                current_chunk.push_str(separator);
                current_chunk.push_str(&split);
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }
}

impl TextSplitter for RecursiveCharacterSplitter {
    fn split_document(
        &self,
        document: &Document,
    ) -> Result<Vec<DocumentChunk>, Box<dyn std::error::Error>> {
        let chunks = self.recursive_split(&document.content, &self.separators);

        let mut result = Vec::new();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let mut chunk_metadata = document.metadata.clone();
            chunk_metadata.insert("chunk_index".to_string(), i.to_string());

            let doc_chunk = DocumentChunk {
                content: chunk,
                metadata: chunk_metadata,
                chunk_index: i,
                document_id: None,
            };
            result.push(doc_chunk);
        }

        Ok(result)
    }

    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Ok(self.recursive_split(text, &self.separators))
    }
}

/// 固定大小文本分割器（简单的按字符数分割）
pub struct FixedSizeSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl FixedSizeSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl TextSplitter for FixedSizeSplitter {
    fn split_document(
        &self,
        document: &Document,
    ) -> Result<Vec<DocumentChunk>, Box<dyn std::error::Error>> {
        let chars: Vec<char> = document.content.chars().collect();
        let mut chunks = Vec::new();

        let mut start = 0;
        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let chunk_chars = chars[start..end].iter().collect::<String>();

            let mut chunk_metadata = document.metadata.clone();
            chunk_metadata.insert("chunk_index".to_string(), chunks.len().to_string());

            let chunk = DocumentChunk {
                content: chunk_chars,
                metadata: chunk_metadata,
                chunk_index: chunks.len(),
                document_id: None,
            };
            chunks.push(chunk);

            // 移动到下一个chunk的开始位置（考虑重叠）
            if end >= chars.len() {
                break;
            }
            start = end - self.chunk_overlap;
        }

        Ok(chunks)
    }

    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();

        let mut start = 0;
        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let chunk_chars = chars[start..end].iter().collect::<String>();
            chunks.push(chunk_chars);

            if end >= chars.len() {
                break;
            }
            start = end - self.chunk_overlap;
        }

        Ok(chunks)
    }
}

/// 正则表达式分割器
pub struct RegexSplitter {
    pattern: Regex,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl RegexSplitter {
    pub fn new(
        pattern: &str,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
            chunk_size,
            chunk_overlap,
        })
    }
}

impl TextSplitter for RegexSplitter {
    fn split_document(
        &self,
        document: &Document,
    ) -> Result<Vec<DocumentChunk>, Box<dyn std::error::Error>> {
        let splits: Vec<String> = self
            .pattern
            .split(&document.content)
            .map(|s| s.to_string())
            .collect();

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for split in splits {
            if current_chunk.len() + split.len() <= self.chunk_size {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(&split);
            } else {
                if !current_chunk.is_empty() {
                    let mut chunk_metadata = document.metadata.clone();
                    chunk_metadata.insert("chunk_index".to_string(), chunks.len().to_string());

                    chunks.push(DocumentChunk {
                        content: current_chunk.clone(),
                        metadata: chunk_metadata,
                        chunk_index: chunks.len(),
                        document_id: None,
                    });

                    // 处理重叠
                    let overlap_size = self.chunk_overlap.min(current_chunk.len());
                    current_chunk = current_chunk
                        .chars()
                        .skip(current_chunk.len() - overlap_size)
                        .collect();
                }
                current_chunk.push_str(&split);
            }
        }

        if !current_chunk.is_empty() {
            let mut chunk_metadata = document.metadata.clone();
            chunk_metadata.insert("chunk_index".to_string(), chunks.len().to_string());

            chunks.push(DocumentChunk {
                content: current_chunk,
                metadata: chunk_metadata,
                chunk_index: chunks.len(),
                document_id: None,
            });
        }

        Ok(chunks)
    }

    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let splits: Vec<String> = self.pattern.split(text).map(|s| s.to_string()).collect();

        // 简单的重新组合逻辑
        let mut chunks = Vec::new();
        let mut current = String::new();

        for split in splits {
            if current.len() + split.len() <= self.chunk_size {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&split);
            } else {
                if !current.is_empty() {
                    chunks.push(current.clone());
                    current.clear();
                }
                current = split;
            }
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        Ok(chunks)
    }
}
