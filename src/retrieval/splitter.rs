// src/retrieval/splitter.rs
//! 文本分割器实现
//!
//! 将长文档分割成较小的块，便于处理和检索。

use crate::vector_stores::Document;

/// 文本分割器 trait
pub trait TextSplitter: Send + Sync {
    /// 分割文本
    ///
    /// # 参数
    /// * `text` - 输入文本
    ///
    /// # 返回
    /// 文本块列表
    fn split_text(&self, text: &str) -> Vec<String>;

    /// 分割文档
    ///
    /// # 参数
    /// * `document` - 输入文档
    ///
    /// # 返回
    /// 文档块列表
    fn split_document(&self, document: &Document) -> Vec<Document> {
        let chunks = self.split_text(&document.content);
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut metadata = document.metadata.clone();
                metadata.insert("chunk".to_string(), i.to_string());

                Document {
                    content: chunk,
                    metadata,
                    id: None,
                }
            })
            .collect()
    }
}

/// 递归字符分割器
///
/// 按照分隔符优先级递归分割文本。
pub struct RecursiveCharacterSplitter {
    /// 块大小（字符数）
    chunk_size: usize,

    /// 块重叠（字符数）
    chunk_overlap: usize,

    /// 分隔符列表（按优先级）
    separators: Vec<String>,
}

impl RecursiveCharacterSplitter {
    /// 创建新的递归字符分割器
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators: vec![
                "\n\n".to_string(), // 段落
                "\n".to_string(),   // 行
                "。".to_string(),   // 中文句号
                ".".to_string(),    // 英文句号
                " ".to_string(),    // 空格
                "".to_string(),     // 字符
            ],
        }
    }

    /// 使用默认参数创建
    pub fn with_defaults() -> Self {
        Self::new(1000, 200)
    }

    /// 设置分隔符
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    /// 分割文本（内部方法）
    fn split_text_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        let mut chunks = Vec::new();

        if text.is_empty() {
            return chunks;
        }

        // 如果文本已经够小，直接返回
        if text.len() <= self.chunk_size {
            chunks.push(text.to_string());
            return chunks;
        }

        // 找到合适的分隔符
        let separator = separators
            .iter()
            .find(|s| text.contains(s.as_str()))
            .cloned()
            .unwrap_or_default();

        // 按分隔符分割
        let splits: Vec<String> = if separator.is_empty() {
            text.chars().map(|c| c.to_string()).collect()
        } else {
            text.split(&separator).map(|s| s.to_string()).collect()
        };

        // 合并分割结果
        let mut current_chunk = String::new();

        for split in splits {
            let split_with_sep = if separator.is_empty() {
                split.clone()
            } else if current_chunk.is_empty() {
                split
            } else {
                format!("{}{}", separator, split)
            };

            // 如果单个分割已经超过块大小，需要递归处理
            if split_with_sep.len() > self.chunk_size {
                // 先保存当前块
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }

                // 递归分割
                let next_separators = if separators.len() > 1 {
                    &separators[1..]
                } else {
                    &[]
                };

                let sub_chunks = self.split_text_recursive(&split_with_sep, next_separators);
                chunks.extend(sub_chunks);
            } else if current_chunk.len() + split_with_sep.len() > self.chunk_size {
                // 当前块已满，保存并开始新块
                chunks.push(current_chunk.clone());
                current_chunk = split_with_sep;
            } else {
                current_chunk.push_str(&split_with_sep);
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }
}

impl TextSplitter for RecursiveCharacterSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut chunks = self.split_text_recursive(text, &self.separators);

        // 处理重叠
        if self.chunk_overlap > 0 && chunks.len() > 1 {
            let mut overlapped = Vec::new();

            for (i, chunk) in chunks.into_iter().enumerate() {
                if i == 0 {
                    overlapped.push(chunk);
                } else {
                    // 从前一个块的末尾取一部分重叠内容（使用字符而非字节）
                    let prev = &overlapped[i - 1];
                    let chars: Vec<char> = prev.chars().collect();
                    let overlap_chars = chars.len().saturating_sub(self.chunk_overlap);
                    let overlap: String = chars[overlap_chars..].iter().collect();

                    overlapped.push(format!("{}{}", overlap, chunk));
                }
            }

            chunks = overlapped;
        }

        chunks
    }
}

/// 简单字符分割器
#[allow(dead_code)]
pub struct CharacterTextSplitter {
    /// 块大小
    chunk_size: usize,

    /// 块重叠
    chunk_overlap: usize,

    /// 分隔符
    separator: String,
}

#[allow(dead_code)]
impl CharacterTextSplitter {
    /// 创建新的字符分割器
    pub fn new(chunk_size: usize, chunk_overlap: usize, separator: &str) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separator: separator.to_string(),
        }
    }
}

impl TextSplitter for CharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let splits: Vec<&str> = text.split(&self.separator).collect();
        let mut chunks = Vec::new();
        let mut current = String::new();

        for split in splits {
            if current.len() + split.len() + self.separator.len() > self.chunk_size
                && !current.is_empty()
            {
                chunks.push(current.clone());

                // 添加重叠
                if self.chunk_overlap > 0 {
                    let overlap_start = current.len().saturating_sub(self.chunk_overlap);
                    current = current[overlap_start..].to_string();
                } else {
                    current.clear();
                }
            }

            if !current.is_empty() {
                current.push_str(&self.separator);
            }
            current.push_str(split);
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_splitter() {
        let splitter = RecursiveCharacterSplitter::new(50, 10);

        let text = "This is a sentence. This is another sentence. And a third one.";
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() <= 60); // 允许一些余量
        }
    }

    #[test]
    fn test_split_document() {
        let splitter = RecursiveCharacterSplitter::new(100, 20);

        let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
            .with_metadata("source", "test");

        let chunks = splitter.split_document(&doc);

        assert!(!chunks.is_empty());
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(chunk.metadata.contains_key("chunk"));
            assert_eq!(chunk.metadata.get("chunk"), Some(&i.to_string()));
            assert_eq!(chunk.metadata.get("source"), Some(&"test".to_string()));
        }
    }

    #[test]
    fn test_character_splitter() {
        let splitter = CharacterTextSplitter::new(20, 5, " ");

        let text = "This is a test sentence with multiple words";
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let splitter = RecursiveCharacterSplitter::new(100, 20);
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_text() {
        let splitter = RecursiveCharacterSplitter::new(1000, 200);
        let text = "Short text";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text");
    }
}
