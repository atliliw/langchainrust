// src/retrieval/semantic_splitter.rs
//! 语义分块器
//!
//! 按语义相关性切分文本:先分句并嵌入,在相邻句向量相似度骤降处断块,
//! 相比字符级分割能更好保留语义完整性,提升检索质量。
//!
//! 注:嵌入是异步操作,而 `TextSplitter` trait 是同步签名。为不破坏现有同步 trait,
//! 本分块器提供独立的异步接口 `split_text` / `split_document`,不实现同步 `TextSplitter`。

use crate::embeddings::{cosine_similarity, Embeddings};
use crate::vector_stores::Document;

/// 语义分块器
///
/// 在相邻句相似度低于 `breakpoint_threshold` 处断块;
/// 累积长度超过 `max_chunk_size` 时强制断。
pub struct SemanticSplitter<E> {
    embeddings: E,
    /// 相邻句相似度低于此阈值则断块
    breakpoint_threshold: f32,
    /// 单块最大字符数,超出强制断
    max_chunk_size: usize,
}

impl<E: Embeddings> SemanticSplitter<E> {
    /// 创建语义分块器
    ///
    /// # 参数
    /// * `embeddings` - 嵌入模型
    /// * `breakpoint_threshold` - 相邻句相似度断点阈值(0.0–1.0,越低越不易断)
    /// * `max_chunk_size` - 单块最大字符数
    pub fn new(embeddings: E, breakpoint_threshold: f32, max_chunk_size: usize) -> Self {
        Self {
            embeddings,
            breakpoint_threshold,
            max_chunk_size,
        }
    }

    /// 使用默认参数创建(threshold=0.5, max=1000)
    pub fn with_defaults(embeddings: E) -> Self {
        Self::new(embeddings, 0.5, 1000)
    }

    /// 分句:支持中文(`。!?;`)与英文(`.!?\n`)
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if matches!(ch, '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?') {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }
        sentences
    }

    /// 异步分块
    pub async fn split_text(
        &self,
        text: &str,
    ) -> Result<Vec<String>, crate::embeddings::EmbeddingError> {
        let sentences = Self::split_sentences(text);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }
        if sentences.len() == 1 {
            return Ok(vec![sentences.into_iter().next().expect("已校验非空")]);
        }

        // 批量嵌入
        let refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let embeddings = self.embeddings.embed_documents(&refs).await?;

        let mut chunks = Vec::new();
        let mut current = sentences[0].clone();

        for i in 1..sentences.len() {
            let sim = cosine_similarity(&embeddings[i - 1], &embeddings[i]).unwrap_or(0.0);
            let would_exceed = current.len() + sentences[i].len() + 1 > self.max_chunk_size;

            if sim < self.breakpoint_threshold || would_exceed {
                chunks.push(std::mem::take(&mut current));
                current = sentences[i].clone();
            } else {
                current.push('\n');
                current.push_str(&sentences[i]);
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        Ok(chunks)
    }

    /// 异步分块文档,保留 metadata(写入 chunk 序号)
    pub async fn split_document(
        &self,
        document: &Document,
    ) -> Result<Vec<Document>, crate::embeddings::EmbeddingError> {
        let chunks = self.split_text(&document.content).await?;
        Ok(chunks
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
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{EmbeddingError, Embeddings, MockEmbeddings};
    use async_trait::async_trait;

    /// 总是失败的嵌入,用于测试回退路径
    struct FailingEmbeddings;
    #[async_trait]
    impl Embeddings for FailingEmbeddings {
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            Err(EmbeddingError::ApiError("intentional failure".to_string()))
        }
        fn dimension(&self) -> usize {
            32
        }
        fn model_name(&self) -> &str {
            "failing"
        }
    }

    fn splitter(threshold: f32, max_chunk: usize) -> SemanticSplitter<MockEmbeddings> {
        SemanticSplitter::new(MockEmbeddings::new(32), threshold, max_chunk)
    }

    #[tokio::test]
    async fn test_empty_text() {
        let s = splitter(0.5, 1000);
        assert!(s.split_text("").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_single_sentence() {
        let s = splitter(0.5, 1000);
        let chunks = s.split_text("只有一句没有标点").await.unwrap();
        assert_eq!(chunks, vec!["只有一句没有标点".to_string()]);
    }

    #[tokio::test]
    async fn test_chunks_contain_all_sentences() {
        let s = splitter(0.5, 1000);
        let text = "苹果是一种水果。香蕉是黄色的。樱桃很小。";
        let chunks = s.split_text(text).await.unwrap();
        assert!(!chunks.is_empty());
        // 无论怎么断,每句都应出现在某个 chunk 中
        let joined = chunks.join("");
        assert!(joined.contains("苹果是一种水果"));
        assert!(joined.contains("香蕉是黄色的"));
        assert!(joined.contains("樱桃很小"));
    }

    #[tokio::test]
    async fn test_max_chunk_size_enforces_break() {
        // max_chunk 设很小,多句必然被强制断成多块
        let s = splitter(0.0, 5);
        let text = "AAAA。BBBB。CCCC。";
        let chunks = s.split_text(text).await.unwrap();
        assert!(
            chunks.len() >= 2,
            "max_chunk=5 应强制断块, 实际 {} 块",
            chunks.len()
        );
    }

    #[tokio::test]
    async fn test_split_document_metadata() {
        let s = splitter(0.0, 5);
        let doc = Document::new("AAAA。BBBB。CCCC。").with_metadata("source", "test");
        let chunks = s.split_document(&doc).await.unwrap();
        assert!(chunks.len() >= 2);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.metadata.get("chunk"), Some(&i.to_string()));
            assert_eq!(c.metadata.get("source"), Some(&"test".to_string()));
            assert!(c.id.is_none());
        }
    }

    #[tokio::test]
    async fn test_embedding_failure_returns_error() {
        // M54: embedding failure now returns an error instead of fallback
        let s = SemanticSplitter::new(FailingEmbeddings, 0.5, 1000);
        let text = "句子一。句子二。句子三。";
        let result = s.split_text(text).await;
        assert!(result.is_err());
    }
}
