// lc-rag/src/bm25/retriever.rs
//! BM25 检索器
//!
//! 关键词统计检索(中文英文都行,纯内存)。v0.15.0 起收敛到 `ChunkedBM25Index`:
//! 文档原文只落在 lc-vector-stores 的 `ChunkedDocumentStoreTrait`,检索结果按
//! parent 聚合返回(旧自持 `Vec<Document>` 的 `BM25Index` 已删除,消除第二落点)。

use super::chunked::{ChunkedBM25Retriever, ChunkedSearchResult};
use crate::retriever::{RetrieverError, RetrieverTrait};
use async_trait::async_trait;
use lc_vector_stores::document_store::{ChunkedDocumentStore, ChunkedDocumentStoreTrait};
use lc_vector_stores::{Document, SearchResult, VectorStoreError};
use std::sync::{Arc, Mutex};

/// 关键词统计检索器。
///
/// P3-1: 内部持 `ChunkedBM25Retriever`(其索引即 `ChunkedBM25Index`),与
/// `UnifiedHybridIndex` 共用同一套 store 落点,不再自持文档原文。
pub struct BM25Retriever<S: ChunkedDocumentStoreTrait = ChunkedDocumentStore> {
    retriever: Mutex<ChunkedBM25Retriever<S>>,
}

impl BM25Retriever<ChunkedDocumentStore> {
    /// 创建检索器(内部自持一个内存 store,无需外部参数)。
    pub fn new() -> Self {
        Self::with_store(Arc::new(ChunkedDocumentStore::new()))
    }

    /// 使用自定义 BM25 参数(k1, b),内部自持内存 store。
    pub fn with_params(k1: f64, b: f64) -> Self {
        Self {
            retriever: Mutex::new(ChunkedBM25Retriever::with_params(
                Arc::new(ChunkedDocumentStore::new()),
                k1,
                b,
            )),
        }
    }
}

impl<S: ChunkedDocumentStoreTrait> BM25Retriever<S> {
    /// 使用共享 store 创建(与 `UnifiedHybridIndex` 等共用文档落点)。
    pub fn with_store(store: Arc<S>) -> Self {
        Self {
            retriever: Mutex::new(ChunkedBM25Retriever::new(store)),
        }
    }

    /// 添加单篇文档(切块后索引,文档原文进 store)。
    pub fn add_document(&self, document: Document) -> Result<(), VectorStoreError> {
        let mut retriever = self.retriever.lock().unwrap_or_else(|e| e.into_inner());
        retriever.add_document(document)
    }

    /// 批量添加文档(同步;单篇失败时跳过并 warn,保持原 `()` 签名)。
    pub fn add_documents_sync(&self, documents: Vec<Document>) {
        for doc in documents {
            if let Err(e) = self.add_document(doc) {
                log::warn!("BM25Retriever::add_documents_sync: 跳过文档,入库失败: {e}");
            }
        }
    }

    /// 关键词检索,返回 parent 级结果(文档 id 为 parent_id)。
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let mut retriever = self.retriever.lock().unwrap_or_else(|e| e.into_inner());

        retriever
            .search(query, k)
            .into_iter()
            .map(|r: ChunkedSearchResult| SearchResult {
                document: Document::new(r.content()).with_id(r.parent_id),
                score: r.score,
            })
            .collect()
    }

    /// 已索引的 chunk 数(每篇文档至少一个 chunk)。
    pub fn len(&self) -> usize {
        self.retriever
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 清空 BM25 索引。共享 store 时,store 中的文档由调用方自行清空
    /// (`ChunkedDocumentStoreTrait::clear`),与本检索器无关。
    pub fn clear(&self) {
        self.retriever
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// P0-1: `BM25Retriever` 实现 `RetrieverTrait`,可与其他检索器统一通过
/// `Arc<dyn RetrieverTrait>` 使用。
#[async_trait]
impl<S: ChunkedDocumentStoreTrait> RetrieverTrait for BM25Retriever<S> {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        Ok(self
            .search(query, k)
            .into_iter()
            .map(|r| r.document)
            .collect())
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        Ok(self.search(query, k))
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        for doc in documents {
            self.add_document(doc).map_err(RetrieverError::StoreError)?;
        }
        Ok(())
    }
}

impl Default for BM25Retriever<ChunkedDocumentStore> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_retriever_basic() {
        let retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ]);

        assert_eq!(retriever.len(), 3);

        let results = retriever.search("programming language", 2);
        assert_eq!(results.len(), 2);

        assert!(results[0].document.content.contains("programming"));
    }

    #[test]
    fn test_bm25_retriever_chinese() {
        let retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust 是一门系统编程语言"),
            Document::new("Python 是脚本语言"),
            Document::new("JavaScript 用于网页开发"),
        ]);

        let results = retriever.search("编程语言", 2);
        assert!(!results.is_empty());

        assert!(results[0].document.content.contains("编程"));
    }

    #[test]
    fn test_bm25_retriever_empty() {
        let retriever = BM25Retriever::new();

        let results = retriever.search("test", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_retriever_params() {
        let retriever = BM25Retriever::with_params(2.0, 0.5);

        retriever.add_documents_sync(vec![
            Document::new("Rust programming"),
            Document::new("Python scripting"),
        ]);

        let results = retriever.search("programming", 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bm25_retriever_no_match() {
        let retriever = BM25Retriever::new();

        retriever.add_documents_sync(vec![
            Document::new("Rust programming language"),
            Document::new("Python scripting language"),
        ]);

        let results = retriever.search("javascript typescript", 5);
        assert!(results.is_empty());
    }
}
