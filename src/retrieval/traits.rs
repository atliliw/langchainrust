use crate::retrieval::document::{Document, DocumentChunk, SearchResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::error::Error;

/// 检索器trait
#[async_trait]
pub trait Retriever: Send + Sync {
    /// 检索相关文档
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<SearchResult>, Box<dyn Error>>;

    /// 带过滤条件的检索
    async fn retrieve_with_filter(
        &self,
        query: &str,
        k: usize,
        _filter: HashMap<String, String>,
    ) -> Result<Vec<SearchResult>, Box<dyn Error>> {
        // 默认实现：忽略过滤条件
        self.retrieve(query, k).await
    }
}

/// 向量存储trait
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// 添加文档到向量存储
    async fn add_documents(
        &mut self,
        documents: Vec<(DocumentChunk, Vec<f32>)>,
    ) -> Result<(), Box<dyn Error>>;

    /// 相似度搜索
    async fn similarity_search(
        &self,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<(DocumentChunk, f32)>, Box<dyn Error>>;

    /// 带元数据过滤的相似度搜索
    async fn similarity_search_with_filter(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: HashMap<String, String>,
    ) -> Result<Vec<(DocumentChunk, f32)>, Box<dyn Error>> {
        // 默认实现：先搜索再过滤
        let results = self.similarity_search(query, k * 2).await?;
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|(chunk, _)| {
                filter
                    .iter()
                    .all(|(key, value)| chunk.metadata.get(key).map_or(false, |v| v == value))
            })
            .take(k)
            .collect();
        Ok(filtered)
    }

    /// 删除文档
    async fn delete_documents(&mut self, ids: Vec<String>) -> Result<(), Box<dyn Error>>;
}

/// 文档加载器trait
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    /// 加载文档
    async fn load(&self) -> Result<Vec<Document>, Box<dyn Error>>;

    /// 加载并分块
    async fn load_and_split(
        &self,
        splitter: &dyn TextSplitter,
    ) -> Result<Vec<DocumentChunk>, Box<dyn Error>> {
        let docs = self.load().await?;
        let mut chunks = Vec::new();

        for (doc_idx, doc) in docs.into_iter().enumerate() {
            let doc_chunks = splitter.split_document(&doc)?;
            for (_chunk_idx, mut chunk) in doc_chunks.into_iter().enumerate() {
                chunk.document_id = Some(format!("doc_{}", doc_idx));
                chunks.push(chunk);
            }
        }

        Ok(chunks)
    }
}

/// 文本分割器trait
pub trait TextSplitter: Send + Sync {
    /// 分割文档
    fn split_document(&self, document: &Document) -> Result<Vec<DocumentChunk>, Box<dyn Error>>;

    /// 分割文本
    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn Error>>;
}

/// 嵌入模型trait
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// 为文本生成嵌入
    async fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>>;

    /// 批量生成嵌入
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;

    /// 获取嵌入维度
    fn embedding_dimension(&self) -> usize;
}

/// 重排序器trait（用于提高检索质量）
#[async_trait]
pub trait Reranker: Send + Sync {
    /// 对搜索结果进行重排序
    async fn rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, Box<dyn Error>>;
}
