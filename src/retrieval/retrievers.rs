use async_trait::async_trait;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::retrieval::document::{DocumentChunk, SearchResult};
use crate::retrieval::traits::{Retriever, VectorStore, EmbeddingModel};

/// 基于相似度的基础检索器
pub struct SimilarityRetriever {
    vector_store: Arc<RwLock<Box<dyn VectorStore>>>,
    embedding_model: Arc<dyn EmbeddingModel>,
}

impl SimilarityRetriever {
    pub fn new(
        vector_store: Box<dyn VectorStore>,
        embedding_model: Arc<dyn EmbeddingModel>,
    ) -> Self {
        Self {
            vector_store: Arc::new(RwLock::new(vector_store)),
            embedding_model,
        }
    }

    /// 添加文档到向量存储
    pub async fn add_documents(
        &self,
        chunks: Vec<DocumentChunk>
    ) -> Result<(), Box<dyn Error>> {
        // 批量生成嵌入
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embedding_model.embed_batch(texts).await?;

        // 创建文档和嵌入的配对
        let documents_with_embeddings: Vec<(DocumentChunk, Vec<f32>)> = chunks
            .into_iter()
            .zip(embeddings.into_iter())
            .collect();

        // 添加到向量存储
        self.vector_store.write().await.add_documents(documents_with_embeddings).await?;
        Ok(())
    }
}

#[async_trait]
impl Retriever for SimilarityRetriever {
    async fn retrieve(
        &self,
        query: &str,
        k: usize
    ) -> Result<Vec<SearchResult>, Box<dyn Error>> {
        // 生成查询嵌入
        let query_embedding = self.embedding_model.embed(query).await?;

        // 在向量存储中搜索
        let results = self.vector_store.read().await
            .similarity_search(query_embedding, k).await?;

        // 转换为SearchResult格式
        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(chunk, score)| SearchResult::new(chunk, score))
            .collect();

        Ok(search_results)
    }

    async fn retrieve_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: HashMap<String, String>
    ) -> Result<Vec<SearchResult>, Box<dyn Error>> {
        // 生成查询嵌入
        let query_embedding = self.embedding_model.embed(query).await?;

        // 在向量存储中搜索并过滤
        let results = self.vector_store.read().await
            .similarity_search_with_filter(query_embedding, k, filter).await?;

        // 转换为SearchResult格式
        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(chunk, score)| SearchResult::new(chunk, score))
            .collect();

        Ok(search_results)
    }
}

/// 重排序检索器包装器
pub struct RerankerRetriever {
    base_retriever: Arc<dyn Retriever>,
    reranker: Arc<dyn super::traits::Reranker>,
    final_k: usize,
}

impl RerankerRetriever {
    pub fn new(
        base_retriever: Arc<dyn Retriever>,
        reranker: Arc<dyn super::traits::Reranker>,
        final_k: usize
    ) -> Self {
        Self {
            base_retriever,
            reranker,
            final_k,
        }
    }
}

#[async_trait]
impl Retriever for RerankerRetriever {
    async fn retrieve(
        &self,
        query: &str,
        k: usize
    ) -> Result<Vec<SearchResult>, Box<dyn Error>> {
        // 先检索更多结果用于重排序
        let initial_results = self.base_retriever.retrieve(query, k * 3).await?;

        // 重排序
        let reranked_results = self.reranker.rerank(query, initial_results).await?;

        // 返回前final_k个结果
        Ok(reranked_results.into_iter().take(self.final_k).collect())
    }
}