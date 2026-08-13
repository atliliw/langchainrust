// lc-rag/src/pipeline.rs
//! RAGPipeline & RAGPipelineBuilder — 一行搞定 RAG 管线
//!
//! 提供流畅的 Builder API，将 LLM + Embeddings + VectorStore + Retriever
//! 组装成完整的 RAG 管线。
//!
//! # Example
//!
//! ```ignore
//! let rag = RAGPipelineBuilder::new()
//!     .llm(OpenAIChat::new(OpenAIConfig::new("sk-...")))
//!     .embeddings(OpenAIEmbeddings::new(config)?)
//!     .vector_store(InMemoryVectorStore::new())
//!     .build()?;
//!
//! rag.index_documents(docs).await?;
//! let answer = rag.query("What is RustB?").await?;
//! ```

use lc_core::language_models::BaseChatModel;
use lc_embeddings::Embeddings;
use lc_providers::ProviderError;
use lc_schema::Message;
use lc_vector_stores::{Document, VectorStore, VectorStoreError};

use crate::retriever::{RetrieverError, RetrieverTrait, SimilarityRetriever};

use std::sync::Arc;

/// RAG Pipeline — 切分 + 嵌入 + 存储 + 检索 + 生成
///
/// 将 LLM 与一个 `RetrieverTrait` 实现(BM25、向量相似度、混合检索等)组装成
/// 完整的 RAG 管线，提供 `index_documents()`、`query()`、`query_with_sources()`
/// 三个核心方法。
///
/// P0-2: 检索路径收敛到 `Arc<dyn RetrieverTrait>`,不再直接依赖
/// `Embeddings + VectorStore`,可无缝切换任意检索器。
pub struct RAGPipeline {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    retriever: Arc<dyn RetrieverTrait>,
    /// 检索文档数量
    retrieve_k: usize,
    /// 系统提示词
    system_prompt: String,
}

impl std::fmt::Debug for RAGPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RAGPipeline")
            .field("model_name", &self.llm.model_name())
            .field("retrieve_k", &self.retrieve_k)
            .field("system_prompt", &self.system_prompt)
            .finish()
    }
}

impl RAGPipeline {
    /// 索引文档
    ///
    /// P0-2: 委托给 `RetrieverTrait::add_documents`(嵌入 + 存储由检索器内部完成)。
    pub async fn index_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        self.retriever.add_documents(documents).await
    }

    /// 查询：检索 + 生成回答
    ///
    /// 1. 将问题嵌入向量
    /// 2. 从 VectorStore 检索相似文档
    /// 3. 将检索结果作为上下文，让 LLM 生成回答
    pub async fn query(&self, question: &str) -> Result<String, RetrieverError> {
        let result = self.query_with_sources(question).await?;
        Ok(result.answer)
    }

    /// 查询并返回来源文档
    ///
    /// 返回生成的答案和检索到的源文档列表。
    pub async fn query_with_sources(
        &self,
        question: &str,
    ) -> Result<RAGQueryResult, RetrieverError> {
        // 1. 检索相关文档(P0-2: 委托给 RetrieverTrait)
        let search_results = self
            .retriever
            .retrieve_with_scores(question, self.retrieve_k)
            .await?;

        let sources: Vec<Document> = search_results.iter().map(|r| r.document.clone()).collect();

        // 3. 构建上下文
        let context = if sources.is_empty() {
            "No relevant documents found.".to_string()
        } else {
            sources
                .iter()
                .enumerate()
                .map(|(i, doc)| format!("[{}] {}", i + 1, doc.page_content()))
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        // 4. 生成回答
        let messages = vec![
            Message::system(format!(
                "{}\n\nUse the following context to answer the question. If the context doesn't contain the answer, say so.",
                self.system_prompt
            )),
            Message::human(format!("Context:\n{}\n\nQuestion: {}", context, question)),
        ];

        let llm_result = self
            .llm
            .chat(messages, None)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(format!("LLM 调用失败: {}", e)))?;

        Ok(RAGQueryResult {
            answer: llm_result.content,
            sources,
        })
    }
}

/// RAG 查询结果
#[derive(Debug, Clone)]
pub struct RAGQueryResult {
    /// 生成的答案
    pub answer: String,
    /// 检索到的源文档
    pub sources: Vec<Document>,
}

// ---------------------------------------------------------------------------
// RAGPipelineBuilder
// ---------------------------------------------------------------------------

/// RAG Pipeline Builder — 流畅 API 创建 RAG 管线
///
/// # Example
///
/// ```ignore
/// let rag = RAGPipelineBuilder::new()
///     .llm(OpenAIChat::new(OpenAIConfig::new("sk-...")))
///     .embeddings(OpenAIEmbeddings::new(config)?)
///     .vector_store(InMemoryVectorStore::new())
///     .build()?;
/// ```
pub struct RAGPipelineBuilder {
    llm: Option<Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>>,
    embeddings: Option<Arc<dyn Embeddings + Send + Sync>>,
    vector_store: Option<Arc<dyn VectorStore + Send + Sync>>,
    /// P0-2: 显式传入的检索器(优先);未提供时用 embeddings + vector_store 构建
    retriever: Option<Arc<dyn RetrieverTrait>>,
    retrieve_k: usize,
    system_prompt: Option<String>,
}

impl RAGPipelineBuilder {
    /// 创建新的 RAGPipelineBuilder
    pub fn new() -> Self {
        Self {
            llm: None,
            embeddings: None,
            vector_store: None,
            retriever: None,
            retrieve_k: 4,
            system_prompt: None,
        }
    }

    /// 设置 LLM（任何实现了 `BaseChatModel` 的类型）
    pub fn llm<L>(mut self, llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        self.llm = Some(lc_providers::wrap_chat_model(llm));
        self
    }

    /// 设置 LLM（从已包装的 `Arc<dyn BaseChatModel>`）
    pub fn llm_from_arc(
        mut self,
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    ) -> Self {
        self.llm = Some(llm);
        self
    }

    /// 设置 LLM（从 `LLMClient`）
    pub fn llm_client(mut self, client: lc_providers::LLMClient) -> Self {
        let provider_arc = client.into_inner();
        self.llm = Some(provider_arc);
        self
    }

    /// 设置 Embeddings
    pub fn embeddings<E: Embeddings + Send + Sync + 'static>(mut self, embeddings: E) -> Self {
        self.embeddings = Some(Arc::new(embeddings));
        self
    }

    /// 设置 VectorStore
    pub fn vector_store<V: VectorStore + Send + Sync + 'static>(mut self, store: V) -> Self {
        self.vector_store = Some(Arc::new(store));
        self
    }

    /// 设置自定义检索器(任何实现了 `RetrieverTrait` 的类型,
    /// 如 BM25、UnifiedHybridIndex、ChunkedHybridRetriever 等)
    ///
    /// P0-2: 显式检索器优先于 `.embeddings() + .vector_store()` 构建的
    /// 相似度检索器。
    pub fn retriever<R>(mut self, retriever: R) -> Self
    where
        R: RetrieverTrait + Send + Sync + 'static,
    {
        self.retriever = Some(Arc::new(retriever));
        self
    }

    /// 设置检索器(从已包装的 `Arc<dyn RetrieverTrait>`)
    pub fn retriever_from_arc(mut self, retriever: Arc<dyn RetrieverTrait>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// 设置检索文档数量
    pub fn retrieve_k(mut self, k: usize) -> Self {
        self.retrieve_k = k;
        self
    }

    /// 设置系统提示词
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// 构建 RAGPipeline
    ///
    /// # Errors
    ///
    /// 如果缺少 LLM、Embeddings 或 VectorStore，返回错误。
    pub fn build(self) -> Result<RAGPipeline, RetrieverError> {
        let llm = self.llm.ok_or_else(|| {
            RetrieverError::EmbeddingError(
                "RAGPipelineBuilder: LLM is required. Call .llm() first.".into(),
            )
        })?;

        // P0-2: 优先使用显式检索器;否则回退到 embeddings + vector_store
        // 构建 SimilarityRetriever,兼容旧用法。
        let retriever = match self.retriever {
            Some(r) => r,
            None => {
                let embeddings = self.embeddings.ok_or_else(|| {
                    RetrieverError::EmbeddingError(
                        "RAGPipelineBuilder: Embeddings is required (or use .retriever()). Call .embeddings() first."
                            .into(),
                    )
                })?;

                let vector_store = self.vector_store.ok_or_else(|| {
                    RetrieverError::StoreError(VectorStoreError::StorageError(
                        "RAGPipelineBuilder: VectorStore is required (or use .retriever()). Call .vector_store() first."
                            .into(),
                    ))
                })?;

                Arc::new(SimilarityRetriever::new(vector_store, embeddings))
            }
        };

        Ok(RAGPipeline {
            llm,
            retriever,
            retrieve_k: self.retrieve_k,
            system_prompt: self.system_prompt.unwrap_or_else(|| {
                "You are a helpful assistant that answers questions based on the provided context.".to_string()
            }),
        })
    }
}

impl Default for RAGPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::MockEmbeddings;
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use lc_vector_stores::InMemoryVectorStore;

    #[test]
    fn test_builder_missing_llm() {
        let result = RAGPipelineBuilder::new()
            .embeddings(MockEmbeddings::new(3))
            .vector_store(InMemoryVectorStore::new())
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("LLM is required"));
    }

    #[test]
    fn test_builder_missing_embeddings() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let result = RAGPipelineBuilder::new()
            .llm(OpenAIChat::new(config))
            .vector_store(InMemoryVectorStore::new())
            .build();

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Embeddings is required"));
    }

    #[test]
    fn test_builder_missing_vector_store() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let result = RAGPipelineBuilder::new()
            .llm(OpenAIChat::new(config))
            .embeddings(MockEmbeddings::new(3))
            .build();

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("VectorStore is required"));
    }

    #[test]
    fn test_builder_success() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let result = RAGPipelineBuilder::new()
            .llm(OpenAIChat::new(config))
            .embeddings(MockEmbeddings::new(3))
            .vector_store(InMemoryVectorStore::new())
            .system("You are a test assistant.")
            .retrieve_k(5)
            .build();

        assert!(result.is_ok());
        let pipeline = result.unwrap();
        assert_eq!(pipeline.retrieve_k, 5);
        assert_eq!(pipeline.system_prompt, "You are a test assistant.");
    }

    #[test]
    fn test_builder_default() {
        let builder = RAGPipelineBuilder::default();
        assert_eq!(builder.retrieve_k, 4);
        assert!(builder.llm.is_none());
    }

    /// P0-2: 支持通过 `.retriever()` 注入自定义 `RetrieverTrait` 实现(BM25),
    /// `index_documents` 委托给该检索器,无需 Embeddings/VectorStore。
    #[tokio::test]
    async fn test_builder_with_custom_retriever() {
        use crate::bm25::BM25Retriever;

        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let rag = RAGPipelineBuilder::new()
            .llm(OpenAIChat::new(config))
            .retriever(BM25Retriever::new())
            .build()
            .expect("build 应成功");

        rag.index_documents(vec![Document::new("Rust is a systems language")])
            .await
            .expect("index_documents 应委托给 BM25 检索器成功");
    }
}
