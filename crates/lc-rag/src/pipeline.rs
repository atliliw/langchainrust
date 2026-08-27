// lc-rag/src/pipeline.rs
//! RAGPipeline & RAGPipelineBuilder — a complete RAG pipeline in one line
//!
//! Provides a fluent Builder API that assembles LLM + Embeddings + VectorStore + Retriever
//! into a complete RAG pipeline.
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

/// RAG Pipeline — chunking + embedding + storage + retrieval + generation
///
/// Assembles an LLM with a `RetrieverTrait` implementation (BM25, vector similarity,
/// hybrid retrieval, etc.) into a complete RAG pipeline, exposing three core methods:
/// `index_documents()`, `query()`, `query_with_sources()`.
///
/// P0-2: The retrieval path converges on `Arc<dyn RetrieverTrait>` instead of depending
/// directly on `Embeddings + VectorStore`, so any retriever can be swapped in seamlessly.
pub struct RAGPipeline {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    retriever: Arc<dyn RetrieverTrait>,
    /// Number of documents to retrieve
    retrieve_k: usize,
    /// System prompt
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
    /// Indexes documents
    ///
    /// P0-2: Delegates to `RetrieverTrait::add_documents` (embedding + storage are handled
    /// internally by the retriever).
    pub async fn index_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        self.retriever.add_documents(documents).await
    }

    /// Query: retrieve + generate an answer
    ///
    /// 1. Embed the question
    /// 2. Retrieve similar documents from the VectorStore
    /// 3. Use the retrieved results as context and let the LLM generate the answer
    pub async fn query(&self, question: &str) -> Result<String, RetrieverError> {
        let result = self.query_with_sources(question).await?;
        Ok(result.answer)
    }

    /// Queries and returns the source documents
    ///
    /// Returns the generated answer and the list of retrieved source documents.
    pub async fn query_with_sources(
        &self,
        question: &str,
    ) -> Result<RAGQueryResult, RetrieverError> {
        // 1. Retrieve relevant documents (P0-2: delegated to RetrieverTrait)
        let search_results = self
            .retriever
            .retrieve_with_scores(question, self.retrieve_k)
            .await?;

        let sources: Vec<Document> = search_results.iter().map(|r| r.document.clone()).collect();

        // 3. Build the context
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

        // 4. Generate the answer
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
            .map_err(|e| RetrieverError::EmbeddingError(format!("LLM call failed: {}", e)))?;

        Ok(RAGQueryResult {
            answer: llm_result.content,
            sources,
        })
    }
}

/// RAG query result
#[derive(Debug, Clone)]
pub struct RAGQueryResult {
    /// The generated answer
    pub answer: String,
    /// The retrieved source documents
    pub sources: Vec<Document>,
}

// ---------------------------------------------------------------------------
// RAGPipelineBuilder
// ---------------------------------------------------------------------------

/// RAG Pipeline Builder — fluent API for creating a RAG pipeline
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
    /// P0-2: An explicitly passed-in retriever (takes priority); when absent, one is built
    /// from embeddings + vector_store
    retriever: Option<Arc<dyn RetrieverTrait>>,
    retrieve_k: usize,
    system_prompt: Option<String>,
}

impl RAGPipelineBuilder {
    /// Creates a new RAGPipelineBuilder
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

    /// Sets the LLM (any type implementing `BaseChatModel`)
    pub fn llm<L>(mut self, llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        self.llm = Some(lc_providers::wrap_chat_model(llm));
        self
    }

    /// Sets the LLM (from an already-wrapped `Arc<dyn BaseChatModel>`)
    pub fn llm_from_arc(
        mut self,
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    ) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Sets the LLM (from an `LLMClient`)
    pub fn llm_client(mut self, client: lc_providers::LLMClient) -> Self {
        let provider_arc = client.into_inner();
        self.llm = Some(provider_arc);
        self
    }

    /// Sets the Embeddings
    pub fn embeddings<E: Embeddings + Send + Sync + 'static>(mut self, embeddings: E) -> Self {
        self.embeddings = Some(Arc::new(embeddings));
        self
    }

    /// Sets the VectorStore
    pub fn vector_store<V: VectorStore + Send + Sync + 'static>(mut self, store: V) -> Self {
        self.vector_store = Some(Arc::new(store));
        self
    }

    /// Sets a custom retriever (any type implementing `RetrieverTrait`,
    /// such as BM25, UnifiedHybridIndex, etc.)
    ///
    /// P0-2: An explicit retriever takes priority over the similarity retriever built from
    /// `.embeddings() + .vector_store()`.
    pub fn retriever<R>(mut self, retriever: R) -> Self
    where
        R: RetrieverTrait + Send + Sync + 'static,
    {
        self.retriever = Some(Arc::new(retriever));
        self
    }

    /// Sets the retriever (from an already-wrapped `Arc<dyn RetrieverTrait>`)
    pub fn retriever_from_arc(mut self, retriever: Arc<dyn RetrieverTrait>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Sets the number of documents to retrieve
    pub fn retrieve_k(mut self, k: usize) -> Self {
        self.retrieve_k = k;
        self
    }

    /// Sets the system prompt
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Builds the RAGPipeline
    ///
    /// # Errors
    ///
    /// Returns an error if the LLM, Embeddings, or VectorStore is missing.
    pub fn build(self) -> Result<RAGPipeline, RetrieverError> {
        let llm = self.llm.ok_or_else(|| {
            RetrieverError::EmbeddingError(
                "RAGPipelineBuilder: LLM is required. Call .llm() first.".into(),
            )
        })?;

        // P0-2: Prefer the explicit retriever; otherwise fall back to building a
        // SimilarityRetriever from embeddings + vector_store for backward compatibility.
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

    /// P0-2: Supports injecting a custom `RetrieverTrait` implementation (BM25) via
    /// `.retriever()`; `index_documents` delegates to it without needing Embeddings/VectorStore.
    #[tokio::test]
    async fn test_builder_with_custom_retriever() {
        use crate::bm25::BM25Retriever;

        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let rag = RAGPipelineBuilder::new()
            .llm(OpenAIChat::new(config))
            .retriever(BM25Retriever::new())
            .build()
            .expect("build should succeed");

        rag.index_documents(vec![Document::new("Rust is a systems language")])
            .await
            .expect("index_documents should delegate to BM25 retriever successfully");
    }
}
