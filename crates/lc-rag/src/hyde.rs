// src/retrieval/hyde.rs
//! HyDE (Hypothetical Document Embedding) Retriever implementation
//!
//! Uses an LLM to generate a hypothetical document, then retrieves with that hypothetical
//! document, improving retrieval recall and precision.

use lc_core::language_models::BaseChatModel;
use lc_prompts::PromptTemplate;
use lc_providers::ProviderError;
use lc_schema::Message;
use lc_vector_stores::{Document, SearchResult};

use crate::retriever::RetrieverTrait;
use std::collections::HashSet;
use std::sync::Arc;

/// HyDE error type
#[derive(Debug)]
#[non_exhaustive]
pub enum HyDEError {
    /// LLM call error
    LLMError(String),
    /// Embedding error
    EmbeddingError(String),
    /// Base retriever error
    RetrieverError(String),
}

impl std::fmt::Display for HyDEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HyDEError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            HyDEError::EmbeddingError(msg) => write!(f, "embedding error: {}", msg),
            HyDEError::RetrieverError(msg) => write!(f, "retrieval error: {}", msg),
        }
    }
}

impl std::error::Error for HyDEError {}

/// HyDE configuration
pub struct HyDEConfig {
    /// Prompt used to generate the hypothetical document
    pub prompt_template: String,

    /// Number of documents to retrieve
    pub k: usize,

    /// Whether to include the original query results
    pub include_original_query: bool,
}

impl Default for HyDEConfig {
    fn default() -> Self {
        Self {
            prompt_template: DEFAULT_HYDE_PROMPT.to_string(),
            k: 5,
            include_original_query: true,
        }
    }
}

impl HyDEConfig {
    /// Creates a `HyDEConfig` with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of documents to retrieve
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the prompt template used to generate the hypothetical document
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_template = prompt.into();
        self
    }

    /// Sets whether to include the original query results
    pub fn with_include_original_query(mut self, include: bool) -> Self {
        self.include_original_query = include;
        self
    }
}

const DEFAULT_HYDE_PROMPT: &str = r#"Please write a passage to answer the question.

Question: {question}

Passage:"#;

/// HyDE Retriever
///
/// Workflow:
/// 1. The user asks a question
/// 2. The LLM generates a hypothetical document (an ideal answer)
/// 3. The hypothetical document is embedded
/// 4. The hypothetical document vector retrieves real documents
/// 5. Returns the relevant documents
pub struct HyDERetriever {
    /// The LLM used to generate the hypothetical document
    ///
    /// P0-3: no longer hardcodes `OpenAIChat`; accepts any LLM implementing `BaseChatModel`.
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    base_retriever: Arc<dyn RetrieverTrait>,
    config: HyDEConfig,
}

impl HyDERetriever {
    /// Creates a HyDERetriever (accepting any LLM implementing `BaseChatModel`)
    ///
    /// P0-3: removes the dead `_embeddings` parameter — embedding the hypothetical document
    /// is handled internally by the `base_retriever`, so no external Embeddings is needed.
    pub fn new<L>(llm: L, base_retriever: Arc<dyn RetrieverTrait>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: lc_providers::wrap_chat_model(llm),
            base_retriever,
            config: HyDEConfig::default(),
        }
    }

    /// P0-3: builds from an already-wrapped `Arc<dyn BaseChatModel<Error = ProviderError>>`
    pub fn new_arc(
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
        base_retriever: Arc<dyn RetrieverTrait>,
    ) -> Self {
        Self {
            llm,
            base_retriever,
            config: HyDEConfig::default(),
        }
    }

    /// Sets the HyDE configuration
    pub fn with_config(mut self, config: HyDEConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the number of documents to retrieve
    pub fn with_k(mut self, k: usize) -> Self {
        self.config.k = k;
        self
    }

    /// Sets whether to include the original query results
    pub fn with_include_original_query(mut self, include: bool) -> Self {
        self.config.include_original_query = include;
        self
    }

    async fn generate_hypothetical_document(&self, query: &str) -> Result<String, HyDEError> {
        let template = PromptTemplate::new(&self.config.prompt_template);
        let mut vars = std::collections::HashMap::new();
        vars.insert("question", query);
        let prompt = template
            .format(&vars)
            .unwrap_or_else(|_| self.config.prompt_template.clone());

        let messages = vec![Message::human(prompt)];

        let response = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| HyDEError::LLMError(e.to_string()))?;

        Ok(response.content)
    }

    /// Retrieves documents via the HyDE flow: generates a hypothetical document first, then retrieves and merges deduped results
    pub async fn retrieve(&self, query: &str) -> Result<Vec<Document>, HyDEError> {
        let hyde_doc = self.generate_hypothetical_document(query).await?;

        let mut all_docs = Vec::new();
        let mut seen_content: HashSet<String> = HashSet::new();

        let hyde_results = self
            .base_retriever
            .retrieve(&hyde_doc, self.config.k)
            .await
            .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;

        for doc in &hyde_results {
            seen_content.insert(doc.content.clone());
        }
        all_docs.extend(hyde_results);

        if self.config.include_original_query {
            let query_results = self
                .base_retriever
                .retrieve(query, self.config.k)
                .await
                .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;

            for doc in query_results {
                if seen_content.insert(doc.content.clone()) {
                    all_docs.push(doc);
                }
            }
        }

        Ok(all_docs)
    }

    /// Retrieves documents with scores (HyDE flow)
    pub async fn retrieve_with_scores(&self, query: &str) -> Result<Vec<SearchResult>, HyDEError> {
        let hyde_doc = self.generate_hypothetical_document(query).await?;

        let mut all_results: Vec<SearchResult> = Vec::new();
        let mut seen_content: HashSet<String> = HashSet::new();

        let hyde_results = self
            .base_retriever
            .retrieve_with_scores(&hyde_doc, self.config.k)
            .await
            .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;

        for r in &hyde_results {
            seen_content.insert(r.document.content.clone());
        }
        all_results.extend(hyde_results);

        if self.config.include_original_query {
            let query_results = self
                .base_retriever
                .retrieve_with_scores(query, self.config.k)
                .await
                .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;

            for result in query_results {
                if seen_content.insert(result.document.content.clone()) {
                    all_results.push(result);
                }
            }
        }

        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_results)
    }

    /// Returns the LLM-generated hypothetical document (without retrieving)
    pub async fn get_hypothetical_document(&self, query: &str) -> Result<String, HyDEError> {
        self.generate_hypothetical_document(query).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyde_config_default() {
        let config = HyDEConfig::default();

        assert_eq!(config.k, 5);
        assert!(config.include_original_query);
        assert!(config.prompt_template.contains("{question}"));
    }

    #[test]
    fn test_hyde_config_custom() {
        let config = HyDEConfig::new()
            .with_k(10)
            .with_include_original_query(false);

        assert_eq!(config.k, 10);
        assert!(!config.include_original_query);
    }

    #[test]
    fn test_hyde_config_prompt() {
        let custom_prompt = "Answer this: {question}".to_string();
        let config = HyDEConfig::new().with_prompt(custom_prompt.clone());

        assert!(config.prompt_template.contains("{question}"));
    }
}
