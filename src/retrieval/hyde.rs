// src/retrieval/hyde.rs
//! HyDE (Hypothetical Document Embedding) Retriever 实现
//!
//! 使用 LLM 生成假设文档，然后用假设文档进行检索，
//! 提升语义检索的召回率和精确度。

use crate::embeddings::Embeddings;
use crate::language_models::OpenAIChat;
use crate::retrieval::RetrieverTrait;
use crate::schema::Message;
use crate::vector_stores::{Document, SearchResult};
use crate::Runnable;
use std::sync::Arc;

/// HyDE 错误类型
#[derive(Debug)]
pub enum HyDEError {
    LLMError(String),
    EmbeddingError(String),
    RetrieverError(String),
}

impl std::fmt::Display for HyDEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HyDEError::LLMError(msg) => write!(f, "LLM 错误: {}", msg),
            HyDEError::EmbeddingError(msg) => write!(f, "Embedding 错误: {}", msg),
            HyDEError::RetrieverError(msg) => write!(f, "检索错误: {}", msg),
        }
    }
}

impl std::error::Error for HyDEError {}

/// HyDE 配置
pub struct HyDEConfig {
    /// 假设文档生成的 prompt
    pub prompt_template: String,
    
    /// 检索文档数量
    pub k: usize,
    
    /// 是否包含原始查询结果
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
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    
    pub fn with_prompt(mut self, prompt: String) -> Self {
        self.prompt_template = prompt;
        self
    }
    
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
/// 工作流程：
/// 1. 用户提问
/// 2. LLM 生成假设文档（一个理想的答案）
/// 3. 将假设文档向量化
/// 4. 用假设文档向量检索真实文档
/// 5. 返回相关文档
pub struct HyDERetriever {
    llm: OpenAIChat,
    #[allow(dead_code)]
    embeddings: Arc<dyn Embeddings>,
    base_retriever: Arc<dyn RetrieverTrait>,
    config: HyDEConfig,
}

impl HyDERetriever {
    pub fn new(
        llm: OpenAIChat,
        embeddings: Arc<dyn Embeddings>,
        base_retriever: Arc<dyn RetrieverTrait>,
    ) -> Self {
        Self {
            llm,
            embeddings,
            base_retriever,
            config: HyDEConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: HyDEConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn with_k(mut self, k: usize) -> Self {
        self.config.k = k;
        self
    }
    
    pub fn with_include_original_query(mut self, include: bool) -> Self {
        self.config.include_original_query = include;
        self
    }
    
    async fn generate_hypothetical_document(&self, query: &str) -> Result<String, HyDEError> {
        let prompt = self.config.prompt_template.replace("{question}", query);
        
        let messages = vec![
            Message::human(prompt),
        ];
        
        let response = self.llm
            .invoke(messages, None)
            .await
            .map_err(|e| HyDEError::LLMError(e.to_string()))?;
        
        Ok(response.content)
    }
    
    pub async fn retrieve(&self, query: &str) -> Result<Vec<Document>, HyDEError> {
        let hyde_doc = self.generate_hypothetical_document(query).await?;
        
        let mut all_docs = Vec::new();
        
        let hyde_results = self.base_retriever
            .retrieve(&hyde_doc, self.config.k)
            .await
            .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;
        
        all_docs.extend(hyde_results);
        
        if self.config.include_original_query {
            let query_results = self.base_retriever
                .retrieve(query, self.config.k)
                .await
                .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;
            
            for doc in query_results {
                if !all_docs.iter().any(|d| d.content == doc.content) {
                    all_docs.push(doc);
                }
            }
        }
        
        Ok(all_docs)
    }
    
    pub async fn retrieve_with_scores(&self, query: &str) -> Result<Vec<SearchResult>, HyDEError> {
        let hyde_doc = self.generate_hypothetical_document(query).await?;
        
        let mut all_results: Vec<SearchResult> = Vec::new();
        
        let hyde_results = self.base_retriever
            .retrieve_with_scores(&hyde_doc, self.config.k)
            .await
            .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;
        
        all_results.extend(hyde_results);
        
        if self.config.include_original_query {
            let query_results = self.base_retriever
                .retrieve_with_scores(query, self.config.k)
                .await
                .map_err(|e| HyDEError::RetrieverError(e.to_string()))?;
            
            for result in query_results {
                if !all_results.iter().any(|r| r.document.content == result.document.content) {
                    all_results.push(result);
                }
            }
        }
        
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(all_results)
    }
    
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
        let config = HyDEConfig::new()
            .with_prompt(custom_prompt.clone());
        
        assert!(config.prompt_template.contains("{question}"));
    }
}