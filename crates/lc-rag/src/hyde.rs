// src/retrieval/hyde.rs
//! HyDE (Hypothetical Document Embedding) Retriever 实现
//!
//! 使用 LLM 生成假设文档，然后用假设文档进行检索，
//! 提升语义检索的召回率和精确度。

use lc_core::language_models::BaseChatModel;
use lc_prompts::PromptTemplate;
use lc_providers::ProviderError;
use lc_schema::Message;
use lc_vector_stores::{Document, SearchResult};

use crate::retriever::RetrieverTrait;
use std::collections::HashSet;
use std::sync::Arc;

/// HyDE 错误类型
#[derive(Debug)]
#[non_exhaustive]
pub enum HyDEError {
    /// LLM 调用错误
    LLMError(String),
    /// 向量化错误
    EmbeddingError(String),
    /// 基础检索器错误
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
    /// 创建使用默认配置的 `HyDEConfig`
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置检索文档数量
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// 设置假设文档生成的 prompt 模板
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_template = prompt.into();
        self
    }

    /// 设置是否包含原始查询结果
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
    /// LLM 用于生成假设文档
    ///
    /// P0-3: 不再硬编码 `OpenAIChat`,接受任意实现 `BaseChatModel` 的 LLM。
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    base_retriever: Arc<dyn RetrieverTrait>,
    config: HyDEConfig,
}

impl HyDERetriever {
    /// 创建 HyDERetriever(接受任意实现 `BaseChatModel` 的 LLM)
    ///
    /// P0-3: 移除死参数 `_embeddings`——假设文档的向量化由内部
    /// `base_retriever` 完成,不需要外部传入 Embeddings。
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

    /// P0-3: 从已包装的 `Arc<dyn BaseChatModel<Error = ProviderError>>` 构建
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

    /// 设置 HyDE 配置
    pub fn with_config(mut self, config: HyDEConfig) -> Self {
        self.config = config;
        self
    }

    /// 设置检索文档数量
    pub fn with_k(mut self, k: usize) -> Self {
        self.config.k = k;
        self
    }

    /// 设置是否包含原始查询结果
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

    /// 用 HyDE 流程检索文档：先生成假设文档，再检索并去重合并结果
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

    /// 带分数检索文档（HyDE 流程）
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

    /// 获取 LLM 生成的假设文档（不执行检索）
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
