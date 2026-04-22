// src/retrieval/multi_query.rs
//! MultiQueryRetriever 实现
//!
//! 使用 LLM 生成多个查询变体，提高检索召回率。

use crate::language_models::OpenAIChat;
use crate::vector_stores::{Document, SearchResult};
use crate::retrieval::RetrieverTrait;
use crate::schema::Message;
use crate::Runnable;
use std::collections::HashMap;
use std::sync::Arc;

/// MultiQueryRetriever 错误类型
#[derive(Debug)]
pub enum MultiQueryError {
    /// LLM 错误
    LLMError(String),
    
    /// 检索错误
    RetrieverError(String),
    
    /// 解析错误
    ParseError(String),
}

impl std::fmt::Display for MultiQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultiQueryError::LLMError(msg) => write!(f, "LLM 错误: {}", msg),
            MultiQueryError::RetrieverError(msg) => write!(f, "检索错误: {}", msg),
            MultiQueryError::ParseError(msg) => write!(f, "解析错误: {}", msg),
        }
    }
}

impl std::error::Error for MultiQueryError {}

/// MultiQueryRetriever 配置
pub struct MultiQueryConfig {
    /// 生成的查询数量
    pub num_queries: usize,
    
    /// 每个查询返回的文档数
    pub k_per_query: usize,
    
    /// 最终返回的文档数
    pub final_k: usize,
    
    /// 查询生成 prompt
    pub prompt_template: String,
}

impl Default for MultiQueryConfig {
    fn default() -> Self {
        Self {
            num_queries: 3,
            k_per_query: 5,
            final_k: 10,
            prompt_template: DEFAULT_MULTI_QUERY_PROMPT.to_string(),
        }
    }
}

impl MultiQueryConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_num_queries(mut self, n: usize) -> Self {
        self.num_queries = n;
        self
    }
    
    pub fn with_k_per_query(mut self, k: usize) -> Self {
        self.k_per_query = k;
        self
    }
    
    pub fn with_final_k(mut self, k: usize) -> Self {
        self.final_k = k;
        self
    }
    
    pub fn with_prompt(mut self, prompt: String) -> Self {
        self.prompt_template = prompt;
        self
    }
}

const DEFAULT_MULTI_QUERY_PROMPT: &str = r#"You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.

By generating multiple perspectives on the user question, your goal is to help overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}

Alternative questions:"#;

/// MultiQueryRetriever
///
/// 使用 LLM 生成多个查询变体，然后用基础检索器分别检索，
/// 最后合并去重结果返回。
pub struct MultiQueryRetriever {
    /// LLM 用于生成查询变体
    llm: OpenAIChat,
    
    /// 基础检索器
    base_retriever: Arc<dyn RetrieverTrait>,
    
    /// 配置
    config: MultiQueryConfig,
}

impl MultiQueryRetriever {
    pub fn new(
        llm: OpenAIChat,
        base_retriever: Arc<dyn RetrieverTrait>,
    ) -> Self {
        Self {
            llm,
            base_retriever,
            config: MultiQueryConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: MultiQueryConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn with_num_queries(mut self, n: usize) -> Self {
        self.config.num_queries = n;
        self
    }
    
    pub fn with_k_per_query(mut self, k: usize) -> Self {
        self.config.k_per_query = k;
        self
    }
    
    pub fn with_final_k(mut self, k: usize) -> Self {
        self.config.final_k = k;
        self
    }
    
    async fn generate_queries(&self, original_query: &str) -> Result<Vec<String>, MultiQueryError> {
        let prompt = self.config.prompt_template.replace("{question}", original_query);
        
        let messages = vec![
            Message::human(prompt),
        ];
        
        let response = self.llm
            .invoke(messages, None)
            .await
            .map_err(|e| MultiQueryError::LLMError(e.to_string()))?;
        
        let content = response.content;
        
        let queries: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();
        
        if queries.is_empty() {
            return Err(MultiQueryError::ParseError("LLM 未生成有效的查询变体".to_string()));
        }
        
        Ok(queries)
    }
    
    pub async fn retrieve_multi(&self, query: &str) -> Result<Vec<Document>, MultiQueryError> {
        let queries = self.generate_queries(query).await?;
        
        let all_queries: Vec<String> = std::iter::once(query.to_string())
            .chain(queries)
            .collect();
        
        let mut doc_scores: HashMap<String, (Document, f32)> = HashMap::new();
        
        for q in &all_queries {
            let results = self.base_retriever
                .retrieve_with_scores(q, self.config.k_per_query)
                .await
                .map_err(|e| MultiQueryError::RetrieverError(e.to_string()))?;
            
            for result in results {
                let doc_id = result.document.id.clone().unwrap_or_else(|| {
                    result.document.content.chars()
                        .take(50)
                        .collect::<String>()
                        .replace(" ", "_")
                });
                
                doc_scores
                    .entry(doc_id)
                    .and_modify(|(_, score)| {
                        *score = (*score + result.score).max(*score);
                    })
                    .or_insert((result.document.clone(), result.score));
            }
        }
        
        let mut scored_docs: Vec<(Document, f32)> = doc_scores.values()
            .map(|(doc, score)| (doc.clone(), *score))
            .collect();
        
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let final_docs: Vec<Document> = scored_docs
            .into_iter()
            .take(self.config.final_k)
            .map(|(doc, _)| doc)
            .collect();
        
        Ok(final_docs)
    }
    
    pub async fn retrieve_multi_with_scores(
        &self,
        query: &str,
    ) -> Result<Vec<SearchResult>, MultiQueryError> {
        let queries = self.generate_queries(query).await?;
        
        let all_queries: Vec<String> = std::iter::once(query.to_string())
            .chain(queries)
            .collect();
        
        let mut doc_scores: HashMap<String, (Document, f32, usize)> = HashMap::new();
        
        for q in &all_queries {
            let results = self.base_retriever
                .retrieve_with_scores(q, self.config.k_per_query)
                .await
                .map_err(|e| MultiQueryError::RetrieverError(e.to_string()))?;
            
            for result in results {
                let doc_id = result.document.id.clone().unwrap_or_else(|| {
                    result.document.content.chars()
                        .take(50)
                        .collect::<String>()
                        .replace(" ", "_")
                });
                
                doc_scores
                    .entry(doc_id)
                    .and_modify(|(_, score, count)| {
                        *score = (*score + result.score).max(*score);
                        *count += 1;
                    })
                    .or_insert((result.document.clone(), result.score, 1));
            }
        }
        
        let mut scored_docs: Vec<SearchResult> = doc_scores.values()
            .map(|(doc, score, count)| {
                let combined_score = score * (1.0 + 0.1 * *count as f32);
                SearchResult {
                    document: doc.clone(),
                    score: combined_score,
                }
            })
            .collect();
        
        scored_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        let final_results: Vec<SearchResult> = scored_docs
            .into_iter()
            .take(self.config.final_k)
            .collect();
        
        Ok(final_results)
    }
    
    pub async fn get_generated_queries(&self, query: &str) -> Result<Vec<String>, MultiQueryError> {
        self.generate_queries(query).await
    }
}

/// 静态查询生成器（不依赖 LLM）
pub struct StaticQueryGenerator {
    expansions: Vec<Box<dyn Fn(&str) -> Vec<String> + Send + Sync>>,
}

impl StaticQueryGenerator {
    pub fn new() -> Self {
        Self {
            expansions: Vec::new(),
        }
    }
    
    pub fn with_synonym_expansion(mut self, synonyms: HashMap<String, Vec<String>>) -> Self {
        self.expansions.push(Box::new(move |query: &str| {
            let mut expanded = Vec::new();
            for (word, syns) in &synonyms {
                if query.contains(word) {
                    for syn in syns {
                        expanded.push(query.replace(word, syn));
                    }
                }
            }
            expanded
        }));
        self
    }
    
    pub fn with_prefix_expansion(mut self, prefixes: Vec<String>) -> Self {
        self.expansions.push(Box::new(move |query: &str| {
            prefixes.iter()
                .map(|p| format!("{} {}", p, query))
                .collect()
        }));
        self
    }
    
    pub fn generate(&self, query: &str) -> Vec<String> {
        self.expansions.iter()
            .flat_map(|exp| exp(query))
            .filter(|q| q != query)
            .collect()
    }
}

impl Default for StaticQueryGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_static_query_generator_synonym() {
        let synonyms: HashMap<String, Vec<String>> = HashMap::from([
            ("数据库".to_string(), vec!["DB".to_string(), "存储".to_string()]),
        ]);
        
        let generator = StaticQueryGenerator::new()
            .with_synonym_expansion(synonyms);
        
        let queries = generator.generate("数据库连接失败");
        
        assert!(queries.contains(&"DB连接失败".to_string()));
        assert!(queries.contains(&"存储连接失败".to_string()));
    }
    
    #[test]
    fn test_static_query_generator_prefix() {
        let generator = StaticQueryGenerator::new()
            .with_prefix_expansion(vec!["如何".to_string(), "怎么".to_string()]);
        
        let queries = generator.generate("处理错误");
        
        assert!(queries.contains(&"如何 处理错误".to_string()));
        assert!(queries.contains(&"怎么 处理错误".to_string()));
    }
    
    #[test]
    fn test_multi_query_config() {
        let config = MultiQueryConfig::new()
            .with_num_queries(5)
            .with_k_per_query(10)
            .with_final_k(20);
        
        assert_eq!(config.num_queries, 5);
        assert_eq!(config.k_per_query, 10);
        assert_eq!(config.final_k, 20);
    }
    
    #[test]
    fn test_multi_query_config_default() {
        let config = MultiQueryConfig::default();
        
        assert_eq!(config.num_queries, 3);
        assert_eq!(config.k_per_query, 5);
        assert_eq!(config.final_k, 10);
    }
}