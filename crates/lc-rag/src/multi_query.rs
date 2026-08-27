// src/retrieval/multi_query.rs
//! MultiQueryRetriever implementation
//!
//! Uses an LLM to generate multiple query variants, improving retrieval recall.

use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_prompts::PromptTemplate;
use lc_providers::ProviderError;
use lc_schema::Message;
use lc_vector_stores::{Document, SearchResult};
use serde_json::json;

use crate::retriever::RetrieverTrait;
use crate::structured::chat_structured;
use std::collections::HashMap;
use std::sync::Arc;

/// Generate a stable document ID from content hash (M58).
///
/// P2-3: Replaces `DefaultHasher` with FNV-1a 64-bit (the std internal algorithm is not
/// guaranteed stable across processes; FNV-1a is a fully-specified deterministic hash).
fn doc_content_hash(content: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = fnv::FnvHasher::default();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// MultiQueryRetriever error type
#[derive(Debug)]
#[non_exhaustive]
pub enum MultiQueryError {
    /// LLM error
    LLMError(String),

    /// Retriever error
    RetrieverError(String),

    /// Parse error
    ParseError(String),
}

impl std::fmt::Display for MultiQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultiQueryError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            MultiQueryError::RetrieverError(msg) => write!(f, "Retriever error: {}", msg),
            MultiQueryError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for MultiQueryError {}

/// MultiQueryRetriever configuration
pub struct MultiQueryConfig {
    /// Number of generated queries
    pub num_queries: usize,

    /// Number of documents returned per query
    pub k_per_query: usize,

    /// Number of documents returned in the final result
    pub final_k: usize,

    /// The query-generation prompt
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
    /// Creates a `MultiQueryConfig` with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of generated queries
    pub fn with_num_queries(mut self, n: usize) -> Self {
        self.num_queries = n;
        self
    }

    /// Sets the number of documents returned per query
    pub fn with_k_per_query(mut self, k: usize) -> Self {
        self.k_per_query = k;
        self
    }

    /// Sets the number of documents returned in the final result
    pub fn with_final_k(mut self, k: usize) -> Self {
        self.final_k = k;
        self
    }

    /// Sets the query-generation prompt
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_template = prompt.into();
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
/// Uses an LLM to generate multiple query variants, retrieves with the base retriever for
/// each one, then merges and dedups the results before returning.
pub struct MultiQueryRetriever {
    /// The LLM used to generate query variants
    ///
    /// P0-3: no longer hardcodes `OpenAIChat`; accepts any LLM implementing `BaseChatModel`.
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,

    /// The base retriever
    base_retriever: Arc<dyn RetrieverTrait>,

    /// Configuration
    config: MultiQueryConfig,
}

impl MultiQueryRetriever {
    /// Creates a MultiQueryRetriever (accepting any LLM implementing `BaseChatModel`)
    pub fn new<L>(llm: L, base_retriever: Arc<dyn RetrieverTrait>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: lc_providers::wrap_chat_model(llm),
            base_retriever,
            config: MultiQueryConfig::default(),
        }
    }

    /// P0-3: Builds from an already-wrapped `Arc<dyn BaseChatModel<Error = ProviderError>>`
    pub fn new_arc(
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
        base_retriever: Arc<dyn RetrieverTrait>,
    ) -> Self {
        Self {
            llm,
            base_retriever,
            config: MultiQueryConfig::default(),
        }
    }

    /// Sets the MultiQuery configuration
    pub fn with_config(mut self, config: MultiQueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the number of generated queries
    pub fn with_num_queries(mut self, n: usize) -> Self {
        self.config.num_queries = n;
        self
    }

    /// Sets the number of documents returned per query
    pub fn with_k_per_query(mut self, k: usize) -> Self {
        self.config.k_per_query = k;
        self
    }

    /// Sets the number of documents returned in the final result
    pub fn with_final_k(mut self, k: usize) -> Self {
        self.config.final_k = k;
        self
    }

    async fn generate_queries(&self, original_query: &str) -> Result<Vec<String>, MultiQueryError> {
        let template = PromptTemplate::new(&self.config.prompt_template);
        let mut vars = HashMap::new();
        vars.insert("question", original_query);
        let prompt = template
            .format(&vars)
            .unwrap_or_else(|_| self.config.prompt_template.clone());

        // P2-1: Prefer the structured query list from tool_calls; on text-parse failure,
        // retry once with a hint.
        const MAX_RETRIES: usize = 1;
        let mut current_prompt = prompt;

        for attempt in 0..=MAX_RETRIES {
            let result = chat_structured(
                self.llm.as_ref(),
                Some(queries_tool()),
                vec![Message::human(&current_prompt)],
            )
            .await
            .map_err(|e| MultiQueryError::LLMError(e.to_string()))?;

            // Prefer tool_calls: the query string array
            if let Some(args) = &result.tool_args {
                if let Some(queries) = parse_queries(args) {
                    if !queries.is_empty() {
                        return Ok(queries);
                    }
                }
            }

            // Text fallback: split line by line, cleaning numbered/quote/bullet noise
            let queries = parse_query_lines(&result.content, self.config.num_queries);
            if !queries.is_empty() {
                return Ok(queries);
            }

            if attempt < MAX_RETRIES {
                current_prompt = format!(
                    "上次的输出不是有效的查询列表。请重新为原问题生成 {} 个不同的查询变体,\
                     每行一个,不要编号、不要项目符号、不要解释或多余文字。\n\n原问题:{}\n\n\
                     上次输出(无效):\n{}\n\n新的查询变体:",
                    self.config.num_queries, original_query, result.content
                );
            }
        }

        Err(MultiQueryError::ParseError(
            "LLM did not generate valid query variants".to_string(),
        ))
    }

    /// Generates multiple query variants, retrieves each one, and returns the merged deduped documents
    pub async fn retrieve_multi(&self, query: &str) -> Result<Vec<Document>, MultiQueryError> {
        let queries = self.generate_queries(query).await?;

        let all_queries: Vec<String> = std::iter::once(query.to_string()).chain(queries).collect();

        let mut doc_scores: HashMap<String, (Document, f32)> = HashMap::new();

        for q in &all_queries {
            let results = self
                .base_retriever
                .retrieve_with_scores(q, self.config.k_per_query)
                .await
                .map_err(|e| MultiQueryError::RetrieverError(e.to_string()))?;

            for result in results {
                let doc_id = result
                    .document
                    .id
                    .clone()
                    .unwrap_or_else(|| doc_content_hash(&result.document.content));

                doc_scores
                    .entry(doc_id)
                    .and_modify(|(_, score)| {
                        // M3: use addition instead of no-op .max()
                        *score += result.score;
                    })
                    .or_insert((result.document.clone(), result.score));
            }
        }

        let mut scored_docs: Vec<(Document, f32)> = doc_scores
            .values()
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

    /// Generates multiple query variants, retrieves each one, and returns the merged deduped results with scores
    pub async fn retrieve_multi_with_scores(
        &self,
        query: &str,
    ) -> Result<Vec<SearchResult>, MultiQueryError> {
        let queries = self.generate_queries(query).await?;

        let all_queries: Vec<String> = std::iter::once(query.to_string()).chain(queries).collect();

        let mut doc_scores: HashMap<String, (Document, f32, usize)> = HashMap::new();

        for q in &all_queries {
            let results = self
                .base_retriever
                .retrieve_with_scores(q, self.config.k_per_query)
                .await
                .map_err(|e| MultiQueryError::RetrieverError(e.to_string()))?;

            for result in results {
                let doc_id = result
                    .document
                    .id
                    .clone()
                    .unwrap_or_else(|| doc_content_hash(&result.document.content));

                doc_scores
                    .entry(doc_id)
                    .and_modify(|(_, score, count)| {
                        // M3: use addition instead of no-op .max()
                        *score += result.score;
                        *count += 1;
                    })
                    .or_insert((result.document.clone(), result.score, 1));
            }
        }

        let mut scored_docs: Vec<SearchResult> = doc_scores
            .values()
            .map(|(doc, score, count)| {
                let combined_score = score * (1.0 + 0.1 * *count as f32);
                SearchResult {
                    document: doc.clone(),
                    score: combined_score,
                }
            })
            .collect();

        scored_docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let final_results: Vec<SearchResult> =
            scored_docs.into_iter().take(self.config.final_k).collect();

        Ok(final_results)
    }

    /// Returns the LLM-generated query variants (without retrieving)
    pub async fn get_generated_queries(&self, query: &str) -> Result<Vec<String>, MultiQueryError> {
        self.generate_queries(query).await
    }
}

/// Query-variant tool definition (P2-1): forces the LLM to output a query string array.
fn queries_tool() -> ToolDefinition {
    ToolDefinition::new(
        "generate_queries",
        "为原问题生成多个不同的检索查询变体,返回查询字符串数组",
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["queries"]
    }))
}

/// Extracts the query array from tool_call arguments.
fn parse_queries(args: &serde_json::Value) -> Option<Vec<String>> {
    args.get("queries")?
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(|s| s.trim().to_string()))
        .collect()
}

/// Text-line parsing: strips numbering ("1. xxx"), bullets, and surrounding quotes so dirty
/// text never becomes a query.
///
/// Takes only the first `limit` lines (usually `num_queries`): the LLM typically puts the
/// queries first, and trailing explanatory prose gets truncated instead of leaking in.
fn parse_query_lines(content: &str, limit: usize) -> Vec<String> {
    content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(|line| {
            let stripped = line.trim_start_matches(['-', '•', '*', ' ']);
            let stripped = stripped.trim_start_matches(|c: char| {
                c.is_ascii_digit() || c == '.' || c == '、' || c == ')' || c == ' '
            });
            stripped
                .trim_matches(['"', '\'', '“', '”'])
                .trim()
                .to_string()
        })
        .filter(|q| !q.is_empty())
        .take(limit)
        .collect()
}

/// A static query generator (no LLM dependency)
#[allow(clippy::type_complexity)]
pub struct StaticQueryGenerator {
    expansions: Vec<Box<dyn Fn(&str) -> Vec<String> + Send + Sync>>,
}

impl StaticQueryGenerator {
    /// Creates an empty static query generator
    pub fn new() -> Self {
        Self {
            expansions: Vec::new(),
        }
    }

    /// Adds a synonym-expansion rule
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

    /// Adds a prefix-expansion rule
    pub fn with_prefix_expansion(mut self, prefixes: Vec<String>) -> Self {
        self.expansions.push(Box::new(move |query: &str| {
            prefixes
                .iter()
                .map(|p| format!("{} {}", p, query))
                .collect()
        }));
        self
    }

    /// Applies all expansion rules to generate query variants
    pub fn generate(&self, query: &str) -> Vec<String> {
        self.expansions
            .iter()
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
        let synonyms: HashMap<String, Vec<String>> = HashMap::from([(
            "数据库".to_string(),
            vec!["DB".to_string(), "存储".to_string()],
        )]);

        let generator = StaticQueryGenerator::new().with_synonym_expansion(synonyms);

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

    /// P2-1: The query tool definition carries a queries-array schema.
    #[test]
    fn test_queries_tool_schema() {
        let tool = queries_tool();
        assert_eq!(tool.function.name, "generate_queries");
        let params = tool.function.parameters.expect("parameters should exist");
        assert_eq!(params["properties"]["queries"]["type"], "array");
    }

    /// P2-1: The tool_call arguments parse into a query array.
    #[test]
    fn test_parse_queries() {
        let args = json!({ "queries": ["数据库连接失败怎么办", "DB 连接错误排查"] });
        let queries = parse_queries(&args).expect("should parse successfully");
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0], "数据库连接失败怎么办");
    }

    /// P2-1: A missing `queries` key → None.
    #[test]
    fn test_parse_queries_missing_key() {
        let args = json!({ "other": 1 });
        assert!(parse_queries(&args).is_none());
    }

    /// P2-1: Text-line parsing strips numbering/bullets/quotes and caps trailing prose by limit.
    #[test]
    fn test_parse_query_lines_cleanup() {
        let content = "1. 数据库连接失败\n- 如何排查 DB 错误\n• \"连接超时怎么办\"\n\n补充解释";
        let queries = parse_query_lines(content, 3);
        assert_eq!(
            queries,
            vec![
                "数据库连接失败".to_string(),
                "如何排查 DB 错误".to_string(),
                "连接超时怎么办".to_string(),
            ]
        );
    }

    /// P2-1: Empty text → empty array.
    #[test]
    fn test_parse_query_lines_empty() {
        assert!(parse_query_lines("  \n\n", 3).is_empty());
    }

    /// P2-1: Extra lines beyond the limit are truncated.
    #[test]
    fn test_parse_query_lines_capped() {
        let content = "a\nb\nc\nd";
        assert_eq!(
            parse_query_lines(content, 2),
            vec!["a".to_string(), "b".to_string()]
        );
    }
}
