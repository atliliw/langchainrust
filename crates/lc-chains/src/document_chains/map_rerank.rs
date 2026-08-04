// lc-chains/src/document_chains/map_rerank.rs
//! MapRerankDocumentsChain - processes documents in parallel then ranks by relevance.

use async_trait::async_trait;
use futures_util::future::try_join_all;
use lc_core::language_models::LLMResult;
use lc_core::{BaseChatModel, Runnable};
use lc_schema::Message;
use lc_shared::document::Document;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::base::{BaseChain, ChainError, ChainResult};

/// Default Map + Rerank prompt template.
pub(crate) const DEFAULT_MAP_RERANK_PROMPT: &str = "Answer the question based on the following document, and provide a relevance score (0-100, higher is more relevant).

Document content:
{context}

Question: {input}

Please output in the following format:
Relevance score: <score>
Answer: <your answer>";

/// MapRerankDocumentsChain
///
/// First calls LLM independently for each document to generate an answer and score,
/// then ranks by relevance score and returns the highest-scoring answer.
pub struct MapRerankDocumentsChain<M: BaseChatModel> {
    llm: M,
    map_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Return top k results (default 1, i.e. only the highest score).
    top_k: usize,
}

// Pre-compiled regex patterns for score extraction.
static SCORE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:relevance\s*score|相关性评分)\s*[:：]\s*(\d+)").unwrap());
static SCORE_RE2: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)score\s*[:：]\s*(\d+)").unwrap());

/// Truncate a string to at most `max_len` characters, respecting char boundaries.
fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.chars().count() <= max_len {
        s
    } else {
        let end = s
            .char_indices()
            .nth(max_len)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        &s[..end]
    }
}

/// Extract score and answer from LLM output.
pub fn extract_score(text: &str) -> (u32, String) {
    if let Some(caps) = SCORE_RE.captures(text) {
        if let Ok(score) = caps[1].parse::<u32>() {
            let cleaned = SCORE_RE.replace(text, "").trim().to_string();
            let cleaned = cleaned
                .trim_start_matches("Answer")
                .trim_start_matches("答案")
                .trim_start_matches(&[':', '：'][..])
                .trim()
                .to_string();
            return (
                std::cmp::min(score, 100),
                if cleaned.is_empty() {
                    text.to_string()
                } else {
                    cleaned
                },
            );
        }
    }

    if let Some(caps) = SCORE_RE2.captures(text) {
        if let Ok(score) = caps[1].parse::<u32>() {
            let cleaned = SCORE_RE2.replace(text, "").trim().to_string();
            return (
                std::cmp::min(score, 100),
                if cleaned.is_empty() {
                    text.to_string()
                } else {
                    cleaned
                },
            );
        }
    }

    (50, text.to_string())
}

impl<M: BaseChatModel> MapRerankDocumentsChain<M> {
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            map_prompt_template: DEFAULT_MAP_RERANK_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "map_rerank_documents".to_string(),
            verbose: false,
            top_k: 1,
        }
    }

    pub fn with_map_prompt(mut self, template: impl Into<String>) -> Self {
        self.map_prompt_template = template.into();
        self
    }

    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
        self
    }

    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the number of top results to return.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Build Map stage prompt.
    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        self.map_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<(u32, String), ChainError>
    where
        <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
    {
        let prompt = self.build_map_prompt(&doc.content, input);
        if self.verbose {
            println!("\n--- Map document {} ---", index + 1);
        }
        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await.map_err(|e| {
            ChainError::ExecutionError(format!("Map call failed (document {}): {}", index + 1, e))
        })?;
        let (score, answer) = extract_score(&response.content);
        if self.verbose {
            println!(
                "Document {} score: {}, answer: {}",
                index + 1,
                score,
                truncate_str(&answer, 80)
            );
        }
        Ok((score, answer))
    }

    /// Invoke with documents and input directly.
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<Vec<(u32, String)>, ChainError>
    where
        <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
    {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
        }

        if self.verbose {
            println!("\n=== MapRerankDocumentsChain ===");
            println!("Document count: {}, Input: {}", documents.len(), input);
            println!("\n--- Map phase ---");
        }

        let mut map_futures = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document(doc, input, i));
        }
        let mut results: Vec<(u32, String)> = try_join_all(map_futures).await?;

        results.sort_by(|a, b| b.0.cmp(&a.0));

        if self.verbose {
            println!("\n--- Rerank phase ---");
            for (i, (score, answer)) in results.iter().enumerate() {
                println!(
                    "Rank {}: score={}, answer={}",
                    i + 1,
                    score,
                    truncate_str(answer, 100)
                );
            }
        }

        let top_results: Vec<(u32, String)> = results.into_iter().take(self.top_k).collect();
        if self.verbose {
            println!("Selected {} best results", top_results.len());
            println!("=== MapRerankDocumentsChain complete ===\n");
        }
        Ok(top_results)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for MapRerankDocumentsChain<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }
    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs
            .get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let results = self.invoke_with_documents(documents, input).await?;
        let output_json: Vec<serde_json::Value> = results
            .iter()
            .map(|(score, answer)| serde_json::json!({"score": score, "answer": answer}))
            .collect();

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::Array(output_json));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
