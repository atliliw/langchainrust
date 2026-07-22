// src/chains/document_chains.rs
//! Document processing chains
//!
//! Document processing chains that provide LLM processing capabilities for multiple documents:
//! - StuffDocumentsChain: Stuff all documents into a single prompt
//! - RefineDocumentsChain: Iteratively refine the answer document by document
//! - MapReduceDocumentsChain: Process documents in parallel then merge results
//! - MapRerankDocumentsChain: Process documents in parallel then rank by relevance

use async_trait::async_trait;
use futures_util::future::try_join_all;
use std::collections::HashMap;
use serde_json::Value;


use super::base::{BaseChain, ChainResult, ChainError};
use crate::BaseChatModel;
use crate::retrieval::Document;
use crate::schema::Message;
use crate::Runnable;

// ============================================================
// StuffDocumentsChain
// ============================================================

/// Default Stuff prompt template
const DEFAULT_STUFF_PROMPT: &str = "Answer the user's question based on the following reference information.

Reference information:
{context}

Question: {input}

Answer:";

/// StuffDocumentsChain
///
/// Stuffs all documents into a single prompt for LLM processing.
/// Suitable when the total document content fits within the LLM context window.
///
/// # Example
/// ```ignore
/// use langchainrust::{StuffDocumentsChain, OpenAIChat};
///
/// let chain = StuffDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "question").await?;
/// ```
pub struct StuffDocumentsChain<M: BaseChatModel> {
    llm: M,
    prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Maximum character count per document (truncated if exceeded)
    max_doc_length: Option<usize>,
}

impl<M: BaseChatModel> StuffDocumentsChain<M> {
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            prompt_template: DEFAULT_STUFF_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "stuff_documents".to_string(),
            verbose: false,
            max_doc_length: None,
        }
    }

    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
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

    pub fn with_max_doc_length(mut self, max: usize) -> Self {
        self.max_doc_length = Some(max);
        self
    }

    /// Format document list into context text
    pub fn format_documents(&self, documents: &[Document]) -> String {
        let mut parts = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            let mut content = doc.content.clone();
            if let Some(max_len) = self.max_doc_length {
                let char_count: usize = content.chars().count();
                if char_count > max_len {
                    content = content.chars().take(max_len).collect::<String>();
                    content.push_str("...\n[document truncated]");
                }
            }
            parts.push(format!("Document {}:\n{}", i + 1, content));
        }
        parts.join("\n\n---\n\n")
    }

    /// Build prompt
    pub fn build_prompt(&self, context: &str, input: &str) -> String {
        let template = self.prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input);
        template
    }

    /// Invoke with documents and input directly
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
    {
        let context = self.format_documents(&documents);

        if self.verbose {
            println!("\n=== StuffDocumentsChain ===");
            println!("Document count: {}", documents.len());
            println!("Context length: {} characters", context.len());
        }

        let prompt = self.build_prompt(&context, input);

        if self.verbose {
            println!("Prompt length: {} characters", prompt.len());
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        let output = response.content;

        if self.verbose {
            println!("Output: {}", output);
            println!("=== StuffDocumentsChain complete ===\n");
        }

        Ok(output)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for StuffDocumentsChain<M>
where
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================
// RefineDocumentsChain
// ============================================================

/// Default initial processing prompt template
const DEFAULT_REFINE_INITIAL_PROMPT: &str = "Answer the question based on the following reference information.

Reference information:
{context}

Question: {input}

Answer:";

/// Default iterative refinement prompt template
const DEFAULT_REFINE_PROMPT: &str = "You have provided an answer based on partial information. Here is additional reference information.

Existing answer:
{existing_answer}

New reference information:
{context}

Please refine or modify your answer based on the new information. If the new information does not conflict with the existing answer, merge them. If the new information conflicts with the existing answer, prioritize the new information.

Question: {input}

Refined answer:";

/// RefineDocumentsChain
///
/// Iteratively refines the answer document by document.
/// Generates an initial answer from the first document, then refines with each subsequent document.
/// Suitable when the total document content is large (processing one at a time avoids exceeding the context window).
///
/// # Example
/// ```ignore
/// use langchainrust::{RefineDocumentsChain, OpenAIChat};
///
/// let chain = RefineDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "question").await?;
/// ```
pub struct RefineDocumentsChain<M: BaseChatModel> {
    llm: M,
    initial_prompt_template: String,
    refine_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
}

impl<M: BaseChatModel> RefineDocumentsChain<M> {
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            initial_prompt_template: DEFAULT_REFINE_INITIAL_PROMPT.to_string(),
            refine_prompt_template: DEFAULT_REFINE_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "refine_documents".to_string(),
            verbose: false,
        }
    }

    pub fn with_initial_prompt(mut self, template: impl Into<String>) -> Self {
        self.initial_prompt_template = template.into();
        self
    }

    pub fn with_refine_prompt(mut self, template: impl Into<String>) -> Self {
        self.refine_prompt_template = template.into();
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

    pub fn build_initial_prompt(&self, context: &str, input: &str) -> String {
        self.initial_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    pub fn build_refine_prompt(&self, context: &str, input: &str, existing_answer: &str) -> String {
        self.refine_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
            .replace("{existing_answer}", existing_answer)
    }

    /// Invoke with documents and input directly (iterative refinement)
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
    {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError("Document list is empty".to_string()));
        }

        if self.verbose {
            println!("\n=== RefineDocumentsChain ===");
            println!("Document count: {}", documents.len());
            println!("Input: {}", input);
        }

        // Step 1: Generate initial answer from the first document
        let first_context = &documents[0].content;
        let initial_prompt = self.build_initial_prompt(first_context, input);

        if self.verbose {
            println!("\n--- Initial processing (document 1) ---");
        }

        let messages = vec![Message::human(&initial_prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM initial call failed: {}", e)))?;
        let mut answer = response.content;

        if self.verbose {
            println!("Initial answer: {}", answer);
        }

        // Subsequent steps: iteratively refine with remaining documents
        for (i, doc) in documents[1..].iter().enumerate() {
            if self.verbose {
                println!("\n--- Refinement step {} (document {}) ---", i + 1, i + 2);
            }

            let refine_prompt = self.build_refine_prompt(&doc.content, input, &answer);

            let messages = vec![Message::human(&refine_prompt)];
            let response = self.llm.invoke(messages, None).await
                .map_err(|e| ChainError::ExecutionError(format!("LLM refinement call failed: {}", e)))?;
            answer = response.content;

            if self.verbose {
                println!("Refined answer: {}", answer);
            }
        }

        if self.verbose {
            println!("=== RefineDocumentsChain complete ===\n");
        }

        Ok(answer)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for RefineDocumentsChain<M>
where
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================
// MapRerankDocumentsChain
// ============================================================

/// Default Map + Rerank prompt template
const DEFAULT_MAP_RERANK_PROMPT: &str = "Answer the question based on the following document, and provide a relevance score (0-100, higher is more relevant).

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
///
/// Suitable for scenarios where the best answer needs to be selected from multiple documents.
///
/// # Example
/// ```ignore
/// use langchainrust::{MapRerankDocumentsChain, OpenAIChat};
///
/// let chain = MapRerankDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "question").await?;
/// ```
pub struct MapRerankDocumentsChain<M: BaseChatModel> {
    llm: M,
    map_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Return top k results (default 1, i.e. only the highest score)
    top_k: usize,
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

    /// Set the number of top results to return
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Build Map stage prompt
    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        self.map_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    /// Extract score and answer from LLM output
    pub fn extract_score(text: &str) -> (u32, String) {
        let score_re = regex::Regex::new(r"(?i)(?:relevance\s*score|相关性评分)\s*[:：]\s*(\d+)").unwrap();
        if let Some(caps) = score_re.captures(text) {
            if let Ok(score) = caps[1].parse::<u32>() {
                let cleaned = score_re.replace(text, "").trim().to_string();
                let cleaned = cleaned.trim_start_matches("Answer")
                    .trim_start_matches("答案")
                    .trim_start_matches(&[':', '：'][..])
                    .trim()
                    .to_string();
                return (std::cmp::min(score, 100), if cleaned.is_empty() { text.to_string() } else { cleaned });
            }
        }

        let score_re2 = regex::Regex::new(r"(?i)score\s*[:：]\s*(\d+)").unwrap();
        if let Some(caps) = score_re2.captures(text) {
            if let Ok(score) = caps[1].parse::<u32>() {
                let cleaned = score_re2.replace(text, "").trim().to_string();
                return (std::cmp::min(score, 100), if cleaned.is_empty() { text.to_string() } else { cleaned });
            }
        }

        (50, text.to_string())
    }

    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<(u32, String), ChainError>
    where
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
    {
        let prompt = self.build_map_prompt(&doc.content, input);
        if self.verbose {
            println!("\n--- Map document {} ---", index + 1);
        }
        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("Map call failed (document {}): {}", index + 1, e)))?;
        let (score, answer) = Self::extract_score(&response.content);
        if self.verbose {
            println!("Document {} score: {}, answer: {}", index + 1, score,
                if answer.len() > 80 { &answer[..80] } else { &answer });
        }
        Ok((score, answer))
    }

    /// Invoke with documents and input directly
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<Vec<(u32, String)>, ChainError>
    where
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
    {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError("Document list is empty".to_string()));
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
                println!("Rank {}: score={}, answer={}", i + 1, score,
                    if answer.len() > 100 { &answer[..100] } else { &answer });
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
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> { vec![&self.input_key, "documents"] }
    fn output_keys(&self) -> Vec<&str> { vec![&self.output_key] }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| serde_json::from_value(v.clone()).ok()).collect())
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let results = self.invoke_with_documents(documents, input).await?;
        let output_json: Vec<serde_json::Value> = results.iter()
            .map(|(score, answer)| serde_json::json!({"score": score, "answer": answer}))
            .collect();

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::Array(output_json));
        Ok(result)
    }

    fn name(&self) -> &str { &self.name }
}

// ============================================================
// MapReduceDocumentsChain
// ============================================================

/// Default Map processing prompt template
const DEFAULT_MAP_PROMPT: &str = "Answer the user's question based on the following document content. Provide a concise answer based on the document content.

Document content:
{context}

Question: {input}

Answer based on this document:";

/// Default Reduce merge prompt template
const DEFAULT_REDUCE_PROMPT: &str = "Below are answers from multiple documents. Please merge them into a single complete and coherent final answer.

Answers from each document:
{summaries}

Original question: {input}

Final consolidated answer:";

/// MapReduceDocumentsChain
///
/// Processes documents in two steps:
/// 1. Map: Calls LLM independently for each document to generate an answer
/// 2. Reduce: Merges all independent answers into a final answer
///
/// Suitable for scenarios with a very large number of documents, as the map phase can process in parallel.
///
/// # Example
/// ```ignore
/// use langchainrust::{MapReduceDocumentsChain, OpenAIChat};
///
/// let chain = MapReduceDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "question").await?;
/// ```
pub struct MapReduceDocumentsChain<M: BaseChatModel> {
    llm: M,
    map_prompt_template: String,
    reduce_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
}

impl<M: BaseChatModel> MapReduceDocumentsChain<M> {
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            map_prompt_template: DEFAULT_MAP_PROMPT.to_string(),
            reduce_prompt_template: DEFAULT_REDUCE_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "map_reduce_documents".to_string(),
            verbose: false,
        }
    }

    pub fn with_map_prompt(mut self, template: impl Into<String>) -> Self {
        self.map_prompt_template = template.into();
        self
    }

    pub fn with_reduce_prompt(mut self, template: impl Into<String>) -> Self {
        self.reduce_prompt_template = template.into();
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

    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        self.map_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    pub fn build_reduce_prompt(&self, summaries: &[String], input: &str) -> String {
        let summaries_text = summaries.iter()
            .enumerate()
            .map(|(i, s)| format!("Answer from document {}:\n{}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n\n");

        self.reduce_prompt_template
            .replace("{summaries}", &summaries_text)
            .replace("{input}", input)
    }

    /// Map phase: call LLM for a single document
    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
    {
        let prompt = self.build_map_prompt(&doc.content, input);

        if self.verbose {
            println!("\n--- Map document {} ---", index + 1);
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("Map call failed (document {}): {}", index + 1, e)))?;

        if self.verbose {
            println!("Document {} answer: {}", index + 1, response.content);
        }

        Ok(response.content)
    }

    /// Invoke with documents and input directly
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
    {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError("Document list is empty".to_string()));
        }

        if self.verbose {
            println!("\n=== MapReduceDocumentsChain ===");
            println!("Document count: {}", documents.len());
            println!("Input: {}", input);
        }

        // Map phase: process each document in parallel
        if self.verbose {
            println!("\n--- Map phase ---");
        }

        let mut map_futures = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document(doc, input, i));
        }
        let summaries: Vec<String> = try_join_all(map_futures).await?;

        if self.verbose {
            println!("\n--- Reduce phase ---");
        }

        // Reduce phase: merge all answers
        let reduce_prompt = self.build_reduce_prompt(&summaries, input);

        if self.verbose {
            println!("Merging answers from {} documents", summaries.len());
        }

        let messages = vec![Message::human(&reduce_prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("Reduce call failed: {}", e)))?;

        let final_answer = response.content;

        if self.verbose {
            println!("Final answer: {}", final_answer);
            println!("=== MapReduceDocumentsChain complete ===\n");
        }

        Ok(final_answer)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for MapReduceDocumentsChain<M>
where
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::OpenAIChat;
    use crate::OpenAIConfig;

    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string(),
            base_url: "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1".to_string(),
            model: "glm-5.2".to_string(),
            streaming: false,
            organization: None,
            frequency_penalty: None,
            max_tokens: None,
            presence_penalty: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
        }
    }

    #[test]
    fn test_stuff_documents_format_documents() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: StuffDocumentsChain<OpenAIChat> = StuffDocumentsChain::new(llm);

        let docs = vec![
            Document { content: "Hello world".to_string(), metadata: HashMap::new(), id: None },
            Document { content: "Rust programming".to_string(), metadata: HashMap::new(), id: None },
        ];

        let formatted = chain.format_documents(&docs);
        assert!(formatted.contains("Document 1:"));
        assert!(formatted.contains("Hello world"));
        assert!(formatted.contains("Document 2:"));
        assert!(formatted.contains("Rust programming"));
    }

    #[test]
    fn test_stuff_documents_build_prompt() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: StuffDocumentsChain<OpenAIChat> = StuffDocumentsChain::new(llm);

        let prompt = chain.build_prompt("some context", "what is Rust?");
        assert!(prompt.contains("some context"));
        assert!(prompt.contains("what is Rust?"));
    }

    #[test]
    fn test_stuff_documents_truncation() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: StuffDocumentsChain<OpenAIChat> = StuffDocumentsChain::new(llm)
            .with_max_doc_length(5);

        let docs = vec![
            Document { content: "Hello world this is a long string".to_string(), metadata: HashMap::new(), id: None },
        ];

        let formatted = chain.format_documents(&docs);
        assert!(formatted.contains("[document truncated]"));
    }

    #[test]
    fn test_refine_documents_build_prompts() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: RefineDocumentsChain<OpenAIChat> = RefineDocumentsChain::new(llm);

        let initial = chain.build_initial_prompt("context1", "question");
        assert!(initial.contains("context1"));
        assert!(initial.contains("question"));

        let refine = chain.build_refine_prompt("context2", "question", "existing answer");
        assert!(refine.contains("context2"));
        assert!(refine.contains("question"));
        assert!(refine.contains("existing answer"));
    }

    #[test]
    fn test_map_rerank_extract_score() {
        let text1 = "Relevance score: 85\nAnswer: The answer is 42";
        let (score1, answer1) = MapRerankDocumentsChain::<OpenAIChat>::extract_score(text1);
        assert_eq!(score1, 85);
        assert!(answer1.contains("42"));

        let text2 = "Score: 70\nSome answer text";
        let (score2, _) = MapRerankDocumentsChain::<OpenAIChat>::extract_score(text2);
        assert_eq!(score2, 70);

        let text3 = "No score here, just an answer";
        let (score3, _) = MapRerankDocumentsChain::<OpenAIChat>::extract_score(text3);
        assert_eq!(score3, 50);
    }

    #[test]
    fn test_map_rerank_build_map_prompt() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: MapRerankDocumentsChain<OpenAIChat> = MapRerankDocumentsChain::new(llm);

        let prompt = chain.build_map_prompt("doc content", "what is this?");
        assert!(prompt.contains("doc content"));
        assert!(prompt.contains("what is this?"));
    }

    #[test]
    fn test_map_reduce_build_prompts() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: MapReduceDocumentsChain<OpenAIChat> = MapReduceDocumentsChain::new(llm);

        let map_prompt = chain.build_map_prompt("doc content", "question");
        assert!(map_prompt.contains("doc content"));
        assert!(map_prompt.contains("question"));

        let summaries = vec!["Answer 1".to_string(), "Answer 2".to_string()];
        let reduce_prompt = chain.build_reduce_prompt(&summaries, "question");
        assert!(reduce_prompt.contains("Answer 1"));
        assert!(reduce_prompt.contains("Answer 2"));
        assert!(reduce_prompt.contains("question"));
    }

    /// Real API test - StuffDocumentsChain
    /// Run: cargo test test_stuff_documents_chain_invoke -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_stuff_documents_chain_invoke() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: StuffDocumentsChain<OpenAIChat> = StuffDocumentsChain::new(llm);

        let docs = vec![
            Document { content: "Rust is a systems programming language.".to_string(), metadata: HashMap::new(), id: None },
            Document { content: "Rust emphasizes safety and performance.".to_string(), metadata: HashMap::new(), id: None },
        ];

        println!("\n=== Test StuffDocumentsChain ===");
        let result = chain.invoke_with_documents(docs, "What is Rust?").await.unwrap();
        println!("Output: {}", result);
        assert!(!result.is_empty());
    }

    /// Real API test - RefineDocumentsChain
    /// Run: cargo test test_refine_documents_chain_invoke -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_refine_documents_chain_invoke() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: RefineDocumentsChain<OpenAIChat> = RefineDocumentsChain::new(llm);

        let docs = vec![
            Document { content: "Rust is a systems programming language.".to_string(), metadata: HashMap::new(), id: None },
            Document { content: "Rust was created by Mozilla.".to_string(), metadata: HashMap::new(), id: None },
        ];

        println!("\n=== Test RefineDocumentsChain ===");
        let result = chain.invoke_with_documents(docs, "What is Rust and who created it?").await.unwrap();
        println!("Output: {}", result);
        assert!(!result.is_empty());
    }

    /// Real API test - MapRerankDocumentsChain
    /// Run: cargo test test_map_rerank_documents_chain_invoke -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_map_rerank_documents_chain_invoke() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: MapRerankDocumentsChain<OpenAIChat> = MapRerankDocumentsChain::new(llm);

        let docs = vec![
            Document { content: "Python is a high-level programming language.".to_string(), metadata: HashMap::new(), id: None },
            Document { content: "Rust is a systems programming language focused on safety.".to_string(), metadata: HashMap::new(), id: None },
        ];

        println!("\n=== Test MapRerankDocumentsChain ===");
        let result = chain.invoke_with_documents(docs, "What is Rust?").await.unwrap();
        println!("Output: {:?}", result);
        assert!(!result.is_empty());
    }

    /// Real API test - MapReduceDocumentsChain
    /// Run: cargo test test_map_reduce_documents_chain_invoke -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_map_reduce_documents_chain_invoke() {
        let llm = OpenAIChat::new(create_test_config());
        let chain: MapReduceDocumentsChain<OpenAIChat> = MapReduceDocumentsChain::new(llm);

        let docs = vec![
            Document { content: "Rust is a systems programming language.".to_string(), metadata: HashMap::new(), id: None },
            Document { content: "Rust emphasizes memory safety without garbage collection.".to_string(), metadata: HashMap::new(), id: None },
        ];

        println!("\n=== Test MapReduceDocumentsChain ===");
        let result = chain.invoke_with_documents(docs, "What is Rust?").await.unwrap();
        println!("Output: {}", result);
        assert!(!result.is_empty());
    }
}
