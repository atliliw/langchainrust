// src/chains/document_chains/map_reduce.rs
//! MapReduceDocumentsChain - processes documents in parallel then merges results.

use async_trait::async_trait;
use futures_util::future::try_join_all;
use serde_json::Value;
use std::collections::HashMap;

use crate::chains::base::{BaseChain, ChainError, ChainResult};
use crate::retrieval::Document;
use crate::schema::Message;
use crate::BaseChatModel;
use crate::Runnable;

/// Default Map processing prompt template
pub(crate) const DEFAULT_MAP_PROMPT: &str = "Answer the user's question based on the following document content. Provide a concise answer based on the document content.

Document content:
{context}

Question: {input}

Answer based on this document:";

/// Default Reduce merge prompt template
pub(crate) const DEFAULT_REDUCE_PROMPT: &str = "Below are answers from multiple documents. Please merge them into a single complete and coherent final answer.

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
        let summaries_text = summaries
            .iter()
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
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error:
            std::fmt::Display,
    {
        let prompt = self.build_map_prompt(&doc.content, input);

        if self.verbose {
            println!("\n--- Map document {} ---", index + 1);
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await.map_err(|e| {
            ChainError::ExecutionError(format!("Map call failed (document {}): {}", index + 1, e))
        })?;

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
        <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error:
            std::fmt::Display,
    {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
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
        let response = self
            .llm
            .invoke(messages, None)
            .await
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
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error:
        std::fmt::Display,
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

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
