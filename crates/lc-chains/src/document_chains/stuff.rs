// lc-chains/src/document_chains/stuff.rs
//! StuffDocumentsChain - stuffs all documents into a single prompt.

use async_trait::async_trait;
use lc_core::language_models::LLMResult;
use lc_core::{BaseChatModel, Runnable};
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;

use crate::base::{BaseChain, ChainError, ChainResult};

/// Default Stuff prompt template.
pub(crate) const DEFAULT_STUFF_PROMPT: &str =
    "Answer the user's question based on the following reference information.

Reference information:
{context}

Question: {input}

Answer:";

/// StuffDocumentsChain
///
/// Stuffs all documents into a single prompt for LLM processing.
/// Suitable when the total document content fits within the LLM context window.
pub struct StuffDocumentsChain<M: BaseChatModel> {
    llm: M,
    prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Maximum character count per document (truncated if exceeded).
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

    /// Format document list into context text.
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

    /// Build prompt.
    pub fn build_prompt(&self, context: &str, input: &str) -> String {
        self.prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    /// Invoke with documents and input directly.
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
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
        let response = self
            .llm
            .invoke(messages, None)
            .await
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

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
