// lc-chains/src/document_chains/refine.rs
//! RefineDocumentsChain - iteratively refines the answer document by document.

use async_trait::async_trait;
use lc_core::language_models::LLMResult;
use lc_core::{BaseChatModel, Runnable};
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;

use crate::base::{BaseChain, ChainError, ChainResult};

/// Default initial processing prompt template.
pub(crate) const DEFAULT_REFINE_INITIAL_PROMPT: &str =
    "Answer the question based on the following reference information.

Reference information:
{context}

Question: {input}

Answer:";

/// Default iterative refinement prompt template.
pub(crate) const DEFAULT_REFINE_PROMPT: &str = "You have provided an answer based on partial information. Here is additional reference information.

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

    /// Invoke with documents and input directly (iterative refinement).
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
    {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
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
        let response =
            self.llm.invoke(messages, None).await.map_err(|e| {
                ChainError::ExecutionError(format!("LLM initial call failed: {}", e))
            })?;
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
            let response = self.llm.invoke(messages, None).await.map_err(|e| {
                ChainError::ExecutionError(format!("LLM refinement call failed: {}", e))
            })?;
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
