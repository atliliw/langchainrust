// src/chains/retrieval_qa.rs
//! RetrievalQA Chain
//!
//! One-stop retrieval QA chain that encapsulates the complete RAG workflow.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::base::{BaseChain, ChainError, ChainResult};
use crate::retrieval::{Document, RetrieverTrait};
use crate::schema::Message;
use crate::BaseChatModel;
use crate::Runnable;

/// Default QA prompt template
const DEFAULT_QA_PROMPT: &str = "Answer the question based on the following context. If the context does not contain relevant information, say 'I don't know'.

Context:
{context}

Question: {question}

Answer:";

/// RetrievalQA Chain
///
/// One-stop retrieval QA chain that automatically:
/// 1. Retrieves relevant documents
/// 2. Assembles prompt (context + question)
/// 3. LLM generates answer
///
/// # Examples
/// ```ignore
/// use langchainrust::{RetrievalQA, OpenAIChat, SimilarityRetriever};
///
/// let llm = OpenAIChat::new(config);
/// let retriever = SimilarityRetriever::new(store, embeddings);
///
/// let qa = RetrievalQA::new(llm, retriever);
///
/// // One line to complete document Q&A
/// let answer = qa.invoke("What is Rust?").await?;
/// ```
pub struct RetrievalQA<M: BaseChatModel> {
    llm: M,
    retriever: Arc<dyn RetrieverTrait>,

    prompt_template: String,
    input_key: String,
    output_key: String,
    name: String,

    k: usize,
    verbose: bool,

    return_source_documents: bool,
    source_document_key: String,
}

impl<M: BaseChatModel + 'static> RetrievalQA<M> {
    pub fn new(llm: M, retriever: Arc<dyn RetrieverTrait>) -> Self {
        Self {
            llm,
            retriever,
            prompt_template: DEFAULT_QA_PROMPT.to_string(),
            input_key: "query".to_string(),
            output_key: "result".to_string(),
            name: "retrieval_qa".to_string(),
            k: 4,
            verbose: false,
            return_source_documents: false,
            source_document_key: "source_documents".to_string(),
        }
    }

    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
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

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_return_source_documents(mut self, return_source: bool) -> Self {
        self.return_source_documents = return_source;
        self
    }

    pub fn with_source_document_key(mut self, key: impl Into<String>) -> Self {
        self.source_document_key = key.into();
        self
    }

    pub fn retriever(&self) -> &Arc<dyn RetrieverTrait> {
        &self.retriever
    }

    pub fn k(&self) -> usize {
        self.k
    }

    fn format_context(&self, documents: &[Document]) -> String {
        documents
            .iter()
            .map(|doc| doc.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn build_prompt(&self, context: &str, question: &str) -> String {
        self.prompt_template
            .replace("{context}", context)
            .replace("{question}", question)
    }

    pub async fn query(&self, question: impl Into<String>) -> Result<String, ChainError> {
        let inputs = HashMap::from([(self.input_key.clone(), Value::String(question.into()))]);

        let result = self.invoke(inputs).await?;

        result
            .get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("Missing output result".to_string()))
    }

    pub async fn query_with_sources(
        &self,
        question: impl Into<String>,
    ) -> Result<(String, Vec<Document>), ChainError> {
        let inputs = HashMap::from([(self.input_key.clone(), Value::String(question.into()))]);

        let was_returning_sources = self.return_source_documents;
        if !was_returning_sources {
            // We need to invoke with source documents enabled, but we can't mutate self.
            // Instead, we directly perform the retrieval and LLM call here.
            let question_str = inputs
                .get(&self.input_key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

            let documents = self
                .retriever
                .retrieve(question_str, self.k)
                .await
                .map_err(|e| ChainError::ExecutionError(format!("Retrieval failed: {}", e)))?;

            let context = self.format_context(&documents);
            let prompt = self.build_prompt(&context, question_str);
            let messages = vec![Message::human(&prompt)];
            let response = self
                .llm
                .invoke(messages, None)
                .await
                .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

            return Ok((response.content, documents));
        }

        let result = self.invoke(inputs).await?;

        let answer = result
            .get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("Missing output result".to_string()))?;

        let sources: Vec<Document> = result
            .get(&self.source_document_key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        Ok((answer, sources))
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for RetrievalQA<M>
where
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error:
        std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        if self.return_source_documents {
            vec![&self.output_key, &self.source_document_key]
        } else {
            vec![&self.output_key]
        }
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;

        let question = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        if self.verbose {
            println!("\n=== RetrievalQA Execution ===");
            println!("Question: {}", question);
            println!("Retrieval count (k): {}", self.k);
        }

        if self.verbose {
            println!("\n--- Step 1: Retrieve relevant documents ---");
        }

        let documents = self
            .retriever
            .retrieve(question, self.k)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Retrieval failed: {}", e)))?;

        if self.verbose {
            println!("Retrieved {} documents", documents.len());
            for (i, doc) in documents.iter().enumerate() {
                // H66: Use char-boundary-safe truncation instead of byte slicing
                let preview: String = doc.content.chars().take(100).collect();
                println!("Document {}: {}", i + 1, preview);
            }
        }

        if documents.is_empty() && self.verbose {
            println!("Warning: No relevant documents retrieved");
        }

        if self.verbose {
            println!("\n--- Step 2: Assemble Prompt ---");
        }

        let context = self.format_context(&documents);
        let prompt = self.build_prompt(&context, question);

        if self.verbose {
            println!("Context length: {} characters", context.len());
            println!("Prompt length: {} characters", prompt.len());
        }

        if self.verbose {
            println!("\n--- Step 3: LLM generates answer ---");
        }

        let messages = vec![Message::human(&prompt)];
        let response = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        let answer = response.content;

        if self.verbose {
            println!("Answer: {}", answer);
            println!("=== RetrievalQA Complete ===\n");
        }

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(answer));

        if self.return_source_documents {
            let sources: Vec<Value> = documents
                .iter()
                .map(|doc| serde_json::to_value(doc).unwrap_or(Value::Null))
                .collect();
            result.insert(self.source_document_key.clone(), Value::Array(sources));
        }

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

    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));

        let qa = RetrievalQA::new(llm, retriever);

        assert_eq!(qa.input_keys(), vec!["query"]);
        assert_eq!(qa.output_keys(), vec!["result"]);
        assert_eq!(qa.name(), "retrieval_qa");
        assert_eq!(qa.k(), 4);
    }

    #[test]
    fn test_with_options() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));

        let qa = RetrievalQA::new(llm, retriever)
            .with_k(5)
            .with_input_key("question")
            .with_output_key("answer")
            .with_return_source_documents(true)
            .with_verbose(true);

        assert_eq!(qa.input_keys(), vec!["question"]);
        assert_eq!(qa.output_keys(), vec!["answer", "source_documents"]);
        assert_eq!(qa.k(), 5);
        assert!(qa.verbose);
    }

    #[test]
    fn test_format_context() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));

        let qa = RetrievalQA::new(llm, retriever);

        let docs = vec![
            Document::new("Document 1 content"),
            Document::new("Document 2 content"),
        ];

        let context = qa.format_context(&docs);
        assert!(context.contains("Document 1 content"));
        assert!(context.contains("Document 2 content"));
    }

    #[test]
    fn test_build_prompt() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));

        let qa = RetrievalQA::new(llm, retriever);

        let prompt = qa.build_prompt("This is the context", "What is Rust?");

        assert!(prompt.contains("This is the context"));
        assert!(prompt.contains("What is Rust?"));
    }

    #[test]
    fn test_custom_prompt_template() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));

        let custom_template = "Background: {context}\nPlease answer: {question}";

        let qa = RetrievalQA::new(llm, retriever).with_prompt_template(custom_template);

        let prompt = qa.build_prompt("Test context", "Test question");

        assert!(prompt.contains("Background"));
        assert!(prompt.contains("Test context"));
        assert!(prompt.contains("Test question"));
    }
}
