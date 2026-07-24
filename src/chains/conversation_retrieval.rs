// src/chains/conversation_retrieval.rs
//! ConversationRetrieval Chain
//!
//! Retrieval-augmented generation chain with memory, combining conversation
//! history with document retrieval. Suitable for Q&A scenarios that require
//! both conversational context and external knowledge.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::base::{BaseChain, ChainError, ChainResult};
use crate::memory::{BaseMemory, ConversationBufferMemory};
use crate::retrieval::{Document, RetrieverTrait};
use crate::schema::Message;
use crate::BaseChatModel;
use crate::Runnable;
use tokio::sync::Mutex;

/// Default retrieval-augmented conversation prompt template
const DEFAULT_QA_PROMPT: &str = "You are an AI assistant. Please answer the user's question based on the conversation history and reference information.

Conversation history:
{history}

Reference information:
{context}

Question: {question}

Answer:";

/// ConversationRetrievalChain
///
/// Retrieval-augmented conversation chain with memory that automatically:
/// 1. Loads conversation history
/// 2. Retrieves relevant documents
/// 3. Combines history + context + question
/// 4. LLM generates answer
/// 5. Saves to conversation memory
///
/// # Examples
/// ```ignore
/// use langchainrust::{ConversationRetrievalChain, OpenAIChat, SimilarityRetriever, ConversationBufferMemory};
///
/// let chain = ConversationRetrievalChain::new(llm, retriever, memory);
/// let answer = chain.query("What is Rust?").await?;
/// ```
pub struct ConversationRetrievalChain<M: BaseChatModel> {
    llm: M,
    retriever: Arc<dyn RetrieverTrait>,
    memory: Arc<Mutex<ConversationBufferMemory>>,

    system_prompt: Option<String>,
    qa_prompt_template: String,
    input_key: String,
    output_key: String,
    memory_key: String,
    name: String,

    k: usize,
    verbose: bool,
    return_source_documents: bool,
    source_document_key: String,
}

impl<M: BaseChatModel + 'static> ConversationRetrievalChain<M> {
    pub fn new(
        llm: M,
        retriever: Arc<dyn RetrieverTrait>,
        memory: ConversationBufferMemory,
    ) -> Self {
        Self {
            llm,
            retriever,
            memory: Arc::new(Mutex::new(memory.with_return_messages(true))),
            system_prompt: None,
            qa_prompt_template: DEFAULT_QA_PROMPT.to_string(),
            input_key: "query".to_string(),
            output_key: "result".to_string(),
            memory_key: "history".to_string(),
            name: "conversation_retrieval".to_string(),
            k: 4,
            verbose: false,
            return_source_documents: false,
            source_document_key: "source_documents".to_string(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_qa_prompt(mut self, template: impl Into<String>) -> Self {
        self.qa_prompt_template = template.into();
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

    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
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

    pub fn memory(&self) -> &Arc<Mutex<ConversationBufferMemory>> {
        &self.memory
    }

    pub async fn clear_memory(&self) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        memory
            .clear()
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Failed to clear memory: {}", e)))?;
        Ok(())
    }

    /// Simplified query interface
    pub async fn query(&self, question: impl Into<String>) -> Result<String, ChainError> {
        let inputs = HashMap::from([(self.input_key.clone(), Value::String(question.into()))]);
        let result = self.invoke(inputs).await?;
        result
            .get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("Missing output result".to_string()))
    }

    fn format_context(&self, documents: &[Document]) -> String {
        documents
            .iter()
            .map(|doc| doc.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    }

    /// Build structured messages for a given history, context, and question.
    /// Useful for testing the message construction logic.
    pub fn build_messages(
        &self,
        history: &[Message],
        context: &str,
        question: &str,
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        // System instruction
        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        } else {
            messages.push(Message::system(
                "You are an AI assistant. Answer the user's question based on the conversation history and reference information."
            ));
        }

        // Conversation history as individual messages
        for msg in history {
            messages.push(msg.clone());
        }

        // Context + question as the final human message
        let human_content = if context.is_empty() {
            question.to_string()
        } else {
            format!(
                "Reference information:\n{}\n\nQuestion: {}",
                context, question
            )
        };
        messages.push(Message::human(&human_content));

        messages
    }

    fn format_history(&self, messages: &[Message]) -> String {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.message_type {
                    crate::schema::MessageType::Human => "User",
                    crate::schema::MessageType::AI => "Assistant",
                    _ => "System",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn load_history(&self) -> Result<Vec<Message>, ChainError> {
        let memory = self.memory.lock().await;
        Ok(memory.chat_memory().messages().to_vec())
    }

    async fn save_context(&self, input: &str, output: &str) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        let inputs = HashMap::from([(self.input_key.clone(), input.to_string())]);
        let outputs = HashMap::from([(self.output_key.clone(), output.to_string())]);
        memory
            .save_context(&inputs, &outputs)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Failed to save context: {}", e)))?;
        Ok(())
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for ConversationRetrievalChain<M>
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
            println!("\n=== ConversationRetrievalChain Execution ===");
            println!("Question: {}", question);
        }

        // Step 1: Load conversation history
        let history_messages = self.load_history().await?;
        let history = self.format_history(&history_messages);

        if self.verbose {
            println!("History messages: {}", history_messages.len());
        }

        // Step 2: Retrieve relevant documents
        if self.verbose {
            println!("\n--- Step 2: Retrieve relevant documents ---");
        }

        let documents = self
            .retriever
            .retrieve(question, self.k)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Retrieval failed: {}", e)))?;

        if self.verbose {
            println!("Retrieved {} documents", documents.len());
            for (i, doc) in documents.iter().enumerate() {
                let preview = if doc.content.len() > 100 {
                    &doc.content[..100]
                } else {
                    &doc.content
                };
                println!("Document {}: {}", i + 1, preview);
            }
        }

        // Step 3: Assemble Prompt
        if self.verbose {
            println!("\n--- Step 3: Assemble Prompt ---");
        }

        let context = self.format_context(&documents);

        if self.verbose {
            println!("History length: {} characters", history.len());
            println!("Context length: {} characters", context.len());
        }

        // Step 4: LLM generates answer
        if self.verbose {
            println!("\n--- Step 4: LLM generates answer ---");
        }

        // H65: Split into structured messages (system + history + context + question)
        // instead of sending the entire prompt as a single human message.
        let mut messages = Vec::new();

        // System instruction
        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        } else {
            messages.push(Message::system(
                "You are an AI assistant. Answer the user's question based on the conversation history and reference information."
            ));
        }

        // Conversation history as individual messages
        for msg in &history_messages {
            messages.push(msg.clone());
        }

        // Context + question as the final human message
        let context_str = self.format_context(&documents);
        let human_content = if context_str.is_empty() {
            question.to_string()
        } else {
            format!(
                "Reference information:\n{}\n\nQuestion: {}",
                context_str, question
            )
        };
        messages.push(Message::human(&human_content));

        let response = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        let answer = response.content;

        if self.verbose {
            println!("Answer: {}", answer);
        }

        // Step 5: Save to memory
        self.save_context(question, &answer).await?;

        if self.verbose {
            println!("=== ConversationRetrievalChain Complete ===\n");
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
        let memory = ConversationBufferMemory::new();

        let chain = ConversationRetrievalChain::new(llm, retriever, memory);

        assert_eq!(chain.input_keys(), vec!["query"]);
        assert_eq!(chain.output_keys(), vec!["result"]);
        assert_eq!(chain.name(), "conversation_retrieval");
    }

    #[test]
    fn test_with_options() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        let memory = ConversationBufferMemory::new();

        let chain = ConversationRetrievalChain::new(llm, retriever, memory)
            .with_k(5)
            .with_input_key("question")
            .with_output_key("answer")
            .with_return_source_documents(true)
            .with_verbose(true);

        assert_eq!(chain.input_keys(), vec!["question"]);
        assert_eq!(chain.output_keys(), vec!["answer", "source_documents"]);
    }

    #[test]
    fn test_format_context() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        let memory = ConversationBufferMemory::new();

        let chain = ConversationRetrievalChain::new(llm, retriever, memory);

        let docs = vec![
            Document::new("Document 1 content"),
            Document::new("Document 2 content"),
        ];

        let context = chain.format_context(&docs);
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
        let memory = ConversationBufferMemory::new();

        let chain = ConversationRetrievalChain::new(llm, retriever, memory);

        let messages = chain.build_messages(&[], "Context text", "What is Rust?");

        // Should have system + context+question human message
        assert!(messages.iter().any(|m| m.content.contains("Context text")));
        assert!(messages.iter().any(|m| m.content.contains("What is Rust?")));
    }

    #[test]
    fn test_custom_prompt_template() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        let memory = ConversationBufferMemory::new();

        let custom_template = "History: {history}\nContext: {context}\nQuestion: {question}";

        let chain =
            ConversationRetrievalChain::new(llm, retriever, memory).with_qa_prompt(custom_template);

        let messages = chain.build_messages(&[], "Test context", "Test question");

        assert!(messages.iter().any(|m| m.content.contains("Test context")));
        assert!(messages.iter().any(|m| m.content.contains("Test question")));
    }
}
