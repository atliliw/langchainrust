// lc-chains/src/conversation_retrieval.rs
//! ConversationRetrieval Chain
//!
//! Retrieval-augmented generation chain with memory, combining conversation
//! history with document retrieval.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::BaseChatModel;
use lc_memory::{BaseMemory, ConversationBufferMemory};
use lc_providers::{wrap_chat_model, ProviderError};
use lc_rag::retriever::RetrieverTrait;
use lc_schema::{Message, MessageType};
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};
use crate::BoxedChatModel;

/// ConversationRetrievalChain
///
/// Retrieval-augmented conversation chain with memory that automatically:
/// 1. Loads conversation history
/// 2. Retrieves relevant documents
/// 3. Combines history + context + question
/// 4. LLM generates answer
/// 5. Saves to conversation memory
pub struct ConversationRetrievalChain {
    llm: BoxedChatModel,
    retriever: Arc<dyn RetrieverTrait>,
    memory: Arc<Mutex<dyn BaseMemory>>,

    system_prompt: Option<String>,
    input_key: String,
    output_key: String,
    name: String,

    k: usize,
    verbose: bool,
    return_source_documents: bool,
    source_document_key: String,
}

impl ConversationRetrievalChain {
    /// Create a new [`ConversationRetrievalChain`] with the given LLM,
    /// retriever, and conversation buffer memory.
    pub fn new<L>(
        llm: L,
        retriever: Arc<dyn RetrieverTrait>,
        memory: ConversationBufferMemory,
    ) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        // Align the memory's input/output keys with this chain's defaults
        // ("query"/"result"). `save_context` addresses the memory by these keys,
        // so without alignment persistence silently fails with `Missing input
        // key 'input'` on both the invoke and stream paths.
        Self::from_memory(
            llm,
            retriever,
            Arc::new(Mutex::new(
                memory
                    .with_return_messages(true)
                    .with_input_key("query".to_string())
                    .with_output_key("result".to_string()),
            )),
        )
    }

    /// Create from any [`BaseMemory`] implementation (window / summary /
    /// vector-store / persistent), mirroring `ConversationChain::from_memory`.
    pub fn from_memory<L>(
        llm: L,
        retriever: Arc<dyn RetrieverTrait>,
        memory: Arc<Mutex<dyn BaseMemory>>,
    ) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            retriever,
            memory,
            system_prompt: None,
            input_key: "query".to_string(),
            output_key: "result".to_string(),
            name: "conversation_retrieval".to_string(),
            k: 4,
            verbose: false,
            return_source_documents: false,
            source_document_key: "source_documents".to_string(),
        }
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the input key.
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set the output key.
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set the chain name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the number of documents to retrieve.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set whether to return the source documents with the answer.
    pub fn with_return_source_documents(mut self, return_source: bool) -> Self {
        self.return_source_documents = return_source;
        self
    }

    /// Get the memory reference.
    pub fn memory(&self) -> &Arc<Mutex<dyn BaseMemory>> {
        &self.memory
    }

    /// Clear the conversation memory.
    pub async fn clear_memory(&self) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        memory
            .clear()
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Failed to clear memory: {}", e)))?;
        Ok(())
    }

    /// Simplified query interface.
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
    pub fn build_messages(
        &self,
        history: &[Message],
        context: &str,
        question: &str,
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        } else {
            messages.push(Message::system(
                "You are an AI assistant. Answer the user's question based on the conversation history and reference information."
            ));
        }

        for msg in history {
            messages.push(msg.clone());
        }

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
                    MessageType::Human => "User",
                    MessageType::AI => "Assistant",
                    _ => "System",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn load_history(&self, question: &str) -> Result<Vec<Message>, ChainError> {
        let memory = self.memory.lock().await;
        let inputs = HashMap::from([(self.input_key.clone(), question.to_string())]);
        let vars = memory
            .load_memory_variables(&inputs)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Failed to load memory: {}", e)))?;
        Ok(crate::base::variables_to_messages(&vars))
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
impl BaseChain for ConversationRetrievalChain {
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
        let history_messages = self.load_history(question).await?;
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

        let context_str = self.format_context(&documents);
        let messages = self.build_messages(&history_messages, &context_str, question);

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
            // Explicit error instead of silently inserting Value::Null (P1-2).
            let sources = crate::base::documents_to_values(&documents)?;
            result.insert(self.source_document_key.clone(), Value::Array(sources));
        }

        Ok(result)
    }

    /// Stream execution for ConversationRetrievalChain -- token by token output.
    ///
    /// P2-2: real streaming — loads history, retrieves and assembles the prompt,
    /// then pushes LLM tokens via `stream_chat`. The full answer is accumulated
    /// through an unbounded channel (the P1-4 pattern) and written to memory
    /// once the stream completes, matching the invoke path's `save_context`.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let question = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        if self.verbose {
            println!("\n=== ConversationRetrievalChain Stream ===");
            println!("Question: {}", question);
        }

        // Step 1: Load conversation history
        let history_messages = self.load_history(question).await?;

        // Step 2: Retrieve relevant documents
        let documents = self
            .retriever
            .retrieve(question, self.k)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Retrieval failed: {}", e)))?;

        if self.verbose {
            println!("Retrieved {} documents", documents.len());
        }

        // Step 3: Assemble messages (history + context + question)
        let context = self.format_context(&documents);
        let messages = self.build_messages(&history_messages, &context, question);

        // Step 4: Stream LLM tokens
        let llm_stream = self
            .llm
            .stream_chat(messages, None)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        let memory = self.memory.clone();
        let input_key = self.input_key.clone();
        let output_key = self.output_key.clone();
        let question_str = question.to_string();

        // Queue tokens through an unbounded channel; the finalizer drains every
        // token and writes the full output to memory (never a truncated one).
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let stream = llm_stream.map(move |result| match result {
            Ok(token) => {
                let _ = tx.send(token.clone());
                Ok(StreamToken {
                    token,
                    is_final: false,
                })
            }
            Err(e) => Err(ChainError::StreamError(format!(
                "Stream token error: {}",
                e
            ))),
        });

        let finalizer_stream = async move {
            let mut output = String::new();
            let mut rx = rx;
            while let Some(token) = rx.recv().await {
                output.push_str(&token);
            }

            // Step 5: Save to memory
            if !output.is_empty() {
                let mut mem = memory.lock().await;
                let ctx_inputs = HashMap::from([(input_key.clone(), question_str.clone())]);
                let ctx_outputs = HashMap::from([(output_key.clone(), output)]);
                if let Err(e) = mem.save_context(&ctx_inputs, &ctx_outputs).await {
                    log::error!("[ConversationRetrievalChain] failed to save context: {}", e);
                }
            }
        };

        let final_stream = stream.chain(futures_util::stream::once(async move {
            finalizer_stream.await;
            Ok(StreamToken {
                token: String::new(),
                is_final: true,
            })
        }));

        Ok(Box::pin(final_stream))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::LLMResult;
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use lc_rag::retriever::RetrieverError;
    use lc_shared::document::SearchResult;
    use std::pin::Pin;

    /// Mock retriever that returns the preloaded documents (up to `k`).
    struct MockRetriever(Vec<Document>);

    #[async_trait]
    impl RetrieverTrait for MockRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
            Ok(self.0.iter().take(k).cloned().collect())
        }
        async fn retrieve_with_scores(
            &self,
            _query: &str,
            _k: usize,
        ) -> Result<Vec<SearchResult>, RetrieverError> {
            Ok(Vec::new())
        }
        async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
            Ok(())
        }
    }

    /// Mock chat model with a deterministic token stream.
    struct MockLLM;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "hello world".to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLLM {
        fn model_name(&self) -> &str {
            "mock"
        }
        fn get_num_tokens(&self, t: &str) -> usize {
            t.len()
        }
        fn with_temperature(self, _: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _: usize) -> Self {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for MockLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "hello world".to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            let tokens = [
                Ok("hello".to_string()),
                Ok(" ".to_string()),
                Ok("world".to_string()),
            ];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    fn doc(content: &str) -> Document {
        Document::new(content.to_string())
    }

    /// P2-2: ConversationRetrieval streams real tokens and persists the full
    /// streamed answer to memory once the stream completes.
    #[tokio::test]
    async fn test_conversation_retrieval_stream_saves_to_memory() {
        let retriever: Arc<dyn RetrieverTrait> = Arc::new(MockRetriever(vec![doc("ctx")]));
        let chain =
            ConversationRetrievalChain::new(MockLLM, retriever, ConversationBufferMemory::new());
        let inputs = HashMap::from([("query".to_string(), Value::String("q".to_string()))]);

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
        assert_eq!(text, "hello world");
        assert!(tokens.last().unwrap().is_final);

        // Step 5: the streamed answer is written to memory (never truncated).
        let memory = chain.memory().clone();
        let mem = memory.lock().await;
        let vars = mem.load_memory_variables(&HashMap::new()).await.unwrap();
        let messages = crate::base::variables_to_messages(&vars);
        assert!(
            messages.iter().any(|m| m.content.contains("hello world")),
            "memory should contain the streamed answer, got {:?}",
            messages
        );
    }

    #[tokio::test]
    async fn test_conversation_retrieval_stream_missing_input() {
        let retriever: Arc<dyn RetrieverTrait> = Arc::new(MockRetriever(vec![]));
        let chain =
            ConversationRetrievalChain::new(MockLLM, retriever, ConversationBufferMemory::new());
        let err = match chain.stream(HashMap::new()).await {
            Ok(_) => panic!("expected a missing-input error"),
            Err(e) => e,
        };
        assert!(matches!(err, ChainError::MissingInput(_)));
    }
}
