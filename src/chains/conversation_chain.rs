// src/chains/conversation_chain.rs
//! Conversation Chain
//!
//! A Chain with memory, supporting multi-turn conversations.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;
use futures_util::StreamExt;

use super::base::{BaseChain, ChainResult, ChainError, ChainStream, StreamToken};
use crate::BaseChatModel;
use crate::memory::{ConversationBufferMemory, BaseMemory};
use crate::schema::Message;
use crate::Runnable;
use tokio::sync::Mutex;

/// Conversation Chain
///
/// A Chain with memory that automatically saves and loads conversation history.
///
/// # Examples
/// ```ignore
/// use langchainrust::{ConversationChain, OpenAIChat, OpenAIConfig, ConversationBufferMemory};
///
/// let llm = OpenAIChat::new(config);
/// let memory = ConversationBufferMemory::new();
///
/// let chain = ConversationChain::new(llm, memory);
///
/// // First turn
/// let result = chain.predict("Hello").await?;
/// println!("AI: {}", result);
///
/// // Second turn - AI remembers previous conversation
/// let result = chain.predict("What's my name?").await?;
/// ```
pub struct ConversationChain<M: BaseChatModel> {
    llm: M,
    memory: Arc<Mutex<ConversationBufferMemory>>,
    system_prompt: Option<String>,
    input_key: String,
    output_key: String,
    memory_key: String,
    name: String,
    verbose: bool,
}

impl<M: BaseChatModel + 'static> ConversationChain<M> {
    /// Create a new ConversationChain
    ///
    /// # Arguments
    /// * `llm` - LLM client (any type implementing BaseChatModel)
    /// * `memory` - Conversation memory
    pub fn new(llm: M, memory: ConversationBufferMemory) -> Self {
        Self {
            llm,
            memory: Arc::new(Mutex::new(memory.with_return_messages(true))),
            system_prompt: None,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            name: "conversation_chain".to_string(),
            verbose: false,
        }
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set input key name
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set output key name
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set memory key name
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set chain name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get memory reference
    pub fn memory(&self) -> &Arc<Mutex<ConversationBufferMemory>> {
        &self.memory
    }

    pub fn builder(llm: M) -> ConversationChainBuilder<M> {
        ConversationChainBuilder::new(llm)
    }

    /// Clear memory
    pub async fn clear_memory(&self) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        memory.clear().await.map_err(|e|
            ChainError::ExecutionError(format!("Failed to clear memory: {}", e))
        )?;
        Ok(())
    }

    /// Simplified prediction interface
    ///
    /// Takes a user input string, returns AI response string.
    pub async fn predict(&self, input: impl Into<String>) -> Result<String, ChainError> {
        let inputs = HashMap::from([
            (self.input_key.clone(), Value::String(input.into()))
        ]);

        let result = self.invoke(inputs).await?;

        result.get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("Missing output".to_string()))
    }

    /// Prepare message list
    ///
    /// Combines system prompt, history messages, and current user input.
    pub fn prepare_messages(
        &self,
        input: &str,
        history_messages: &[Message],
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        // Add system prompt
        if let Some(system_prompt) = &self.system_prompt {
            messages.push(Message::system(system_prompt));
        }

        // Add history messages
        for msg in history_messages {
            messages.push(msg.clone());
        }

        // Add current user input
        messages.push(Message::human(input));

        messages
    }

    /// Load history messages
    async fn load_history(&self) -> Result<Vec<Message>, ChainError> {
        let memory = self.memory.lock().await;
        let messages = memory.chat_memory().messages().to_vec();
        Ok(messages)
    }

    /// Save conversation context
    async fn save_context(&self, input: &str, output: &str) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;

        let inputs = HashMap::from([(self.input_key.clone(), input.to_string())]);
        let outputs = HashMap::from([(self.output_key.clone(), output.to_string())]);

        memory.save_context(&inputs, &outputs).await
            .map_err(|e| ChainError::ExecutionError(format!("Failed to save context: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for ConversationChain<M>
where
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        // Validate inputs
        self.validate_inputs(&inputs)?;

        // Get user input
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        if self.verbose {
            println!("\n=== ConversationChain execution ===");
            println!("User input: {}", input);
        }

        // Load history messages
        let history_messages = self.load_history().await?;

        if self.verbose && !history_messages.is_empty() {
            println!("History message count: {}", history_messages.len());
        }

        // Prepare message list
        let messages = self.prepare_messages(input, &history_messages);

        if self.verbose {
            println!("Total message count: {}", messages.len());
        }

        // Call LLM
        let result = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        let output = result.content;

        if self.verbose {
            println!("AI response: {}", output);
        }

        // Save conversation context
        self.save_context(input, &output).await?;

        if self.verbose {
            println!("=== ConversationChain complete ===\n");
        }

        // Build output
        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));

        Ok(result)
    }

    /// Stream execution for ConversationChain -- token by token output.
    ///
    /// After the stream completes, conversation context is automatically saved.
    async fn stream(
        &self,
        inputs: HashMap<String, Value>,
    ) -> Result<ChainStream, ChainError> {
        // Validate inputs
        self.validate_inputs(&inputs)?;

        // Get user input
        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        // Load history messages
        let history_messages = self.load_history().await?;

        // Prepare message list
        let messages = self.prepare_messages(input, &history_messages);

        // Call LLM stream_chat
        let llm_stream = self
            .llm
            .stream_chat(messages, None)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        // Wrap the stream to collect output and save context on completion
        let memory = self.memory.clone();
        let input_key = self.input_key.clone();
        let output_key = self.output_key.clone();
        let input_str = input.to_string();

        // Accumulate tokens and save context when stream ends
        let accumulated: Arc<tokio::sync::Mutex<String>> = Arc::new(tokio::sync::Mutex::new(String::new()));
        let accumulated_clone = accumulated.clone();

        let stream = llm_stream.map(move |result| {
            match result {
                Ok(token) => {
                    // Accumulate token for later context saving
                    if let Ok(mut acc) = accumulated_clone.try_lock() {
                        acc.push_str(&token);
                    }
                    Ok(StreamToken {
                        token,
                        is_final: false,
                    })
                }
                Err(e) => Err(ChainError::StreamError(format!("Stream token error: {}", e))),
            }
        });

        // Create a finalizer stream that saves context after the LLM stream ends
        let finalizer_stream = async move {
            // Wait a tick to let the main stream be consumed
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;

            // Get accumulated output
            let output = accumulated.lock().await.clone();

            // Save context
            if !output.is_empty() {
                let mut mem = memory.lock().await;
                let ctx_inputs = HashMap::from([(input_key.clone(), input_str.clone())]);
                let ctx_outputs = HashMap::from([(output_key.clone(), output)]);
                let _ = mem.save_context(&ctx_inputs, &ctx_outputs).await;
            }
        };

        // Combine: main stream + finalizer
        let final_stream = stream.chain(futures_util::stream::once(async move {
            finalizer_stream.await;
            // Return a final token to signal completion
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

/// ConversationChain Builder
///
/// Convenience builder for ConversationChain.
pub struct ConversationChainBuilder<M: BaseChatModel> {
    llm: M,
    memory: Option<ConversationBufferMemory>,
    system_prompt: Option<String>,
    input_key: Option<String>,
    output_key: Option<String>,
    memory_key: Option<String>,
    name: Option<String>,
    verbose: Option<bool>,
}

impl<M: BaseChatModel + 'static> ConversationChainBuilder<M> {
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            memory: None,
            system_prompt: None,
            input_key: None,
            output_key: None,
            memory_key: None,
            name: None,
            verbose: None,
        }
    }

    pub fn memory(mut self, memory: ConversationBufferMemory) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = Some(key.into());
        self
    }

    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }

    pub fn memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = Some(key.into());
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }

    pub fn build(self) -> ConversationChain<M> {
        let memory = self.memory.unwrap_or_default();
        let mut chain = ConversationChain::new(self.llm, memory);

        if let Some(prompt) = self.system_prompt {
            chain = chain.with_system_prompt(prompt);
        }

        if let Some(key) = self.input_key {
            chain = chain.with_input_key(key);
        }

        if let Some(key) = self.output_key {
            chain = chain.with_output_key(key);
        }

        if let Some(key) = self.memory_key {
            chain = chain.with_memory_key(key);
        }

        if let Some(name) = self.name {
            chain = chain.with_name(name);
        }

        if let Some(verbose) = self.verbose {
            chain = chain.with_verbose(verbose);
        }

        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::OpenAIChat;
    use crate::OpenAIConfig;
    use crate::memory::ConversationBufferMemory;

    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-test".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
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
    fn test_conversation_chain_new() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory);

        assert_eq!(chain.input_keys(), vec!["input"]);
        assert_eq!(chain.output_keys(), vec!["output"]);
        assert_eq!(chain.name(), "conversation_chain");
    }

    #[test]
    fn test_conversation_chain_with_system_prompt() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("You are a friendly assistant");

        assert!(chain.system_prompt.is_some());
        assert_eq!(chain.system_prompt.unwrap(), "You are a friendly assistant");
    }

    #[test]
    fn test_conversation_chain_builder() {
        let llm = OpenAIChat::new(create_test_config());

        let chain = ConversationChainBuilder::new(llm)
            .system_prompt("You are a Rust expert")
            .input_key("question")
            .output_key("answer")
            .verbose(true)
            .build();

        assert_eq!(chain.input_key, "question");
        assert_eq!(chain.output_key, "answer");
        assert!(chain.verbose);
    }

    #[test]
    fn test_prepare_messages_empty_history() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("You are an assistant");

        let messages = chain.prepare_messages("Hello", &[]);

        // System message + user message
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message_type, crate::schema::MessageType::System);
        assert_eq!(messages[1].message_type, crate::schema::MessageType::Human);
    }

    #[test]
    fn test_prepare_messages_with_history() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory);

        let history = vec![
            Message::human("Hello"),
            Message::ai("Hi! How can I help you?"),
        ];

        let messages = chain.prepare_messages("Tell me about Rust", &history);

        // 2 history + 1 user input
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].message_type, crate::schema::MessageType::Human);
        assert_eq!(messages[1].message_type, crate::schema::MessageType::AI);
        assert_eq!(messages[2].message_type, crate::schema::MessageType::Human);
    }

    /// Real API test - single turn
    /// Run: cargo test test_conversation_chain_single -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_conversation_chain_single() {
        let config = OpenAIConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1".to_string()),
            model: "glm-5.2".to_string(),
            streaming: false,
            ..Default::default()
        };

        let llm = OpenAIChat::new(config);
        let memory = ConversationBufferMemory::new();

        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("You are a friendly assistant")
            .with_verbose(true);

        println!("\n=== Test ConversationChain - single turn ===");

        let result = chain.predict("Hello, introduce yourself").await.unwrap();
        println!("AI response: {}", result);

        assert!(!result.is_empty());
    }

    /// Real API test - multi-turn
    /// Run: cargo test test_conversation_chain_multi_turn -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_conversation_chain_multi_turn() {
        let config = OpenAIConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1".to_string()),
            model: "glm-5.2".to_string(),
            streaming: false,
            ..Default::default()
        };

        let llm = OpenAIChat::new(config);
        let memory = ConversationBufferMemory::new();

        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("You are a friendly assistant, remember the user's name")
            .with_verbose(true);

        println!("\n=== Test ConversationChain - multi-turn ===");

        // First turn: tell name
        println!("\n--- Turn 1 ---");
        let result1 = chain.predict("Hello, my name is Alice").await.unwrap();
        println!("AI: {}", result1);

        // Second turn: ask name (test memory)
        println!("\n--- Turn 2 ---");
        let result2 = chain.predict("What is my name?").await.unwrap();
        println!("AI: {}", result2);

        // Check memory saved the name
        let memory = chain.memory.lock().await;
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();

        println!("\nHistory: {}", history);
        assert!(history.contains("Alice"), "Memory should contain user name");
    }

    /// Real API test - clear memory
    /// Run: cargo test test_conversation_chain_clear_memory -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_conversation_chain_clear_memory() {
        let config = OpenAIConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1".to_string()),
            model: "glm-5.2".to_string(),
            streaming: false,
            ..Default::default()
        };

        let llm = OpenAIChat::new(config);
        let memory = ConversationBufferMemory::new();

        let chain = ConversationChain::new(llm, memory);

        println!("\n=== Test ConversationChain - clear memory ===");

        // First turn
        let result1 = chain.predict("My name is Bob").await.unwrap();
        println!("Turn 1: {}", result1);

        // Clear memory
        chain.clear_memory().await.unwrap();

        // Second turn: should not remember name
        let result2 = chain.predict("What is my name?").await.unwrap();
        println!("Turn 2 (after clear): {}", result2);

        // Check memory is cleared
        let memory = chain.memory.lock().await;
        assert_eq!(memory.chat_memory().len(), 2);
    }
}
