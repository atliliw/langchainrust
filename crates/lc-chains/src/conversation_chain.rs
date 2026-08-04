// lc-chains/src/conversation_chain.rs
//! Conversation Chain
//!
//! A Chain with memory, supporting multi-turn conversations.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::language_models::LLMResult;
use lc_core::{BaseChatModel, Runnable};
use lc_memory::{BaseMemory, ConversationBufferMemory};
use lc_schema::Message;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};

/// Conversation Chain
///
/// A Chain with memory that automatically saves and loads conversation history.
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
    /// Create a new ConversationChain.
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

    /// Set system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set input key name.
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set output key name.
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set memory key name.
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set chain name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get memory reference.
    pub fn memory(&self) -> &Arc<Mutex<ConversationBufferMemory>> {
        &self.memory
    }

    pub fn builder(llm: M) -> ConversationChainBuilder<M> {
        ConversationChainBuilder::new(llm)
    }

    /// Clear memory.
    pub async fn clear_memory(&self) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        memory
            .clear()
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Failed to clear memory: {}", e)))?;
        Ok(())
    }

    /// Simplified prediction interface.
    ///
    /// Takes a user input string, returns AI response string.
    pub async fn predict(&self, input: impl Into<String>) -> Result<String, ChainError> {
        let inputs = HashMap::from([(self.input_key.clone(), Value::String(input.into()))]);

        let result = self.invoke(inputs).await?;

        result
            .get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("Missing output".to_string()))
    }

    /// Prepare message list.
    ///
    /// Combines system prompt, history messages, and current user input.
    pub fn prepare_messages(&self, input: &str, history_messages: &[Message]) -> Vec<Message> {
        let mut messages = Vec::new();

        if let Some(system_prompt) = &self.system_prompt {
            messages.push(Message::system(system_prompt));
        }

        for msg in history_messages {
            messages.push(msg.clone());
        }

        messages.push(Message::human(input));

        messages
    }

    /// Load history messages.
    async fn load_history(&self) -> Result<Vec<Message>, ChainError> {
        let memory = self.memory.lock().await;
        let messages = memory.chat_memory().messages().to_vec();
        Ok(messages)
    }

    /// Save conversation context.
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
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for ConversationChain<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        if self.verbose {
            println!("\n=== ConversationChain execution ===");
            println!("User input: {}", input);
        }

        let history_messages = self.load_history().await?;

        if self.verbose && !history_messages.is_empty() {
            println!("History message count: {}", history_messages.len());
        }

        let messages = self.prepare_messages(input, &history_messages);

        if self.verbose {
            println!("Total message count: {}", messages.len());
        }

        let result = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        let output = result.content;

        if self.verbose {
            println!("AI response: {}", output);
        }

        self.save_context(input, &output).await?;

        if self.verbose {
            println!("=== ConversationChain complete ===\n");
        }

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));

        Ok(result)
    }

    /// Stream execution for ConversationChain -- token by token output.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let history_messages = self.load_history().await?;

        let messages = self.prepare_messages(input, &history_messages);

        let llm_stream = self
            .llm
            .stream_chat(messages, None)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        let memory = self.memory.clone();
        let input_key = self.input_key.clone();
        let output_key = self.output_key.clone();
        let input_str = input.to_string();

        let accumulated: Arc<tokio::sync::Mutex<String>> =
            Arc::new(tokio::sync::Mutex::new(String::new()));
        let accumulated_clone = accumulated.clone();

        let stream = llm_stream.map(move |result| match result {
            Ok(token) => {
                if let Ok(mut acc) = accumulated_clone.try_lock() {
                    acc.push_str(&token);
                }
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
            let output = accumulated.lock().await.clone();

            if !output.is_empty() {
                let mut mem = memory.lock().await;
                let ctx_inputs = HashMap::from([(input_key.clone(), input_str.clone())]);
                let ctx_outputs = HashMap::from([(output_key.clone(), output)]);
                if let Err(e) = mem.save_context(&ctx_inputs, &ctx_outputs).await {
                    eprintln!("[ConversationChain] Warning: failed to save context: {}", e);
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

/// ConversationChain Builder.
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
