// lc-chains/src/conversation_chain.rs
//! Conversation Chain
//!
//! A Chain with memory, supporting multi-turn conversations.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::BaseChatModel;
use lc_memory::{BaseMemory, ConversationBufferMemory};
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};
use crate::BoxedChatModel;

/// Conversation Chain
///
/// A Chain with memory that automatically saves and loads conversation history.
pub struct ConversationChain {
    llm: BoxedChatModel,
    memory: Arc<Mutex<dyn BaseMemory>>,
    system_prompt: Option<String>,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
}

impl ConversationChain {
    /// Create a new ConversationChain.
    ///
    /// # Arguments
    /// * `llm` - LLM client (any type implementing BaseChatModel)
    /// * `memory` - Conversation memory
    pub fn new<L>(llm: L, memory: ConversationBufferMemory) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self::from_memory(llm, Arc::new(Mutex::new(memory.with_return_messages(true))))
    }

    /// Create a ConversationChain from any [`BaseMemory`] implementation.
    ///
    /// Unlike [`ConversationChain::new`] (which takes the concrete
    /// `ConversationBufferMemory`), this accepts any memory — window, summary,
    /// vector-store, persistent — so the chain's memory is pluggable without
    /// changing the chain source.
    pub fn from_memory<L>(llm: L, memory: Arc<Mutex<dyn BaseMemory>>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self::from_wrapped_memory(wrap_chat_model(llm), memory)
    }

    /// Construct from an already-wrapped model (internal builder path).
    pub(crate) fn from_wrapped_memory(
        llm: BoxedChatModel,
        memory: Arc<Mutex<dyn BaseMemory>>,
    ) -> Self {
        Self {
            llm,
            memory,
            system_prompt: None,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
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
    pub fn memory(&self) -> &Arc<Mutex<dyn BaseMemory>> {
        &self.memory
    }

    /// Create a [`ConversationChainBuilder`] from an LLM.
    pub fn builder<L>(llm: L) -> ConversationChainBuilder
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
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

    /// Load history messages through the memory trait's `load_memory_variables`.
    ///
    /// Accepts any memory shape (array of `Message` objects or a rendered
    /// history string); the current input is forwarded so input-aware memories
    /// (e.g. `VectorStoreRetrieverMemory`) can use it as the retrieval query.
    async fn load_history(&self, input: &str) -> Result<Vec<Message>, ChainError> {
        let memory = self.memory.lock().await;
        let inputs = HashMap::from([(self.input_key.clone(), input.to_string())]);
        let vars = memory
            .load_memory_variables(&inputs)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Failed to load memory: {}", e)))?;
        Ok(crate::base::variables_to_messages(&vars))
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
impl BaseChain for ConversationChain {
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

        let history_messages = self.load_history(input).await?;

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

        let history_messages = self.load_history(input).await?;

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

        // P1-4: queue tokens through an unbounded channel instead of `try_lock`
        // on a shared mutex. The map closure's sync sender never drops a token,
        // so the memory write sees the full output rather than a truncated one.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let stream = llm_stream.map(move |result| match result {
            Ok(chunk) => {
                let _ = tx.send(chunk.text.clone());
                Ok(StreamToken {
                    token: chunk.text,
                    is_final: false,
                })
            }
            Err(e) => Err(ChainError::StreamError(format!(
                "Stream token error: {}",
                e
            ))),
        });

        let finalizer_stream = async move {
            // The channel closes once the map closure's sender is dropped (stream
            // exhausted); drain every queued token into the memory write.
            let mut output = String::new();
            let mut rx = rx;
            while let Some(token) = rx.recv().await {
                output.push_str(&token);
            }

            if !output.is_empty() {
                let mut mem = memory.lock().await;
                let ctx_inputs = HashMap::from([(input_key.clone(), input_str.clone())]);
                let ctx_outputs = HashMap::from([(output_key.clone(), output)]);
                if let Err(e) = mem.save_context(&ctx_inputs, &ctx_outputs).await {
                    log::error!("[ConversationChain] failed to save context: {}", e);
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
pub struct ConversationChainBuilder {
    llm: BoxedChatModel,
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    system_prompt: Option<String>,
    input_key: Option<String>,
    output_key: Option<String>,
    name: Option<String>,
    verbose: Option<bool>,
}

impl ConversationChainBuilder {
    /// Create a new [`ConversationChainBuilder`] with the given chat model.
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            memory: None,
            system_prompt: None,
            input_key: None,
            output_key: None,
            name: None,
            verbose: None,
        }
    }

    /// Set the conversation memory.
    pub fn memory<Mem: BaseMemory + 'static>(mut self, memory: Mem) -> Self {
        self.memory = Some(Arc::new(Mutex::new(memory)));
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the input key.
    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = Some(key.into());
        self
    }

    /// Set the output key.
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }

    /// Set the chain name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set verbose mode.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }

    /// Build the final [`ConversationChain`].
    pub fn build(self) -> ConversationChain {
        let mut chain = match self.memory {
            Some(memory) => ConversationChain::from_wrapped_memory(self.llm, memory),
            None => ConversationChain::from_wrapped_memory(
                self.llm,
                Arc::new(Mutex::new(
                    ConversationBufferMemory::new().with_return_messages(true),
                )),
            ),
        };

        if let Some(prompt) = self.system_prompt {
            chain = chain.with_system_prompt(prompt);
        }

        if let Some(key) = self.input_key {
            chain = chain.with_input_key(key);
        }

        if let Some(key) = self.output_key {
            chain = chain.with_output_key(key);
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
