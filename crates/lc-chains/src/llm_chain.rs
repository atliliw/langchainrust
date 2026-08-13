// lc-chains/src/llm_chain.rs
//! LLM Chain
//!
//! The most basic Chain, combining a Prompt and an LLM.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::LLMResult;
use lc_core::runnables::RunnableConfig;
use lc_core::{BaseChatModel, Runnable};
use lc_schema::Message;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};

/// LLM Chain
///
/// Combines a Prompt template and an LLM. The most basic Chain.
///
/// # Examples
/// ```ignore
/// use lc_chains::LLMChain;
///
/// let chain = LLMChain::new(llm, "{question}");
///
/// let inputs = HashMap::from([("question".to_string(), "What is Rust?".into())]);
/// let result = chain.invoke(inputs).await?;
/// ```
pub struct LLMChain<M: BaseChatModel> {
    /// LLM client.
    llm: M,

    /// Prompt template.
    prompt_template: String,

    /// Input key name.
    input_key: String,

    /// Output key name.
    output_key: String,

    /// Chain name.
    name: String,
}

/// Pre-compiled regex for detecting unreplaced template variables.
static TEMPLATE_VAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}").unwrap());

impl<M: BaseChatModel> LLMChain<M> {
    /// Create a new LLMChain.
    ///
    /// # Arguments
    /// * `llm` - LLM client (any type implementing BaseChatModel)
    /// * `prompt_template` - Prompt template string with {variable} placeholders
    pub fn new(llm: M, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
            input_key: "question".to_string(),
            output_key: "text".to_string(),
            name: "llm_chain".to_string(),
        }
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

    /// Render the Prompt template.
    ///
    /// Validates that all {variable} placeholders in the template
    /// have been replaced. Returns an error if any unreplaced placeholders remain.
    fn render_prompt(&self, inputs: &HashMap<String, Value>) -> Result<String, ChainError> {
        let mut prompt = self.prompt_template.clone();

        for (key, value) in inputs {
            let placeholder = format!("{{{}}}", key);
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            prompt = prompt.replace(&placeholder, &value_str);
        }

        // Check for unreplaced {variable} placeholders
        let unreplaced: Vec<&str> = TEMPLATE_VAR_RE
            .captures_iter(&prompt)
            .filter_map(|c| c.get(1).map(|m| m.as_str()))
            .collect();

        if !unreplaced.is_empty() {
            return Err(ChainError::ExecutionError(format!(
                "Prompt template has unreplaced variable(s): {}",
                unreplaced.join(", ")
            )));
        }

        Ok(prompt)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for LLMChain<M>
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

        let prompt = self.render_prompt(&inputs)?;

        let messages = vec![Message::human(&prompt)];
        let result = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), Value::String(result.content));

        Ok(output)
    }

    /// Execute the Chain with callback propagation.
    ///
    /// Fires `on_chain_start` → `on_llm_start` → LLM call → `on_llm_end` → `on_chain_end`.
    /// On error, fires `on_llm_error` / `on_chain_error` instead.
    async fn invoke_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;

        let callbacks = config.as_ref().and_then(|c| c.callbacks.clone());

        // Create root RunTree for this chain invocation
        let mut run = RunTree::new(self.name(), RunType::Chain, json!({ "inputs": inputs }));

        // on_chain_start
        if let Some(ref cb) = callbacks {
            cb.dispatch_chain_start(&run, &run.inputs).await;
        }

        let prompt = self.render_prompt(&inputs)?;
        let messages = vec![Message::human(&prompt)];

        // on_llm_start — single child run reused for both on_llm_end and
        // on_llm_error, so the trace has exactly one LLM node per call
        // (previously each callback created its own child, producing duplicate runs).
        let mut llm_run = run.create_child(
            format!("{}.llm", self.name()),
            RunType::Llm,
            json!({"messages_count": messages.len()}),
        );
        if let Some(ref cb) = callbacks {
            cb.dispatch_llm_start(&llm_run, &messages).await;
        }

        // LLM call with config propagation
        let llm_config = config.clone();
        let result = self.llm.invoke(messages, llm_config).await;

        match result {
            Ok(llm_result) => {
                // on_llm_end
                llm_run.end(json!({"response": &llm_result.content}));
                if let Some(ref cb) = callbacks {
                    cb.dispatch_llm_end(&llm_run, &llm_result.content).await;
                }

                let mut output = HashMap::new();
                output.insert(
                    self.output_key.clone(),
                    Value::String(llm_result.content.clone()),
                );

                run.end(json!({"output": &llm_result.content}));

                // on_chain_end
                if let Some(ref cb) = callbacks {
                    cb.dispatch_chain_end(&run, &json!({"output": llm_result.content}))
                        .await;
                }

                Ok(output)
            }
            Err(e) => {
                let err_msg = e.to_string();

                // on_llm_error
                llm_run.end_with_error(err_msg.clone());
                if let Some(ref cb) = callbacks {
                    cb.dispatch_llm_error(&llm_run, &err_msg).await;
                }

                run.end_with_error(err_msg.clone());

                // on_chain_error
                if let Some(ref cb) = callbacks {
                    cb.dispatch_chain_error(&run, &err_msg).await;
                }

                Err(ChainError::ExecutionError(format!(
                    "LLM call failed: {}",
                    err_msg
                )))
            }
        }
    }

    /// Stream execution for LLMChain -- token by token output.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let prompt = self.render_prompt(&inputs)?;

        let messages = vec![Message::human(&prompt)];
        let llm_stream = self
            .llm
            .stream_chat(messages, None)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        let stream = llm_stream.map(move |result| match result {
            Ok(token) => Ok(StreamToken {
                token,
                is_final: false,
            }),
            Err(e) => Err(ChainError::StreamError(format!(
                "Stream token error: {}",
                e
            ))),
        });

        let final_stream = stream.chain(futures_util::stream::once(async move {
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

/// LLMChain Builder.
///
/// Convenience builder for LLMChain.
pub struct LLMChainBuilder<M: BaseChatModel> {
    llm: M,
    prompt_template: String,
    input_key: Option<String>,
    output_key: Option<String>,
    name: Option<String>,
}

impl<M: BaseChatModel> LLMChainBuilder<M> {
    pub fn new(llm: M, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
            input_key: None,
            output_key: None,
            name: None,
        }
    }

    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = Some(key.into());
        self
    }

    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn build(self) -> LLMChain<M> {
        let mut chain = LLMChain::new(self.llm, self.prompt_template);

        if let Some(key) = self.input_key {
            chain = chain.with_input_key(key);
        }

        if let Some(key) = self.output_key {
            chain = chain.with_output_key(key);
        }

        if let Some(name) = self.name {
            chain = chain.with_name(name);
        }

        chain
    }
}
