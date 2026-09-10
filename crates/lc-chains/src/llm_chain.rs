// lc-chains/src/llm_chain.rs
//! LLM Chain
//!
//! The most basic Chain, combining a Prompt and an LLM.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_callbacks::{RunTree, RunType};
use lc_core::runnables::RunnableConfig;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::base::{
    stream_chain_with_callbacks, substitute_template, BaseChain, ChainError, ChainResult,
    ChainStream, StreamToken,
};
use crate::BoxedChatModel;

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
pub struct LLMChain {
    /// LLM client.
    llm: BoxedChatModel,

    /// Prompt template.
    prompt_template: String,

    /// Input key name.
    input_key: String,

    /// Output key name.
    output_key: String,

    /// Chain name.
    name: String,
}

impl LLMChain {
    /// Shared streaming body used by `stream` (config-less) and
    /// `stream_with_config` (config threaded into the LLM stream).
    async fn stream_body(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;
        if config.as_ref().is_some_and(|c| c.is_cancelled()) {
            return Err(ChainError::StreamError("Operation cancelled".to_string()));
        }
        let prompt = self.render_prompt(&inputs)?;
        let messages = vec![Message::human(&prompt)];
        let llm_stream = self
            .llm
            .stream_chat(messages, config)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;
        let stream = llm_stream.map(move |result| match result {
            Ok(chunk) => Ok(StreamToken {
                token: chunk.text,
                is_final: false,
            }),
            Err(e) => Err(ChainError::StreamError(format!("Stream token error: {}", e))),
        });
        let final_stream = stream.chain(futures_util::stream::once(async move {
            Ok(StreamToken {
                token: String::new(),
                is_final: true,
            })
        }));
        Ok(Box::pin(final_stream))
    }

    /// Create a new LLMChain.
    ///
    /// # Arguments
    /// * `llm` - LLM client (any type implementing BaseChatModel)
    /// * `prompt_template` - Prompt template string with {variable} placeholders
    pub fn new<L>(llm: L, prompt_template: impl Into<String>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self::from_wrapped(wrap_chat_model(llm), prompt_template)
    }

    /// Construct from an already-wrapped model (internal builder path).
    pub(crate) fn from_wrapped(llm: BoxedChatModel, prompt_template: impl Into<String>) -> Self {
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
    /// 0.22.0 audit fix (H-C1): single-pass tokenized replacement mirroring
    /// `lc-prompts` semantics — values are never rescanned (no re-replacement
    /// injection), CJK variable names are recognized, and `{{`/`}}` escape to
    /// literal braces. Missing variables are an error (like lc-prompts), and
    /// ALL of them are reported in one message.
    fn render_prompt(&self, inputs: &HashMap<String, Value>) -> Result<String, ChainError> {
        let mut vars = HashMap::with_capacity(inputs.len());
        for (key, value) in inputs {
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            vars.insert(key.clone(), value_str);
        }

        let (prompt, missing) = substitute_template(&self.prompt_template, &vars);

        if !missing.is_empty() {
            return Err(ChainError::ExecutionError(format!(
                "Prompt template has unreplaced variable(s): {}",
                missing.join(", ")
            )));
        }

        Ok(prompt)
    }
}

#[async_trait]
impl BaseChain for LLMChain {
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

        // 0.22.0 audit fix (H-C5): a render_prompt failure used to `?`-return
        // without ending the run or firing on_chain_error, leaking the run in
        // the observability run tree. End the run and dispatch the error
        // callback, matching the other error paths below.
        let prompt = match self.render_prompt(&inputs) {
            Ok(p) => p,
            Err(e) => {
                let msg = e.to_string();
                run.end_with_error(msg.clone());
                if let Some(ref cb) = callbacks {
                    cb.dispatch_chain_error(&run, &msg).await;
                }
                return Err(e);
            }
        };
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
        self.stream_body(inputs, None).await
    }

    /// Stream with config propagation.
    ///
    /// 0.22.0 audit fix (H-C3): the chain's `RunnableConfig` is threaded into
    /// `stream_chat`, so sampling overrides / cancellation token / callbacks
    /// reach the provider (providers consume config via `apply_overrides`)
    /// instead of every streaming call being hardwired to `None`.
    async fn stream_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        let output_key = Some(self.output_key.clone());
        stream_chain_with_callbacks(
            self.name(),
            inputs,
            config.clone(),
            output_key,
            |inputs| async move { self.stream_body(inputs, config).await },
        )
        .await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// LLMChain Builder.
///
/// Convenience builder for LLMChain.
pub struct LLMChainBuilder {
    llm: BoxedChatModel,
    prompt_template: String,
    input_key: Option<String>,
    output_key: Option<String>,
    name: Option<String>,
}

impl LLMChainBuilder {
    /// Create a new [`LLMChainBuilder`] with the given LLM and prompt template.
    pub fn new<L>(llm: L, prompt_template: impl Into<String>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            prompt_template: prompt_template.into(),
            input_key: None,
            output_key: None,
            name: None,
        }
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

    /// Build the final [`LLMChain`].
    pub fn build(self) -> LLMChain {
        let mut chain = LLMChain::from_wrapped(self.llm, self.prompt_template);

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
