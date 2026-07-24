// src/chains/llm_chain.rs
//! LLM Chain
//!
//! The most basic Chain, combining a Prompt and an LLM.

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::LazyLock;

use super::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};
use crate::schema::Message;
use crate::BaseChatModel;
use crate::Runnable;

/// LLM Chain
///
/// Combines a Prompt template and an LLM. The most basic Chain.
///
/// # Examples
/// ```ignore
/// use langchainrust::{LLMChain, OpenAIChat, OpenAIConfig};
///
/// let llm = OpenAIChat::new(config);
/// let chain = LLMChain::new(llm, "{question}");
///
/// let inputs = HashMap::from([("question".to_string(), "What is Rust?".into())]);
/// let result = chain.invoke(inputs).await?;
/// ```
pub struct LLMChain<M: BaseChatModel> {
    /// LLM client
    llm: M,

    /// Prompt template
    prompt_template: String,

    /// Input key name
    input_key: String,

    /// Output key name
    output_key: String,

    /// Chain name
    name: String,
}

/// Pre-compiled regex for detecting unreplaced template variables
static TEMPLATE_VAR_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}").unwrap());

impl<M: BaseChatModel> LLMChain<M> {
    /// Create a new LLMChain
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

    /// Set chain name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Render the Prompt template
    ///
    /// M78: Validates that all {variable} placeholders in the template
    /// have been replaced. Returns an error if any unreplaced placeholders remain.
    fn render_prompt(&self, inputs: &HashMap<String, Value>) -> Result<String, ChainError> {
        let mut prompt = self.prompt_template.clone();

        for (key, value) in inputs {
            // Replace {key} placeholders
            let placeholder = format!("{{{}}}", key);
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            prompt = prompt.replace(&placeholder, &value_str);
        }

        // M78: Check for unreplaced {variable} placeholders
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
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error:
        std::fmt::Display,
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

        // Render prompt
        let prompt = self.render_prompt(&inputs)?;

        // Call LLM
        let messages = vec![Message::human(&prompt)];
        let result = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        // Build output
        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), Value::String(result.content));

        Ok(output)
    }

    /// Stream execution for LLMChain -- token by token output
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        // Validate inputs
        self.validate_inputs(&inputs)?;

        // Render prompt
        let prompt = self.render_prompt(&inputs)?;

        // Call LLM stream_chat
        let messages = vec![Message::human(&prompt)];
        let llm_stream = self
            .llm
            .stream_chat(messages, None)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        // Map LLM stream to ChainStream, then append a final is_final=true token
        // (H69: stream must emit is_final=true to signal completion)
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

/// LLMChain Builder
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::OpenAIChat;
    use crate::OpenAIConfig;

    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string(),
            base_url:
                "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                    .to_string(),
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
    fn test_render_prompt() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, "Question: {question}");

        let inputs = HashMap::from([(
            "question".to_string(),
            Value::String("What is Rust?".to_string()),
        )]);

        let prompt = chain.render_prompt(&inputs).unwrap();
        assert_eq!(prompt, "Question: What is Rust?");
    }

    #[test]
    fn test_render_prompt_multiple_vars() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, "Name: {name}, Age: {age}");

        let inputs = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Number(25.into())),
        ]);

        let prompt = chain.render_prompt(&inputs).unwrap();
        assert_eq!(prompt, "Name: Alice, Age: 25");
    }

    /// Real API test - simple question
    /// Run: cargo test test_llm_chain_simple -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_llm_chain_simple() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, "Answer in one sentence: {question}");

        let inputs = HashMap::from([(
            "question".to_string(),
            Value::String("What is Rust?".to_string()),
        )]);

        println!("\n=== Test LLMChain - simple question ===");
        let result = chain.invoke(inputs).await.unwrap();

        println!("Output: {:?}", result);
        assert!(result.contains_key("text"));
        assert!(!result.get("text").unwrap().as_str().unwrap().is_empty());
    }

    /// Real API test - multi-variable template
    /// Run: cargo test test_llm_chain_template -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_llm_chain_template() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, "Answer in {style} style: {question}");

        let inputs = HashMap::from([
            ("style".to_string(), Value::String("humorous".to_string())),
            (
                "question".to_string(),
                Value::String("What is programming?".to_string()),
            ),
        ]);

        println!("\n=== Test LLMChain - multi-variable template ===");
        let result = chain.invoke(inputs).await.unwrap();

        println!("Output: {:?}", result);
        assert!(result.contains_key("text"));
    }

    /// Real API test - using Builder
    /// Run: cargo test test_llm_chain_builder -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_llm_chain_builder() {
        let llm = OpenAIChat::new(create_test_config());

        let chain = LLMChainBuilder::new(llm, "Translate to {language}: {text}")
            .input_key("text")
            .output_key("translation")
            .name("translator")
            .build();

        let inputs = HashMap::from([
            ("language".to_string(), Value::String("English".to_string())),
            (
                "text".to_string(),
                Value::String("Hello, world".to_string()),
            ),
        ]);

        println!("\n=== Test LLMChain - Builder ===");
        let result = chain.invoke(inputs).await.unwrap();

        println!("Output: {:?}", result);
        assert!(result.contains_key("translation"));
    }
}
