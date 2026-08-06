// lc-chains/src/base.rs
//! Chain base trait.

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::runnables::RunnableConfig;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;

/// Chain error type.
#[derive(Debug, thiserror::Error)]
pub enum ChainError {
    /// Missing input.
    #[error("Missing input: {0}")]
    MissingInput(String),

    /// Output error.
    #[error("Output error: {0}")]
    OutputError(String),

    /// Execution error.
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Stream error.
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Other error.
    #[error("Chain error: {0}")]
    Other(String),
}

/// Chain execution result.
pub type ChainResult = HashMap<String, Value>;

/// Stream output item: token-by-token output.
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// Token text.
    pub token: String,
    /// Whether this is the final token.
    pub is_final: bool,
}

/// Chain stream output type.
pub type ChainStream = Pin<Box<dyn Stream<Item = Result<StreamToken, ChainError>> + Send>>;

/// Base Chain trait.
///
/// Chain is LangChain's core abstraction, representing a sequence of operations.
#[async_trait]
pub trait BaseChain: Send + Sync {
    /// Get input keys.
    fn input_keys(&self) -> Vec<&str>;

    /// Get output keys.
    fn output_keys(&self) -> Vec<&str>;

    /// Execute the Chain.
    ///
    /// # Arguments
    /// * `inputs` - Input parameter dictionary
    ///
    /// # Returns
    /// Output result dictionary
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError>;

    /// Execute the Chain with a RunnableConfig.
    ///
    /// This method propagates callbacks through the chain execution.
    /// The default implementation delegates to `invoke()` without config,
    /// so chains that don't need callback support work automatically.
    ///
    /// Chains that want to propagate callbacks (on_chain_start/end, on_llm_start/end)
    /// should override this method.
    async fn invoke_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        // Default: fire on_chain_start/end if callbacks are present, then delegate to invoke
        let _ = config; // suppress unused warning
        self.invoke(inputs).await
    }

    /// Stream execute the Chain -- token by token output.
    ///
    /// Default implementation wraps the invoke result as a single-element stream.
    /// Chains that support LLM streaming (LLMChain / ConversationChain) should
    /// override this method, calling `BaseChatModel::stream_chat` internally.
    ///
    /// # Arguments
    /// * `inputs` - Input parameter dictionary
    ///
    /// # Returns
    /// Token stream
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        // Default: wrap invoke result as single-element stream
        let result = self.invoke(inputs).await?;
        let output_text = result
            .values()
            .next()
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let stream = futures_util::stream::once(async move {
            Ok(StreamToken {
                token: output_text,
                is_final: true,
            })
        });
        Ok(Box::pin(stream))
    }

    /// Stream execute the Chain with a RunnableConfig.
    ///
    /// The default implementation delegates to `stream()` without config.
    async fn stream_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        let _ = config;
        self.stream(inputs).await
    }

    /// Validate inputs.
    fn validate_inputs(&self, inputs: &HashMap<String, Value>) -> Result<(), ChainError> {
        for key in self.input_keys() {
            if !inputs.contains_key(key) {
                return Err(ChainError::MissingInput(key.to_string()));
            }
        }
        Ok(())
    }

    /// Get Chain name.
    fn name(&self) -> &str {
        "chain"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_error_display() {
        let error = ChainError::MissingInput("test".to_string());
        assert!(error.to_string().contains("Missing input"));

        let error = ChainError::ExecutionError("test".to_string());
        assert!(error.to_string().contains("Execution error"));
    }

    #[test]
    fn test_chain_error_all_variants() {
        let err = ChainError::MissingInput("key".to_string());
        assert!(err.to_string().contains("key"));

        let err = ChainError::OutputError("bad".to_string());
        assert!(err.to_string().contains("bad"));

        let err = ChainError::ExecutionError("fail".to_string());
        assert!(err.to_string().contains("fail"));

        let err = ChainError::StreamError("broken".to_string());
        assert!(err.to_string().contains("broken"));

        let err = ChainError::Other("misc".to_string());
        assert!(err.to_string().contains("misc"));
    }

    #[test]
    fn test_stream_token_debug() {
        let token = StreamToken {
            token: "hello".to_string(),
            is_final: false,
        };
        assert!(format!("{:?}", token).contains("hello"));
    }

    #[test]
    fn test_validate_inputs_pass() {
        struct PassthroughChain;
        #[async_trait]
        impl BaseChain for PassthroughChain {
            fn input_keys(&self) -> Vec<&str> { vec!["input"] }
            fn output_keys(&self) -> Vec<&str> { vec!["output"] }
            async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
                Ok(inputs)
            }
        }

        let chain = PassthroughChain;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("test".to_string()));
        assert!(chain.validate_inputs(&inputs).is_ok());
    }

    #[test]
    fn test_validate_inputs_missing_key() {
        struct PassthroughChain;
        #[async_trait]
        impl BaseChain for PassthroughChain {
            fn input_keys(&self) -> Vec<&str> { vec!["input"] }
            fn output_keys(&self) -> Vec<&str> { vec!["output"] }
            async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
                Ok(HashMap::new())
            }
        }

        let chain = PassthroughChain;
        let inputs = HashMap::new();
        assert!(chain.validate_inputs(&inputs).is_err());
    }

    #[test]
    fn test_default_chain_name() {
        struct MyChain;
        #[async_trait]
        impl BaseChain for MyChain {
            fn input_keys(&self) -> Vec<&str> { vec![] }
            fn output_keys(&self) -> Vec<&str> { vec![] }
            async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
                Ok(HashMap::new())
            }
        }
        let chain = MyChain;
        assert_eq!(chain.name(), "chain");
    }
}
