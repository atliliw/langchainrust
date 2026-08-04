// lc-chains/src/base.rs
//! Chain base trait.

use async_trait::async_trait;
use futures_util::Stream;
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
}
