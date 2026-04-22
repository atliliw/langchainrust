// src/core/runnables/runnable_trait.rs
//! Runnable trait - foundation of LCEL (LangChain Expression Language).
//!
//! Every LangChain component implements Runnable, enabling
//! chaining, composition, and interoperability.

use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use super::RunnableConfig;

/// Base trait for all LangChain components.
///
/// This trait defines the core interface every component must implement:
/// - Single execution via `invoke`
/// - Batch processing via `batch`
/// - Streaming output via `stream`
///
/// # Example
/// ```rust
/// use langchainrust::core::runnables::Runnable;
/// use langchainrust::RunnableConfig;
/// use async_trait::async_trait;
///
/// // Define a simple Runnable: add one
/// struct AddOne;
///
/// #[async_trait]
/// impl Runnable<i32, i32> for AddOne {
///     type Error = std::convert::Infallible;
///
///     async fn invoke(&self, input: i32, _config: Option<RunnableConfig>) -> Result<i32, Self::Error> {
///         Ok(input + 1)
///     }
/// }
/// ```
#[async_trait]
pub trait Runnable<Input: Send + Sync + 'static, Output: Send + Sync + 'static>: Send + Sync {
    /// Error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Transforms single input to output.
    ///
    /// This is the primary method for single execution.
    ///
    /// # Arguments
    /// * `input` - Input to process.
    /// * `config` - Optional execution configuration.
    ///
    /// # Returns
    /// Execution result.
    async fn invoke(&self, input: Input, config: Option<RunnableConfig>) -> Result<Output, Self::Error>;

    /// Batch processing - transforms multiple inputs to outputs.
    ///
    /// Default implementation processes inputs sequentially.
    /// Override for concurrent execution or batch optimization.
    ///
    /// # Arguments
    /// * `inputs` - Input vector.
    /// * `config` - Optional batch configuration.
    ///
    /// # Returns
    /// Result vector.
    async fn batch(
        &self,
        inputs: Vec<Input>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<Output>, Self::Error> {
        let mut results = Vec::with_capacity(inputs.len());

        // Process each input sequentially
        for input in inputs {
            let result = self.invoke(input, config.clone()).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Streaming output - for real-time responses (LLM, etc).
    ///
    /// Enables real-time stream processing of output,
    /// suitable for chat models, token generation, etc.
    ///
    /// # Arguments
    /// * `input` - Input to process.
    /// * `config` - Optional configuration.
    ///
    /// # Returns
    /// Output stream.
    ///
    /// # Default Implementation
    /// Wraps invoke result as single-element stream.
    /// Types supporting true streaming should override.
    async fn stream(
        &self,
        input: Input,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Output, Self::Error>> + Send>>, Self::Error> {
        // Default: wrap invoke result as single-element stream
        // All Runables automatically get stream capability
        // Types with true streaming (like LLM) should override
        let result = self.invoke(input, config).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    struct TestRunnable;

    #[async_trait]
    impl Runnable<String, String> for TestRunnable {
        type Error = std::convert::Infallible;

        async fn invoke(&self, input: String, _config: Option<RunnableConfig>) -> Result<String, Self::Error> {
            Ok(format!("processed: {}", input))
        }
    }

    #[tokio::test]
    async fn test_default_stream_returns_single_element() {
        let runnable = TestRunnable;
        let mut stream = runnable.stream("test".to_string(), None).await.unwrap();
        
        let first = stream.next().await;
        assert!(first.is_some());
        assert_eq!(first.unwrap().unwrap(), "processed: test");
        
        let second = stream.next().await;
        assert!(second.is_none());
    }

    #[tokio::test]
    async fn test_invoke_matches_stream_result() {
        let runnable = TestRunnable;
        
        let invoke_result = runnable.invoke("hello".to_string(), None).await.unwrap();
        let mut stream = runnable.stream("hello".to_string(), None).await.unwrap();
        let stream_result = stream.next().await.unwrap().unwrap();
        
        assert_eq!(invoke_result, stream_result);
    }
}