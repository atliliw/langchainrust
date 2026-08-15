// src/core/runnables/runnable_trait.rs
//! Runnable trait - foundation of LCEL (LangChain Expression Language).
//!
//! Every LangChain component implements Runnable, enabling
//! chaining, composition, and interoperability.

use super::RunnableConfig;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

/// Base trait for all LangChain components.
///
/// This trait defines the core interface every component must implement:
/// - Single execution via `invoke`
/// - Batch processing via `batch`
/// - Streaming output via `stream`
/// - Stream-to-stream transformation via `transform`
///
/// # Example
/// ```no_run
/// use lc_core::runnables::Runnable;
/// use lc_core::runnables::RunnableConfig;
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
pub trait Runnable<Input: Send + Sync + 'static, Output: Send + Sync + 'static>:
    Send + Sync
{
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
    async fn invoke(
        &self,
        input: Input,
        config: Option<RunnableConfig>,
    ) -> Result<Output, Self::Error>;

    /// Batch processing - transforms multiple inputs to outputs.
    ///
    /// Default implementation processes inputs concurrently with a bounded
    /// concurrency: `config.max_concurrency` items run at once (defaults to
    /// all inputs), and results are returned in input order regardless of
    /// completion order (`buffered`, not `buffer_unordered`). Override for
    /// provider-level batch optimization.
    ///
    /// # Arguments
    /// * `inputs` - Input vector.
    /// * `config` - Optional batch configuration.
    ///
    /// # Returns
    /// Result vector, ordered as the inputs.
    async fn batch(
        &self,
        inputs: Vec<Input>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<Output>, Self::Error> {
        use futures_util::StreamExt;

        // Concurrency cap: `max_concurrency`, clamped to at least 1 so an
        // explicit `Some(0)` (or an empty input list) cannot panic `buffered`.
        let limit = config
            .as_ref()
            .and_then(|c| c.max_concurrency)
            .unwrap_or(inputs.len())
            .max(1);

        let results = futures_util::stream::iter(inputs)
            .map(|input| {
                let config = config.clone();
                async move { self.invoke(input, config).await }
            })
            .buffered(limit)
            .collect::<Vec<Result<_, _>>>()
            .await;

        // Short-circuit on the first error, preserving input order otherwise.
        results.into_iter().collect()
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

    /// Stream-to-stream transformation - the core of LCEL streaming.
    ///
    /// Takes an input stream and produces an output stream, enabling
    /// pipeline streaming without buffering intermediate results.
    ///
    /// # Default Implementation
    /// Drives each input item through `stream` and concatenates the per-item
    /// streams in order — the LangChain default `transform` semantics. A step
    /// that overrides `stream` (e.g. an LLM) yields a real token stream per
    /// item; a step using the default `stream` maps elementwise via `invoke`.
    /// Components that want aggregation (e.g. incremental parsers) should
    /// override this method.
    ///
    /// # Arguments
    /// * `input` - Input stream to transform.
    /// * `config` - Optional execution configuration.
    ///
    /// # Returns
    /// Output stream.
    async fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<Input, Self::Error>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Output, Self::Error>> + Send>>, Self::Error> {
        use futures_util::StreamExt;

        // Buffer the input items, then run each one through `stream` and
        // concatenate the per-item streams (elementwise semantics).
        let mut items = Vec::new();
        let mut input = input;
        while let Some(item) = input.next().await {
            items.push(item?);
        }

        let mut per_item_streams = Vec::with_capacity(items.len());
        for item in items {
            let stream = self.stream(item, config.clone()).await?;
            per_item_streams.push(stream);
        }

        let flattened = futures_util::stream::iter(per_item_streams).flatten();
        Ok(Box::pin(flattened))
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

        async fn invoke(
            &self,
            input: String,
            _config: Option<RunnableConfig>,
        ) -> Result<String, Self::Error> {
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

    #[tokio::test]
    async fn test_default_transform_maps_elementwise() {
        let runnable = TestRunnable;
        let input_stream = Box::pin(futures_util::stream::iter(vec![
            Ok("first".to_string()),
            Ok("second".to_string()),
            Ok("third".to_string()),
        ]))
            as Pin<Box<dyn Stream<Item = Result<String, std::convert::Infallible>> + Send>>;

        let mut output_stream = runnable.transform(input_stream, None).await.unwrap();

        // Default transform maps each item through invoke (elementwise).
        let mut results = Vec::new();
        while let Some(item) = output_stream.next().await {
            results.push(item.unwrap());
        }
        assert_eq!(
            results,
            vec![
                "processed: first".to_string(),
                "processed: second".to_string(),
                "processed: third".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn test_default_transform_empty_input() {
        let runnable = TestRunnable;
        let input_stream = Box::pin(futures_util::stream::empty::<
            Result<String, std::convert::Infallible>,
        >())
            as Pin<Box<dyn Stream<Item = Result<String, std::convert::Infallible>> + Send>>;

        let mut output_stream = runnable.transform(input_stream, None).await.unwrap();

        // Empty input → empty output
        assert!(output_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_default_batch_preserves_order() {
        let runnable = TestRunnable;
        let results = runnable
            .batch(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            results,
            vec![
                "processed: a".to_string(),
                "processed: b".to_string(),
                "processed: c".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn test_default_batch_respects_max_concurrency() {
        let runnable = TestRunnable;
        let config = RunnableConfig::new().with_max_concurrency(1);
        let results = runnable
            .batch(
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
                Some(config),
            )
            .await
            .unwrap();
        assert_eq!(
            results,
            vec![
                "processed: x".to_string(),
                "processed: y".to_string(),
                "processed: z".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn test_default_batch_empty_input() {
        let runnable = TestRunnable;
        let results = runnable.batch(vec![], None).await.unwrap();
        assert!(results.is_empty());
    }
}
