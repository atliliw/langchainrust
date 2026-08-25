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

    /// Batch processing that returns results in *completion* order.
    ///
    /// Rust counterpart of Python LCEL's `batch_as_completed`: each input is
    /// driven through the full chain via `invoke` independently, with
    /// concurrency bounded by `config.max_concurrency` (defaults to all
    /// inputs). The result is a `Vec<(usize, Output)>` ordered by *completion*
    /// time, where the `usize` is the original index in `inputs`.
    ///
    /// Short-circuits on the first error (like `batch`): if any input fails,
    /// the error is returned immediately and the remaining results are
    /// dropped.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = chain.batch_as_completed(inputs, None).await?;
    /// // 最快完成的那项在 results[0],其下标标识它在 inputs 里的位置
    /// for (index, output) in results {
    ///     println!("inputs[{index}] -> {output}");
    /// }
    /// ```
    async fn batch_as_completed(
        &self,
        inputs: Vec<Input>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<(usize, Output)>, Self::Error> {
        use futures_util::StreamExt;

        let limit = config
            .as_ref()
            .and_then(|c| c.max_concurrency)
            .unwrap_or(inputs.len())
            .max(1);

        let results = futures_util::stream::iter(inputs.into_iter().enumerate())
            .map(|(index, input)| {
                let config = config.clone();
                async move {
                    self.invoke(input, config)
                        .await
                        .map(|output| (index, output))
                }
            })
            .buffer_unordered(limit)
            .collect::<Vec<Result<(usize, Output), _>>>()
            .await;

        // Short-circuit on the first error (in completion order).
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
    /// Drives each input item through `stream` **lazily**: as soon as an input
    /// item arrives it is immediately run through `stream` and its output
    /// yielded, before pulling the next input item. This is the LangChain
    /// default `transform` semantics — downstream receives output incrementally
    /// instead of waiting for the entire input stream to finish, and an
    /// infinite/long-lived upstream never accumulates unboundedly in memory.
    /// A step that overrides `stream` (e.g. an LLM) yields a real token stream
    /// per item; a step using the default `stream` maps elementwise via
    /// `invoke`. Components that want aggregation (e.g. incremental parsers)
    /// should override this method.
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Output, Self::Error>> + Send + '_>>, Self::Error>
    {
        // Note: the `+ '_` on the returned `dyn Stream` allows the default lazy
        // implementation to borrow `&self` across the stream's lifetime. Existing
        // implementations that return `'static` streams remain valid (they simply
        // outlive the required bound).
        use futures_util::StreamExt;

        let config = config.clone();
        let stream = async_stream::stream! {
            let mut input = input;
            loop {
                let item = match input.next().await {
                    Some(Ok(item)) => item,
                    Some(Err(e)) => {
                        yield Err(e);
                        return;
                    }
                    None => return,
                };
                // Run the current item through `stream` and drain it to the
                // output before pulling the next input item (lazy elementwise).
                let inner = match self.stream(item, config.clone()).await {
                    Ok(s) => s,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                futures_util::pin_mut!(inner);
                while let Some(res) = inner.next().await {
                    yield res;
                }
            }
        };
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

    /// 带延迟的 Runnable:"slow" 明显慢于其他,用于验证完成顺序 ≠ 输入顺序。
    struct Delayed;

    #[async_trait]
    impl Runnable<&'static str, usize> for Delayed {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: &'static str,
            _config: Option<RunnableConfig>,
        ) -> Result<usize, Self::Error> {
            match input {
                "slow" => {
                    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                    Ok(10)
                }
                _ => {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    Ok(1)
                }
            }
        }
    }

    #[tokio::test]
    async fn test_batch_as_completed_returns_completion_order() {
        let results = Delayed
            .batch_as_completed(vec!["slow", "fast"], None)
            .await
            .unwrap();
        // 快的那项先完成 → results[0] 的下标应是 1("fast")
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1, "完成最快的应带原始下标 1");
        assert_eq!(results[0].1, 1);
        assert_eq!(results[1].0, 0);
        assert_eq!(results[1].1, 10);
    }

    #[tokio::test]
    async fn test_batch_as_completed_respects_max_concurrency() {
        let config = RunnableConfig::new().with_max_concurrency(1);
        let results = Delayed
            .batch_as_completed(vec!["fast", "slow"], Some(config))
            .await
            .unwrap();
        // 并发上限 1 → 串行,完成顺序 = 输入顺序
        assert_eq!(results, vec![(0, 1), (1, 10)]);
    }

    #[tokio::test]
    async fn test_batch_as_completed_empty_input() {
        let runnable = TestRunnable;
        let results = runnable.batch_as_completed(vec![], None).await.unwrap();
        assert!(results.is_empty());
    }

    /// 默认 `transform` 必须惰性:下游收到第一条输出时,上游流尚未产完。
    /// 若仍按旧的"攒齐整条流"实现,本测试会在 `assert!(!produced_last..)` 处失败。
    #[tokio::test]
    async fn default_transform_is_lazy_incremental() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let produced_last = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&produced_last);

        // 上游:产出 3 条,产完最后一条才置位 flag。
        let src = async_stream::stream! {
            yield Ok("first".to_string());
            yield Ok("second".to_string());
            yield Ok("third".to_string());
            flag.store(true, Ordering::SeqCst);
        };
        let input_stream = Box::pin(src)
            as Pin<Box<dyn Stream<Item = Result<String, std::convert::Infallible>> + Send>>;

        let runnable = TestRunnable;
        let mut output_stream = runnable.transform(input_stream, None).await.unwrap();

        // 首条输出到达时,上游尚未产完(惰性逐条拼接,而非攒齐整条流)。
        let first = output_stream.next().await.unwrap().unwrap();
        assert_eq!(first, "processed: first");
        assert!(
            !produced_last.load(Ordering::SeqCst),
            "transform 不应在上游流结束前就攒齐整条输入"
        );

        // 消费剩余全部,此时上游 flag 应已置位(输出序列完整、无丢失)。
        while output_stream.next().await.is_some() {}
        assert!(produced_last.load(Ordering::SeqCst));
    }
}
