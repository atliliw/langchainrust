// lc-core/src/runnables/ext.rs
//! Extension trait for Runnable providing LCEL composition methods.
//!
//! `RunnableExt` adds the `pipe()` method to any `Runnable` whose `Error`
//! type can be converted into `LcelError`. This enables the core LCEL
//! composition pattern:
//!
//! ```rust,ignore
//! let chain = prompt.pipe(llm).pipe(parser);
//! ```

use super::any::into_runnable_any;
use super::configurable::{RunnableConfigurable, RunnableConfigurableFields};
use super::error::LcelError;
use super::fallback::RunnableWithFallbacks;
use super::retry::{RetryConfig, RunnableRetry};
use super::runnable_trait::Runnable;
use super::sequence::RunnableSequence;

/// Extension trait that provides LCEL composition methods for `Runnable`.
///
/// Automatically implemented for all `Runnable<I, O>` types where
/// `Self::Error: Into<LcelError>`.
pub trait RunnableExt<Input: Send + Sync + 'static, Output: Send + Sync + 'static>:
    Runnable<Input, Output>
where
    Self::Error: Into<LcelError>,
    Self: Sized + 'static,
{
    /// Pipe the output of this runnable into another runnable.
    ///
    /// This is the core LCEL composition operator. It creates a
    /// `RunnableSequence` that executes `self` first, then passes
    /// the output to `other`.
    ///
    /// # Type Safety
    ///
    /// The compiler ensures that the output type of `self` matches
    /// the input type of `other`. At runtime, intermediate types
    /// are erased via `RunnableAny`, but the type relationship
    /// was already proven at compile time.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let chain = prompt.pipe(llm).pipe(parser);
    /// let result = chain.invoke("What is Rust?".to_string(), None).await?;
    /// ```
    fn pipe<O2, R2>(self, other: R2) -> RunnableSequence<Input, O2>
    where
        O2: Send + Sync + 'static,
        R2: Runnable<Output, O2> + Send + Sync + 'static,
        R2::Error: Into<LcelError>,
    {
        RunnableSequence::from_pair(self, other)
    }

    /// Create a `RunnableSequence` from this runnable as a single step.
    ///
    /// Useful when you want to start building a pipeline and add
    /// steps later via `pipe()`.
    fn into_sequence(self) -> RunnableSequence<Input, Output> {
        RunnableSequence::from_single(self)
    }

    /// Add fallback runnables that are tried if this one fails.
    ///
    /// If `self` returns an error, each fallback is tried in order.
    /// The first successful result is returned. If all fail, the
    /// primary's error is returned.
    ///
    /// The input type `Input` must be `Clone` so that the input can
    /// be re-boxed for each fallback attempt.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let chain = prompt
    ///     .pipe(openai_llm)
    ///     .with_fallbacks(vec![anthropic_llm, ollama_llm])
    ///     .pipe(parser);
    /// ```
    fn with_fallbacks<R>(self, fallbacks: Vec<R>) -> RunnableWithFallbacks<Input, Output>
    where
        Input: Clone,
        R: Runnable<Input, Output> + Send + Sync + 'static,
        R::Error: Into<LcelError>,
    {
        let fallback_boxes: Vec<Box<dyn super::any::RunnableAny>> = fallbacks
            .into_iter()
            .map(|r| into_runnable_any(r))
            .collect();
        RunnableWithFallbacks::new(self, fallback_boxes)
    }

    /// Wrap this runnable with retry logic using exponential backoff.
    ///
    /// On failure, the runnable is retried up to `max_retries` times
    /// with increasing delays between attempts.
    ///
    /// The input type `Input` must be `Clone` so that the input can
    /// be re-boxed for each retry attempt.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use lc_core::runnables::{RetryConfig, RunnableExt};
    /// use std::time::Duration;
    ///
    /// let chain = prompt.pipe(llm).pipe(parser)
    ///     .with_retry(RetryConfig {
    ///         max_retries: 3,
    ///         initial_delay: Duration::from_millis(500),
    ///         max_delay: Duration::from_secs(10),
    ///         backoff_multiplier: 2.0,
    ///         ..Default::default()
    ///     });
    /// ```
    fn with_retry(self, retry_config: RetryConfig) -> RunnableRetry<Input, Output>
    where
        Input: Clone,
    {
        let runnable_any = into_runnable_any(self);
        RunnableRetry::new(runnable_any, retry_config)
    }

    /// Route between a default runnable and named alternatives at invoke time.
    ///
    /// Rust counterpart of Python LCEL's `Runnable.configurable_alternatives`.
    /// The selector key `which` is read from `config.configurable`; the value
    /// must name the `default_key` (→ this runnable) or one of the
    /// `alternatives`. An unknown value falls back to this runnable.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let chain = llm.configurable_alternatives(
    ///     "provider", "default",
    ///     vec![("anthropic", anthropic_llm)],
    /// );
    /// // invoke(msgs, Some(RunnableConfig::new().with_configurable("provider", json!("anthropic"))))
    /// ```
    fn configurable_alternatives<K, R>(
        self,
        which: impl Into<String>,
        default_key: impl Into<String>,
        alternatives: Vec<(K, R)>,
    ) -> RunnableConfigurable<Input, Output>
    where
        K: Into<String>,
        R: Runnable<Input, Output> + Send + Sync + 'static,
        R::Error: Into<LcelError>,
    {
        let mut configurable = RunnableConfigurable::new(self, which, default_key);
        for (name, runnable) in alternatives {
            configurable = configurable.with_alternative(name, runnable);
        }
        configurable
    }

    /// Override recognized config fields at invoke time from
    /// `config.configurable` (Python's `Runnable.configurable_fields`).
    ///
    /// `temperature` / `max_tokens` in the configurable map are promoted to
    /// the typed config fields the providers consume; other keys are merged
    /// into `config.metadata`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let llm = llm.configurable_fields();
    /// let config = RunnableConfig::new().with_configurable("temperature", json!(0.5));
    /// llm.invoke(msgs, Some(config)).await?;   // 本次调用采样温度 0.5
    /// ```
    fn configurable_fields(self) -> RunnableConfigurableFields<Input, Output> {
        RunnableConfigurableFields::new(self)
    }
}

// Blanket implementation: all Runnables with compatible Error get RunnableExt
impl<I, O, R> RunnableExt<I, O> for R
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    R: Runnable<I, O>,
    R::Error: Into<LcelError>,
    R: Sized + 'static,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunnableConfig;
    use async_trait::async_trait;
    use futures_util::StreamExt;

    struct Double;

    #[async_trait]
    impl Runnable<i32, i32> for Double {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<i32, Self::Error> {
            Ok(input * 2)
        }
    }

    struct AddSuffix;

    #[async_trait]
    impl Runnable<i32, String> for AddSuffix {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<String, Self::Error> {
            Ok(format!("result: {}", input))
        }
    }

    #[tokio::test]
    async fn pipe_creates_sequence() {
        let chain = Double.pipe(AddSuffix);
        let result = chain.invoke(5, None).await.unwrap();
        assert_eq!(result, "result: 10");
    }

    #[tokio::test]
    async fn pipe_chain_multiple() {
        // Double → Double → AddSuffix
        let chain = Double.pipe(Double).pipe(AddSuffix);
        let result = chain.invoke(3, None).await.unwrap();
        assert_eq!(result, "result: 12"); // 3 * 2 * 2 = 12
    }

    #[tokio::test]
    async fn pipe_stream_works() {
        let chain = Double.pipe(AddSuffix);
        let mut stream = chain.stream(5, None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, "result: 10");
    }
}
