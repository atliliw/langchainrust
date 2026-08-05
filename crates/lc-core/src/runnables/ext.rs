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
use super::error::LcelError;
use super::fallback::RunnableWithFallbacks;
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

        async fn invoke(&self, input: i32, _config: Option<RunnableConfig>) -> Result<i32, Self::Error> {
            Ok(input * 2)
        }
    }

    struct AddSuffix;

    #[async_trait]
    impl Runnable<i32, String> for AddSuffix {
        type Error = std::convert::Infallible;

        async fn invoke(&self, input: i32, _config: Option<RunnableConfig>) -> Result<String, Self::Error> {
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
