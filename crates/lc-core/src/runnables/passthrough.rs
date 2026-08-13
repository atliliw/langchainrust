// lc-core/src/runnables/passthrough.rs
//! RunnablePassthrough - a Runnable that passes input through unchanged.
//!
//! Useful in LCEL pipelines where you need to forward the input
//! without transformation, e.g. in `RunnableParallel` branches
//! or as a no-op step.

use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

/// A `Runnable` that passes its input through unchanged.
///
/// # Example
///
/// ```rust,ignore
/// let passthrough = RunnablePassthrough::<i32>::new();
/// let result = passthrough.invoke(42, None).await?; // 42
/// ```
pub struct RunnablePassthrough<I: Send + Sync + 'static> {
    _marker: std::marker::PhantomData<I>,
}

impl<I: Send + Sync + 'static> std::fmt::Debug for RunnablePassthrough<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnablePassthrough")
            .field("type", &std::any::type_name::<I>())
            .finish()
    }
}

impl<I: Send + Sync + 'static> Default for RunnablePassthrough<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Send + Sync + 'static> RunnablePassthrough<I> {
    /// Create a new passthrough runnable.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<I: Clone + Send + Sync + 'static> Clone for RunnablePassthrough<I> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

#[async_trait]
impl<I: Clone + Send + Sync + 'static> Runnable<I, I> for RunnablePassthrough<I> {
    type Error = LcelError;

    async fn invoke(&self, input: I, _config: Option<RunnableConfig>) -> Result<I, LcelError> {
        Ok(input)
    }

    /// Passthrough supports true streaming: each input item
    /// is forwarded immediately without buffering.
    async fn stream(
        &self,
        input: I,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>, LcelError> {
        Ok(Box::pin(futures_util::stream::once(
            async move { Ok(input) },
        )))
    }

    /// Passthrough supports true transform: each item in the
    /// input stream is forwarded immediately.
    async fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>, LcelError> {
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn invoke_passthrough() {
        let passthrough = RunnablePassthrough::<i32>::new();
        let result = passthrough.invoke(42, None).await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn stream_passthrough() {
        let passthrough = RunnablePassthrough::<String>::new();
        let mut stream = passthrough.stream("hello".to_string(), None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn transform_passthrough() {
        let passthrough = RunnablePassthrough::<i32>::new();
        let input = Box::pin(futures_util::stream::iter(vec![
            Ok(1i32),
            Ok(2i32),
            Ok(3i32),
        ])) as Pin<Box<dyn Stream<Item = Result<i32, LcelError>> + Send>>;

        let mut output = passthrough.transform(input, None).await.unwrap();
        assert_eq!(output.next().await.unwrap().unwrap(), 1);
        assert_eq!(output.next().await.unwrap().unwrap(), 2);
        assert_eq!(output.next().await.unwrap().unwrap(), 3);
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn default_works() {
        let passthrough = RunnablePassthrough::<i32>::default();
        let result = passthrough.invoke(99, None).await.unwrap();
        assert_eq!(result, 99);
    }
}
