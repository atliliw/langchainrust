// lc-core/src/runnables/lambda.rs
//! RunnableLambda - wraps closures as Runnable steps.
//!
//! `RunnableLambda` allows inline closures to participate in LCEL
//! pipelines. It supports both synchronous and asynchronous closures.

use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Type alias for the boxed async closure stored in `RunnableLambda`.
type AsyncFn<I, O> =
    Arc<dyn Fn(I) -> Pin<Box<dyn Future<Output = Result<O, LcelError>> + Send>> + Send + Sync>;

/// A `Runnable` that wraps a closure.
///
/// Created via `RunnableLambda::new_sync` or `RunnableLambda::new_async`.
///
/// # Example
///
/// ```rust,ignore
/// let doubler = RunnableLambda::new_sync(|x: i32| x * 2);
/// let result = doubler.invoke(5, None).await?; // 10
/// ```
pub struct RunnableLambda<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    func: AsyncFn<I, O>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> std::fmt::Debug for RunnableLambda<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnableLambda")
            .field("input", &std::any::type_name::<I>())
            .field("output", &std::any::type_name::<O>())
            .finish()
    }
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> RunnableLambda<I, O> {
    /// Create from a synchronous (blocking) closure.
    ///
    /// The closure's output is automatically wrapped in `Ok(...)`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let upper = RunnableLambda::new_sync(|s: String| s.to_uppercase());
    /// ```
    pub fn new_sync<F>(func: F) -> Self
    where
        F: Fn(I) -> O + Send + Sync + 'static,
    {
        let func = Arc::new(move |input: I| {
            let result = func(input);
            Box::pin(async move { Ok(result) })
                as Pin<Box<dyn Future<Output = Result<O, LcelError>> + Send>>
        });
        Self { func }
    }

    /// Create from a synchronous closure that can fail.
    ///
    /// The closure returns `Result<O, LcelError>`.
    pub fn new_sync_fallible<F>(func: F) -> Self
    where
        F: Fn(I) -> Result<O, LcelError> + Send + Sync + 'static,
    {
        let func = Arc::new(move |input: I| {
            let result = func(input);
            Box::pin(async move { result })
                as Pin<Box<dyn Future<Output = Result<O, LcelError>> + Send>>
        });
        Self { func }
    }

    /// Create from an async closure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let fetch = RunnableLambda::new_async(|url: String| async move {
    ///     reqwest::get(&url).await?.text().await.map_err(|e| LcelError::Other(e.to_string()))
    /// });
    /// ```
    pub fn new_async<F, Fut>(func: F) -> Self
    where
        F: Fn(I) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, LcelError>> + Send + 'static,
    {
        let func = Arc::new(move |input: I| {
            let fut = func(input);
            Box::pin(fut) as Pin<Box<dyn Future<Output = Result<O, LcelError>> + Send>>
        });
        Self { func }
    }
}

#[async_trait]
impl<I: Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O> for RunnableLambda<I, O> {
    type Error = LcelError;

    async fn invoke(&self, input: I, _config: Option<RunnableConfig>) -> Result<O, LcelError> {
        (self.func)(input).await
    }

    // batch, stream, transform all use default implementations
    // (sequential invoke, single-element stream, buffer-and-invoke)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn sync_closure_works() {
        let lambda = RunnableLambda::new_sync(|x: i32| x * 3);
        let result = lambda.invoke(7, None).await.unwrap();
        assert_eq!(result, 21);
    }

    #[tokio::test]
    async fn sync_fallible_closure_ok() {
        let lambda = RunnableLambda::new_sync_fallible(|x: i32| {
            if x > 0 {
                Ok(x * 2)
            } else {
                Err(LcelError::Other("must be positive".to_string()))
            }
        });
        assert_eq!(lambda.invoke(5, None).await.unwrap(), 10);
    }

    #[tokio::test]
    async fn sync_fallible_closure_err() {
        let lambda = RunnableLambda::new_sync_fallible(|x: i32| {
            if x > 0 {
                Ok(x * 2)
            } else {
                Err(LcelError::Other("must be positive".to_string()))
            }
        });
        let result = lambda.invoke(-1, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_closure_works() {
        let lambda = RunnableLambda::new_async(|x: i32| async move {
            tokio::task::spawn_blocking(move || x + 100)
                .await
                .map_err(|e| LcelError::Other(e.to_string()))
        });
        let result = lambda.invoke(5, None).await.unwrap();
        assert_eq!(result, 105);
    }

    #[tokio::test]
    async fn stream_uses_default() {
        let lambda = RunnableLambda::new_sync(|x: i32| x + 1);
        let mut stream = lambda.stream(9, None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, 10);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn batch_uses_default() {
        let lambda = RunnableLambda::new_sync(|x: i32| x * 10);
        let results = lambda.batch(vec![1, 2, 3], None).await.unwrap();
        assert_eq!(results, vec![10, 20, 30]);
    }
}
