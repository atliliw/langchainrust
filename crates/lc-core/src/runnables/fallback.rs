// lc-core/src/runnables/fallback.rs
//! RunnableWithFallbacks - fallback composition for LCEL pipelines.
//!
//! `RunnableWithFallbacks` wraps a primary `Runnable` with a list of fallback
//! runnables. If the primary fails, each fallback is tried in order until one
//! succeeds. If all fail, the primary's error is returned.
//!
//! # Example
//!
//! ```rust,ignore
//! let chain = prompt
//!     .pipe(openai_llm)
//!     .with_fallbacks(vec![anthropic_llm, ollama_llm])
//!     .pipe(parser);
//! ```

use super::any::{into_runnable_any, RunnableAny};
use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use std::any::Any;
use std::marker::PhantomData;
use std::pin::Pin;

/// A `Runnable` that tries fallbacks when the primary fails.
///
/// If the primary runnable returns an error on `invoke`, each fallback is
/// tried in order. The first successful result is returned. If all fail,
/// the **primary's** error is returned (so the user sees the error from
/// the runnable they explicitly chose).
///
/// The input type `I` must be `Clone` so that the input can be re-boxed
/// for each fallback attempt.
pub struct RunnableWithFallbacks<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    primary: Box<dyn RunnableAny>,
    fallbacks: Vec<Box<dyn RunnableAny>>,
    _marker: PhantomData<(I, O)>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> std::fmt::Debug
    for RunnableWithFallbacks<I, O>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnableWithFallbacks")
            .field("fallbacks", &self.fallbacks.len())
            .field("input", &std::any::type_name::<I>())
            .field("output", &std::any::type_name::<O>())
            .finish()
    }
}

impl<I: Clone + Send + Sync + 'static, O: Send + Sync + 'static> RunnableWithFallbacks<I, O> {
    /// Create a new fallback runnable with the given primary and fallbacks.
    pub fn new<R>(primary: R, fallbacks: Vec<Box<dyn RunnableAny>>) -> Self
    where
        R: Runnable<I, O> + 'static,
        R::Error: Into<LcelError>,
    {
        Self {
            primary: into_runnable_any(primary),
            fallbacks,
            _marker: PhantomData,
        }
    }

    /// Number of fallback runnables.
    pub fn fallback_count(&self) -> usize {
        self.fallbacks.len()
    }

    /// Try all runnables (primary then fallbacks) on the given input.
    /// Returns the first successful result, or the primary's error if all fail.
    async fn try_all(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<O, LcelError> {
        // Try primary
        let boxed_input = Box::new(input.clone()) as Box<dyn Any + Send>;
        match self.primary.invoke_any(boxed_input, config.clone()).await {
            Ok(result) => {
                return result
                    .downcast::<O>()
                    .map(|b| *b)
                    .map_err(|_| LcelError::TypeMismatch(format!(
                        "fallback primary output downcast: expected {}",
                        std::any::type_name::<O>()
                    )));
            }
            Err(primary_error) => {
                // Try each fallback
                for fallback in &self.fallbacks {
                    let boxed_input = Box::new(input.clone()) as Box<dyn Any + Send>;
                    match fallback.invoke_any(boxed_input, config.clone()).await {
                        Ok(result) => {
                            return result
                                .downcast::<O>()
                                .map(|b| *b)
                                .map_err(|_| LcelError::TypeMismatch(format!(
                                    "fallback output downcast: expected {}",
                                    std::any::type_name::<O>()
                                )));
                        }
                        Err(_) => continue,
                    }
                }
                // All failed, return the primary's error
                Err(primary_error)
            }
        }
    }
}

#[async_trait]
impl<I: Clone + Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O>
    for RunnableWithFallbacks<I, O>
{
    type Error = LcelError;

    /// Try the primary, then fallbacks on failure.
    async fn invoke(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<O, LcelError> {
        self.try_all(input, config).await
    }

    /// Stream: try the primary first. On failure, try fallbacks.
    /// Returns the first successful stream, or the primary's error.
    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        // Try primary stream
        let boxed_input = Box::new(input.clone()) as Box<dyn Any + Send>;
        match self.primary.stream_any(boxed_input, config.clone()).await {
            Ok(stream) => {
                let output_stream = stream.map(|result| {
                    result.and_then(|boxed| {
                        boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                            LcelError::TypeMismatch(format!(
                                "fallback stream downcast: expected {}",
                                std::any::type_name::<O>()
                            ))
                        })
                    })
                });
                return Ok(Box::pin(output_stream));
            }
            Err(primary_error) => {
                // Try each fallback
                for fallback in &self.fallbacks {
                    let boxed_input = Box::new(input.clone()) as Box<dyn Any + Send>;
                    match fallback.stream_any(boxed_input, config.clone()).await {
                        Ok(stream) => {
                            let output_stream = stream.map(|result| {
                                result.and_then(|boxed| {
                                    boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                                        LcelError::TypeMismatch(format!(
                                            "fallback stream downcast: expected {}",
                                            std::any::type_name::<O>()
                                        ))
                                    })
                                })
                            });
                            return Ok(Box::pin(output_stream));
                        }
                        Err(_) => continue,
                    }
                }
                Err(primary_error)
            }
        }
    }

    /// Batch: apply fallback logic per-item.
    async fn batch(
        &self,
        inputs: Vec<I>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<O>, LcelError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.invoke(input, config.clone()).await?);
        }
        Ok(results)
    }

    /// Transform: use invoke with fallback, wrap as stream.
    async fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        // Default: buffer all input, take the last item, invoke with fallback
        let mut items = Vec::new();
        let mut input = input;
        while let Some(item) = input.next().await {
            items.push(item?);
        }

        if let Some(last) = items.into_iter().last() {
            let result = self.invoke(last, config).await?;
            Ok(Box::pin(futures_util::stream::once(async move { Ok(result) })))
        } else {
            Ok(Box::pin(futures_util::stream::empty()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::{RunnableExt, RunnableLambda};
    use futures_util::StreamExt;

    #[tokio::test]
    async fn primary_succeeds_no_fallback() {
        let primary = RunnableLambda::new_sync(|x: i32| x * 2);
        let fallback = RunnableLambda::new_sync(|x: i32| x * 3);

        let with_fallbacks = primary.with_fallbacks(vec![fallback]);
        let result = with_fallbacks.invoke(5, None).await.unwrap();
        assert_eq!(result, 10); // 5 * 2, primary succeeded
    }

    #[tokio::test]
    async fn primary_fails_fallback_succeeds() {
        let primary = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Other("primary failed".to_string()))
        });
        let fallback = RunnableLambda::new_sync(|x: i32| x * 3);

        let with_fallbacks = primary.with_fallbacks(vec![fallback]);
        let result = with_fallbacks.invoke(5, None).await.unwrap();
        assert_eq!(result, 15); // 5 * 3, fallback succeeded
    }

    #[tokio::test]
    async fn all_fail_returns_primary_error() {
        let primary = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Provider("openai timeout".to_string()))
        });
        let fallback = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Provider("anthropic timeout".to_string()))
        });

        let with_fallbacks = primary.with_fallbacks(vec![fallback]);
        let err = with_fallbacks.invoke(5, None).await.unwrap_err();
        // Should return the primary's error
        assert!(matches!(err, LcelError::Provider(msg) if msg.contains("openai")));
    }

    #[tokio::test]
    async fn multiple_fallbacks_first_wins() {
        let primary = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Other("primary failed".to_string()))
        });
        let fb1 = RunnableLambda::new_sync(|x: i32| x + 100);
        let fb2 = RunnableLambda::new_sync(|x: i32| x + 200);

        let with_fallbacks = primary.with_fallbacks(vec![fb1, fb2]);
        let result = with_fallbacks.invoke(5, None).await.unwrap();
        assert_eq!(result, 105); // 5 + 100, first fallback succeeded
    }

    #[tokio::test]
    async fn first_fallback_fails_second_succeeds() {
        let primary = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Other("primary failed".to_string()))
        });
        let fb1 = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Other("fb1 failed".to_string()))
        });
        let fb2 = RunnableLambda::new_sync(|x: i32| x + 200);

        let with_fallbacks = primary.with_fallbacks(vec![fb1, fb2]);
        let result = with_fallbacks.invoke(5, None).await.unwrap();
        assert_eq!(result, 205); // 5 + 200, second fallback succeeded
    }

    #[tokio::test]
    async fn stream_primary_succeeds() {
        let primary = RunnableLambda::new_sync(|x: i32| x * 2);
        let with_fallbacks = primary.with_fallbacks(Vec::<RunnableLambda<i32, i32>>::new());

        let mut stream = with_fallbacks.stream(5, None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn stream_primary_fails_fallback_succeeds() {
        let primary = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Other("primary failed".to_string()))
        });
        let fallback = RunnableLambda::new_sync(|x: i32| x * 3);

        let with_fallbacks = primary.with_fallbacks(vec![fallback]);
        let mut stream = with_fallbacks.stream(5, None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn batch_works() {
        let primary = RunnableLambda::new_sync(|x: i32| x * 2);
        let fallback = RunnableLambda::new_sync(|x: i32| x * 3);

        let with_fallbacks = primary.with_fallbacks(vec![fallback]);
        let results = with_fallbacks.batch(vec![1, 2, 3], None).await.unwrap();
        assert_eq!(results, vec![2, 4, 6]); // primary succeeds for all
    }

    #[tokio::test]
    async fn transform_works() {
        let primary = RunnableLambda::new_sync(|x: i32| x * 2);
        let with_fallbacks = primary.with_fallbacks(Vec::<RunnableLambda<i32, i32>>::new());

        let input = Box::pin(futures_util::stream::iter(vec![
            Ok(1i32),
            Ok(2i32),
            Ok(3i32),
        ])) as Pin<Box<dyn Stream<Item = Result<i32, LcelError>> + Send>>;

        let mut output = with_fallbacks.transform(input, None).await.unwrap();
        // Default transform takes the last item and invokes: 3 * 2 = 6
        let result = output.next().await.unwrap().unwrap();
        assert_eq!(result, 6);
    }

    #[tokio::test]
    async fn debug_format() {
        let primary = RunnableLambda::new_sync(|x: i32| x);
        let with_fallbacks = primary.with_fallbacks(Vec::<RunnableLambda<i32, i32>>::new());
        let debug_str = format!("{:?}", with_fallbacks);
        assert!(debug_str.contains("RunnableWithFallbacks"));
    }

    #[tokio::test]
    async fn pipe_with_fallbacks() {
        let primary = RunnableLambda::new_sync_fallible(|x: i32| -> Result<i32, LcelError> {
            Err(LcelError::Other("fail".to_string()))
        });
        let fallback = RunnableLambda::new_sync(|x: i32| x + 10);

        // pipe: (primary | fallback) -> double
        let double = RunnableLambda::new_sync(|x: i32| x * 2);
        let chain = primary.with_fallbacks(vec![fallback]).pipe(double);

        let result = chain.invoke(5, None).await.unwrap();
        assert_eq!(result, 30); // fallback: 5+10=15, double: 15*2=30
    }
}
