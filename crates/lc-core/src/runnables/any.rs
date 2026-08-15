// lc-core/src/runnables/any.rs
//! Type-erased Runnable trait for LCEL pipeline internals.
//!
//! `RunnableAny` erases the generic `Input`/`Output` types of `Runnable`
//! into `Box<dyn Any + Send>`, allowing heterogeneous steps to be stored
//! in a single `RunnableSequence`.
//!
//! # Safety Guarantee
//!
//! Type safety is maintained at the `pipe()` boundary: the compiler
//! ensures that `A: Runnable<I, M>` and `B: Runnable<M, O>` have matching
//! intermediate types. The `Any` downcast only happens internally within
//! `RunnableSequence`, where the type relationship is already proven.

use super::config::RunnableConfig;
use super::error::LcelError;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use std::any::Any;
use std::pin::Pin;

/// Type-erased Runnable with unified `LcelError`.
///
/// This trait is the runtime representation of a `Runnable<I, O>` step
/// inside a `RunnableSequence`. All generic types are erased to `Box<dyn Any + Send>`.
#[async_trait]
pub trait RunnableAny: Send + Sync {
    /// Type-erased invoke: `Box<dyn Any + Send>` → `Box<dyn Any + Send>`.
    async fn invoke_any(
        &self,
        input: Box<dyn Any + Send>,
        config: Option<RunnableConfig>,
    ) -> Result<Box<dyn Any + Send>, LcelError>;

    /// Type-erased stream: `Box<dyn Any + Send>` → Stream of `Box<dyn Any + Send>`.
    async fn stream_any(
        &self,
        input: Box<dyn Any + Send>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>>, LcelError>;

    /// Type-erased transform: Stream of `Box<dyn Any + Send>` → Stream of `Box<dyn Any + Send>`.
    ///
    /// This is the core of LCEL streaming: each step takes an input stream
    /// and produces an output stream, enabling pipeline streaming without
    /// buffering intermediate results.
    async fn transform_any(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>>, LcelError>;

    /// Type-erased batch: `Vec<Box<dyn Any + Send>>` → `Vec<Box<dyn Any + Send>>`.
    async fn batch_any(
        &self,
        inputs: Vec<Box<dyn Any + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<Box<dyn Any + Send>>, LcelError>;
}

/// Wrapper that implements `RunnableAny` for any `Runnable<I, O>`.
///
/// We use a concrete wrapper struct instead of a blanket impl because
/// Rust's orphan rules and type parameter constraints make a blanket
/// `impl<I, O, R> RunnableAny for R` impossible (I and O would be
/// unconstrained).
pub struct RunnableAnyWrapper<I, O, R>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    R: super::Runnable<I, O>,
{
    inner: R,
    _marker: std::marker::PhantomData<(I, O)>,
}

impl<I, O, R> RunnableAnyWrapper<I, O, R>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    R: super::Runnable<I, O>,
{
    /// Create a new wrapper.
    pub fn new(runnable: R) -> Self {
        Self {
            inner: runnable,
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, R> RunnableAny for RunnableAnyWrapper<I, O, R>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    R: super::Runnable<I, O> + 'static,
    R::Error: Into<LcelError>,
{
    async fn invoke_any(
        &self,
        input: Box<dyn Any + Send>,
        config: Option<RunnableConfig>,
    ) -> Result<Box<dyn Any + Send>, LcelError> {
        let typed_input = input.downcast::<I>().map_err(|_| {
            LcelError::TypeMismatch(format!(
                "invoke_any: expected {}, got unknown type",
                std::any::type_name::<I>()
            ))
        })?;
        let result = self
            .inner
            .invoke(*typed_input, config)
            .await
            .map_err(Into::into)?;
        Ok(Box::new(result) as Box<dyn Any + Send>)
    }

    async fn stream_any(
        &self,
        input: Box<dyn Any + Send>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>>, LcelError>
    {
        let typed_input = input.downcast::<I>().map_err(|_| {
            LcelError::TypeMismatch(format!(
                "stream_any: expected {}, got unknown type",
                std::any::type_name::<I>()
            ))
        })?;
        let stream = self
            .inner
            .stream(*typed_input, config)
            .await
            .map_err(Into::into)?;
        let any_stream = stream.map(|result| {
            result
                .map(|output| Box::new(output) as Box<dyn Any + Send>)
                .map_err(Into::into)
        });
        Ok(Box::pin(any_stream))
    }

    async fn transform_any(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>>, LcelError>
    {
        // We can't forward the input stream to `self.inner.transform()` directly
        // because the error types don't match (LcelError vs `R::Error` and
        // `R::Error: From<LcelError>` does not hold for e.g. `Infallible`).
        //
        // Instead we drive each input item through `self.inner.stream(item, ..)`
        // and concatenate the per-item streams in order. This is the LangChain
        // default `transform` semantics: a step that overrides `stream` (e.g. an
        // LLM) yields a real token stream per item, while a step using the
        // default `stream` (single-element) degrades to elementwise `invoke`.
        use futures_util::StreamExt;

        let mut items = Vec::new();
        let mut input = input;
        while let Some(item) = input.next().await {
            let boxed = item?;
            let typed = boxed.downcast::<I>().map_err(|_| {
                LcelError::TypeMismatch(format!(
                    "transform_any input: expected {}",
                    std::any::type_name::<I>()
                ))
            })?;
            items.push(*typed);
        }

        let mut per_item_streams = Vec::with_capacity(items.len());
        for item in items {
            let stream = self
                .inner
                .stream(item, config.clone())
                .await
                .map_err(Into::into)?;
            let any_stream = stream.map(|result| {
                result
                    .map(|output| Box::new(output) as Box<dyn Any + Send>)
                    .map_err(Into::into)
            });
            per_item_streams.push(any_stream);
        }

        let flattened = futures_util::stream::iter(per_item_streams).flatten();
        Ok(Box::pin(flattened))
    }

    async fn batch_any(
        &self,
        inputs: Vec<Box<dyn Any + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<Box<dyn Any + Send>>, LcelError> {
        let typed_inputs: Vec<I> = inputs
            .into_iter()
            .map(|boxed| {
                boxed.downcast::<I>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "batch_any: expected {}",
                        std::any::type_name::<I>()
                    ))
                })
            })
            .collect::<Result<Vec<I>, LcelError>>()?;
        let results = self
            .inner
            .batch(typed_inputs, config)
            .await
            .map_err(Into::into)?;
        Ok(results
            .into_iter()
            .map(|r| Box::new(r) as Box<dyn Any + Send>)
            .collect())
    }
}

/// Helper function to convert any `Runnable` into `Box<dyn RunnableAny>`.
///
/// This is used internally by `RunnableSequence` and `RunnableExt`
/// to wrap typed runnables into type-erased boxes.
pub fn into_runnable_any<I, O, R>(runnable: R) -> Box<dyn RunnableAny>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    R: super::Runnable<I, O> + 'static,
    R::Error: Into<LcelError>,
{
    Box::new(RunnableAnyWrapper::new(runnable))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    struct AddOne;

    #[async_trait]
    impl super::super::Runnable<i32, i32> for AddOne {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<i32, Self::Error> {
            Ok(input + 1)
        }
    }

    #[tokio::test]
    async fn invoke_any_works() {
        let wrapper = RunnableAnyWrapper::new(AddOne);
        let input: Box<dyn Any + Send> = Box::new(41i32);
        let result = wrapper.invoke_any(input, None).await.unwrap();
        let output: i32 = *result.downcast::<i32>().unwrap();
        assert_eq!(output, 42);
    }

    #[tokio::test]
    async fn batch_any_works() {
        let wrapper = RunnableAnyWrapper::new(AddOne);
        let inputs: Vec<Box<dyn Any + Send>> = vec![Box::new(1i32), Box::new(2i32), Box::new(3i32)];
        let results = wrapper.batch_any(inputs, None).await.unwrap();
        let outputs: Vec<i32> = results
            .into_iter()
            .map(|b| *b.downcast::<i32>().unwrap())
            .collect();
        assert_eq!(outputs, vec![2, 3, 4]);
    }

    #[tokio::test]
    async fn stream_any_works() {
        let wrapper = RunnableAnyWrapper::new(AddOne);
        let input: Box<dyn Any + Send> = Box::new(9i32);
        let mut stream = wrapper.stream_any(input, None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        let output: i32 = *result.downcast::<i32>().unwrap();
        assert_eq!(output, 10);
    }

    #[tokio::test]
    async fn invoke_any_type_mismatch() {
        let wrapper = RunnableAnyWrapper::new(AddOne);
        let wrong_input: Box<dyn Any + Send> = Box::new("not an i32");
        let result = wrapper.invoke_any(wrong_input, None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LcelError::TypeMismatch(_)));
    }

    #[tokio::test]
    async fn into_runnable_any_works() {
        let boxed: Box<dyn RunnableAny> = into_runnable_any::<i32, i32, _>(AddOne);
        let input: Box<dyn Any + Send> = Box::new(5i32);
        let result = boxed.invoke_any(input, None).await.unwrap();
        let output: i32 = *result.downcast::<i32>().unwrap();
        assert_eq!(output, 6);
    }
}
