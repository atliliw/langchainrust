// lc-core/src/runnables/binding.rs
//! RunnableBinding - bind runtime parameters and config to a Runnable.
//!
//! `RunnableBinding` wraps a `Runnable` and attaches runtime kwargs
//! and/or a `RunnableConfig`, allowing pre-configuration of steps
//! in an LCEL pipeline.

use super::any::{into_runnable_any, RunnableAny};
use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use std::any::Any;
use std::collections::HashMap;
use std::pin::Pin;

/// A `Runnable` that binds runtime parameters and config to an inner runnable.
///
/// # Example
///
/// ```rust,ignore
/// let chain = llm
///     .pipe(parser)
///     .bind("stop", json!("\n"))
///     .with_config(RunnableConfig::new().with_tag("production"));
/// ```
pub struct RunnableBinding<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    bound: Box<dyn RunnableAny>,
    kwargs: HashMap<String, Value>,
    config: RunnableConfig,
    _marker: std::marker::PhantomData<(I, O)>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> std::fmt::Debug for RunnableBinding<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnableBinding")
            .field("kwargs", &self.kwargs)
            .field("input", &std::any::type_name::<I>())
            .field("output", &std::any::type_name::<O>())
            .finish()
    }
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> RunnableBinding<I, O> {
    /// Create a new binding wrapping the given runnable.
    pub fn new<R>(runnable: R) -> Self
    where
        R: Runnable<I, O> + 'static,
        R::Error: Into<LcelError>,
    {
        Self {
            bound: into_runnable_any(runnable),
            kwargs: HashMap::new(),
            config: RunnableConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Bind a runtime keyword argument.
    ///
    /// The kwargs are merged into the `RunnableConfig` metadata
    /// when the runnable is invoked.
    pub fn bind(mut self, key: impl Into<String>, value: Value) -> Self {
        self.kwargs.insert(key.into(), value);
        self
    }

    /// Set the execution config.
    ///
    /// This config is merged with any config passed at invocation time.
    pub fn with_config(mut self, config: RunnableConfig) -> Self {
        self.config = config;
        self
    }

    /// Merge bound kwargs and config into the invocation config.
    fn merged_config(&self, invocation_config: Option<RunnableConfig>) -> RunnableConfig {
        let mut base = self.config.clone();

        // Add kwargs as metadata
        for (key, value) in &self.kwargs {
            base = base.with_metadata(key.clone(), value.clone());
        }

        // Merge with invocation config
        if let Some(inv) = invocation_config {
            base.merge(inv)
        } else {
            base
        }
    }
}

#[async_trait]
impl<I: Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O> for RunnableBinding<I, O> {
    type Error = LcelError;

    async fn invoke(&self, input: I, config: Option<RunnableConfig>) -> Result<O, LcelError> {
        let merged = self.merged_config(config);
        let result = self
            .bound
            .invoke_any(Box::new(input) as Box<dyn Any + Send>, Some(merged))
            .await?;
        result.downcast::<O>().map(|b| *b).map_err(|_| {
            LcelError::TypeMismatch(format!(
                "binding output downcast: expected {}",
                std::any::type_name::<O>()
            ))
        })
    }

    async fn batch(
        &self,
        inputs: Vec<I>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<O>, LcelError> {
        let merged = self.merged_config(config);
        let boxed_inputs: Vec<Box<dyn Any + Send>> = inputs
            .into_iter()
            .map(|i| Box::new(i) as Box<dyn Any + Send>)
            .collect();
        let results = self.bound.batch_any(boxed_inputs, Some(merged)).await?;
        results
            .into_iter()
            .map(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "binding batch downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
            .collect()
    }

    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        let merged = self.merged_config(config);
        let stream = self
            .bound
            .stream_any(Box::new(input) as Box<dyn Any + Send>, Some(merged))
            .await?;
        let output_stream = stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "binding stream downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });
        Ok(Box::pin(output_stream))
    }

    async fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        let merged = self.merged_config(config);
        let any_input: Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>> =
            Box::pin(input.map(|result| result.map(|item| Box::new(item) as Box<dyn Any + Send>)));

        let output_stream = self.bound.transform_any(any_input, Some(merged)).await?;

        let typed_output = output_stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "binding transform downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });
        Ok(Box::pin(typed_output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    struct EchoRunnable;

    #[async_trait]
    impl Runnable<String, String> for EchoRunnable {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: String,
            config: Option<RunnableConfig>,
        ) -> Result<String, Self::Error> {
            let tags = config.map(|c| c.tags.join(",")).unwrap_or_default();
            if tags.is_empty() {
                Ok(input)
            } else {
                Ok(format!("[{}] {}", tags, input))
            }
        }
    }

    #[tokio::test]
    async fn binding_with_config() {
        let binding = RunnableBinding::new(EchoRunnable)
            .with_config(RunnableConfig::default().with_tag("prod"));

        let result = binding.invoke("hello".to_string(), None).await.unwrap();
        assert_eq!(result, "[prod] hello");
    }

    #[tokio::test]
    async fn binding_with_kwargs() {
        let binding =
            RunnableBinding::new(EchoRunnable).bind("stop", Value::String("\n".to_string()));

        // kwargs are stored in metadata
        let result = binding.invoke("test".to_string(), None).await.unwrap();
        assert_eq!(result, "test"); // EchoRunnable doesn't use metadata
    }

    #[tokio::test]
    async fn binding_stream_works() {
        let binding = RunnableBinding::new(EchoRunnable)
            .with_config(RunnableConfig::default().with_tag("stream"));

        let mut stream = binding.stream("hello".to_string(), None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, "[stream] hello");
    }
}
