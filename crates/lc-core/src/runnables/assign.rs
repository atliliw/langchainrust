// lc-core/src/runnables/assign.rs
//! RunnableAssign - inject new fields into a HashMap pipeline.
//!
//! `RunnableAssign` adds new key-value pairs to a `HashMap<String, Value>`,
//! enabling RAG pipelines where the context is injected alongside the
//! original question.
//!
//! # Example
//!
//! ```rust,ignore
//! let chain = RunnableParallel::<String>::new()
//!     .assign("context", retriever.pipe(format_docs))
//!     .assign("question", RunnablePassthrough)
//!     .pipe(prompt_template)
//!     .pipe(llm);
//! ```

use super::any::RunnableAny;
use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::Stream;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;

/// A `Runnable` that adds new key-value pairs to a `HashMap<String, Value>`.
///
/// Each mapping runs a `Runnable` on the input HashMap and merges the
/// result back into the HashMap under the specified key.
///
/// This is the LCEL equivalent of Python's `RunnableAssign`.
pub struct RunnableAssign {
    mappings: Vec<(String, Box<dyn RunnableAny>)>,
}

impl std::fmt::Debug for RunnableAssign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys: Vec<&str> = self.mappings.iter().map(|(k, _)| k.as_str()).collect();
        f.debug_struct("RunnableAssign")
            .field("mappings", &keys)
            .finish()
    }
}

impl RunnableAssign {
    /// Create an empty assign runnable.
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
        }
    }

    /// Add a mapping that runs the given runnable and stores the result
    /// under the specified key.
    ///
    /// The runnable takes `HashMap<String, Value>` as input and produces
    /// any output that implements `serde::Serialize`. The output is
    /// serialized to `serde_json::Value` and merged into the HashMap.
    pub fn with<O, R>(mut self, key: &str, runnable: R) -> Self
    where
        O: serde::Serialize + Send + Sync + 'static,
        R: Runnable<HashMap<String, Value>, O> + Send + Sync + 'static,
        R::Error: Into<LcelError>,
    {
        use super::any::into_runnable_any;

        // Wrap the runnable so that its output is serialized to Value
        // and then we can store it in the HashMap
        let wrapped = AssignStepWrapper {
            inner: runnable,
            serialize: |output: &O| serde_json::to_value(output),
            _marker: std::marker::PhantomData,
        };

        self.mappings.push((key.to_string(), into_runnable_any(wrapped)));
        self
    }

    /// Number of mappings.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Whether there are no mappings.
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }
}

impl Default for RunnableAssign {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal wrapper that serializes the output of a Runnable to Value.
struct AssignStepWrapper<O, R>
where
    O: serde::Serialize + Send + Sync + 'static,
    R: Runnable<HashMap<String, Value>, O>,
{
    inner: R,
    serialize: fn(&O) -> Result<Value, serde_json::Error>,
    _marker: std::marker::PhantomData<O>,
}

#[async_trait]
impl<O, R> Runnable<HashMap<String, Value>, Value> for AssignStepWrapper<O, R>
where
    O: serde::Serialize + Send + Sync + 'static,
    R: Runnable<HashMap<String, Value>, O>,
    R::Error: Into<LcelError>,
{
    type Error = LcelError;

    async fn invoke(
        &self,
        input: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<Value, LcelError> {
        let result = self.inner.invoke(input, config).await.map_err(Into::into)?;
        (self.serialize)(&result)
            .map_err(|e| LcelError::Other(format!("assign serialization: {}", e)))
    }
}

#[async_trait]
impl Runnable<HashMap<String, Value>, HashMap<String, Value>> for RunnableAssign {
    type Error = LcelError;

    /// Execute all mappings and merge results into the input HashMap.
    async fn invoke(
        &self,
        mut input: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<HashMap<String, Value>, LcelError> {
        use std::any::Any;

        for (key, step) in &self.mappings {
            let boxed_input = Box::new(input.clone()) as Box<dyn Any + Send>;
            let result = step.invoke_any(boxed_input, config.clone()).await?;

            // The result should be a Value (from AssignStepWrapper)
            let value = result
                .downcast::<Value>()
                .map(|b| *b)
                .map_err(|_| LcelError::TypeMismatch(format!(
                    "assign step output downcast: expected Value, got unknown type for key '{}'",
                    key
                )))?;

            input.insert(key.clone(), value);
        }

        Ok(input)
    }

    /// Stream: invoke and return single-element stream.
    async fn stream(
        &self,
        input: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<HashMap<String, Value>, LcelError>> + Send>>, LcelError> {
        let result = self.invoke(input, config).await?;
        Ok(Box::pin(futures_util::stream::once(async move { Ok(result) })))
    }

    /// Batch: invoke per input.
    async fn batch(
        &self,
        inputs: Vec<HashMap<String, Value>>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<HashMap<String, Value>>, LcelError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.invoke(input, config.clone()).await?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::RunnableLambda;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn assign_single_field() {
        let assign = RunnableAssign::new()
            .with("length", RunnableLambda::new_sync(|m: HashMap<String, Value>| {
                m.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.len() as i64)
                    .unwrap_or(0)
            }));

        let mut input = HashMap::new();
        input.insert("text".to_string(), Value::String("hello".to_string()));

        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result.get("text").unwrap(), &Value::String("hello".to_string()));
        assert_eq!(result.get("length").unwrap(), &Value::Number(serde_json::Number::from(5)));
    }

    #[tokio::test]
    async fn assign_multiple_fields() {
        let assign = RunnableAssign::new()
            .with("length", RunnableLambda::new_sync(|m: HashMap<String, Value>| {
                m.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.len() as i64)
                    .unwrap_or(0)
            }))
            .with("upper", RunnableLambda::new_sync(|m: HashMap<String, Value>| {
                m.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_uppercase())
                    .unwrap_or_default()
            }));

        let mut input = HashMap::new();
        input.insert("text".to_string(), Value::String("hello".to_string()));

        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result.get("length").unwrap(), &Value::Number(serde_json::Number::from(5)));
        assert_eq!(result.get("upper").unwrap(), &Value::String("HELLO".to_string()));
        // Original field preserved
        assert_eq!(result.get("text").unwrap(), &Value::String("hello".to_string()));
    }

    #[tokio::test]
    async fn assign_overwrites_existing_key() {
        let assign = RunnableAssign::new()
            .with("text", RunnableLambda::new_sync(|_m: HashMap<String, Value>| {
                "replaced".to_string()
            }));

        let mut input = HashMap::new();
        input.insert("text".to_string(), Value::String("original".to_string()));

        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result.get("text").unwrap(), &Value::String("replaced".to_string()));
    }

    #[tokio::test]
    async fn assign_stream_works() {
        let assign = RunnableAssign::new()
            .with("length", RunnableLambda::new_sync(|m: HashMap<String, Value>| {
                m.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.len() as i64)
                    .unwrap_or(0)
            }));

        let mut input = HashMap::new();
        input.insert("text".to_string(), Value::String("hi".to_string()));

        let mut stream = assign.stream(input, None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result.get("length").unwrap(), &Value::Number(serde_json::Number::from(2)));
    }

    #[tokio::test]
    async fn assign_batch_works() {
        let assign = RunnableAssign::new()
            .with("length", RunnableLambda::new_sync(|m: HashMap<String, Value>| {
                m.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.len() as i64)
                    .unwrap_or(0)
            }));

        let mut input1 = HashMap::new();
        input1.insert("text".to_string(), Value::String("hi".to_string()));

        let mut input2 = HashMap::new();
        input2.insert("text".to_string(), Value::String("hello".to_string()));

        let results = assign.batch(vec![input1, input2], None).await.unwrap();
        assert_eq!(results[0].get("length").unwrap(), &Value::Number(serde_json::Number::from(2)));
        assert_eq!(results[1].get("length").unwrap(), &Value::Number(serde_json::Number::from(5)));
    }

    #[tokio::test]
    async fn assign_empty_works() {
        let assign = RunnableAssign::new();
        let mut input = HashMap::new();
        input.insert("key".to_string(), Value::String("value".to_string()));

        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result.len(), 1);
    }
}
