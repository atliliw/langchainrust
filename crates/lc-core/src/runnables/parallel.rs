// lc-core/src/runnables/parallel.rs
//! RunnableParallel - fan-out/fan-in composition.
//!
//! `RunnableParallel` runs multiple `Runnable` steps concurrently on
//! the same input, collecting results into a `HashMap<String, Value>`.
//! This is the LCEL equivalent of Python's `RunnableParallel` / `RunnableMap`.

use super::assign::RunnableAssign;
use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::Stream;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

/// A `Runnable` that runs multiple steps in parallel on the same input.
///
/// Each step is identified by a string key. The output is a
/// `HashMap<String, Value>` where each key maps to the corresponding
/// step's output (serialized as `serde_json::Value`).
///
/// # Example
///
/// ```rust,ignore
/// let parallel = RunnableParallel::<String>::new()
///     .with("length", RunnableLambda::new_sync(|s: String| s.len() as i64))
///     .with("upper", RunnableLambda::new_sync(|s: String| s.to_uppercase()));
///
/// let result = parallel.invoke("hello".to_string(), None).await?;
/// // result = {"length": 5, "upper": "HELLO"}
/// ```
pub struct RunnableParallel<I: Send + Sync + 'static> {
    steps: Vec<(String, Arc<dyn ParallelStep<I>>)>,
}

impl<I: Send + Sync + 'static> std::fmt::Debug for RunnableParallel<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys: Vec<&str> = self.steps.iter().map(|(k, _)| k.as_str()).collect();
        f.debug_struct("RunnableParallel")
            .field("steps", &keys)
            .field("input", &std::any::type_name::<I>())
            .finish()
    }
}

impl<I: Clone + Send + Sync + 'static> Default for RunnableParallel<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Clone + Send + Sync + 'static> RunnableParallel<I> {
    /// Create an empty parallel runnable.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step with the given key.
    ///
    /// The step's output will be serialized to `serde_json::Value`
    /// and stored under the key in the output HashMap.
    pub fn with<O, R>(mut self, key: &str, runnable: R) -> Self
    where
        O: serde::Serialize + Send + Sync + 'static,
        R: Runnable<I, O> + Send + Sync + 'static,
        R::Error: Into<LcelError>,
    {
        self.steps.push((
            key.to_string(),
            Arc::new(ParallelStepImpl {
                inner: runnable,
                serialize: |output: &O| serde_json::to_value(output),
                _marker: std::marker::PhantomData,
            }),
        ));
        self
    }

    /// Number of parallel steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether there are no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Add an assign step that injects a new key into the output HashMap.
    ///
    /// This is the LCEL equivalent of Python's `RunnableParallel.assign()`.
    /// It pipes the parallel output (a `HashMap<String, Value>`) through
    /// a `RunnableAssign` that runs the given runnable on the HashMap
    /// and merges the result under the specified key.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let chain = RunnableParallel::<String>::new()
    ///     .with("context", retriever.pipe(format_docs))
    ///     .assign("question", RunnableLambda::new_sync(|m: HashMap<String, Value>| {
    ///         m.get("context").map(|c| c.to_string()).unwrap_or_default()
    ///     }))
    ///     .pipe(prompt_template)
    ///     .pipe(llm);
    /// ```
    ///
    /// # How it works
    ///
    /// `assign()` returns `self.pipe(RunnableAssign)`. The RunnableAssign
    /// receives the HashMap output from the parallel step, runs the
    /// provided runnable on it, and merges the result back.
    pub fn assign<O, R>(self, key: &str, runnable: R) -> RunnableSequence<I, HashMap<String, Value>>
    where
        I: 'static,
        O: serde::Serialize + Send + Sync + 'static,
        R: Runnable<HashMap<String, Value>, O> + Send + Sync + 'static,
        R::Error: Into<LcelError>,
    {
        use super::ext::RunnableExt;

        let assign = RunnableAssign::new().with(key, runnable);
        self.pipe(assign)
    }
}

use super::sequence::RunnableSequence;

/// Trait for a single parallel step that produces a `serde_json::Value`.
#[async_trait]
trait ParallelStep<I: Send + Sync + 'static>: Send + Sync {
    async fn invoke(&self, input: I, config: Option<RunnableConfig>) -> Result<Value, LcelError>;
}

/// Concrete implementation of `ParallelStep` for any `Runnable<I, O>`.
struct ParallelStepImpl<I, O, R>
where
    I: Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
    R: Runnable<I, O>,
{
    inner: R,
    serialize: fn(&O) -> Result<Value, serde_json::Error>,
    _marker: std::marker::PhantomData<I>,
}

#[async_trait]
impl<I, O, R> ParallelStep<I> for ParallelStepImpl<I, O, R>
where
    I: Clone + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
    R: Runnable<I, O>,
    R::Error: Into<LcelError>,
{
    async fn invoke(&self, input: I, config: Option<RunnableConfig>) -> Result<Value, LcelError> {
        let result = self.inner.invoke(input, config).await.map_err(Into::into)?;
        (self.serialize)(&result)
            .map_err(|e| LcelError::Other(format!("parallel serialization: {}", e)))
    }
}

#[async_trait]
impl<I: Clone + Send + Sync + 'static> Runnable<I, HashMap<String, Value>> for RunnableParallel<I> {
    type Error = LcelError;

    /// Execute all steps in parallel using tokio tasks.
    async fn invoke(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<HashMap<String, Value>, LcelError> {
        let mut handles = Vec::with_capacity(self.steps.len());

        for (key, step) in &self.steps {
            let key = key.clone();
            let step = step.clone();
            let input = input.clone();
            let config = config.clone();

            let handle = tokio::spawn(async move {
                let value = step.invoke(input, config).await?;
                Ok::<(String, Value), LcelError>((key, value))
            });

            handles.push(handle);
        }

        let mut results = HashMap::new();
        for handle in handles {
            let (k, v) = handle
                .await
                .map_err(|e| LcelError::Other(format!("parallel task join error: {}", e)))??;
            results.insert(k, v);
        }

        Ok(results)
    }

    /// Batch: each step processes all inputs independently.
    async fn batch(
        &self,
        inputs: Vec<I>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<HashMap<String, Value>>, LcelError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.invoke(input, config.clone()).await?);
        }
        Ok(results)
    }

    /// Stream: invoke and return single-element stream.
    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<HashMap<String, Value>, LcelError>> + Send>>,
        LcelError,
    > {
        let result = self.invoke(input, config).await?;
        Ok(Box::pin(futures_util::stream::once(
            async move { Ok(result) },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunnableLambda;

    #[tokio::test]
    async fn parallel_invoke() {
        let parallel = RunnableParallel::<String>::new()
            .with("len", RunnableLambda::new_sync(|s: String| s.len() as i64))
            .with(
                "upper",
                RunnableLambda::new_sync(|s: String| s.to_uppercase()),
            );

        let result = parallel.invoke("hello".to_string(), None).await.unwrap();
        assert_eq!(
            result.get("len").unwrap(),
            &Value::Number(serde_json::Number::from(5))
        );
        assert_eq!(
            result.get("upper").unwrap(),
            &Value::String("HELLO".to_string())
        );
    }

    #[tokio::test]
    async fn parallel_empty() {
        let parallel = RunnableParallel::<i32>::new();
        let result = parallel.invoke(42, None).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn parallel_batch() {
        let parallel = RunnableParallel::<String>::new()
            .with("len", RunnableLambda::new_sync(|s: String| s.len() as i64));

        let results = parallel
            .batch(vec!["hi".to_string(), "hello".to_string()], None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].get("len").unwrap(),
            &Value::Number(serde_json::Number::from(2))
        );
        assert_eq!(
            results[1].get("len").unwrap(),
            &Value::Number(serde_json::Number::from(5))
        );
    }

    #[tokio::test]
    async fn parallel_assign_adds_field() {
        let chain = RunnableParallel::<String>::new()
            .with("len", RunnableLambda::new_sync(|s: String| s.len() as i64))
            .assign(
                "upper",
                RunnableLambda::new_sync(|m: HashMap<String, Value>| {
                    // Use the "len" field from the parallel output
                    m.get("len")
                        .and_then(|v| v.as_i64())
                        .map(|n| format!("length={}", n))
                        .unwrap_or_default()
                }),
            );

        let result = chain.invoke("hello".to_string(), None).await.unwrap();
        // Original parallel step result
        assert_eq!(
            result.get("len").unwrap(),
            &Value::Number(serde_json::Number::from(5))
        );
        // Assign step result — can reference previous parallel output
        assert_eq!(
            result.get("upper").unwrap(),
            &Value::String("length=5".to_string())
        );
    }
}
