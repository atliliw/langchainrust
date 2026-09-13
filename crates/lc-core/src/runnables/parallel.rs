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
use futures_util::future::join_all;
use futures_util::Stream;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Semaphore;

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
    ///
    /// Concurrency is bounded by `config.max_concurrency` (Semaphore), and all
    /// tasks are awaited via `join_all` so an early error cannot orphan the
    /// remaining in-flight tasks (a dropped `JoinHandle` only detaches — it
    /// does not cancel). (A2)
    async fn invoke(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<HashMap<String, Value>, LcelError> {
        let limit = config
            .as_ref()
            .and_then(|c| c.max_concurrency)
            .unwrap_or(self.steps.len())
            .max(1);
        let semaphore = Arc::new(Semaphore::new(limit));

        let mut handles = Vec::with_capacity(self.steps.len());

        for (key, step) in &self.steps {
            let key = key.clone();
            let step = step.clone();
            let input = input.clone();
            let config = config.clone();
            let sem = semaphore.clone();

            let handle = tokio::spawn(async move {
                // Tasks beyond the limit park on the permit, so max_concurrency
                // is a true cap on in-flight step execution.
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|e| LcelError::Other(format!("parallel semaphore: {e}")))?;
                let value = step.invoke(input, config).await?;
                Ok::<(String, Value), LcelError>((key, value))
            });

            handles.push(handle);
        }

        let joined = join_all(handles).await;
        let mut results = HashMap::new();
        for res in joined {
            // Outer = JoinError (task panicked/cancelled), inner = LcelError.
            let inner =
                res.map_err(|e| LcelError::Other(format!("parallel task join error: {e}")))?;
            let (k, v) = inner?;
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

    #[tokio::test]
    async fn parallel_respects_max_concurrency() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let mk = |in_flight: Arc<AtomicUsize>, peak: Arc<AtomicUsize>| {
            RunnableLambda::new_async(move |_: String| {
                let a = in_flight.clone();
                let b = peak.clone();
                async move {
                    let cur = a.fetch_add(1, Ordering::SeqCst) + 1;
                    b.fetch_max(cur, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    a.fetch_sub(1, Ordering::SeqCst);
                    Ok::<i32, LcelError>(1)
                }
            })
        };

        let parallel = RunnableParallel::<String>::new()
            .with("a", mk(in_flight.clone(), peak.clone()))
            .with("b", mk(in_flight.clone(), peak.clone()))
            .with("c", mk(in_flight.clone(), peak.clone()))
            .with("d", mk(in_flight.clone(), peak.clone()));

        let config = RunnableConfig::new().with_max_concurrency(2);
        let result = parallel
            .invoke("x".to_string(), Some(config))
            .await
            .unwrap();
        assert_eq!(result.len(), 4);

        // With a cap of 2, we should never see more than 2 steps in flight.
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "peak concurrency {} exceeded cap 2",
            peak.load(Ordering::SeqCst)
        );
    }

    /// A2: when one step fails, `invoke` must still await the other spawned
    /// tasks before returning the error. A dropped `JoinHandle` only detaches a
    /// task (it does not cancel it), so the previous early-return-on-first-error
    /// implementation orphaned in-flight work: the call returned before the
    /// surviving steps' side effects happened.
    #[tokio::test]
    async fn parallel_failure_waits_for_other_steps_instead_of_orphaning() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::{Duration, Instant};

        let completed = Arc::new(AtomicUsize::new(0));

        let slow = |completed: Arc<AtomicUsize>| {
            RunnableLambda::new_async(move |_: String| {
                let done = completed.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(60)).await;
                    done.fetch_add(1, Ordering::SeqCst);
                    Ok::<i32, LcelError>(1)
                }
            })
        };

        let failing = RunnableLambda::new_async(|_: String| async move {
            // Fails immediately — much faster than the three sleeping steps.
            Err::<i32, LcelError>(LcelError::Other("deliberate step failure".to_string()))
        });

        let parallel = RunnableParallel::<String>::new()
            .with("a", slow(completed.clone()))
            .with("boom", failing)
            .with("c", slow(completed.clone()))
            .with("d", slow(completed.clone()));

        let start = Instant::now();
        let err = parallel.invoke("x".to_string(), None).await.unwrap_err();
        let elapsed = start.elapsed();

        assert!(
            err.to_string().contains("deliberate step failure"),
            "expected the step error, got: {err}"
        );
        // When the error surfaces, all three surviving steps have run to
        // completion — join_all folded the full task set.
        assert_eq!(
            completed.load(Ordering::SeqCst),
            3,
            "surviving steps must finish before invoke returns the error"
        );
        // The old detach-on-first-error implementation returned in ~0ms while
        // the surviving tasks were still sleeping.
        assert!(
            elapsed >= Duration::from_millis(45),
            "invoke returned after {elapsed:?} — orphaned steps were not awaited"
        );
    }
}
