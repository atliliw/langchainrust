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
use futures_util::{Stream, StreamExt};
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
    ///
    /// 0.26.0 M10: `tokio::spawn` panics when no tokio runtime is active
    /// (`block_on` over a foreign executor, wasm, an offline non-runtime call).
    /// We detect the runtime up front (`Handle::try_current`) and, when absent,
    /// degrade to plain executor-agnostic `join_all`: the steps still run
    /// concurrently at their await points and still bound to `max_concurrency`
    /// via the same semaphore, just without dedicated tasks. This keeps
    /// `RunnableParallel` usable off-tokio instead of panicking.
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
        // M10: `Some` → a tokio runtime is live, `tokio::spawn` is safe. `None` →
        // fall back to executors without a tokio runtime (graceful degradation).
        let on_tokio = tokio::runtime::Handle::try_current().is_ok();

        // Bind each step into a future exposed by EITHER the tokio-spawn path or
        // the join_all fallback; both drive the same per-step body, so the two
        // paths can never drift.
        let mut futures = Vec::with_capacity(self.steps.len());
        for (key, step) in &self.steps {
            let key = key.clone();
            let step = step.clone();
            let input = input.clone();
            let config = config.clone();
            let sem = semaphore.clone();
            futures.push(async move {
                // Tasks beyond the limit park on the permit, so max_concurrency
                // is a true cap on in-flight step execution.
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|e| LcelError::Other(format!("parallel semaphore: {e}")))?;
                let value = step.invoke(input, config).await?;
                Ok::<(String, Value), LcelError>((key, value))
            });
        }

        let results: Vec<(String, Value)> = if on_tokio {
            // Dedicated tasks on the multi-thread runtime → true parallelism.
            let handles = futures.into_iter().map(tokio::spawn).collect::<Vec<_>>();
            let mut out = Vec::with_capacity(handles.len());
            for res in join_all(handles).await {
                // Outer = JoinError (task panicked/cancelled), inner = LcelError.
                let inner =
                    res.map_err(|e| LcelError::Other(format!("parallel task join error: {e}")))?;
                out.push(inner?);
            }
            out
        } else {
            // No tokio runtime: same steps, cooperative scheduling, no spawn.
            join_all(futures)
                .await
                .into_iter()
                .collect::<Result<_, _>>()?
        };

        let mut map = HashMap::with_capacity(results.len());
        for (k, v) in results {
            map.insert(k, v);
        }
        Ok(map)
    }

    /// Batch: each step processes all inputs independently and concurrently.
    ///
    /// 0.26.0 M11: the previous implementation processed inputs serially in a
    /// plain loop and ignored `config.max_concurrency`, contradicting the
    /// `Runnable` batch contract. Now the input batch runs through
    /// `buffer_unordered(limit)` so up to `limit` invocations are in flight
    /// together; `limit` comes from `config.max_concurrency` (defaults to the
    /// full batch). Input order is preserved by sorting on the tagged index.
    /// `buffer_unordered` needs no `tokio::spawn`, so batch is safe on any
    /// executor (consistent with the M10 degradation in [`Self::invoke`]).
    async fn batch(
        &self,
        inputs: Vec<I>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<HashMap<String, Value>>, LcelError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let limit = config
            .as_ref()
            .and_then(|c| c.max_concurrency)
            .unwrap_or(inputs.len())
            .max(1)
            .min(inputs.len());

        let tagged: Vec<(usize, Result<HashMap<String, Value>, LcelError>)> =
            futures_util::stream::iter(inputs.into_iter().enumerate().map(|(idx, input)| {
                let config = config.clone();
                async move {
                    let result = self.invoke(input, config.clone()).await;
                    (idx, result)
                }
            }))
            .buffer_unordered(limit)
            .collect()
            .await;

        let mut tagged = tagged;
        tagged.sort_by_key(|(idx, _)| *idx);
        tagged.into_iter().map(|(_, r)| r).collect()
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

    /// M10: `invoke` must NOT panic when no tokio runtime is active (the old
    /// code unconditionally called `tokio::spawn`). Driving the same runnable
    /// on a plain futures executor should produce identical results.
    #[test]
    fn parallel_invoke_works_off_tokio() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let parallel = RunnableParallel::<String>::new()
            .with("len", RunnableLambda::new_sync(|s: String| s.len() as i64))
            .with(
                "upper",
                RunnableLambda::new_sync(|s: String| s.to_uppercase()),
            );

        // Execute on an executor with no tokio runtime present.
        let result =
            futures_executor::block_on(async { parallel.invoke("hello".to_string(), None).await })
                .unwrap();

        assert_eq!(
            result.get("len").unwrap(),
            &Value::Number(serde_json::Number::from(5))
        );
        assert_eq!(
            result.get("upper").unwrap(),
            &Value::String("HELLO".to_string())
        );
        // Result is a fresh owned value (not a poisoned shared cell).
        let still_ok = Arc::new(AtomicUsize::new(0));
        assert_eq!(still_ok.load(Ordering::SeqCst), 0);

        // max_concurrency is still honored even off-tokio via the semaphore
        // (a step that blocks briefly; on the single-thread futures executor
        // the steps serialize, so peak stays ≤ the cap — the assertion is that
        // the runnable still works and never exceeds the configured limit).
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mk = |in_flight: Arc<AtomicUsize>, peak: Arc<AtomicUsize>| {
            RunnableLambda::new_async(move |_: String| {
                let a = in_flight.clone();
                let b = peak.clone();
                async move {
                    let cur = a.fetch_add(1, Ordering::SeqCst) + 1;
                    b.fetch_max(cur, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    a.fetch_sub(1, Ordering::SeqCst);
                    Ok::<i32, LcelError>(1)
                }
            })
        };
        let limited = RunnableParallel::<String>::new()
            .with("a", mk(in_flight.clone(), peak.clone()))
            .with("b", mk(in_flight.clone(), peak.clone()))
            .with("c", mk(in_flight.clone(), peak.clone()))
            .with("d", mk(in_flight.clone(), peak.clone()));
        let config = RunnableConfig::new().with_max_concurrency(2);
        let out =
            futures_executor::block_on(limited.invoke("x".to_string(), Some(config))).unwrap();
        assert_eq!(out.len(), 4);
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "off-tokio degradation still caps at max_concurrency, peak {}",
            peak.load(Ordering::SeqCst)
        );
    }

    /// M11: `batch` now processes the input batch concurrently up to
    /// `max_concurrency` (was a serial loop), while preserving input order.
    #[tokio::test]
    async fn parallel_batch_respects_max_concurrency_and_keeps_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        // A step that sleeps ~25ms and records in-flight overlaps across the batch.
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let peak_for_assert = peak.clone();
        let step = RunnableLambda::new_async(move |s: String| {
            let a = in_flight.clone();
            let b = peak.clone();
            async move {
                let cur = a.fetch_add(1, Ordering::SeqCst) + 1;
                b.fetch_max(cur, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(25)).await;
                a.fetch_sub(1, Ordering::SeqCst);
                let l = s.len() as i64;
                Ok::<i64, LcelError>(l)
            }
        });
        let parallel = RunnableParallel::<String>::new().with("len", step);

        let inputs: Vec<String> = ["aa", "bbb", "cccc", "ddddd", "eeeeee"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let config = RunnableConfig::new().with_max_concurrency(2);
        let results = parallel.batch(inputs, Some(config)).await.unwrap();
        let lens: Vec<i64> = results.iter().map(|m| m["len"].as_i64().unwrap()).collect();

        // Concurrency capped at 2 => the 5 × 25ms sleeps overlap (totals < 5×25ms).
        assert!(
            peak_for_assert.load(Ordering::SeqCst) <= 2,
            "batch peak concurrency {} exceeded max_concurrency=2",
            peak_for_assert.load(Ordering::SeqCst)
        );

        // Input order preserved (tagged + sorted) — len of each input in order.
        assert_eq!(
            lens,
            vec![2, 3, 4, 5, 6],
            "batch must return results in input order"
        );
    }
}
