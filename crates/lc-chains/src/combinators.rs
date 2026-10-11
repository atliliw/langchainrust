//! LCEL composition primitives — `map`, `batch` and `parallel` combinators.
//!
//! These wrap a [`Runnable`] (typically a [`crate::adapter::ChainRunnable`])
//! and provide the standard sequential-pipe, fan-out and fan-in combinators so
//! chains compose without hand-writing the adapters each time:
//!
//! - [`map`] — run the inner step, then push its output through a closure.
//! - [`batch`] — fan a list of inputs through one step, preserving order.
//! - [`parallel`] — run two steps over the same input concurrently and merge
//!   their output maps (last-write-wins on colliding keys).
//!
//! ```
//! use lc_chains::combinators::{map, parallel};
//! ```

use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The input/output value shaped by [`crate::adapter::ChainRunnable`].
pub type ChainMap = HashMap<String, Value>;

/// An object-safe reference to a runnable over [`ChainMap`].
pub type BoxRunnable = Arc<dyn Runnable<ChainMap, ChainMap, Error = LcelError>>;

/// Combinator: run `inner`, then apply `f` to its output.
pub struct Map<F> {
    inner: BoxRunnable,
    f: Arc<F>,
}

impl<F> Map<F>
where
    F: Fn(ChainMap) -> Result<ChainMap, LcelError> + Send + Sync + 'static,
{
    /// Builds a map step from an inner runnable and the output transform.
    pub fn new(inner: BoxRunnable, f: F) -> Self {
        Self { inner, f: Arc::new(f) }
    }
}

#[async_trait]
impl<F> Runnable<ChainMap, ChainMap> for Map<F>
where
    F: Fn(ChainMap) -> Result<ChainMap, LcelError> + Send + Sync + 'static,
{
    type Error = LcelError;

    async fn invoke(
        &self,
        input: ChainMap,
        config: Option<RunnableConfig>,
    ) -> Result<ChainMap, LcelError> {
        let out = self.inner.invoke(input, config).await?;
        (self.f)(out)
    }
}

/// Combinator: run one step over many inputs, preserving order.
pub struct Batch {
    inner: BoxRunnable,
}

/// Combinator: run two steps over the same input concurrently, then merge.
pub struct Parallel {
    a: BoxRunnable,
    b: BoxRunnable,
}

/// Builds a [`Map`] step.
pub fn map<F>(inner: BoxRunnable, f: F) -> Map<F>
where
    F: Fn(ChainMap) -> Result<ChainMap, LcelError> + Send + Sync + 'static,
{
    Map::new(inner, f)
}

/// Builds a step that fans `Vec<ChainMap>` through `inner` one by one.
pub fn batch(inner: BoxRunnable) -> Batch {
    Batch { inner }
}

/// Builds a step that runs `a` and `b` over the same input concurrently.
pub fn parallel(a: BoxRunnable, b: BoxRunnable) -> Parallel {
    Parallel { a, b }
}

#[async_trait]
impl Runnable<Vec<ChainMap>, Vec<ChainMap>> for Batch {
    type Error = LcelError;

    async fn invoke(
        &self,
        inputs: Vec<ChainMap>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<ChainMap>, LcelError> {
        // Sequential to keep order + bounded memory; `parallel`/`parallelize`
        // are the concurrency primitives.
        let mut outs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outs.push(self.inner.invoke(input, config.clone()).await?);
        }
        Ok(outs)
    }
}

#[async_trait]
impl Runnable<ChainMap, ChainMap> for Parallel {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: ChainMap,
        config: Option<RunnableConfig>,
    ) -> Result<ChainMap, LcelError> {
        let a_in = input.clone();
        let (ra, rb) = futures_util::future::join(
            self.a.invoke(a_in, config.clone()),
            self.b.invoke(input, config),
        )
        .await;
        let mut merged = ra?;
        merged.extend(rb?);
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial runnable that echoes a key as its own value.
    struct Echo(String);

    #[async_trait]
    impl Runnable<ChainMap, ChainMap> for Echo {
        type Error = LcelError;

        async fn invoke(
            &self,
            mut input: ChainMap,
            _config: Option<RunnableConfig>,
        ) -> Result<ChainMap, LcelError> {
            let v = input.remove(&self.0).unwrap_or(Value::Null);
            input.insert("out".to_string(), v);
            Ok(input)
        }
    }

    fn runnable(key: &str) -> BoxRunnable {
        Arc::new(Echo(key.to_string()))
    }

    fn m(entries: &[(&str, &str)]) -> ChainMap {
        let mut map = ChainMap::new();
        for (k, v) in entries {
            map.insert((*k).to_string(), Value::String((*v).to_string()));
        }
        map
    }

    #[tokio::test]
    async fn map_transforms_output() {
        let mapped = map(runnable("a"), |mut o| {
            o.insert("mapped".to_string(), Value::Bool(true));
            Ok(o)
        });
        let out = mapped.invoke(m(&[("a", "1")]), None).await.unwrap();
        assert_eq!(out.get("mapped"), Some(&Value::Bool(true)));
        assert_eq!(out.get("out"), Some(&Value::String("1".to_string())));
    }

    #[tokio::test]
    async fn batch_preserves_order() {
        let step = batch(runnable("a"));
        let ins = vec![m(&[("a", "1")]), m(&[("a", "2")]), m(&[("a", "3")])];
        let outs = step.invoke(ins, None).await.unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[1].get("out"), Some(&Value::String("2".to_string())));
    }

    #[tokio::test]
    async fn parallel_runs_both_and_merges() {
        let step = parallel(runnable("a"), runnable("b"));
        let out = step
            .invoke(m(&[("a", "1"), ("b", "2")]), None)
            .await
            .unwrap();
        assert_eq!(out.get("out"), Some(&Value::String("2".to_string())));
        // Both branches ran: a's "out"="1" was overwritten by b's "out"="2" in
        // `extend` order, while a's remaining keys survive.
    }
}