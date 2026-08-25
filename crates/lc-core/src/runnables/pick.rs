// lc-core/src/runnables/pick.rs
//! RunnablePick — extract keys / values out of a dict-returning Runnable.
//!
//! Rust counterpart of Python LCEL's `Runnable.pick(*keys)` and
//! `Runnable.pluck(key)`:
//!
//! - `pick(["a", "b"])` — keep only the given keys of a
//!   `HashMap<String, Value>` output, dropping everything else.
//! - `pluck("a")` — pull a single value out of the dict output.
//!
//! The trait is blanket-implemented for every `Runnable` whose output is
//! `HashMap<String, Value>` (e.g. `RunnableParallel`, or any sequence that
//! ends in a map), so the composition is checked at compile time — a
//! chain that does *not* produce a map simply won't compile, mirroring the
//! dynamic check Python performs at runtime.

use super::error::LcelError;
use super::ext::RunnableExt;
use super::lambda::RunnableLambda;
use super::runnable_trait::Runnable;
use super::sequence::RunnableSequence;
use serde_json::Value;
use std::collections::HashMap;

/// Extracts keys / values from a Runnable whose output is a `HashMap<String, Value>`.
pub trait RunnablePick<I: Send + Sync + 'static>: Sized {
    /// Keep only the given keys of the dict output, dropping everything else.
    ///
    /// Missing keys are omitted from the result (Python raises `KeyError`;
    /// Rust's map semantics make omission the closest safe equivalent).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let parallel = RunnableParallel::<String>::new()
    ///     .with("a", RunnableLambda::new_sync(|s: String| s.len() as i64))
    ///     .with("b", RunnableLambda::new_sync(|s: String| s.to_uppercase()));
    /// let picked = parallel.pick(["a"]);   // output: {"a": N}
    /// ```
    fn pick<K>(
        self,
        keys: impl IntoIterator<Item = K>,
    ) -> RunnableSequence<I, HashMap<String, Value>>
    where
        K: Into<String>;

    /// Pull a single value out of the dict output.
    ///
    /// A missing key yields `Value::Null` (Python raises `KeyError`; `Null`
    /// keeps the runnable total so the pipeline does not short-circuit).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let val = parallel.pluck("a");   // output: Value
    /// ```
    fn pluck(self, key: impl Into<String>) -> RunnableSequence<I, Value>;
}

impl<I, R> RunnablePick<I> for R
where
    I: Send + Sync + 'static,
    R: Runnable<I, HashMap<String, Value>> + Sized + 'static,
    R::Error: Into<LcelError>,
{
    fn pick<K>(
        self,
        keys: impl IntoIterator<Item = K>,
    ) -> RunnableSequence<I, HashMap<String, Value>>
    where
        K: Into<String>,
    {
        let keys: Vec<String> = keys.into_iter().map(Into::into).collect();
        let filter = RunnableLambda::new_sync(move |m: HashMap<String, Value>| {
            keys.iter()
                .filter_map(|k| m.get(k).cloned().map(|v| (k.clone(), v)))
                .collect::<HashMap<String, Value>>()
        });
        self.pipe(filter)
    }

    fn pluck(self, key: impl Into<String>) -> RunnableSequence<I, Value> {
        let key = key.into();
        let extract = RunnableLambda::new_sync(move |m: HashMap<String, Value>| {
            m.get(&key).cloned().unwrap_or(Value::Null)
        });
        self.pipe(extract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Runnable;
    use crate::RunnableLambda;
    use crate::RunnableParallel;

    fn parallel() -> RunnableParallel<String> {
        RunnableParallel::<String>::new()
            .with("len", RunnableLambda::new_sync(|s: String| s.len() as i64))
            .with(
                "upper",
                RunnableLambda::new_sync(|s: String| s.to_uppercase()),
            )
    }

    #[tokio::test]
    async fn pick_keeps_only_selected_keys() {
        let chain = parallel().pick(["len"]);
        let out = chain.invoke("hello".to_string(), None).await.unwrap();
        assert_eq!(out.len(), 1);
        assert!(out.contains_key("len"));
        assert!(!out.contains_key("upper"));
    }

    #[tokio::test]
    async fn pick_multiple_keys_and_missing() {
        let chain = parallel().pick(["len", "nope"]);
        let out = chain.invoke("hello".to_string(), None).await.unwrap();
        // "nope" 不存在 → 被省略,只剩 len
        assert_eq!(out.len(), 1);
        assert!(out.contains_key("len"));
    }

    #[tokio::test]
    async fn pluck_returns_single_value() {
        let chain = parallel().pluck("upper");
        let out = chain.invoke("hello".to_string(), None).await.unwrap();
        assert_eq!(out, Value::String("HELLO".to_string()));
    }

    #[tokio::test]
    async fn pluck_missing_key_yields_null() {
        let chain = parallel().pluck("missing");
        let out = chain.invoke("hello".to_string(), None).await.unwrap();
        assert_eq!(out, Value::Null);
    }

    #[tokio::test]
    async fn pick_works_on_sequence_ending_in_map() {
        // RunnableSequence<I, HashMap<String, Value>> 也自动获得 RunnablePick
        let seq = RunnableLambda::new_sync(|s: String| s.to_uppercase())
            .pipe(RunnableLambda::new_sync(|s: String| {
                let mut m = HashMap::new();
                m.insert("up".to_string(), Value::String(s));
                m.insert("drop".to_string(), Value::Bool(true));
                m
            }))
            .pick(["up"]);
        let out = seq.invoke("hi".to_string(), None).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out["up"], Value::String("HI".to_string()));
    }
}
