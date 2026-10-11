// lc-chains/src/adapter.rs
//! ChainRunnable adapter - bridges BaseChain to the Runnable trait.
//!
//! This allows chains to participate in LCEL pipelines via `pipe()`.

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use crate::base::BaseChain;

/// Adapter that wraps a `BaseChain` as a `Runnable<HashMap<String, Value>, HashMap<String, Value>>`.
///
/// This enables chains to participate in LCEL pipelines:
///
/// ```rust,ignore
/// let chain_runnable = ChainRunnable::new(Arc::new(my_chain));
/// let pipeline = chain_runnable.pipe(parser);
/// ```
pub struct ChainRunnable {
    chain: Arc<dyn BaseChain>,
}

impl ChainRunnable {
    /// Create a new adapter wrapping the given chain.
    pub fn new(chain: Arc<dyn BaseChain>) -> Self {
        Self { chain }
    }
}

#[async_trait]
impl Runnable<HashMap<String, Value>, HashMap<String, Value>> for ChainRunnable {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<HashMap<String, Value>, LcelError> {
        // Propagate config (including callbacks) through to the chain.
        // P2-1: single conversion point — ChainError → LcelError via the
        // `From` impl below, so `?` applies it uniformly.
        let result = self.chain.invoke_with_config(input, config).await?;
        Ok(result)
    }

    async fn stream(
        &self,
        input: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<HashMap<String, Value>, LcelError>> + Send>>,
        LcelError,
    > {
        // Propagate config (including callbacks) through to the chain.
        // P2-1: same single `From` conversion as `invoke` — outer error and
        // every stream item error both convert via `LcelError::from`, so the
        // invoke/stream paths surface identical error variants.
        let chain_stream = self.chain.stream_with_config(input, config).await?;

        // L57 (v0.27.0): unify the stream output key with the chain's declared
        // output key instead of hardcoding a `"text"` fragment that disagrees
        // with `invoke`. The same `output_keys().first()` convention is used by
        // `BaseChain::stream_with_config`'s final-callback payload, so a streamed
        // pipeline and its identical `invoke` now emit the same — and correct —
        // key. Falls back to `"text"` only for chains that declare no output key.
        let key = self
            .chain
            .output_keys()
            .first()
            .map(|k| (*k).to_string())
            .unwrap_or_else(|| "text".to_string());

        // Convert StreamToken stream to HashMap stream
        let mapped = chain_stream.map(move |result| {
            result
                .map(|token| {
                    let mut map = HashMap::new();
                    map.insert(key.clone(), Value::String(token.token));
                    map
                })
                .map_err(LcelError::from)
        });

        Ok(Box::pin(mapped))
    }

    // batch and transform use default implementations
}

/// Allow `ChainError` to convert into `LcelError`.
impl From<crate::base::ChainError> for LcelError {
    fn from(err: crate::base::ChainError) -> Self {
        // T1: preserve the full error chain. `ChainError::Nested`'s Display
        // (`{context}: {source}`) already inlines the framework's own nested
        // chain into `to_string()` — but only one Display level deep for a
        // *foreign* source. Walk `source()` so deeper foreign contexts (e.g. an
        // `io::Error` wrapping another) survive conversion, while skipping
        // `ChainError` levels that the Display already rendered (avoids
        // duplicating the inline chain). The variant stays a `String` per the
        // documented `LcelError` design.
        let mut msg = err.to_string();
        let mut source = std::error::Error::source(&err);
        while let Some(src) = source {
            if src.downcast_ref::<crate::base::ChainError>().is_some() {
                // Our own Display inlined this level and its source already.
                source = src.source();
                continue;
            }
            msg = format!("{msg}: {src}");
            source = src.source();
        }
        LcelError::Chain(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::ChainError;
    use lc_core::runnables::Runnable;

    /// T1: converting a deeply-nested `ChainError` into `LcelError` preserves
    /// every wrapped context, not just the top-level Display.
    #[test]
    fn test_from_chain_error_preserves_nested_chain() {
        let inner = ChainError::MissingInput("text".to_string());
        let mid = ChainError::Nested {
            context: "sub-step failed".to_string(),
            source: Box::new(inner),
        };
        let outer = ChainError::Nested {
            context: "pipeline failed".to_string(),
            source: Box::new(mid),
        };

        let lcel: LcelError = outer.into();
        let msg = lcel.to_string();
        assert!(msg.contains("pipeline failed"), "outer context: {msg}");
        // The framework chain is inlined by Display; the chain walk must not
        // append an already-rendered `ChainError` level a second time.
        let mid_count = msg.matches("sub-step failed").count();
        assert_eq!(mid_count, 1, "no duplicate middle context: {msg}");
        assert!(msg.contains("sub-step failed"), "middle context: {msg}");
        assert!(msg.contains("Missing input"), "leaf: {msg}");
    }

    struct TestChain;

    #[async_trait]
    impl BaseChain for TestChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }

        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }

        async fn invoke(
            &self,
            inputs: HashMap<String, Value>,
        ) -> Result<HashMap<String, Value>, crate::base::ChainError> {
            let input = inputs.get("input").and_then(|v| v.as_str()).unwrap_or("");
            let mut result = HashMap::new();
            result.insert(
                "output".to_string(),
                Value::String(format!("echo: {}", input)),
            );
            Ok(result)
        }
    }

    #[tokio::test]
    async fn chain_runnable_invoke() {
        let chain = ChainRunnable::new(Arc::new(TestChain));
        let mut input = HashMap::new();
        input.insert("input".to_string(), Value::String("hello".to_string()));
        let result = chain.invoke(input, None).await.unwrap();
        assert_eq!(
            result.get("output").unwrap(),
            &Value::String("echo: hello".to_string())
        );
    }
}
