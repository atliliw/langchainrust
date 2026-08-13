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

        // Convert StreamToken stream to HashMap stream
        let mapped = chain_stream.map(|result| {
            result
                .map(|token| {
                    let mut map = HashMap::new();
                    map.insert("text".to_string(), Value::String(token.token));
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
        LcelError::Chain(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::runnables::Runnable;

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
