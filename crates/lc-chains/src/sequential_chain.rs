// lc-chains/src/sequential_chain.rs
//! Sequential Chain
//!
//! Execute multiple chains sequentially.

use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::base::{
    run_chain_with_callbacks, stream_chain_with_callbacks, BaseChain, ChainError, ChainResult,
    ChainStream,
};

/// Sequential Chain
///
/// Executes multiple chains sequentially, where the output of one chain
/// can be used as the input of the next.
pub struct SequentialChain {
    /// Chain list.
    chains: Vec<ChainStep>,

    /// Chain name.
    name: String,
}

/// Chain step.
struct ChainStep {
    /// Chain instance.
    chain: Arc<dyn BaseChain>,

    /// Input mapping (from global input or previous output).
    input_mapping: HashMap<String, String>,

    /// Output mapping (to global result).
    output_mapping: HashMap<String, String>,
}

impl SequentialChain {
    /// Create an empty SequentialChain.
    pub fn new() -> Self {
        Self {
            chains: Vec::new(),
            name: "sequential_chain".to_string(),
        }
    }

    /// Set name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a Chain.
    ///
    /// # Arguments
    /// * `chain` - Chain to add
    /// * `input_keys` - Input keys (from global input)
    /// * `output_keys` - Output keys (to global result)
    pub fn add_chain(
        mut self,
        chain: Arc<dyn BaseChain>,
        input_keys: Vec<&str>,
        output_keys: Vec<&str>,
    ) -> Self {
        let input_mapping = input_keys
            .into_iter()
            .map(|k| (k.to_string(), k.to_string()))
            .collect();

        let output_mapping = output_keys
            .into_iter()
            .map(|k| (k.to_string(), k.to_string()))
            .collect();

        self.chains.push(ChainStep {
            chain,
            input_mapping,
            output_mapping,
        });

        self
    }

    /// Add a Chain with mapping.
    ///
    /// # Arguments
    /// * `chain` - Chain to add
    /// * `input_mapping` - Input mapping {chain_input_key: global_key}
    /// * `output_mapping` - Output mapping {chain_output_key: global_key}
    pub fn add_chain_with_mapping(
        mut self,
        chain: Arc<dyn BaseChain>,
        input_mapping: HashMap<String, String>,
        output_mapping: HashMap<String, String>,
    ) -> Self {
        self.chains.push(ChainStep {
            chain,
            input_mapping,
            output_mapping,
        });

        self
    }

    /// Run every step in order, threading `config` into each sub-chain via
    /// `invoke_with_config` (never silently dropping it at the composition
    /// boundary).
    async fn run_steps(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        let mut current_state = inputs.clone();
        let mut final_output = HashMap::new();

        for (step_index, step) in self.chains.iter().enumerate() {
            let mut chain_inputs = HashMap::new();
            for (chain_key, global_key) in &step.input_mapping {
                if let Some(value) = current_state.get(global_key) {
                    chain_inputs.insert(chain_key.clone(), value.clone());
                } else {
                    return Err(ChainError::MissingInput(format!(
                        "Step {}: missing input '{}' (mapped from '{}')",
                        step_index, chain_key, global_key
                    )));
                }
            }

            let chain_output = step
                .chain
                .invoke_with_config(chain_inputs, config.clone())
                .await
                .map_err(|e| ChainError::Nested {
                    context: format!(
                        "Step {} ({}) execution failed",
                        step_index,
                        step.chain.name()
                    ),
                    source: Box::new(e),
                })?;

            for (chain_key, global_key) in &step.output_mapping {
                if let Some(value) = chain_output.get(chain_key) {
                    current_state.insert(global_key.clone(), value.clone());
                    final_output.insert(global_key.clone(), value.clone());
                } else {
                    return Err(ChainError::OutputError(format!(
                        "Step {} ({}) did not produce expected output key '{}' (mapped to '{}')",
                        step_index,
                        step.chain.name(),
                        chain_key,
                        global_key
                    )));
                }
            }
        }

        Ok(final_output)
    }

    /// Stream body: run all chains except the last via `invoke_with_config`
    /// (so callbacks flow through), stream the last chain via
    /// `stream_with_config`.
    async fn stream_steps(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        if self.chains.is_empty() {
            return Err(ChainError::ExecutionError(
                "SequentialChain has no chains".to_string(),
            ));
        }

        let mut current_state = inputs.clone();

        // Run all chains except the last via invoke
        let last_idx = self.chains.len() - 1;
        for (step_index, step) in self.chains[..last_idx].iter().enumerate() {
            let mut chain_inputs = HashMap::new();
            for (chain_key, global_key) in &step.input_mapping {
                if let Some(value) = current_state.get(global_key) {
                    chain_inputs.insert(chain_key.clone(), value.clone());
                } else {
                    return Err(ChainError::MissingInput(format!(
                        "Step {}: missing input '{}' (mapped from '{}')",
                        step_index, chain_key, global_key
                    )));
                }
            }

            let chain_output = step
                .chain
                .invoke_with_config(chain_inputs, config.clone())
                .await
                .map_err(|e| ChainError::Nested {
                    context: format!(
                        "Step {} ({}) execution failed",
                        step_index,
                        step.chain.name()
                    ),
                    source: Box::new(e),
                })?;

            for (chain_key, global_key) in &step.output_mapping {
                if let Some(value) = chain_output.get(chain_key) {
                    current_state.insert(global_key.clone(), value.clone());
                } else {
                    return Err(ChainError::OutputError(format!(
                        "Step {} ({}) did not produce expected output key '{}'",
                        step_index,
                        step.chain.name(),
                        chain_key,
                    )));
                }
            }
        }

        // Stream the last chain
        let last_step = &self.chains[last_idx];
        let mut chain_inputs = HashMap::new();
        for (chain_key, global_key) in &last_step.input_mapping {
            if let Some(value) = current_state.get(global_key) {
                chain_inputs.insert(chain_key.clone(), value.clone());
            } else {
                return Err(ChainError::MissingInput(format!(
                    "Last step: missing input '{}' (mapped from '{}')",
                    chain_key, global_key
                )));
            }
        }

        // P2-3: apply the last step's output_mapping to the streamed output.
        // The token stream IS the last chain's output under its mapped global
        // keys, so the mapping must be satisfiable — a reference to an output
        // key the chain cannot produce is caught here. The invoke path already
        // errors on this against the actual result dict; the stream path only
        // sees an unkeyed token stream, so it validates against the chain's
        // declared `output_keys()` instead. Without this, `output_keys()` (which
        // reports the mapped global keys) could advertise an output the stream
        // would never actually produce.
        for (chain_key, global_key) in &last_step.output_mapping {
            if !last_step.chain.output_keys().contains(&chain_key.as_str()) {
                return Err(ChainError::OutputError(format!(
                    "Last step ({}) did not produce expected output key '{}' (mapped to '{}')",
                    last_step.chain.name(),
                    chain_key,
                    global_key
                )));
            }
        }

        last_step
            .chain
            .stream_with_config(chain_inputs, config.clone())
            .await
    }
}

impl Default for SequentialChain {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseChain for SequentialChain {
    fn input_keys(&self) -> Vec<&str> {
        if let Some(first) = self.chains.first() {
            first.input_mapping.values().map(|s| s.as_str()).collect()
        } else {
            vec![]
        }
    }

    fn output_keys(&self) -> Vec<&str> {
        if let Some(last) = self.chains.last() {
            last.output_mapping.values().map(|s| s.as_str()).collect()
        } else {
            vec![]
        }
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.run_steps(inputs, None).await
    }

    /// Execute the Chain with config propagation.
    ///
    /// Dispatches this chain's `on_chain_start`/`on_chain_end` and threads
    /// `config` (and thus callbacks) into every sub-chain via
    /// `invoke_with_config` — previously sub-chains were called with plain
    /// `invoke`, dropping the config at the composition boundary.
    async fn invoke_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        run_chain_with_callbacks(self.name(), inputs, config.clone(), |inputs| async move {
            self.run_steps(inputs, config).await
        })
        .await
    }

    /// Stream execution for SequentialChain.
    ///
    /// Runs all chains except the last via invoke (since their output feeds
    /// into subsequent chains). The last chain's output is streamed token
    /// by token by delegating to its `stream()` method; its `output_mapping`
    /// names the global output keys that stream stands for (validated in
    /// `Self::stream_steps` so a broken mapping fails loudly, P2-3).
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.stream_steps(inputs, None).await
    }

    /// Stream execute the Chain with config propagation.
    ///
    /// Dispatches this chain's `on_chain_start`/`on_chain_end` and threads
    /// `config` into every sub-chain (invoke for intermediate steps,
    /// stream for the last step).
    async fn stream_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        stream_chain_with_callbacks(self.name(), inputs, config.clone(), |inputs| async move {
            self.stream_steps(inputs, config).await
        })
        .await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Debug for SequentialChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequentialChain")
            .field("steps", &self.chains.len())
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use serde_json::json;

    /// A simple mock chain that echoes input to output.
    struct EchoChain {
        input_key: String,
        output_key: String,
    }

    impl EchoChain {
        fn new(input_key: &str, output_key: &str) -> Self {
            Self {
                input_key: input_key.to_string(),
                output_key: output_key.to_string(),
            }
        }
    }

    #[async_trait]
    impl BaseChain for EchoChain {
        fn input_keys(&self) -> Vec<&str> {
            vec![&self.input_key]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec![&self.output_key]
        }
        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let mut result = HashMap::new();
            if let Some(v) = inputs.get(&self.input_key) {
                result.insert(self.output_key.clone(), v.clone());
            }
            Ok(result)
        }
    }

    /// A mock chain that always fails with a specific `ChainError` variant.
    struct FailingChain;

    #[async_trait]
    impl BaseChain for FailingChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["text"]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec![]
        }
        async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            Err(ChainError::MissingInput("nested_key".to_string()))
        }
    }

    /// A mock chain that transforms input (uppercases).
    struct UppercaseChain {
        input_key: String,
        output_key: String,
    }

    #[async_trait]
    impl BaseChain for UppercaseChain {
        fn input_keys(&self) -> Vec<&str> {
            vec![&self.input_key]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec![&self.output_key]
        }
        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let mut result = HashMap::new();
            if let Some(Value::String(s)) = inputs.get(&self.input_key) {
                result.insert(self.output_key.clone(), Value::String(s.to_uppercase()));
            }
            Ok(result)
        }
    }

    #[tokio::test]
    async fn test_sequential_chain_single_step() {
        let chain = SequentialChain::new().add_chain(
            Arc::new(EchoChain::new("text", "result")),
            vec!["text"],
            vec!["result"],
        );

        let mut inputs = HashMap::new();
        inputs.insert("text".to_string(), json!("hello"));

        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(result.get("result").unwrap(), &json!("hello"));
    }

    #[tokio::test]
    async fn test_sequential_chain_two_steps() {
        let chain = SequentialChain::new()
            .add_chain(
                Arc::new(EchoChain::new("text", "intermediate")),
                vec!["text"],
                vec!["intermediate"],
            )
            .add_chain(
                Arc::new(UppercaseChain {
                    input_key: "intermediate".to_string(),
                    output_key: "result".to_string(),
                }),
                vec!["intermediate"],
                vec!["result"],
            );

        let mut inputs = HashMap::new();
        inputs.insert("text".to_string(), json!("hello"));

        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(result.get("result").unwrap(), &json!("HELLO"));
    }

    #[tokio::test]
    async fn test_sequential_chain_missing_input() {
        let chain = SequentialChain::new().add_chain(
            Arc::new(EchoChain::new("text", "result")),
            vec!["text"],
            vec!["result"],
        );

        let inputs = HashMap::new();
        let result = chain.invoke(inputs).await;
        assert!(result.is_err());
    }

    /// P2-1: sub-chain failures are wrapped in `Nested` and the original
    /// `ChainError` variant stays inspectable via `source()` (not flattened
    /// into a string).
    #[tokio::test]
    async fn test_sequential_chain_preserves_subchain_error() {
        let chain = SequentialChain::new().add_chain(Arc::new(FailingChain), vec!["text"], vec![]);
        let mut inputs = HashMap::new();
        inputs.insert("text".to_string(), json!("hello"));
        let err = chain.invoke(inputs).await.unwrap_err();
        match &err {
            ChainError::Nested { context, source } => {
                assert!(context.contains("Step 0"), "context: {context}");
                let downcast = source.downcast_ref::<ChainError>();
                assert!(
                    matches!(downcast, Some(ChainError::MissingInput(k)) if k == "nested_key"),
                    "source should downcast to the original variant, got {downcast:?}"
                );
            }
            other => panic!("expected ChainError::Nested, got {other:?}"),
        }
    }

    /// P2-3: the streamed output corresponds to the last step's mapped global
    /// output keys. Intermediate outputs feed `current_state` via their
    /// `output_mapping`, and the last step's `output_mapping` names the global
    /// output the token stream stands for.
    #[tokio::test]
    async fn test_sequential_chain_stream_applies_output_mapping() {
        let chain = SequentialChain::new()
            .add_chain_with_mapping(
                Arc::new(EchoChain::new("text", "intermediate")),
                HashMap::from([("text".to_string(), "text".to_string())]),
                HashMap::from([("intermediate".to_string(), "intermediate".to_string())]),
            )
            .add_chain_with_mapping(
                Arc::new(UppercaseChain {
                    input_key: "intermediate".to_string(),
                    output_key: "answer".to_string(),
                }),
                HashMap::from([("intermediate".to_string(), "intermediate".to_string())]),
                HashMap::from([("answer".to_string(), "final_result".to_string())]),
            );

        // The last step's output_mapping names the global output keys.
        assert_eq!(chain.output_keys(), vec!["final_result"]);

        let mut inputs = HashMap::new();
        inputs.insert("text".to_string(), json!("hello"));

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
        assert_eq!(text, "HELLO");
        assert!(tokens.last().unwrap().is_final);
    }

    /// P2-3: a last-step `output_mapping` referencing an output key the chain
    /// cannot produce is caught on the stream path. Previously the mapping was
    /// silently ignored — the stream emitted anyway and `output_keys()` (which
    /// reports the mapped global keys) could advertise an output that would
    /// never materialize.
    #[tokio::test]
    async fn test_sequential_chain_stream_validates_output_mapping() {
        let chain = SequentialChain::new().add_chain_with_mapping(
            Arc::new(EchoChain::new("text", "result")),
            HashMap::from([("text".to_string(), "text".to_string())]),
            HashMap::from([("missing".to_string(), "final_result".to_string())]),
        );

        let mut inputs = HashMap::new();
        inputs.insert("text".to_string(), json!("hello"));

        let err = match chain.stream(inputs).await {
            Ok(_) => panic!("expected an OutputError"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ChainError::OutputError(_)),
            "expected OutputError, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_sequential_chain_with_name() {
        let chain = SequentialChain::new().with_name("my_chain");
        assert_eq!(chain.name(), "my_chain");
    }

    #[tokio::test]
    async fn test_sequential_chain_default() {
        let chain = SequentialChain::default();
        assert_eq!(chain.name(), "sequential_chain");
    }

    #[tokio::test]
    async fn test_sequential_chain_debug() {
        let chain = SequentialChain::new().with_name("test_chain");
        let debug_str = format!("{:?}", chain);
        assert!(debug_str.contains("test_chain"));
        assert!(debug_str.contains("0")); // 0 steps
    }

    #[tokio::test]
    async fn test_sequential_chain_input_keys() {
        let chain = SequentialChain::new().add_chain(
            Arc::new(EchoChain::new("query", "result")),
            vec!["query"],
            vec!["result"],
        );
        assert_eq!(chain.input_keys(), vec!["query"]);
    }

    #[tokio::test]
    async fn test_sequential_chain_output_keys() {
        let chain = SequentialChain::new().add_chain(
            Arc::new(EchoChain::new("query", "answer")),
            vec!["query"],
            vec!["answer"],
        );
        assert_eq!(chain.output_keys(), vec!["answer"]);
    }
}
