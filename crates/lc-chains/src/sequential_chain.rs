// lc-chains/src/sequential_chain.rs
//! Sequential Chain
//!
//! Execute multiple chains sequentially.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::runnables::RunnableConfig;
use lc_memory::BaseMemory;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;

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

    /// Optional pluggable memory (L65: generic `.with_memory<M>` accepts any
    /// `BaseMemory`). When present, its variables are injected into the chain
    /// input state before execution and context is saved after each step.
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
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
            memory: None,
        }
    }

    /// Set name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// L65: inject a memory generically — any `BaseMemory` implementation
    /// (buffer, window, summary, persistent, vector-store, …) is accepted, not
    /// just one concrete type. Loaded memory variables are merged into the
    /// chain input state (so any step whose `input_mapping` references a
    /// memory variable, e.g. `history`, can consume it), and context is saved
    /// after execution.
    ///
    /// `save_context` operates on string payloads keyed by the memory's own
    /// contract — to be saved, the chain's `input`/`output`-mapped keys must
    /// appear as string values in the inputs / final output.
    pub fn with_memory<M: BaseMemory + 'static>(mut self, memory: M) -> Self {
        self.memory = Some(Arc::new(Mutex::new(memory)));
        self
    }

    /// Access the injected memory, if any.
    pub fn injected_memory(&self) -> Option<&Arc<Mutex<dyn BaseMemory>>> {
        self.memory.as_ref()
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

    /// Reduce a value map to its string-valued entries — the `BaseMemory`
    /// trait is `String`-based (`load_memory_variables`/`save_context` operate
    /// on `HashMap<String, String>`), so non-string state (nested arrays,
    /// documents, …) cannot be passed to memory.
    fn string_pairs(map: &HashMap<String, Value>) -> HashMap<String, String> {
        map.iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect()
    }

    /// L65: load memory variables and merge them into the chain input `state`
    /// so any step whose `input_mapping` references a memory variable (e.g.
    /// `history`) can consume it. Memory-agnostic: every `BaseMemory`
    /// publishes its own variables.
    async fn inject_memory(
        &self,
        state: &mut HashMap<String, Value>,
        inputs: &HashMap<String, Value>,
    ) -> Result<(), ChainError> {
        if let Some(memory) = &self.memory {
            let mem = memory.lock().await;
            let vars = mem
                .load_memory_variables(&Self::string_pairs(inputs))
                .await
                .map_err(|e| ChainError::ExecutionError(format!("Failed to load memory: {e}")))?;
            drop(mem);
            for (k, v) in vars {
                state.insert(k, v);
            }
        }
        Ok(())
    }

    /// L65: persist the executed turn into the injected memory, if any.
    async fn save_memory(
        &self,
        inputs: &HashMap<String, Value>,
        outputs: &HashMap<String, Value>,
    ) -> Result<(), ChainError> {
        if let Some(memory) = &self.memory {
            let mut mem = memory.lock().await;
            mem.save_context(&Self::string_pairs(inputs), &Self::string_pairs(outputs))
                .await
                .map_err(|e| ChainError::ExecutionError(format!("Failed to save context: {e}")))?;
        }
        Ok(())
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
        self.inject_memory(&mut current_state, &inputs).await?;
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

        self.save_memory(&inputs, &final_output).await?;

        Ok(final_output)
    }

    /// Stream body.
    ///
    /// L64 全正向流式: every step's *own* stream is driven forward — a
    /// single-output intermediate step is streamed (so its LLM callbacks/
    /// tokens fire per token instead of a blocking `invoke`) and its streamed
    /// text is accumulated back into `current_state` to feed the next step.
    /// Steps with multiple mapped outputs or non-text output fall back to
    /// `invoke_with_config` (a token stream can only represent one text value).
    ///
    /// The last chain's stream is passed through to the caller; when memory is
    /// injected, the stream is wrapped so the turn is saved on completion.
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
        self.inject_memory(&mut current_state, &inputs).await?;

        // L64: stream intermediate steps forward; fall back to invoke when a
        // step cannot be represented as a single text stream.
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

            let wrapped = |e: ChainError| ChainError::Nested {
                context: format!(
                    "Step {} ({}) execution failed",
                    step_index,
                    step.chain.name()
                ),
                source: Box::new(e),
            };

            // A single-output-mapping step whose sole mapped output key is one
            // of the chain's declared outputs can be forward-streamed: its
            // token text is that output.
            let streamable = match step.output_mapping.iter().next() {
                Some((chain_key, _)) if step.output_mapping.len() == 1 => step
                    .chain
                    .output_keys()
                    .iter()
                    .any(|k| k == &chain_key.as_str()),
                _ => false,
            };

            if streamable {
                let global_key = step.output_mapping.iter().next().unwrap().1;
                let mut s = step
                    .chain
                    .stream_with_config(chain_inputs, config.clone())
                    .await
                    .map_err(wrapped)?;
                let mut text = String::new();
                while let Some(token) = s.next().await {
                    let token = token.map_err(wrapped)?;
                    text.push_str(&token.token);
                }
                current_state.insert((*global_key).clone(), Value::String(text));
            } else {
                let chain_output = step
                    .chain
                    .invoke_with_config(chain_inputs, config.clone())
                    .await
                    .map_err(wrapped)?;

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

        let last_stream = last_step
            .chain
            .stream_with_config(chain_inputs, config.clone())
            .await?;

        // L65: when memory is injected, wrap the last stream so the completed
        // turn is saved once the token stream ends. A save failure at this
        // point cannot fail the already-streamed run, so it is logged rather
        // than silently dropped.
        if self.memory.is_some() {
            let memory = self.memory.clone();
            let out_key = self
                .output_keys()
                .first()
                .map(|k| (*k).to_string())
                .unwrap_or_else(|| "output".to_string());
            let wrapped = futures_util::stream::unfold(
                (last_stream, memory, inputs, out_key, String::new()),
                |(mut inner, memory, save_inputs, out_key, mut acc)| async move {
                    match inner.next().await {
                        Some(Ok(token)) => {
                            acc.push_str(&token.token);
                            Some((Ok(token), (inner, memory, save_inputs, out_key, acc)))
                        }
                        Some(Err(e)) => Some((Err(e), (inner, memory, save_inputs, out_key, acc))),
                        None => {
                            if let Some(memory) = memory {
                                let mut mem = memory.lock().await;
                                let mut outputs = HashMap::new();
                                outputs.insert(out_key, Value::String(acc));
                                if let Err(e) = mem
                                    .save_context(
                                        &Self::string_pairs(&save_inputs),
                                        &Self::string_pairs(&outputs),
                                    )
                                    .await
                                {
                                    log::warn!(
                                        "SequentialChain: failed to save context on stream end: {e}"
                                    );
                                }
                            }
                            None
                        }
                    }
                },
            );
            return Ok(Box::pin(wrapped));
        }

        Ok(last_stream)
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
        // L13: report every *external* input, not just the first step's. A global key
        // is external iff the caller must supply it — it appears in some step's input
        // mapping and is not produced by any step's output mapping (keys produced
        // inside the chain are intermediates, not inputs). A later step that maps a
        // global key directly is otherwise silently under-reported.
        let produced: HashSet<&str> = self
            .chains
            .iter()
            .flat_map(|s| s.output_mapping.values().map(String::as_str))
            .collect();
        let mut seen = HashSet::new();
        let mut keys = Vec::new();
        for step in &self.chains {
            for k in step.input_mapping.values() {
                if !produced.contains(k.as_str()) && seen.insert(k.as_str()) {
                    keys.push(k.as_str());
                }
            }
        }
        keys
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
    /// L64 全正向流式: single-output intermediate steps are driven through
    /// their own `stream_with_config` (tokens/callbacks fire per token instead
    /// of a blocking `invoke`) and the accumulated text feeds the next step;
    /// steps with multiple mapped outputs or non-text output fall back to
    /// `invoke_with_config`. The last chain's output is streamed token by token
    /// by delegating to its `stream()` method; its `output_mapping` names the
    /// global output keys that stream stands for (validated in
    /// `Self::stream_steps` so a broken mapping fails loudly, P2-3).
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.stream_steps(inputs, None).await
    }

    /// Stream execute the Chain with config propagation.
    ///
    /// Dispatches this chain's `on_chain_start`/`on_chain_end` and threads
    /// `config` into every sub-chain (single-output intermediates via stream,
    /// multi-output/non-text intermediates via invoke, last step via stream).
    async fn stream_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        let output_key = self.output_keys().first().map(|k| (*k).to_string());
        stream_chain_with_callbacks(
            self.name(),
            inputs,
            config.clone(),
            output_key,
            |inputs| async move { self.stream_steps(inputs, config).await },
        )
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
            .field("memory", &self.memory.is_some())
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

    /// L13: `input_keys` reports every *external* input. The second step maps the global
    /// key `topic` directly (via mapping), so it is a real caller-supplied input even
    /// though it is not the first step's input; the intermediate key `text`/`intermediate`
    /// is produced inside the chain and must NOT be reported.
    #[tokio::test]
    async fn test_sequential_chain_input_keys_union_across_steps() {
        let chain = SequentialChain::new()
            .add_chain_with_mapping(
                Arc::new(EchoChain::new("text", "intermediate")),
                HashMap::from([("text".to_string(), "query".to_string())]),
                HashMap::from([("intermediate".to_string(), "intermediate".to_string())]),
            )
            .add_chain_with_mapping(
                Arc::new(UppercaseChain {
                    input_key: "topic".to_string(),
                    output_key: "answer".to_string(),
                }),
                HashMap::from([
                    ("topic".to_string(), "topic".to_string()),
                    ("intermediate".to_string(), "intermediate".to_string()),
                ]),
                HashMap::from([("answer".to_string(), "final_result".to_string())]),
            );

        let mut keys = chain.input_keys();
        keys.sort();
        // `intermediate` is chain-internal (produced by step 1) and must be excluded;
        // `query` (step 1 mapped from global) and `topic` (step 2 mapped directly) are external.
        assert_eq!(keys, vec!["query", "topic"]);
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

    /// Minimal in-memory `BaseMemory` returning a fixed variable so L65's
    /// generic injection is observable without a concrete memory crate.
    struct StubMemory(HashMap<String, Value>);

    #[async_trait]
    impl BaseMemory for StubMemory {
        fn memory_variables(&self) -> Vec<&str> {
            self.0.keys().map(String::as_str).collect()
        }
        async fn load_memory_variables(
            &self,
            _inputs: &HashMap<String, String>,
        ) -> Result<HashMap<String, Value>, lc_memory::MemoryError> {
            Ok(self.0.clone())
        }
        async fn save_context(
            &mut self,
            _inputs: &HashMap<String, String>,
            _outputs: &HashMap<String, String>,
        ) -> Result<(), lc_memory::MemoryError> {
            Ok(())
        }
        async fn clear(&mut self) -> Result<(), lc_memory::MemoryError> {
            self.0.clear();
            Ok(())
        }
    }

    /// L65: `.with_memory<M>` accepts ANY `BaseMemory`, and its loaded variables
    /// are injected into the input state so a step's `input_mapping` can
    /// consume them — here the step consumes the injected `history` variable.
    #[tokio::test]
    async fn test_sequential_chain_with_memory_generic_injection() {
        let memory = StubMemory(HashMap::from([("history".to_string(), json!("HIST"))]));
        let chain = SequentialChain::new()
            .with_memory(memory)
            .add_chain_with_mapping(
                Arc::new(EchoChain::new("history", "seen")),
                HashMap::from([("history".to_string(), "history".to_string())]),
                HashMap::from([("seen".to_string(), "seen".to_string())]),
            );
        assert!(chain.injected_memory().is_some());
        assert!(format!("{chain:?}").contains("memory: true"));

        let inputs = HashMap::from([("unused".to_string(), json!("x"))]);
        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(result.get("seen").unwrap(), &json!("HIST"));
    }

    /// L64 全正向流式: an intermediate step is driven through its own stream
    /// (not a blocking invoke) and its accumulated text feeds the final step —
    /// the piped result must equal the sum of both transformations.
    #[tokio::test]
    async fn test_sequential_chain_stream_forward_streams_intermediates() {
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
}
