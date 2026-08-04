// lc-chains/src/sequential_chain.rs
//! Sequential Chain
//!
//! Execute multiple chains sequentially.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::base::{BaseChain, ChainError, ChainResult};

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

            let chain_output = step.chain.invoke(chain_inputs).await.map_err(|e| {
                ChainError::ExecutionError(format!(
                    "Step {} ({}) execution failed: {}",
                    step_index,
                    step.chain.name(),
                    e
                ))
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
