// lc-chains/src/router_chain/base.rs
//! Keyword-based [`RouterChain`] and shared routing helpers.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use lc_core::tools::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use serde_json::Value;

use crate::base::{
    run_chain_with_callbacks, stream_chain_with_callbacks, BaseChain, ChainError, ChainResult,
    ChainStream,
};

use super::destination::RouteDestination;

/// Router Chain
///
/// Automatically routes to different Chains based on input content.
pub struct RouterChain {
    /// Route destination list.
    destinations: Vec<RouteDestination>,

    /// Default Chain (used when no match is found).
    default_chain: Option<Arc<dyn BaseChain>>,

    /// Input key name.
    input_key: String,

    /// Chain name.
    name: String,

    /// Whether to print verbose information.
    verbose: bool,
}

impl RouterChain {
    /// Create a new empty [`RouterChain`].
    pub fn new() -> Self {
        Self {
            destinations: Vec::new(),
            default_chain: None,
            input_key: "input".to_string(),
            name: "router_chain".to_string(),
            verbose: false,
        }
    }

    /// Add a route destination.
    pub fn add_route(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        chain: Arc<dyn BaseChain>,
    ) -> Self {
        self.destinations
            .push(RouteDestination::new(name, description, chain));
        self
    }

    /// Add a route destination with a keyword list for keyword-based routing.
    pub fn add_route_with_keywords(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        chain: Arc<dyn BaseChain>,
        keywords: Vec<&str>,
    ) -> Self {
        self.destinations
            .push(RouteDestination::new(name, description, chain).with_keywords(keywords));
        self
    }

    /// Set the default chain used when no route matches.
    pub fn with_default(mut self, chain: Arc<dyn BaseChain>) -> Self {
        self.default_chain = Some(chain);
        self
    }

    /// Set the input key.
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set the chain name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get the route destinations.
    pub fn destinations(&self) -> &[RouteDestination] {
        &self.destinations
    }

    /// Get the default chain, if set.
    pub fn default_chain(&self) -> Option<&Arc<dyn BaseChain>> {
        self.default_chain.as_ref()
    }

    /// Keyword-based routing.
    ///
    /// Longest-match-first instead of first-match-wins.
    fn route_by_keywords(&self, input: &str) -> Option<&RouteDestination> {
        let mut best_match: Option<(&RouteDestination, usize)> = None;
        for dest in &self.destinations {
            for keyword in dest.keywords() {
                if input.contains(keyword) {
                    let len = keyword.len();
                    if best_match.is_none() || len > best_match.unwrap().1 {
                        best_match = Some((dest, len));
                    }
                }
            }
        }
        best_match.map(|(dest, _)| dest)
    }

    /// Select a route destination.
    fn select_route(&self, input: &str) -> Result<Option<&RouteDestination>, ChainError> {
        if let Some(dest) = self.route_by_keywords(input) {
            return Ok(Some(dest));
        }

        Ok(None)
    }

    /// Route to a destination/default chain and invoke it, threading `config`
    /// through `invoke_with_config` (never silently dropping it).
    async fn route_and_invoke(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        if self.verbose {
            println!("\n=== RouterChain execution ===");
            println!("Input: {}", input);
            println!("Route destination count: {}", self.destinations.len());
        }

        let route_result = self.select_route(input)?;

        let chain = match route_result {
            Some(dest) => {
                if self.verbose {
                    println!("Routed to: {} ({})", dest.name(), dest.description());
                }
                dest.chain()
            }
            None => {
                if let Some(default) = &self.default_chain {
                    if self.verbose {
                        println!("No keyword match, using default Chain");
                    }
                    default
                } else {
                    return Err(ChainError::ExecutionError(
                        "No matching route destination and no default Chain configured".to_string(),
                    ));
                }
            }
        };

        let result = chain.invoke_with_config(inputs, config).await?;

        if self.verbose {
            println!("=== RouterChain complete ===\n");
        }

        Ok(result)
    }

    /// Route to a destination/default chain and stream it, threading `config`
    /// through `stream_with_config`.
    async fn route_and_stream(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let route_result = self.select_route(input)?;

        let chain = match route_result {
            Some(dest) => dest.chain(),
            None => self.default_chain.as_ref().ok_or_else(|| {
                ChainError::ExecutionError(
                    "No matching route destination and no default Chain configured".to_string(),
                )
            })?,
        };

        chain.stream_with_config(inputs, config).await
    }
}

impl Default for RouterChain {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseChain for RouterChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        let mut seen = std::collections::HashSet::new();
        let mut result: Vec<&str> = Vec::new();

        for dest in &self.destinations {
            for key in dest.chain().output_keys() {
                if seen.insert(key.to_string()) {
                    result.push(key);
                }
            }
        }
        if let Some(default) = &self.default_chain {
            for key in default.output_keys() {
                if seen.insert(key.to_string()) {
                    result.push(key);
                }
            }
        }

        if result.is_empty() {
            vec!["output"]
        } else {
            result
        }
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.route_and_invoke(inputs, None).await
    }

    /// Execute the Chain with config propagation.
    ///
    /// Dispatches this chain's `on_chain_start`/`on_chain_end` and threads
    /// `config` into the routed destination chain via `invoke_with_config`.
    async fn invoke_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        run_chain_with_callbacks(self.name(), inputs, config.clone(), |inputs| async move {
            self.route_and_invoke(inputs, config).await
        })
        .await
    }

    /// Stream execution for RouterChain.
    ///
    /// After routing (keyword matching), delegates to the selected chain's
    /// `stream()` method.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.route_and_stream(inputs, None).await
    }

    /// Stream execute the Chain with config propagation.
    async fn stream_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        stream_chain_with_callbacks(self.name(), inputs, config.clone(), |inputs| async move {
            self.route_and_stream(inputs, config).await
        })
        .await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Structured routing decision returned by the LLM (P2-5).
///
/// Preferred source is the `route_to_destination` tool call's JSON arguments
/// (`{"destination": ..., "reason": ...}`). `from_text` is a lenient fallback
/// for providers without tool binding: it tries the same JSON object shape,
/// then a bare destination name.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RouteDecision {
    /// Handler name — must match a configured destination.
    pub destination: String,
    /// Optional explanation for the choice (used in verbose diagnostics).
    pub reason: Option<String>,
}

impl RouteDecision {
    /// Parse a lenient text reply: JSON object first, bare name fallback.
    pub(crate) fn from_text(text: &str) -> Self {
        let trimmed = text.trim();
        if let Ok(decision) = serde_json::from_str::<RouteDecision>(trimmed) {
            return decision;
        }
        Self {
            destination: trimmed.to_string(),
            reason: None,
        }
    }
}

/// Tool definition that forces the routing LLM to emit a structured
/// `{destination, reason}` object instead of free text.
pub(crate) fn route_tool() -> ToolDefinition {
    ToolDefinition::new(
        "route_to_destination",
        "根据用户输入选择最合适的处理 handler,返回目标名称与理由",
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "destination": { "type": "string", "description": "目标 handler 名称" },
            "reason": { "type": "string", "description": "选择该 handler 的理由" }
        },
        "required": ["destination"]
    }))
}
