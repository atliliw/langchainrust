// lc-chains/src/router_chain/llm.rs
//! LLM-based routing chain.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use serde_json::Value;

use crate::base::{
    run_chain_with_callbacks, stream_chain_with_callbacks, BaseChain, ChainError, ChainResult,
    ChainStream,
};
use crate::BoxedChatModel;

use super::destination::RouteDestination;
use super::{route_tool, RouteDecision};

/// LLM Router Chain
///
/// Uses an LLM to intelligently determine the routing destination.
pub struct LLMRouterChain {
    /// LLM used for routing decisions.
    llm: BoxedChatModel,

    /// Route destinations.
    destinations: Vec<RouteDestination>,

    /// Default Chain.
    default_chain: Option<Arc<dyn BaseChain>>,

    /// Input key name.
    input_key: String,

    /// Chain name.
    name: String,

    /// Whether to print verbose information.
    verbose: bool,
}

impl LLMRouterChain {
    /// Create a new empty [`LLMRouterChain`] with the given LLM.
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            destinations: Vec::new(),
            default_chain: None,
            input_key: "input".to_string(),
            name: "llm_router_chain".to_string(),
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

    /// Build the LLM routing prompt.
    fn build_router_prompt(&self, input: &str) -> String {
        let mut prompt =
            String::from("Based on the user input, select the most appropriate handler.\n\n");
        prompt.push_str("Available handlers:\n");

        for (i, dest) in self.destinations.iter().enumerate() {
            prompt.push_str(&format!(
                "{}. {}: {}\n",
                i + 1,
                dest.name(),
                dest.description()
            ));
        }

        prompt.push_str("\nUser input: ");
        prompt.push_str(input);
        // P2-5: ask for a structured object rather than a bare name so the
        // decision (destination + reason) survives parsing reliably.
        prompt.push_str(
            "\n\nReturn only the chosen handler as JSON: {\"destination\": \"<handler name>\", \"reason\": \"<why this handler>\"}",
        );

        prompt
    }

    /// Use LLM to determine the route (P2-5: structured output).
    ///
    /// Binds `route_to_destination` so the provider emits a structured
    /// `{destination, reason}` tool-call argument instead of free text; the
    /// first tool call's parsed arguments win. Providers without tool binding
    /// (or a response without `tool_calls`) fall back to the same call's text,
    /// parsed leniently by [`RouteDecision::from_text`]. One LLM call, no
    /// retry.
    async fn route_with_llm(
        &self,
        input: &str,
        config: Option<RunnableConfig>,
    ) -> Result<RouteDecision, ChainError> {
        let prompt = self.build_router_prompt(input);

        let messages = vec![Message::human(&prompt)];

        let map_err = |e| ChainError::Nested {
            context: "LLM routing call failed".to_string(),
            source: Box::new(e),
        };

        let result = match self.llm.bind_tools(vec![route_tool()]) {
            Some(bound) => bound.chat(messages, config).await.map_err(map_err)?,
            None => self.llm.chat(messages, config).await.map_err(map_err)?,
        };

        if let Some(decision) = result
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .and_then(|call| call.parse_arguments::<RouteDecision>().ok())
        {
            return Ok(decision);
        }

        Ok(RouteDecision::from_text(&result.content))
    }

    /// Find a route destination by name.
    fn find_destination(&self, name: &str) -> Option<&RouteDestination> {
        let name_lower = name.to_lowercase();
        // 1. Exact case-insensitive match
        if let Some(dest) = self
            .destinations
            .iter()
            .find(|d| d.name().eq_ignore_ascii_case(name))
        {
            return Some(dest);
        }
        // 2. The LLM result starts or ends with the destination name
        self.destinations.iter().find(|d| {
            let d_lower = d.name().to_lowercase();
            name_lower.starts_with(&d_lower)
                || name_lower.ends_with(&d_lower)
                || name_lower
                    .split_whitespace()
                    .any(|word| word.eq_ignore_ascii_case(&d_lower))
        })
    }

    /// LLM routing takes priority over keyword matching.
    async fn select_route(
        &self,
        input: &str,
        config: Option<RunnableConfig>,
    ) -> Result<&RouteDestination, ChainError> {
        if self.destinations.is_empty() {
            return Err(ChainError::ExecutionError(
                "No route destinations configured".to_string(),
            ));
        }

        if self.destinations.len() == 1 {
            return Ok(&self.destinations[0]);
        }

        // Try LLM routing first (primary strategy).
        // P1-6: the LLM error/unknown-name is retained rather than swallowed, so
        // keyword fallback stays as a legitimate safety net but the final error
        // carries the real routing diagnostics.
        // P2-5: the LLM now returns a structured decision {destination, reason};
        // the reason is surfaced in verbose mode only.
        let llm_note: Option<String> = {
            let llm_result = self.route_with_llm(input, config).await;
            match llm_result {
                Ok(decision) => {
                    if let Some(reason) = &decision.reason {
                        if self.verbose {
                            println!("LLM route reason: {}", reason);
                        }
                    }
                    if let Some(dest) = self.find_destination(&decision.destination) {
                        return Ok(dest);
                    }
                    Some(format!(
                        "LLM returned an unknown route destination {:?}",
                        decision.destination
                    ))
                }
                Err(e) => Some(format!("LLM routing call failed: {}", e)),
            }
        };

        // Fallback: keyword matching (longest match first)
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
        if let Some((dest, _)) = best_match {
            return Ok(dest);
        }

        Err(ChainError::ExecutionError(format!(
            "No matching route destination found ({})",
            llm_note.unwrap_or_else(|| "LLM and keyword matching both failed".to_string())
        )))
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
            println!("\n=== LLMRouterChain execution ===");
            println!("Input: {}", input);
            println!("Route destination count: {}", self.destinations.len());
        }

        let route_result = self.select_route(input, config.clone()).await;

        let chain = match route_result {
            Ok(dest) => {
                if self.verbose {
                    println!("Routed to: {} ({})", dest.name(), dest.description());
                }
                dest.chain()
            }
            Err(e) => {
                if let Some(default) = &self.default_chain {
                    // 路由失败走默认链:不静默,记 error 日志说明原因,
                    // 避免调用方把 fallback 答案当成路由选择的正确结果
                    log::error!(
                        "routing failed, falling back to default chain (caller may receive an \
                         answer that does not match the input): {e}"
                    );
                    default
                } else {
                    return Err(e);
                }
            }
        };

        let result = chain.invoke_with_config(inputs, config).await?;

        if self.verbose {
            println!("=== LLMRouterChain complete ===\n");
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

        let route_result = self.select_route(input, config.clone()).await;

        let chain = match route_result {
            Ok(dest) => dest.chain(),
            Err(e) => {
                if let Some(default) = &self.default_chain {
                    default
                } else {
                    return Err(e);
                }
            }
        };

        chain.stream_with_config(inputs, config).await
    }
}

#[async_trait]
impl BaseChain for LLMRouterChain {
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
    /// Dispatches this chain's `on_chain_start`/`on_chain_end`, threads
    /// `config` into the routing LLM call, and into the routed destination
    /// chain via `invoke_with_config`.
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

    /// Stream execution for LLMRouterChain.
    ///
    /// The routing LLM call must complete first (to determine the destination),
    /// then delegates to the selected chain's `stream()` method.
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
