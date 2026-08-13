// lc-chains/src/router_chain.rs
//! Router Chain
//!
//! Automatically routes to different Chains based on input content.

use async_trait::async_trait;
use lc_core::language_models::LLMResult;
use lc_core::runnables::RunnableConfig;
use lc_core::tools::ToolDefinition;
use lc_core::{BaseChatModel, Runnable};
use lc_schema::Message;
use serde::Deserialize;
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::base::{
    run_chain_with_callbacks, stream_chain_with_callbacks, BaseChain, ChainError, ChainResult,
    ChainStream,
};

/// Route destination.
pub struct RouteDestination {
    /// Destination name.
    name: String,
    /// Destination description (used for routing decisions).
    description: String,
    /// Destination Chain.
    chain: Arc<dyn BaseChain>,
    /// Keyword list (used for keyword-based routing).
    keywords: Vec<String>,
}

impl RouteDestination {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        chain: Arc<dyn BaseChain>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            chain,
            keywords: Vec::new(),
        }
    }

    pub fn with_keywords(mut self, keywords: Vec<&str>) -> Self {
        self.keywords = keywords.into_iter().map(String::from).collect();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn chain(&self) -> &Arc<dyn BaseChain> {
        &self.chain
    }

    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }
}

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
    pub fn new() -> Self {
        Self {
            destinations: Vec::new(),
            default_chain: None,
            input_key: "input".to_string(),
            name: "router_chain".to_string(),
            verbose: false,
        }
    }

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

    pub fn with_default(mut self, chain: Arc<dyn BaseChain>) -> Self {
        self.default_chain = Some(chain);
        self
    }

    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn destinations(&self) -> &[RouteDestination] {
        &self.destinations
    }

    pub fn default_chain(&self) -> Option<&Arc<dyn BaseChain>> {
        self.default_chain.as_ref()
    }

    /// Keyword-based routing.
    ///
    /// Longest-match-first instead of first-match-wins.
    fn route_by_keywords(&self, input: &str) -> Option<&RouteDestination> {
        let mut best_match: Option<(&RouteDestination, usize)> = None;
        for dest in &self.destinations {
            for keyword in &dest.keywords {
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
    fn from_text(text: &str) -> Self {
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
fn route_tool() -> ToolDefinition {
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

/// LLM Router Chain
///
/// Uses an LLM to intelligently determine the routing destination.
pub struct LLMRouterChain<M: BaseChatModel> {
    /// LLM used for routing decisions.
    llm: M,

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

impl<M: BaseChatModel + Send + Sync + 'static> LLMRouterChain<M> {
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            destinations: Vec::new(),
            default_chain: None,
            input_key: "input".to_string(),
            name: "llm_router_chain".to_string(),
            verbose: false,
        }
    }

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

    pub fn with_default(mut self, chain: Arc<dyn BaseChain>) -> Self {
        self.default_chain = Some(chain);
        self
    }

    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn destinations(&self) -> &[RouteDestination] {
        &self.destinations
    }

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
                    if self.verbose {
                        println!("Routing failed: {}, using default Chain", e);
                    }
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
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for LLMRouterChain<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use std::pin::Pin;
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockError(String);
    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for MockError {}

    /// Mock chat model that returns a canned `LLMResult`. `supports_tools`
    /// controls whether `bind_tools` yields a bound copy (tool-call path) or
    /// `None` (plain-text fallback path) — so both P2-5 branches are testable.
    #[derive(Clone)]
    struct MockRouterLLM {
        response: LLMResult,
        supports_tools: bool,
    }

    impl MockRouterLLM {
        fn tools(response: LLMResult) -> Self {
            Self {
                response,
                supports_tools: true,
            }
        }
        fn plain(response: LLMResult) -> Self {
            Self {
                response,
                supports_tools: false,
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockRouterLLM {
        type Error = MockError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(self.response.clone())
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockRouterLLM {
        fn model_name(&self) -> &str {
            "mock"
        }
        fn get_num_tokens(&self, t: &str) -> usize {
            t.len()
        }
        fn with_temperature(self, _: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _: usize) -> Self {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for MockRouterLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(self.response.clone())
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            let tokens = [Ok(self.response.content.clone())];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
        fn bind_tools(
            &self,
            _tools: Vec<ToolDefinition>,
        ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
            if self.supports_tools {
                Some(Box::new(self.clone()))
            } else {
                None
            }
        }
    }

    /// Simple destination chain that echoes the input under `output`.
    struct EchoChain;

    #[async_trait]
    impl BaseChain for EchoChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }
        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let mut result = HashMap::new();
            if let Some(v) = inputs.get("input") {
                result.insert("output".to_string(), v.clone());
            }
            Ok(result)
        }
    }

    fn router_with(llm: MockRouterLLM) -> LLMRouterChain<MockRouterLLM> {
        LLMRouterChain::new(llm)
            .add_route(
                "math",
                "handles mathematical questions",
                Arc::new(EchoChain),
            )
            .add_route(
                "science",
                "handles scientific questions",
                Arc::new(EchoChain),
            )
    }

    fn decision_response(destination: &str, reason: &str) -> LLMResult {
        LLMResult {
            content: String::new(),
            model: "mock".to_string(),
            token_usage: None,
            tool_calls: Some(vec![lc_core::tools::ToolCall::new(
                "call_1",
                "route_to_destination",
                format!(
                    r#"{{"destination": "{}", "reason": "{}"}}"#,
                    destination, reason
                ),
            )]),
            thinking_content: None,
        }
    }

    fn text_response(content: &str) -> LLMResult {
        LLMResult {
            content: content.to_string(),
            model: "mock".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        }
    }

    /// P2-5: the routing LLM's `route_to_destination` tool-call arguments (the
    /// structured `{destination, reason}` object) drive the route.
    #[tokio::test]
    async fn test_llm_router_routes_via_tool_call() {
        let chain = router_with(MockRouterLLM::tools(decision_response(
            "math",
            "user asked a calculation",
        )));
        let inputs = HashMap::from([("input".to_string(), json!("what is 2 + 2?"))]);

        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(
            result.get("output").unwrap(),
            &json!("what is 2 + 2?"),
            "routed to the math destination"
        );
    }

    /// P2-5: a provider without tool binding falls back to the same call's
    /// text, parsed as a JSON `{destination, reason}` object.
    #[tokio::test]
    async fn test_llm_router_json_text_fallback() {
        let chain = router_with(MockRouterLLM::plain(text_response(
            r#"{"destination": "science", "reason": "explains a phenomenon"}"#,
        )));
        let inputs = HashMap::from([("input".to_string(), json!("why is the sky blue?"))]);

        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(
            result.get("output").unwrap(),
            &json!("why is the sky blue?"),
            "routed to the science destination"
        );
    }

    /// P2-5: a bare destination name still works (lenient text fallback).
    #[tokio::test]
    async fn test_llm_router_bare_name_fallback() {
        let chain = router_with(MockRouterLLM::plain(text_response("math")));
        let inputs = HashMap::from([("input".to_string(), json!("calculate something"))]);

        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(result.get("output").unwrap(), &json!("calculate something"));
    }

    /// P2-5: an unknown destination from the LLM does not abort routing — the
    /// real routing diagnostics stay in the error (P1-6).
    #[tokio::test]
    async fn test_llm_router_unknown_destination_reports_diagnostics() {
        let chain = router_with(MockRouterLLM::tools(decision_response(
            "physics", "closest",
        )));
        let inputs = HashMap::from([("input".to_string(), json!("what is 2+2"))]);

        let err = match chain.invoke(inputs).await {
            Ok(_) => panic!("expected a routing failure"),
            Err(e) => e,
        };
        let msg = format!("{err:?}");
        assert!(
            msg.contains("unknown route destination"),
            "expected the LLM diagnostics in the error, got: {msg}"
        );
    }

    /// P2-5: when the LLM names a nonexistent destination but a keyword
    /// matches, the keyword fallback still lands a destination.
    #[tokio::test]
    async fn test_llm_router_unknown_destination_keyword_still_routes() {
        let chain = LLMRouterChain::new(MockRouterLLM::tools(decision_response(
            "physics", "nonsense",
        )))
        .add_route_with_keywords(
            "math",
            "handles mathematical questions",
            Arc::new(EchoChain),
            vec!["calculate", "2+2"],
        )
        .add_route(
            "science",
            "handles scientific questions",
            Arc::new(EchoChain),
        );

        let inputs = HashMap::from([("input".to_string(), json!("please calculate 2+2"))]);
        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(
            result.get("output").unwrap(),
            &json!("please calculate 2+2")
        );
    }

    #[test]
    fn test_route_decision_from_text_json() {
        let d = RouteDecision::from_text(r#"{"destination": "math", "reason": "why"}"#);
        assert_eq!(d.destination, "math");
        assert_eq!(d.reason.as_deref(), Some("why"));
    }

    #[test]
    fn test_route_decision_from_text_bare_name() {
        let d = RouteDecision::from_text("  science  ");
        assert_eq!(d.destination, "science");
        assert!(d.reason.is_none());
    }

    #[test]
    fn test_route_tool_schema() {
        let tool = route_tool();
        assert_eq!(tool.function.name, "route_to_destination");
        assert!(tool.function.parameters.is_some());
    }
}
