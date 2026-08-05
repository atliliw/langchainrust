// lc-chains/src/router_chain.rs
//! Router Chain
//!
//! Automatically routes to different Chains based on input content.

use async_trait::async_trait;
use lc_core::language_models::LLMResult;
use lc_core::{BaseChatModel, Runnable};
use lc_schema::Message;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream};

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

        let result = chain.invoke(inputs).await?;

        if self.verbose {
            println!("=== RouterChain complete ===\n");
        }

        Ok(result)
    }

    /// Stream execution for RouterChain.
    ///
    /// After routing (keyword matching), delegates to the selected chain's
    /// `stream()` method.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let route_result = self.select_route(input)?;

        let chain = match route_result {
            Some(dest) => dest.chain(),
            None => self
                .default_chain
                .as_ref()
                .ok_or_else(|| ChainError::ExecutionError(
                    "No matching route destination and no default Chain configured".to_string(),
                ))?,
        };

        chain.stream(inputs).await
    }

    fn name(&self) -> &str {
        &self.name
    }
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

impl<M: BaseChatModel> LLMRouterChain<M> {
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
        prompt
            .push_str("\n\nReturn only the name of the most appropriate handler (no explanation).");

        prompt
    }

    /// Use LLM to determine the route.
    async fn route_with_llm(&self, input: &str) -> Result<String, ChainError> {
        let prompt = self.build_router_prompt(input);

        let messages = vec![Message::human(&prompt)];

        let result = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        Ok(result.content.trim().to_string())
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
    async fn select_route(&self, input: &str) -> Result<&RouteDestination, ChainError> {
        if self.destinations.is_empty() {
            return Err(ChainError::ExecutionError(
                "No route destinations configured".to_string(),
            ));
        }

        if self.destinations.len() == 1 {
            return Ok(&self.destinations[0]);
        }

        // Try LLM routing first (primary strategy)
        let llm_result = self.route_with_llm(input).await;
        match llm_result {
            Ok(route_name) => {
                if let Some(dest) = self.find_destination(&route_name) {
                    return Ok(dest);
                }
            }
            Err(_) => {
                // LLM call failed; fall through to keyword matching
            }
        }

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

        Err(ChainError::ExecutionError(
            "No matching route destination found (LLM and keyword matching both failed)"
                .to_string(),
        ))
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

        let route_result = self.select_route(input).await;

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

        let result = chain.invoke(inputs).await?;

        if self.verbose {
            println!("=== LLMRouterChain complete ===\n");
        }

        Ok(result)
    }

    /// Stream execution for LLMRouterChain.
    ///
    /// The routing LLM call must complete first (to determine the destination),
    /// then delegates to the selected chain's `stream()` method.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let route_result = self.select_route(input).await;

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

        chain.stream(inputs).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}
