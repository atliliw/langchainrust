// src/chains/router_chain.rs
//! Router Chain
//!
//! Automatically routes to different Chains based on input content.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;

use super::base::{BaseChain, ChainResult, ChainError};
use crate::BaseChatModel;
use crate::schema::Message;
use crate::Runnable;

/// Route destination
pub struct RouteDestination {
    /// Destination name
    name: String,
    /// Destination description (used for routing decisions)
    description: String,
    /// Destination Chain
    chain: Arc<dyn BaseChain>,
    /// Keyword list (used for keyword-based routing)
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
///
/// # Example
/// ```ignore
/// use langchainrust::{RouterChain, LLMChain, OpenAIChat};
///
/// let llm = OpenAIChat::new(config);
///
/// let math_chain = LLMChain::new(llm.clone(), "Calculate: {question}");
/// let code_chain = LLMChain::new(llm.clone(), "Programming question: {question}");
/// let general_chain = LLMChain::new(llm, "Answer: {question}");
///
/// let router = RouterChain::new()
///     .add_route("math", "Handle math calculation problems", Arc::new(math_chain))
///     .add_route("code", "Handle programming-related questions", Arc::new(code_chain))
///     .with_default(Arc::new(general_chain));
///
/// // "What is 1+1?" -> automatically routes to math_chain
/// // "How to write Rust?" -> automatically routes to code_chain
/// ```
pub struct RouterChain {
    /// Route destination list
    destinations: Vec<RouteDestination>,

    /// Default Chain (used when no match is found)
    default_chain: Option<Arc<dyn BaseChain>>,

    /// Input key name
    input_key: String,

    /// Chain name
    name: String,

    /// Whether to print verbose information
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
        self.destinations.push(RouteDestination::new(name, description, chain));
        self
    }
    
    pub fn add_route_with_keywords(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        chain: Arc<dyn BaseChain>,
        keywords: Vec<&str>,
    ) -> Self {
        self.destinations.push(
            RouteDestination::new(name, description, chain).with_keywords(keywords)
        );
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
    
    /// Keyword-based routing
    ///
    /// Checks if the input contains any destination keywords and returns the matching destination.
    fn route_by_keywords(&self, input: &str) -> Option<&RouteDestination> {
        for dest in &self.destinations {
            for keyword in &dest.keywords {
                if input.contains(keyword) {
                    return Some(dest);
                }
            }
        }
        None
    }
    
    /// Select a route destination
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
        if let Some(default) = &self.default_chain {
            default.output_keys()
        } else if let Some(first) = self.destinations.first() {
            first.chain().output_keys()
        } else {
            vec!["output"]
        }
    }
    
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;
        
        let input = inputs.get(&self.input_key)
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
                        "No matching route destination and no default Chain configured".to_string()
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
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// LLM Router Chain
///
/// Uses an LLM to intelligently determine the routing destination.
pub struct LLMRouterChain<M: BaseChatModel> {
    /// LLM used for routing decisions
    llm: M,

    /// Route destinations
    destinations: Vec<RouteDestination>,

    /// Default Chain
    default_chain: Option<Arc<dyn BaseChain>>,

    /// Input key name
    input_key: String,

    /// Chain name
    name: String,

    /// Whether to print verbose information
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
        self.destinations.push(RouteDestination::new(name, description, chain));
        self
    }
    
    pub fn add_route_with_keywords(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        chain: Arc<dyn BaseChain>,
        keywords: Vec<&str>,
    ) -> Self {
        self.destinations.push(
            RouteDestination::new(name, description, chain).with_keywords(keywords)
        );
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
    
    /// Build the LLM routing prompt
    fn build_router_prompt(&self, input: &str) -> String {
        let mut prompt = String::from("Based on the user input, select the most appropriate handler.\n\n");
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
        prompt.push_str("\n\nReturn only the name of the most appropriate handler (no explanation).");
        
        prompt
    }
    
    /// Use LLM to determine the route
    async fn route_with_llm(&self, input: &str) -> Result<String, ChainError> {
        let prompt = self.build_router_prompt(input);
        
        let messages = vec![Message::human(&prompt)];
        
        let result = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;
        
        Ok(result.content.trim().to_string())
    }
    
    /// Find a route destination by name
    fn find_destination(&self, name: &str) -> Option<&RouteDestination> {
        self.destinations.iter().find(|d| {
            d.name().eq_ignore_ascii_case(name) || 
            name.contains(d.name()) ||
            d.name().contains(name)
        })
    }
    
    /// Try keyword matching first, fall back to LLM routing
    async fn select_route(&self, input: &str) -> Result<&RouteDestination, ChainError> {
        if self.destinations.is_empty() {
            return Err(ChainError::ExecutionError("No route destinations configured".to_string()));
        }
        
        if self.destinations.len() == 1 {
            return Ok(&self.destinations[0]);
        }
        
        // Try keyword matching first
        for dest in &self.destinations {
            for keyword in dest.keywords() {
                if input.contains(keyword) {
                    return Ok(dest);
                }
            }
        }
        
        // Fall back to LLM routing
        let llm_result = self.route_with_llm(input).await?;
        
        self.find_destination(&llm_result)
            .ok_or_else(|| ChainError::ExecutionError(
                format!("LLM returned unknown route destination '{}'", llm_result)
            ))
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for LLMRouterChain<M>
where
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }
    
    fn output_keys(&self) -> Vec<&str> {
        if let Some(default) = &self.default_chain {
            default.output_keys()
        } else if let Some(first) = self.destinations.first() {
            first.chain().output_keys()
        } else {
            vec!["output"]
        }
    }
    
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;
        
        let input = inputs.get(&self.input_key)
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
    
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct MockChain {
        name: String,
        output: String,
    }
    
    impl MockChain {
        fn new(name: impl Into<String>, output: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                output: output.into(),
            }
        }
    }
    
    #[async_trait]
    impl BaseChain for MockChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }
        
        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }
        
        async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let mut result = HashMap::new();
            result.insert("output".to_string(), Value::String(self.output.clone()));
            Ok(result)
        }
        
        fn name(&self) -> &str {
            &self.name
        }
    }
    
    #[test]
    fn test_router_chain_new() {
        let router = RouterChain::new();
        assert_eq!(router.name(), "router_chain");
        assert_eq!(router.destinations().len(), 0);
    }
    
    #[test]
    fn test_router_chain_add_route() {
        let chain = Arc::new(MockChain::new("math", "数学答案"));
        
        let router = RouterChain::new()
            .add_route("数学", "处理数学问题", chain);
        
        assert_eq!(router.destinations().len(), 1);
        assert_eq!(router.destinations()[0].name(), "数学");
    }
    
    #[test]
    fn test_router_chain_with_keywords() {
        let chain = Arc::new(MockChain::new("math", "数学答案"));
        
        let router = RouterChain::new()
            .add_route_with_keywords(
                "数学", 
                "处理数学问题", 
                chain,
                vec!["计算", "加", "减", "乘", "除"]
            );
        
        assert_eq!(router.destinations()[0].keywords().len(), 5);
    }
    
    #[test]
    fn test_route_by_keywords() {
        let math_chain = Arc::new(MockChain::new("math", "数学答案"));
        let code_chain = Arc::new(MockChain::new("code", "编程答案"));
        
        let router = RouterChain::new()
            .add_route_with_keywords("数学", "处理数学问题", math_chain, vec!["计算", "加", "数学"])
            .add_route_with_keywords("编程", "处理编程问题", code_chain, vec!["代码", "Rust", "编程"]);
        
        let dest = router.route_by_keywords("帮我计算一下");
        assert!(dest.is_some());
        assert_eq!(dest.unwrap().name(), "数学");
        
        let dest2 = router.route_by_keywords("如何写Rust代码");
        assert!(dest2.is_some());
        assert_eq!(dest2.unwrap().name(), "编程");
        
        let dest3 = router.route_by_keywords("你好");
        assert!(dest3.is_none());
    }
    
    #[tokio::test]
    async fn test_router_chain_invoke_keywords_match() {
        let math_chain = Arc::new(MockChain::new("math", "数学答案: 42"));
        let code_chain = Arc::new(MockChain::new("code", "编程答案"));
        let default_chain = Arc::new(MockChain::new("default", "通用答案"));
        
        let router = RouterChain::new()
            .add_route_with_keywords("数学", "处理数学问题", math_chain, vec!["计算", "加", "数学"])
            .add_route_with_keywords("编程", "处理编程问题", code_chain, vec!["代码", "Rust"])
            .with_default(default_chain);
        
        let inputs = HashMap::from([
            ("input".to_string(), Value::String("帮我计算一下".to_string()))
        ]);
        
        let result = router.invoke(inputs).await.unwrap();
        let output = result.get("output").unwrap().as_str().unwrap();
        
        assert!(output.contains("数学"));
    }
    
    #[tokio::test]
    async fn test_router_chain_invoke_default() {
        let math_chain = Arc::new(MockChain::new("math", "数学答案"));
        let default_chain = Arc::new(MockChain::new("default", "通用答案"));
        
        let router = RouterChain::new()
            .add_route_with_keywords("数学", "处理数学问题", math_chain, vec!["计算", "数学"])
            .with_default(default_chain);
        
        let inputs = HashMap::from([
            ("input".to_string(), Value::String("你好".to_string()))
        ]);
        
        let result = router.invoke(inputs).await.unwrap();
        let output = result.get("output").unwrap().as_str().unwrap();
        
        assert!(output.contains("通用"));
    }
    
    #[tokio::test]
    async fn test_router_chain_no_match_no_default() {
        let math_chain = Arc::new(MockChain::new("math", "数学答案"));
        
        let router = RouterChain::new()
            .add_route_with_keywords("数学", "处理数学问题", math_chain, vec!["计算", "数学"]);
        
        let inputs = HashMap::from([
            ("input".to_string(), Value::String("你好".to_string()))
        ]);
        
        let result = router.invoke(inputs).await;
        assert!(result.is_err());
    }
}