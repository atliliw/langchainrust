// src/chains/router_chain.rs
//! Router Chain
//!
//! 根据输入内容自动路由到不同的 Chain。

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;

use super::base::{BaseChain, ChainResult, ChainError};
use crate::language_models::OpenAIChat;
use crate::schema::Message;
use crate::Runnable;

/// 路由目标
pub struct RouteDestination {
    /// 目标名称
    name: String,
    /// 目标描述（用于路由判断）
    description: String,
    /// 目标 Chain
    chain: Arc<dyn BaseChain>,
    /// 关键词列表（用于关键词匹配路由）
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
/// 根据输入内容自动路由到不同的 Chain。
///
/// # 示例
/// ```ignore
/// use langchainrust::{RouterChain, LLMChain, OpenAIChat};
///
/// let llm = OpenAIChat::new(config);
///
/// let math_chain = LLMChain::new(llm.clone(), "计算: {question}");
/// let code_chain = LLMChain::new(llm.clone(), "编程问题: {question}");
/// let general_chain = LLMChain::new(llm, "回答: {question}");
///
/// let router = RouterChain::new()
///     .add_route("数学", "处理数学计算问题", Arc::new(math_chain))
///     .add_route("编程", "处理编程相关问题", Arc::new(code_chain))
///     .with_default(Arc::new(general_chain));
///
/// // "1+1等于几？" → 自动路由到 math_chain
/// // "如何写 Rust？" → 自动路由到 code_chain
/// ```
pub struct RouterChain {
    /// 路由目标列表
    destinations: Vec<RouteDestination>,
    
    /// 默认 Chain（没有匹配时使用）
    default_chain: Option<Arc<dyn BaseChain>>,
    
    /// 输入键名
    input_key: String,
    
    /// Chain 名称
    name: String,
    
    /// 是否打印详细信息
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
    
    /// 关键词匹配路由
    ///
    /// 检查输入是否包含目标的关键词，返回匹配的目标。
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
    
    /// 选择路由目标
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
            println!("\n=== RouterChain 执行 ===");
            println!("输入: {}", input);
            println!("路由目标数量: {}", self.destinations.len());
        }
        
        let route_result = self.select_route(input)?;
        
        let chain = match route_result {
            Some(dest) => {
                if self.verbose {
                    println!("路由到: {} ({})", dest.name(), dest.description());
                }
                dest.chain()
            }
            None => {
                if let Some(default) = &self.default_chain {
                    if self.verbose {
                        println!("关键词未匹配，使用默认 Chain");
                    }
                    default
                } else {
                    return Err(ChainError::ExecutionError(
                        "无法匹配路由目标，且没有配置默认 Chain".to_string()
                    ));
                }
            }
        };
        
        let result = chain.invoke(inputs).await?;
        
        if self.verbose {
            println!("=== RouterChain 完成 ===\n");
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// LLM Router Chain
///
/// 使用 LLM 智能判断路由目标。
pub struct LLMRouterChain {
    /// LLM 用于路由判断
    llm: OpenAIChat,
    
    /// 路由目标列表
    destinations: Vec<RouteDestination>,
    
    /// 默认 Chain
    default_chain: Option<Arc<dyn BaseChain>>,
    
    /// 输入键名
    input_key: String,
    
    /// Chain 名称
    name: String,
    
    /// 是否打印详细信息
    verbose: bool,
}

impl LLMRouterChain {
    pub fn new(llm: OpenAIChat) -> Self {
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
    
    /// 构建 LLM 路由提示词
    fn build_router_prompt(&self, input: &str) -> String {
        let mut prompt = String::from("根据用户输入，选择最合适的处理方式。\n\n");
        prompt.push_str("可选的处理方式：\n");
        
        for (i, dest) in self.destinations.iter().enumerate() {
            prompt.push_str(&format!(
                "{}. {}: {}\n",
                i + 1,
                dest.name(),
                dest.description()
            ));
        }
        
        prompt.push_str("\n用户输入：");
        prompt.push_str(input);
        prompt.push_str("\n\n请只返回最合适的处理方式的名称（不要解释）。");
        
        prompt
    }
    
    /// 使用 LLM 判断路由
    async fn route_with_llm(&self, input: &str) -> Result<String, ChainError> {
        let prompt = self.build_router_prompt(input);
        
        let messages = vec![Message::human(&prompt)];
        
        let result = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM 路由判断失败: {}", e)))?;
        
        Ok(result.content.trim().to_string())
    }
    
    /// 根据名称查找路由目标
    fn find_destination(&self, name: &str) -> Option<&RouteDestination> {
        self.destinations.iter().find(|d| {
            d.name().eq_ignore_ascii_case(name) || 
            name.contains(d.name()) ||
            d.name().contains(name)
        })
    }
    
    /// 先尝试关键词匹配，失败则使用 LLM
    async fn select_route(&self, input: &str) -> Result<&RouteDestination, ChainError> {
        if self.destinations.is_empty() {
            return Err(ChainError::ExecutionError("没有配置路由目标".to_string()));
        }
        
        if self.destinations.len() == 1 {
            return Ok(&self.destinations[0]);
        }
        
        // 先尝试关键词匹配
        for dest in &self.destinations {
            for keyword in dest.keywords() {
                if input.contains(keyword) {
                    return Ok(dest);
                }
            }
        }
        
        // 使用 LLM 判断
        let llm_result = self.route_with_llm(input).await?;
        
        self.find_destination(&llm_result)
            .ok_or_else(|| ChainError::ExecutionError(
                format!("LLM 返回的路由目标 '{}' 不存在", llm_result)
            ))
    }
}

#[async_trait]
impl BaseChain for LLMRouterChain {
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
            println!("\n=== LLMRouterChain 执行 ===");
            println!("输入: {}", input);
            println!("路由目标数量: {}", self.destinations.len());
        }
        
        let route_result = self.select_route(input).await;
        
        let chain = match route_result {
            Ok(dest) => {
                if self.verbose {
                    println!("路由到: {} ({})", dest.name(), dest.description());
                }
                dest.chain()
            }
            Err(e) => {
                if let Some(default) = &self.default_chain {
                    if self.verbose {
                        println!("路由失败: {}, 使用默认 Chain", e);
                    }
                    default
                } else {
                    return Err(e);
                }
            }
        };
        
        let result = chain.invoke(inputs).await?;
        
        if self.verbose {
            println!("=== LLMRouterChain 完成 ===\n");
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
        
        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
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