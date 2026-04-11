// src/chains/llm_chain.rs
//! LLM Chain
//!
//! 最基础的 Chain，组合 Prompt 和 LLM。

use async_trait::async_trait;
use std::collections::HashMap;
use serde_json::Value;

use super::base::{BaseChain, ChainResult, ChainError};
use crate::language_models::OpenAIChat;
use crate::schema::Message;
use crate::Runnable;

/// LLM Chain
/// 
/// 组合 Prompt 模板和 LLM，是最基础的 Chain。
/// 
/// # 示例
/// ```ignore
/// use langchainrust::{LLMChain, OpenAIChat, OpenAIConfig};
/// 
/// let llm = OpenAIChat::new(config);
/// let chain = LLMChain::new(llm, "{question}");
/// 
/// let inputs = HashMap::from([("question".to_string(), "什么是Rust?".into())]);
/// let result = chain.invoke(inputs).await?;
/// ```
pub struct LLMChain {
    /// LLM 客户端
    llm: OpenAIChat,
    
    /// Prompt 模板
    prompt_template: String,
    
    /// 输入键名
    input_key: String,
    
    /// 输出键名
    output_key: String,
    
    /// Chain 名称
    name: String,
}

impl LLMChain {
    /// 创建新的 LLMChain
    /// 
    /// # 参数
    /// * `llm` - LLM 客户端
    /// * `prompt_template` - Prompt 模板字符串，使用 {variable} 作为占位符
    pub fn new(llm: OpenAIChat, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
            input_key: "question".to_string(),
            output_key: "text".to_string(),
            name: "llm_chain".to_string(),
        }
    }
    
    /// 设置输入键名
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }
    
    /// 设置输出键名
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }
    
    /// 设置 Chain 名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    /// 渲染 Prompt 模板
    fn render_prompt(&self, inputs: &HashMap<String, Value>) -> Result<String, ChainError> {
        let mut prompt = self.prompt_template.clone();
        
        for (key, value) in inputs {
            // 替换 {key} 占位符
            let placeholder = format!("{{{}}}", key);
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            prompt = prompt.replace(&placeholder, &value_str);
        }
        
        Ok(prompt)
    }
}

#[async_trait]
impl BaseChain for LLMChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }
    
    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }
    
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        // 验证输入
        self.validate_inputs(&inputs)?;
        
        // 渲染 Prompt
        let prompt = self.render_prompt(&inputs)?;
        
        // 调用 LLM
        let messages = vec![Message::human(&prompt)];
        let result = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM 调用失败: {}", e)))?;
        
        // 构造输出
        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), Value::String(result.content));
        
        Ok(output)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// LLMChain Builder
/// 
/// 方便构建 LLMChain。
pub struct LLMChainBuilder {
    llm: OpenAIChat,
    prompt_template: String,
    input_key: Option<String>,
    output_key: Option<String>,
    name: Option<String>,
}

impl LLMChainBuilder {
    pub fn new(llm: OpenAIChat, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
            input_key: None,
            output_key: None,
            name: None,
        }
    }
    
    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = Some(key.into());
        self
    }
    
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }
    
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    pub fn build(self) -> LLMChain {
        let mut chain = LLMChain::new(self.llm, self.prompt_template);
        
        if let Some(key) = self.input_key {
            chain = chain.with_input_key(key);
        }
        
        if let Some(key) = self.output_key {
            chain = chain.with_output_key(key);
        }
        
        if let Some(name) = self.name {
            chain = chain.with_name(name);
        }
        
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpenAIConfig;
    
    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".to_string(),
            base_url: "https://api.openai-proxy.org/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            streaming: false,
            organization: None,
            frequency_penalty: None,
            max_tokens: None,
            presence_penalty: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
        }
    }
    
    #[test]
    fn test_render_prompt() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, "问题: {question}");
        
        let inputs = HashMap::from([
            ("question".to_string(), Value::String("什么是Rust?".to_string()))
        ]);
        
        let prompt = chain.render_prompt(&inputs).unwrap();
        assert_eq!(prompt, "问题: 什么是Rust?");
    }
    
    #[test]
    fn test_render_prompt_multiple_vars() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, "名字: {name}, 年龄: {age}");
        
        let inputs = HashMap::from([
            ("name".to_string(), Value::String("张三".to_string())),
            ("age".to_string(), Value::Number(25.into())),
        ]);
        
        let prompt = chain.render_prompt(&inputs).unwrap();
        assert_eq!(prompt, "名字: 张三, 年龄: 25");
    }
    
    /// 真实 API 测试 - 简单问题
    /// 运行: cargo test test_llm_chain_simple -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_llm_chain_simple() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, "请用一句话回答: {question}");
        
        let inputs = HashMap::from([
            ("question".to_string(), Value::String("什么是 Rust?".to_string()))
        ]);
        
        println!("\n=== 测试 LLMChain - 简单问题 ===");
        let result = chain.invoke(inputs).await.unwrap();
        
        println!("输出: {:?}", result);
        assert!(result.contains_key("text"));
        assert!(!result.get("text").unwrap().as_str().unwrap().is_empty());
    }
    
    /// 真实 API 测试 - 多变量模板
    /// 运行: cargo test test_llm_chain_template -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_llm_chain_template() {
        let llm = OpenAIChat::new(create_test_config());
        let chain = LLMChain::new(llm, 
            "请用{style}的语气回答问题: {question}"
        );
        
        let inputs = HashMap::from([
            ("style".to_string(), Value::String("幽默".to_string())),
            ("question".to_string(), Value::String("什么是编程?".to_string()))
        ]);
        
        println!("\n=== 测试 LLMChain - 多变量模板 ===");
        let result = chain.invoke(inputs).await.unwrap();
        
        println!("输出: {:?}", result);
        assert!(result.contains_key("text"));
    }
    
    /// 真实 API 测试 - 使用 Builder
    /// 运行: cargo test test_llm_chain_builder -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_llm_chain_builder() {
        let llm = OpenAIChat::new(create_test_config());
        
        let chain = LLMChainBuilder::new(llm, "翻译以下内容到{language}: {text}")
            .input_key("text")
            .output_key("translation")
            .name("translator")
            .build();
        
        let inputs = HashMap::from([
            ("language".to_string(), Value::String("英文".to_string())),
            ("text".to_string(), Value::String("你好，世界".to_string()))
        ]);
        
        println!("\n=== 测试 LLMChain - Builder ===");
        let result = chain.invoke(inputs).await.unwrap();
        
        println!("输出: {:?}", result);
        assert!(result.contains_key("translation"));
    }
}