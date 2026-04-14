// src/chains/conversation_chain.rs
//! Conversation Chain
//!
//! 带记忆的对话 Chain，支持多轮对话。

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;

use super::base::{BaseChain, ChainResult, ChainError};
use crate::language_models::OpenAIChat;
use crate::memory::{ConversationBufferMemory, BaseMemory};
use crate::schema::Message;
use crate::Runnable;
use tokio::sync::Mutex;

/// Conversation Chain
///
/// 带记忆的对话 Chain，自动保存和加载对话历史。
///
/// # 示例
/// ```ignore
/// use langchainrust::{ConversationChain, OpenAIChat, OpenAIConfig, ConversationBufferMemory};
///
/// let llm = OpenAIChat::new(config);
/// let memory = ConversationBufferMemory::new();
///
/// let chain = ConversationChain::new(llm, memory);
///
/// // 第一轮对话
/// let result = chain.predict("你好").await?;
/// println!("AI: {}", result);
///
/// // 第二轮对话 - AI 会记住之前的对话
/// let result = chain.predict("我叫什么？").await?;
/// ```
pub struct ConversationChain {
    llm: OpenAIChat,
    memory: Arc<Mutex<ConversationBufferMemory>>,
    system_prompt: Option<String>,
    input_key: String,
    output_key: String,
    memory_key: String,
    name: String,
    verbose: bool,
}

impl ConversationChain {
    /// 创建新的 ConversationChain
    ///
    /// # 参数
    /// * `llm` - LLM 客户端
    /// * `memory` - 对话记忆
    pub fn new(llm: OpenAIChat, memory: ConversationBufferMemory) -> Self {
        Self {
            llm,
            memory: Arc::new(Mutex::new(memory.with_return_messages(true))),
            system_prompt: None,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            name: "conversation_chain".to_string(),
            verbose: false,
        }
    }
    
    /// 设置系统提示词
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
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
    
    /// 设置记忆键名
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }
    
    /// 设置 Chain 名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    /// 设置是否打印详细信息
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// 获取记忆
    pub fn memory(&self) -> &Arc<Mutex<ConversationBufferMemory>> {
        &self.memory
    }
    
    pub fn builder(llm: OpenAIChat) -> ConversationChainBuilder {
        ConversationChainBuilder::new(llm)
    }
    
    /// 清空记忆
    pub async fn clear_memory(&self) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        memory.clear().await.map_err(|e|
            ChainError::ExecutionError(format!("清空记忆失败: {}", e))
        )?;
        Ok(())
    }
    
    /// 简化的预测接口
    ///
    /// 直接传入用户输入字符串，返回 AI 响应字符串。
    ///
    /// # 参数
    /// * `input` - 用户输入
    ///
    /// # 返回
    /// AI 响应字符串
    pub async fn predict(&self, input: impl Into<String>) -> Result<String, ChainError> {
        let inputs = HashMap::from([
            (self.input_key.clone(), Value::String(input.into()))
        ]);
        
        let result = self.invoke(inputs).await?;
        
        result.get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("缺少输出".to_string()))
    }
    
    /// 准备消息列表
    ///
    /// 组合系统提示词、历史消息和当前用户输入。
    pub fn prepare_messages(
        &self,
        input: &str,
        history_messages: &[Message],
    ) -> Vec<Message> {
        let mut messages = Vec::new();
        
        // 添加系统提示词
        if let Some(system_prompt) = &self.system_prompt {
            messages.push(Message::system(system_prompt));
        }
        
        // 添加历史消息
        for msg in history_messages {
            messages.push(msg.clone());
        }
        
        // 添加当前用户输入
        messages.push(Message::human(input));
        
        messages
    }
    
    /// 加载历史消息
    async fn load_history(&self) -> Result<Vec<Message>, ChainError> {
        let memory = self.memory.lock().await;
        
        let messages = memory.chat_memory().messages().to_vec();
        
        Ok(messages)
    }
    
    /// 保存对话上下文
    async fn save_context(&self, input: &str, output: &str) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        
        let inputs = HashMap::from([(self.input_key.clone(), input.to_string())]);
        let outputs = HashMap::from([(self.output_key.clone(), output.to_string())]);
        
        memory.save_context(&inputs, &outputs).await
            .map_err(|e| ChainError::ExecutionError(format!("保存上下文失败: {}", e)))?;
        
        Ok(())
    }
}

#[async_trait]
impl BaseChain for ConversationChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }
    
    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }
    
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        // 验证输入
        self.validate_inputs(&inputs)?;
        
        // 获取用户输入
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;
        
        if self.verbose {
            println!("\n=== ConversationChain 执行 ===");
            println!("用户输入: {}", input);
        }
        
        // 加载历史消息
        let history_messages = self.load_history().await?;
        
        if self.verbose && !history_messages.is_empty() {
            println!("历史消息数量: {}", history_messages.len());
        }
        
        // 准备消息列表
        let messages = self.prepare_messages(input, &history_messages);
        
        if self.verbose {
            println!("总消息数量: {}", messages.len());
        }
        
        // 调用 LLM
        let result = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM 调用失败: {}", e)))?;
        
        let output = result.content;
        
        if self.verbose {
            println!("AI 响应: {}", output);
        }
        
        // 保存对话上下文
        self.save_context(input, &output).await?;
        
        if self.verbose {
            println!("=== ConversationChain 完成 ===\n");
        }
        
        // 构造输出
        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// ConversationChain Builder
///
/// 方便构建 ConversationChain。
pub struct ConversationChainBuilder {
    llm: OpenAIChat,
    memory: Option<ConversationBufferMemory>,
    system_prompt: Option<String>,
    input_key: Option<String>,
    output_key: Option<String>,
    memory_key: Option<String>,
    name: Option<String>,
    verbose: Option<bool>,
}

impl ConversationChainBuilder {
    pub fn new(llm: OpenAIChat) -> Self {
        Self {
            llm,
            memory: None,
            system_prompt: None,
            input_key: None,
            output_key: None,
            memory_key: None,
            name: None,
            verbose: None,
        }
    }
    
    pub fn memory(mut self, memory: ConversationBufferMemory) -> Self {
        self.memory = Some(memory);
        self
    }
    
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
    
    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = Some(key.into());
        self
    }
    
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }
    
    pub fn memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = Some(key.into());
        self
    }
    
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }
    
    pub fn build(self) -> ConversationChain {
        let memory = self.memory.unwrap_or_else(ConversationBufferMemory::new);
        let mut chain = ConversationChain::new(self.llm, memory);
        
        if let Some(prompt) = self.system_prompt {
            chain = chain.with_system_prompt(prompt);
        }
        
        if let Some(key) = self.input_key {
            chain = chain.with_input_key(key);
        }
        
        if let Some(key) = self.output_key {
            chain = chain.with_output_key(key);
        }
        
        if let Some(key) = self.memory_key {
            chain = chain.with_memory_key(key);
        }
        
        if let Some(name) = self.name {
            chain = chain.with_name(name);
        }
        
        if let Some(verbose) = self.verbose {
            chain = chain.with_verbose(verbose);
        }
        
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpenAIConfig;
    use crate::memory::ConversationBufferMemory;
    
    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-test".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
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
    fn test_conversation_chain_new() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory);
        
        assert_eq!(chain.input_keys(), vec!["input"]);
        assert_eq!(chain.output_keys(), vec!["output"]);
        assert_eq!(chain.name(), "conversation_chain");
    }
    
    #[test]
    fn test_conversation_chain_with_system_prompt() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("你是一个友好的助手");
        
        assert!(chain.system_prompt.is_some());
        assert_eq!(chain.system_prompt.unwrap(), "你是一个友好的助手");
    }
    
    #[test]
    fn test_conversation_chain_builder() {
        let llm = OpenAIChat::new(create_test_config());
        
        let chain = ConversationChainBuilder::new(llm)
            .system_prompt("你是一个 Rust 专家")
            .input_key("question")
            .output_key("answer")
            .verbose(true)
            .build();
        
        assert_eq!(chain.input_key, "question");
        assert_eq!(chain.output_key, "answer");
        assert!(chain.verbose);
    }
    
    #[test]
    fn test_prepare_messages_empty_history() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("你是一个助手");
        
        let messages = chain.prepare_messages("你好", &[]);
        
        // 系统消息 + 用户消息
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message_type, crate::schema::MessageType::System);
        assert_eq!(messages[1].message_type, crate::schema::MessageType::Human);
    }
    
    #[test]
    fn test_prepare_messages_with_history() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationBufferMemory::new();
        let chain = ConversationChain::new(llm, memory);
        
        let history = vec![
            Message::human("你好"),
            Message::ai("你好！有什么可以帮助你的？"),
        ];
        
        let messages = chain.prepare_messages("介绍一下 Rust", &history);
        
        // 2 条历史 + 1 条用户输入
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].message_type, crate::schema::MessageType::Human);
        assert_eq!(messages[1].message_type, crate::schema::MessageType::AI);
        assert_eq!(messages[2].message_type, crate::schema::MessageType::Human);
    }
    
    /// 真实 API 测试 - 单轮对话
    /// 运行: cargo test test_conversation_chain_single -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_conversation_chain_single() {
        let config = OpenAIConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            model: "gpt-3.5-turbo".to_string(),
            streaming: false,
            ..Default::default()
        };
        
        let llm = OpenAIChat::new(config);
        let memory = ConversationBufferMemory::new();
        
        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("你是一个友好的助手")
            .with_verbose(true);
        
        println!("\n=== 测试 ConversationChain - 单轮对话 ===");
        
        let result = chain.predict("你好，介绍一下自己").await.unwrap();
        println!("AI 响应: {}", result);
        
        assert!(!result.is_empty());
    }
    
    /// 真实 API 测试 - 多轮对话
    /// 运行: cargo test test_conversation_chain_multi_turn -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_conversation_chain_multi_turn() {
        let config = OpenAIConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            model: "gpt-3.5-turbo".to_string(),
            streaming: false,
            ..Default::default()
        };
        
        let llm = OpenAIChat::new(config);
        let memory = ConversationBufferMemory::new();
        
        let chain = ConversationChain::new(llm, memory)
            .with_system_prompt("你是一个友好的助手，请记住用户的名字")
            .with_verbose(true);
        
        println!("\n=== 测试 ConversationChain - 多轮对话 ===");
        
        // 第一轮：告诉名字
        println!("\n--- 第一轮 ---");
        let result1 = chain.predict("你好，我叫张三").await.unwrap();
        println!("AI: {}", result1);
        
        // 第二轮：问名字（测试记忆）
        println!("\n--- 第二轮 ---");
        let result2 = chain.predict("我叫什么名字？").await.unwrap();
        println!("AI: {}", result2);
        
        // 检查记忆是否保存了名字
        let memory = chain.memory.lock().await;
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();
        
        println!("\n历史记录: {}", history);
        assert!(history.contains("张三"), "记忆应该包含用户名字");
    }
    
    /// 真实 API 测试 - 清空记忆
    /// 运行: cargo test test_conversation_chain_clear_memory -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_conversation_chain_clear_memory() {
        let config = OpenAIConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            model: "gpt-3.5-turbo".to_string(),
            streaming: false,
            ..Default::default()
        };
        
        let llm = OpenAIChat::new(config);
        let memory = ConversationBufferMemory::new();
        
        let chain = ConversationChain::new(llm, memory);
        
        println!("\n=== 测试 ConversationChain - 清空记忆 ===");
        
        // 第一轮
        let result1 = chain.predict("我叫李四").await.unwrap();
        println!("第一轮: {}", result1);
        
        // 清空记忆
        chain.clear_memory().await.unwrap();
        
        // 第二轮：应该不记得名字了
        let result2 = chain.predict("我叫什么？").await.unwrap();
        println!("第二轮 (清空后): {}", result2);
        
        // 检查记忆已清空
        let memory = chain.memory.lock().await;
        assert_eq!(memory.chat_memory().len(), 2);
    }
}