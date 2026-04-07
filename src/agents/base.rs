// src/agents/base.rs
//! Agent 基础 trait 定义

use super::types::{AgentAction, AgentFinish, AgentOutput, AgentStep};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use crate::core::tools::BaseTool;
use crate::memory::BaseMemory;

/// Agent 错误类型
#[derive(Debug)]
pub enum AgentError {
    /// 输出解析错误
    OutputParsingError(String),
    
    /// 工具未找到
    ToolNotFound(String),
    
    /// 工具执行错误
    ToolExecutionError(String),
    
    /// 达到最大迭代次数
    MaxIterationsReached,
    
    /// 其他错误
    Other(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::OutputParsingError(msg) => write!(f, "输出解析错误: {}", msg),
            AgentError::ToolNotFound(name) => write!(f, "工具未找到: {}", name),
            AgentError::ToolExecutionError(msg) => write!(f, "工具执行错误: {}", msg),
            AgentError::MaxIterationsReached => write!(f, "达到最大迭代次数"),
            AgentError::Other(msg) => write!(f, "Agent 错误: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {}

/// Base Agent trait
/// 
/// 定义 Agent 的核心接口。Agent 负责决策（plan），不负责执行。
/// 执行由 AgentExecutor 处理。
#[async_trait]
pub trait BaseAgent: Send + Sync {
    /// 规划下一步行动
    /// 
    /// # 参数
    /// * `intermediate_steps` - 已执行的步骤历史
    /// * `inputs` - 用户输入
    /// 
    /// # 返回
    /// * `AgentOutput::Action` - 需要执行的动作
    /// * `AgentOutput::Finish` - 最终答案
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError>;
    
    /// 获取输入键
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }
    
    /// 获取允许的工具列表
    fn get_allowed_tools(&self) -> Option<Vec<&str>> {
        None
    }
    
    /// 当达到最大迭代次数时的停止响应
    fn return_stopped_response(
        &self,
        _intermediate_steps: &[AgentStep],
    ) -> AgentFinish {
        AgentFinish::new(
            "Agent stopped due to iteration limit or time limit.".to_string(),
            String::new(),
        )
    }
}

/// Agent 执行器
/// 
/// 负责执行 Agent 的决策循环：Plan → Act → Observe
pub struct AgentExecutor {
    /// Agent 实例
    agent: Arc<dyn BaseAgent>,
    
    /// 可用工具
    tools: Vec<Arc<dyn BaseTool>>,
    
    /// 最大迭代次数
    max_iterations: usize,
    
    /// 是否详细输出
    verbose: bool,
    
    /// 记忆（可选）
    memory: Option<Arc<tokio::sync::Mutex<dyn BaseMemory>>>,
}

impl AgentExecutor {
    /// 创建新的 AgentExecutor
    pub fn new(agent: Arc<dyn BaseAgent>, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        Self {
            agent,
            tools,
            max_iterations: 10,
            verbose: false,
            memory: None,
        }
    }
    
    /// 设置最大迭代次数
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    
    /// 设置详细输出
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// 设置记忆
    pub fn with_memory(mut self, memory: Arc<tokio::sync::Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }
    
    /// 执行 Agent
    /// 
    /// # 参数
    /// * `input` - 用户输入
    /// 
    /// # 返回
    /// 最终答案
    pub async fn invoke(&self, input: String) -> Result<String, AgentError> {
        // 准备输入
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input.clone());
        
        // 如果有 memory，加载历史
        if let Some(memory) = &self.memory {
            let memory_vars = memory.lock().await
                .load_memory_variables(&inputs).await
                .map_err(|e| AgentError::Other(format!("加载记忆失败: {}", e)))?;
            
            // 将历史添加到 inputs
            if let Some(history) = memory_vars.get("history") {
                if let Some(history_str) = history.as_str() {
                    inputs.insert("history".to_string(), history_str.to_string());
                }
            }
        }
        
        // 中间步骤历史
        let intermediate_steps: Vec<AgentStep> = Vec::new();
        
        // 执行循环
        let result = self.run_agent_loop(inputs.clone(), intermediate_steps).await;
        
        // 如果有 memory，保存对话
        if let Some(memory) = &self.memory {
            if let Ok(ref output) = result {
                let mut outputs = HashMap::new();
                outputs.insert("output".to_string(), output.clone());
                
                memory.lock().await
                    .save_context(&inputs, &outputs).await
                    .map_err(|e| AgentError::Other(format!("保存记忆失败: {}", e)))?;
            }
        }
        
        result
    }
    
    /// 运行 Agent 循环
    async fn run_agent_loop(
        &self,
        inputs: HashMap<String, String>,
        mut intermediate_steps: Vec<AgentStep>,
    ) -> Result<String, AgentError> {
        // 执行循环
        for iteration in 0..self.max_iterations {
            if self.verbose {
                println!("\n=== 迭代 {} ===", iteration + 1);
            }
            
            // 1. PLAN: Agent 决策
            let output = self.agent.plan(&intermediate_steps, &inputs).await?;
            
            match output {
                AgentOutput::Finish(finish) => {
                    // 返回最终答案
                    if self.verbose {
                        println!("最终答案: {:?}", finish.return_values);
                    }
                    return Ok(finish.output().unwrap_or("").to_string());
                }
                
                AgentOutput::Action(action) => {
                    if self.verbose {
                        println!("动作: {}({})", action.tool, action.tool_input);
                    }
                    
                    // 2. ACT: 执行工具
                    let observation = self.execute_tool(&action).await?;
                    
                    if self.verbose {
                        println!("观察: {}", observation);
                    }
                    
                    // 3. OBSERVE: 记录结果
                    intermediate_steps.push(AgentStep::new(action, observation));
                }
            }
        }
        
        // 达到最大迭代次数
        if self.verbose {
            println!("达到最大迭代次数: {}", self.max_iterations);
        }
        
        let finish = self.agent.return_stopped_response(&intermediate_steps);
        Ok(finish.output().unwrap_or("").to_string())
    }
    
    /// 执行工具
    async fn execute_tool(&self, action: &AgentAction) -> Result<String, AgentError> {
        // 查找工具
        let tool = self.tools.iter()
            .find(|t| t.name() == action.tool)
            .ok_or_else(|| AgentError::ToolNotFound(action.tool.clone()))?;
        
        // 准备输入
        let input_str = match &action.tool_input {
            super::types::ToolInput::String(s) => s.clone(),
            super::types::ToolInput::Object(v) => serde_json::to_string(v)
                .unwrap_or_else(|_| v.to_string()),
        };
        
        // 执行工具
        tool.run(input_str).await
            .map_err(|e| AgentError::ToolExecutionError(e.to_string()))
    }
}

impl std::fmt::Debug for AgentExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentExecutor")
            .field("max_iterations", &self.max_iterations)
            .field("verbose", &self.verbose)
            .field("tools_count", &self.tools.len())
            .field("has_memory", &self.memory.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ConversationBufferMemory;
    
    /// 测试 AgentExecutor with memory
    #[tokio::test]
    async fn test_agent_executor_with_memory() {
        // 创建简单的 mock agent
        struct TestAgent;
        
        #[async_trait]
        impl BaseAgent for TestAgent {
            async fn plan(
                &self,
                _intermediate_steps: &[AgentStep],
                inputs: &HashMap<String, String>,
            ) -> Result<AgentOutput, AgentError> {
                // 如果有历史，检查是否包含之前的信息
                if let Some(history) = inputs.get("history") {
                    if history.contains("张三") {
                        return Ok(AgentOutput::Finish(AgentFinish::new(
                            "你叫张三".to_string(),
                            String::new(),
                        )));
                    }
                }
                
                // 否则返回输入内容
                let input = inputs.get("input").unwrap();
                Ok(AgentOutput::Finish(AgentFinish::new(
                    format!("收到: {}", input),
                    String::new(),
                )))
            }
        }
        
        // 创建 memory
        let memory = Arc::new(tokio::sync::Mutex::new(ConversationBufferMemory::new()));
        
        // 创建 executor
        let executor = AgentExecutor::new(Arc::new(TestAgent), vec![])
            .with_memory(memory);
        
        // 第一轮对话
        let result1 = executor.invoke("我叫张三".to_string()).await.unwrap();
        println!("第一轮: {}", result1);
        
        // 第二轮对话 - 应该记得名字
        let result2 = executor.invoke("我叫什么名字？".to_string()).await.unwrap();
        println!("第二轮: {}", result2);
        
        assert!(result2.contains("张三"));
    }
}