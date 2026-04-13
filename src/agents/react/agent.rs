// src/agents/react/agent.rs
//! ReAct Agent 实现
//!
//! 基于 "ReAct: Synergizing Reasoning and Acting in Language Models" 论文。

use crate::agents::{AgentError, AgentOutput, AgentStep, BaseAgent};
use crate::core::tools::BaseTool;
use crate::language_models::OpenAIChat;
use crate::schema::Message;
use crate::core::language_models::BaseChatModel;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use super::parser::ReActOutputParser;
use super::prompt::{build_react_prompt, format_scratchpad};

/// ReAct Agent
///
/// 使用 ReAct (Reasoning + Acting) 模式的 Agent。
/// 会先思考，然后决定执行什么工具，最后观察结果。
pub struct ReActAgent {
    /// LLM 客户端
    llm: OpenAIChat,
    
    /// 可用工具列表
    tools: Vec<Arc<dyn BaseTool>>,
    
    /// 输出解析器
    parser: ReActOutputParser,
    
    /// 自定义系统提示词（可选）
    system_prompt: Option<String>,
}

impl ReActAgent {
    /// 创建新的 ReAct Agent
    ///
    /// # 参数
    /// * `llm` - LLM 客户端
    /// * `tools` - 可用工具列表
    /// * `system_prompt` - 自定义系统提示词（可选）
    pub fn new(llm: OpenAIChat, tools: Vec<Arc<dyn BaseTool>>, system_prompt: Option<String>) -> Self {
        Self {
            llm,
            tools,
            parser: ReActOutputParser::new(),
            system_prompt,
        }
    }
    
    /// 格式化工具描述
    ///
    /// 将工具列表格式化为 ReAct prompt 需要的格式
    fn format_tools(&self) -> String {
        self.tools
            .iter()
            .map(|tool| format!("{}: {}", tool.name(), tool.description()))
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    /// 获取工具名称列表
    fn get_tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }
    
    /// 构建 ReAct prompt
    ///
    /// # 参数
    /// * `input` - 用户问题
    /// * `intermediate_steps` - 已执行的步骤历史
    /// * `history` - 对话历史（可选）
    fn build_prompt(&self, input: &str, intermediate_steps: &[AgentStep], history: Option<&str>) -> String {
        // 格式化工具描述
        let tools_description = self.format_tools();
        let tool_names = self.get_tool_names();
        
        // 格式化思考历史
        let scratchpad = format_scratchpad(intermediate_steps);
        
        // 构建基础 prompt
        let mut prompt = build_react_prompt(&tools_description, &tool_names, input, &scratchpad);
        
        // 如果有对话历史，添加到 prompt 开头
        if let Some(h) = history {
            if !h.is_empty() {
                prompt = format!("之前的对话历史:\n{}\n\n{}", h, prompt);
            }
        }
        
        // 如果有自定义系统提示词，添加到 prompt 开头
        if let Some(sys) = &self.system_prompt {
            prompt = format!("{}\n\n{}", sys, prompt);
        }
        
        prompt
    }
}

#[async_trait]
impl BaseAgent for ReActAgent {
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
    ) -> Result<AgentOutput, AgentError> {
        // 获取用户输入
        let input = inputs.get("input")
            .ok_or_else(|| AgentError::Other("缺少输入参数 'input'".to_string()))?;
        
        // 获取对话历史（如果有）
        let history = inputs.get("history").map(|s| s.as_str());
        
        // 构建 prompt
        let prompt_text = self.build_prompt(input, intermediate_steps, history);
        
        // 创建消息
        let messages = vec![Message::human(prompt_text)];
        
        // 调用 LLM
        let result = self.llm.chat(messages, None)
            .await
            .map_err(|e| AgentError::Other(format!("LLM 调用失败: {}", e)))?;
        
        // 解析输出
        self.parser.parse(&result.content)
    }
    
    /// 获取允许的工具列表
    fn get_allowed_tools(&self) -> Option<Vec<&str>> {
        Some(self.get_tool_names())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Calculator;
    use crate::language_models::OpenAIConfig;
    use crate::agents::{AgentAction, ToolInput};
    
    /// 创建测试用的 OpenAI 配置
    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".to_string(),
            base_url: "https://api.openai-proxy.org/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            temperature: Some(0.0),
            max_tokens: Some(500),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            streaming: false,
            organization: None,
            tools: None,
            tool_choice: None,
        }
    }
    
    #[test]
    fn test_format_tools_description() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator)];
        let agent = ReActAgent::new(llm, tools, None);
        
        let desc = agent.format_tools();
        assert!(desc.contains("calculator"));
    }
    
    #[test]
    fn test_get_tool_names() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator)];
        let agent = ReActAgent::new(llm, tools, None);
        
        let names = agent.get_tool_names();
        assert_eq!(names, vec!["calculator"]);
    }
    
    #[test]
    fn test_build_prompt() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator)];
        let agent = ReActAgent::new(llm, tools, None);
        
        let prompt = agent.build_prompt("计算 2 + 2", &[], None);
        
        assert!(prompt.contains("计算 2 + 2"));
        assert!(prompt.contains("calculator"));
        assert!(prompt.contains("Question:"));
        assert!(prompt.contains("Thought:"));
    }
    
    #[test]
    fn test_build_prompt_with_history() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator)];
        let agent = ReActAgent::new(llm, tools, None);
        
        let prompt = agent.build_prompt("计算 3 + 3", &[], Some("用户: 你好\n助手: 你好！"));
        
        assert!(prompt.contains("之前的对话历史"));
        assert!(prompt.contains("你好"));
    }
    
    #[test]
    fn test_build_prompt_with_system_prompt() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator)];
        let agent = ReActAgent::new(llm, tools, Some("你是一个数学助手".to_string()));
        
        let prompt = agent.build_prompt("计算 4 + 4", &[], None);
        
        assert!(prompt.contains("你是一个数学助手"));
    }
    
    /// 真实 API 测试：简单问题（无工具调用）
    #[tokio::test]
    #[ignore = "需要真实 API 调用"]
    async fn test_real_api_simple() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![];
        let agent = ReActAgent::new(llm, tools, None);
        
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "什么是 Rust 语言？".to_string());
        
        let result = agent.plan(&[], &inputs).await.unwrap();
        
        // 应该直接返回最终答案（因为没有工具）
        match result {
            AgentOutput::Finish(finish) => {
                println!("答案: {:?}", finish.return_values);
                assert!(finish.output().is_some());
            }
            AgentOutput::Action(_) => {
                println!("LLM 尝试调用工具");
            }
            AgentOutput::Actions(_) => {
                println!("ReActAgent 不支持并行工具调用");
            }
        }
    }
    
    /// 真实 API 测试：使用计算器
    #[tokio::test]
    #[ignore = "需要真实 API 调用"]
    async fn test_real_api_with_calculator() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator)];
        let agent = ReActAgent::new(llm, tools, None);
        
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "计算 37 加 48 等于多少？".to_string());
        
        let result = agent.plan(&[], &inputs).await.unwrap();
        
        match result {
            AgentOutput::Action(action) => {
                println!("动作: {}({})", action.tool, action.tool_input);
                assert_eq!(action.tool, "calculator");
            }
            AgentOutput::Finish(finish) => {
                println!("直接答案: {:?}", finish.return_values);
            }
            AgentOutput::Actions(_) => {
                println!("ReActAgent 不支持并行工具调用");
            }
        }
    }
    
    /// 真实 API 测试：多步问题
    #[tokio::test]
    #[ignore = "需要真实 API 调用"]
    async fn test_real_api_multi_step() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator)];
        let agent = ReActAgent::new(llm, tools, None);
        
        // 创建一个已执行的动作历史
        let steps = vec![
            AgentStep::new(
                AgentAction {
                    tool: "calculator".to_string(),
                    tool_input: ToolInput::String("37 + 48".to_string()),
                    log: "我需要先计算 37 + 48".to_string(),
                },
                "85".to_string(),
            ),
        ];
        
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "计算 (37 + 48) * 2 等于多少？".to_string());
        
        let result = agent.plan(&steps, &inputs).await.unwrap();
        
        match result {
            AgentOutput::Action(action) => {
                println!("下一步动作: {}({})", action.tool, action.tool_input);
                assert_eq!(action.tool, "calculator");
            }
            AgentOutput::Finish(finish) => {
                println!("最终答案: {:?}", finish.return_values);
            }
            AgentOutput::Actions(_) => {
                println!("ReActAgent 不支持并行工具调用");
            }
        }
    }
    
    /// 真实 API 测试：带对话历史
    #[tokio::test]
    #[ignore = "需要真实 API 调用"]
    async fn test_real_api_with_memory() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![];
        let agent = ReActAgent::new(llm, tools, None);
        
        // 模拟对话历史
        let history = "Human: 我叫张三\nAI: 好的，张三，我记住了。";
        
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "我叫什么名字？".to_string());
        inputs.insert("history".to_string(), history.to_string());
        
        let result = agent.plan(&[], &inputs).await.unwrap();
        
        match result {
            AgentOutput::Finish(finish) => {
                println!("答案: {:?}", finish.return_values);
                let output = finish.output().unwrap_or("");
                assert!(output.contains("张三"), "应该记住用户名字");
            }
            AgentOutput::Action(_) => {
                println!("LLM 尝试调用工具");
            }
            AgentOutput::Actions(_) => {
                println!("ReActAgent 不支持并行工具调用");
            }
        }
    }
}