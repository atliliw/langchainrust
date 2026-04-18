// src/agents/function_calling/agent.rs
//! Function Calling Agent 实现
//!
//! 使用 OpenAI 原生 Function Calling 的 Agent，不依赖文本解析。

use crate::agents::{AgentAction, AgentError, AgentFinish, AgentOutput, AgentStep, BaseAgent, ToolInput};
use crate::core::tools::{BaseTool, ToolCall, ToolDefinition, to_tool_definition};
use crate::language_models::OpenAIChat;
use crate::schema::Message;
use crate::core::language_models::BaseChatModel;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Function Calling Agent
///
/// 使用 OpenAI 原生 Function Calling 的 Agent。
/// 不依赖文本解析，直接处理 tool_calls。
pub struct FunctionCallingAgent {
    /// LLM 客户端（已绑定工具）
    llm: OpenAIChat,
    
    /// 可用工具列表
    tools: Vec<Arc<dyn BaseTool>>,
    
    /// 自定义系统提示词
    system_prompt: Option<String>,
}

impl FunctionCallingAgent {
    /// 创建新的 Function Calling Agent
    ///
    /// # 参数
    /// * `llm` - LLM 客户端
    /// * `tools` - 可用工具列表
    /// * `system_prompt` - 自定义系统提示词（可选）
    pub fn new(llm: OpenAIChat, tools: Vec<Arc<dyn BaseTool>>, system_prompt: Option<String>) -> Self {
        let tool_definitions: Vec<ToolDefinition> = tools
            .iter()
            .map(|t| to_tool_definition(t.as_ref()))
            .collect();
        
        let llm_with_tools = llm.bind_tools(tool_definitions);
        
        Self {
            llm: llm_with_tools,
            tools,
            system_prompt,
        }
    }
    
    /// 构建消息
    fn build_messages(
        &self,
        inputs: &HashMap<String, String>,
        intermediate_steps: &[AgentStep],
    ) -> Vec<Message> {
        let mut messages = Vec::new();
        
        let system_content = self.system_prompt
            .clone()
            .unwrap_or_else(|| "你是一个助手，可以使用工具回答问题。".to_string());
        messages.push(Message::system(&system_content));
        
        let default_input = String::new();
        let input = inputs.get("input").unwrap_or(&default_input);
        messages.push(Message::human(input));
        
        for step in intermediate_steps {
            let tool_call = ToolCall::new(
                &step.action.log,
                &step.action.tool,
                match &step.action.tool_input {
                    ToolInput::String(s) => s.clone(),
                    ToolInput::Object(v) => v.to_string(),
                },
            );
            messages.push(Message::ai_with_tool_calls("", vec![tool_call]));
            messages.push(Message::tool(&step.action.log, &step.observation));
        }
        
        messages
    }
}

#[async_trait]
impl BaseAgent for FunctionCallingAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        let messages = self.build_messages(inputs, intermediate_steps);
        
        let result = self.llm.chat(messages, None)
            .await
            .map_err(|e| AgentError::Other(format!("LLM 调用失败: {}", e)))?;
        
        if let Some(tool_calls) = &result.tool_calls {
            if !tool_calls.is_empty() {
                let actions: Vec<AgentAction> = tool_calls.iter().map(|call| {
                    let tool_input = match serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
                        Ok(v) => ToolInput::Object(v),
                        Err(_) => ToolInput::String(call.function.arguments.clone()),
                    };
                    
                    AgentAction {
                        tool: call.function.name.clone(),
                        tool_input,
                        log: call.id.clone(),
                    }
                }).collect();
                
                if actions.len() == 1 {
                    return Ok(AgentOutput::Action(actions.into_iter().next().unwrap()));
                } else {
                    return Ok(AgentOutput::Actions(actions));
                }
            }
        }
        
        Ok(AgentOutput::Finish(AgentFinish::new(
            result.content.clone(),
            String::new(),
        )))
    }
    
    fn get_allowed_tools(&self) -> Option<Vec<&str>> {
        Some(self.tools.iter().map(|t| t.name()).collect())
    }
}

impl std::fmt::Debug for FunctionCallingAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionCallingAgent")
            .field("tools_count", &self.tools.len())
            .field("system_prompt", &self.system_prompt)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::OpenAIChat;
    use crate::language_models::openai::OpenAIConfig;
    use crate::tools::Calculator;
    
    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig::new("test_key")
            .with_base_url("http://localhost:8080/v1")
    }
    
    #[test]
    fn test_function_calling_agent_creation() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
        
        let agent = FunctionCallingAgent::new(llm, tools, None);
        assert_eq!(agent.tools.len(), 1);
    }
    
#[test]
    fn test_get_allowed_tools() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
        
        let agent = FunctionCallingAgent::new(llm, tools, None);
        
        assert_eq!(agent.tools.len(), 1);
        assert!(agent.system_prompt.is_none());
    }
    
#[test]
    fn test_new_with_system_prompt() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
        
        let agent = FunctionCallingAgent::new(llm, tools, Some("你是一个数学助手".to_string()));
        
        assert_eq!(agent.system_prompt, Some("你是一个数学助手".to_string()));
    }
    
    #[test]
    fn test_build_messages_empty() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
        
        let agent = FunctionCallingAgent::new(llm, tools, None);
        
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "计算 2 + 3".to_string());
        
        let messages = agent.build_messages(&inputs, &[]);
        
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "你是一个助手，可以使用工具回答问题。");
        assert_eq!(messages[1].content, "计算 2 + 3");
    }
    
    #[test]
    fn test_build_messages_with_history() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
        
        let agent = FunctionCallingAgent::new(llm, tools, None);
        
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "继续计算".to_string());
        
        let steps = vec![
            AgentStep::new(
                AgentAction {
                    tool: "calculator".to_string(),
                    tool_input: ToolInput::String("2 + 3".to_string()),
                    log: "call_123".to_string(),
                },
                "5".to_string(),
            ),
        ];
        
        let messages = agent.build_messages(&inputs, &steps);
        
        assert_eq!(messages.len(), 4);
        assert!(messages[2].has_tool_calls());
    }
}