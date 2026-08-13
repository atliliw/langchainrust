// src/agents/function_calling/agent.rs
//! Function Calling Agent 实现
//!
//! 使用 LLM 原生 Function Calling 的 Agent，不依赖文本解析。
//! 支持任何实现了 `BaseChatModel` 的 LLM Provider。

use crate::{AgentAction, AgentError, AgentFinish, AgentOutput, AgentStep, BaseAgent, ToolInput};
use async_trait::async_trait;
use lc_core::language_models::{BaseChatModel, LLMResult, TokenUsage};
use lc_core::tools::{to_tool_definition, BaseTool, ToolCall, ToolDefinition};
use lc_providers::ProviderError;
use lc_schema::Message;
use std::collections::HashMap;
use std::sync::Arc;

/// Function Calling Agent
///
/// 使用 LLM 原生 Function Calling 的 Agent。
/// 不依赖文本解析，直接处理 tool_calls。
/// 支持任何实现了 `BaseChatModel` 的 LLM Provider。
pub struct FunctionCallingAgent {
    /// LLM 客户端（已绑定工具）
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,

    /// 可用工具列表
    tools: Vec<Arc<dyn BaseTool>>,

    /// 自定义系统提示词
    system_prompt: Option<String>,

    /// 最近一次 `plan()` 的 token 用量(P1-5)。
    last_token_usage: std::sync::Mutex<Option<TokenUsage>>,
}

impl FunctionCallingAgent {
    /// 创建新的 Function Calling Agent
    ///
    /// # 参数
    /// * `llm` - LLM 客户端（任何实现了 `BaseChatModel` 的类型）
    /// * `tools` - 可用工具列表
    /// * `system_prompt` - 自定义系统提示词（可选）
    ///
    /// # 向后兼容
    /// 旧代码 `FunctionCallingAgent::new(openai_chat, tools, None)` 仍然可用，
    /// 因为 `OpenAIChat: BaseChatModel` 且 `OpenAIError: Into<Error>`。
    pub fn new<L>(llm: L, tools: Vec<Arc<dyn BaseTool>>, system_prompt: Option<String>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        // 先包装 LLM，将错误类型统一为 ProviderError
        let wrapped = lc_providers::ChatModelWrapper::new(llm);

        let tool_definitions: Vec<ToolDefinition> = tools
            .iter()
            .map(|t| to_tool_definition(t.as_ref()))
            .collect();

        // 优先用 trait bind_tools（返回 Box<dyn BaseChatModel<Error = ProviderError>>）
        // Provider 不支持则直接用包装后的 LLM
        let llm_with_tools: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync> = wrapped
            .bind_tools(tool_definitions)
            .map(|boxed| {
                Arc::from(boxed) as Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>
            })
            .unwrap_or_else(|| Arc::new(wrapped));

        Self {
            llm: llm_with_tools,
            tools,
            system_prompt,
            last_token_usage: std::sync::Mutex::new(None),
        }
    }

    /// 从已包装的 `Arc<dyn BaseChatModel>` 创建 Agent
    ///
    /// 适用于已通过 `wrap_chat_model()` 或 `LLMClient` 创建的 LLM 实例。
    pub fn from_arc(
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
        tools: Vec<Arc<dyn BaseTool>>,
        system_prompt: Option<String>,
    ) -> Self {
        let tool_definitions: Vec<ToolDefinition> = tools
            .iter()
            .map(|t| to_tool_definition(t.as_ref()))
            .collect();

        let llm_with_tools = llm
            .bind_tools(tool_definitions)
            .map(|boxed| {
                Arc::from(boxed) as Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>
            })
            .unwrap_or(llm);

        Self {
            llm: llm_with_tools,
            tools,
            system_prompt,
            last_token_usage: std::sync::Mutex::new(None),
        }
    }

    /// 获取工具数量
    pub fn tools_count(&self) -> usize {
        self.tools.len()
    }

    /// 获取系统提示词
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// 构建消息
    fn build_messages(
        &self,
        inputs: &HashMap<String, String>,
        intermediate_steps: &[AgentStep],
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        let system_content = self
            .system_prompt
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
                    ToolInput::String { value: s } => s.clone(),
                    ToolInput::Object { value: v } => {
                        serde_json::to_string(v).unwrap_or_else(|_| v.to_string())
                    }
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

        let result: LLMResult = crate::retry::retry_chat(
            self.llm.as_ref(),
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AgentError::Other(format!("LLM 调用失败: {}", e)))?;

        // P1-5: record token usage for the executor's metrics.
        if let Ok(mut guard) = self.last_token_usage.lock() {
            *guard = result.token_usage.clone();
        }

        if let Some(tool_calls) = &result.tool_calls {
            if !tool_calls.is_empty() {
                let actions: Vec<AgentAction> = tool_calls
                    .iter()
                    .map(|call| {
                        let tool_input = match serde_json::from_str::<serde_json::Value>(
                            &call.function.arguments,
                        ) {
                            Ok(v) => ToolInput::Object { value: v },
                            Err(_) => ToolInput::String {
                                value: call.function.arguments.clone(),
                            },
                        };

                        AgentAction {
                            tool: call.function.name.clone(),
                            tool_input,
                            log: call.id.clone(),
                        }
                    })
                    .collect();

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

    /// Reports the token usage from the most recent `plan()` call (P1-5).
    fn last_token_usage(&self) -> Option<TokenUsage> {
        self.last_token_usage.lock().ok().and_then(|g| g.clone())
    }
}

impl std::fmt::Debug for FunctionCallingAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionCallingAgent")
            .field("tools_count", &self.tools.len())
            .field("system_prompt", &self.system_prompt)
            .field(
                "has_token_usage",
                &self.last_token_usage.lock().ok().is_some(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use lc_tools::Calculator;

    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1")
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

        let steps = vec![AgentStep::new(
            AgentAction {
                tool: "calculator".to_string(),
                tool_input: ToolInput::String {
                    value: "2 + 3".to_string(),
                },
                log: "call_123".to_string(),
            },
            "5".to_string(),
        )];

        let messages = agent.build_messages(&inputs, &steps);

        assert_eq!(messages.len(), 4);
        assert!(messages[2].has_tool_calls());
    }

    #[test]
    fn test_from_arc_creation() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let llm_arc: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync> =
            lc_providers::wrap_chat_model(llm);
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];

        let agent = FunctionCallingAgent::from_arc(llm_arc, tools, Some("test".into()));
        assert_eq!(agent.tools.len(), 1);
        assert_eq!(agent.system_prompt, Some("test".to_string()));
    }
}
