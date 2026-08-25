// src/agents/react/agent.rs
//! ReAct Agent 实现
//!
//! 基于 "ReAct: Synergizing Reasoning and Acting in Language Models" 论文。
//! 支持任何实现了 `BaseChatModel` 的 LLM Provider。

use super::parser::ReActOutputParser;
use super::prompt::{build_react_prompt, format_scratchpad};
use crate::{AgentError, AgentOutput, AgentStep, BaseAgent};
use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::language_models::{BaseChatModel, TokenUsage};
use lc_core::tools::BaseTool;
use lc_providers::ProviderError;
use lc_schema::Message;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// ReAct Agent
///
/// 使用 ReAct (Reasoning + Acting) 模式的 Agent。
/// 会先思考，然后决定执行什么工具，最后观察结果。
/// 支持任何实现了 `BaseChatModel` 的 LLM Provider。
pub struct ReActAgent {
    /// LLM 客户端
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,

    /// 可用工具列表
    tools: Vec<Arc<dyn BaseTool>>,

    /// 输出解析器
    parser: ReActOutputParser,

    /// 自定义系统提示词（可选）
    system_prompt: Option<String>,

    /// 最近一次 `plan()` 的 token 用量(P1-5)。
    last_token_usage: std::sync::Mutex<Option<TokenUsage>>,
}

impl ReActAgent {
    /// 创建新的 ReAct Agent
    ///
    /// # 参数
    /// * `llm` - LLM 客户端（任何实现了 `BaseChatModel` 的类型）
    /// * `tools` - 可用工具列表
    /// * `system_prompt` - 自定义系统提示词（可选）
    ///
    /// # 向后兼容
    /// 旧代码 `ReActAgent::new(openai_chat, tools, None)` 仍然可用，
    /// 因为 `OpenAIChat: BaseChatModel` 且 `OpenAIError: Into<Error>`。
    pub fn new<L>(llm: L, tools: Vec<Arc<dyn BaseTool>>, system_prompt: Option<String>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: lc_providers::wrap_chat_model(llm),
            tools,
            parser: ReActOutputParser::new(),
            system_prompt,
            last_token_usage: std::sync::Mutex::new(None),
        }
    }

    /// 从已包装的 `Arc<dyn BaseChatModel>` 创建 Agent
    pub fn from_arc(
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
        tools: Vec<Arc<dyn BaseTool>>,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            llm,
            tools,
            parser: ReActOutputParser::new(),
            system_prompt,
            last_token_usage: std::sync::Mutex::new(None),
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
    fn build_prompt(
        &self,
        input: &str,
        intermediate_steps: &[AgentStep],
        history: Option<&str>,
    ) -> String {
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
        let input = inputs
            .get("input")
            .ok_or_else(|| AgentError::Other("Missing input parameter 'input'".to_string()))?;

        // 获取对话历史（如果有）
        let history = inputs.get("history").map(|s| s.as_str());

        // 构建 prompt
        let prompt_text = self.build_prompt(input, intermediate_steps, history);

        // 创建消息
        let messages = vec![Message::human(prompt_text)];

        // 调用 LLM
        let result = crate::retry::retry_chat(
            self.llm.as_ref(),
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AgentError::Other(format!("LLM call failed: {}", e)))?;

        // P1-5: record token usage for the executor's metrics.
        if let Ok(mut guard) = self.last_token_usage.lock() {
            *guard = result.token_usage.clone();
        }

        // 解析输出
        self.parser.parse(&result.content)
    }

    /// 流式规划(F3):逐 token 转发模型输出,累积为完整文本后解析。
    ///
    /// `plan()` 走非流式 `chat`(带重试、记录 token 用量);这里走 `stream_chat`
    /// 把每个 chunk 经 `on_token` 实时转发为 `Text` 事件,同时累积成完整文本
    /// 供 Action / Final Answer 解析。
    ///
    /// 权衡:流式 `stream_chat` 只回传文本字符串,拿不到 token 用量——流式路径
    /// 的 metrics 用量由非流式 `invoke` 路径补齐。`stream_chat` 立即可用即失败
    /// (如 provider 未实现流式)时回退到非流式 `plan()`,保证 agent 循环不中断。
    async fn plan_stream(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
        on_token: &mut (dyn FnMut(String) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send),
    ) -> Result<AgentOutput, AgentError> {
        let input = inputs
            .get("input")
            .ok_or_else(|| AgentError::Other("Missing input parameter 'input'".to_string()))?;
        let history = inputs.get("history").map(|s| s.as_str());
        let prompt_text = self.build_prompt(input, intermediate_steps, history);
        let messages = vec![Message::human(prompt_text)];

        let mut stream = match self.llm.stream_chat(messages, None).await {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "stream_chat unavailable ({}), falling back to non-streaming plan",
                    e
                );
                let output = self.plan(intermediate_steps, inputs).await?;
                if let AgentOutput::Finish(finish) = &output {
                    on_token(finish.output().unwrap_or("").to_string()).await;
                }
                return Ok(output);
            }
        };

        // 逐 token:先实时转发(clone 出自有 String),再拼进完整文本。
        // 解析只在流结束后进行,因此一个 ReAct 步骤的 Action / Final Answer
        // 判定不受影响。
        let mut full = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AgentError::Other(format!("LLM stream error: {}", e)))?;
            on_token(chunk.clone()).await;
            full.push_str(&chunk);
        }
        self.parser.parse(&full)
    }

    /// 获取允许的工具列表
    fn get_allowed_tools(&self) -> Option<Vec<&str>> {
        Some(self.get_tool_names())
    }

    /// Reports the token usage from the most recent `plan()` call (P1-5).
    fn last_token_usage(&self) -> Option<TokenUsage> {
        self.last_token_usage.lock().ok().and_then(|g| g.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use lc_tools::Calculator;

    /// 创建测试用的 OpenAI 配置
    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string(),
            base_url:
                "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                    .to_string(),
            model: "glm-5.2".to_string(),
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
}
