// src/agents/function_calling/agent.rs
//! Function Calling Agent 实现
//!
//! 使用 LLM 原生 Function Calling 的 Agent，不依赖文本解析。
//! 支持任何实现了 `BaseChatModel` 的 LLM Provider。

use crate::{AgentAction, AgentError, AgentFinish, AgentOutput, AgentStep, BaseAgent, ToolInput};
use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::language_models::{BaseChatModel, LLMResult, TokenUsage};
use lc_core::tools::{to_tool_definition, BaseTool, ToolCall, ToolDefinition};
use lc_providers::ProviderError;
use lc_schema::Message;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
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
            let tool_call = ToolCall::builder(&step.action.log)
                .name(&step.action.tool)
                .arguments(match &step.action.tool_input {
                    ToolInput::String { value: s } => s.clone(),
                    ToolInput::Object { value: v } => {
                        serde_json::to_string(v).unwrap_or_else(|_| v.to_string())
                    }
                })
                .build();
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
        .map_err(|e| AgentError::Other(format!("LLM call failed: {}", e)))?;

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

    /// 流式规划(S2):走 `stream_chat` 逐 token 转发模型文本 + 累积 usage。
    ///
    /// 函数调用 Agent 的**最终答案**以文本逐 token 流出(打字机效果);但
    /// **工具调用步骤**模型不回文本(内容为空、tool_calls 在增量里),而当前
    /// `stream_chat` 的 chunk 只携带文本与 usage、不携带 tool_calls —— 流式路径
    /// 无法重构工具调用,故累积文本为空时回退非流式 [`BaseAgent::plan`] 拿原生
    /// `tool_calls`,避免"空文本 → 空 Finish"的假流式把 agent 循环提前终止。
    /// `stream_chat` 立即可用即失败时同样回退非流式 `plan()`。
    /// 诚实边界:模型在单个步骤同时输出文本与工具调用时,当前只保留文本
    /// (工具调用丢失)——与 provider 层 chunk 不带 tool_calls 的现状一致,已在
    /// v0.18 计划书 "chunk 内携带 thinking / tool_calls 增量" 留作后续演进。
    async fn plan_stream(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
        on_token: &mut (dyn FnMut(String) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send),
    ) -> Result<AgentOutput, AgentError> {
        let messages = self.build_messages(inputs, intermediate_steps);

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

        // 逐 token:转发非空文本,累积完整文本 + 最后一个非 None 的 usage。
        let mut full = String::new();
        let mut usage: Option<TokenUsage> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AgentError::Other(format!("LLM stream error: {}", e)))?;
            if !chunk.text.is_empty() {
                on_token(chunk.text.clone()).await;
            }
            full.push_str(&chunk.text);
            if chunk.token_usage.is_some() {
                usage = chunk.token_usage;
            }
        }
        if let Ok(mut guard) = self.last_token_usage.lock() {
            *guard = usage;
        }

        // 工具调用步骤模型不回文本:流式 chunk 拿不到 tool_calls,回退非流式
        // plan() 拿原生工具调用,保证 agent 循环不因"空 Finish"提前终止。
        if full.trim().is_empty() {
            log::debug!(
                "streamed plan produced no text (likely a tool call), \
                 falling back to non-streaming plan for tool_calls"
            );
            let output = self.plan(intermediate_steps, inputs).await?;
            if let AgentOutput::Finish(finish) = &output {
                on_token(finish.output().unwrap_or("").to_string()).await;
            }
            return Ok(output);
        }

        Ok(AgentOutput::Finish(AgentFinish::new(full, String::new())))
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
    use futures_util::Stream;
    use lc_core::language_models::{BaseLanguageModel, StreamChunk};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use lc_providers::{AssistantError, OpenAIChat, OpenAIConfig};
    use lc_tools::Calculator;
    use std::sync::Mutex;

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

    /// S2 流式 mock:可配置 `stream_chat` 返回(正常 chunks / 立即失败)与 `chat`
    /// 返回(工具调用 / 最终答案),并记录方法调用序列,验证 FunctionCallingAgent
    /// 的 `plan_stream` 覆写:逐 token 转发、空流回退非流式 plan、立即失败回退。
    struct MockFuncLLM {
        stream_chunks: Option<Vec<StreamChunk>>,
        chat_result: LLMResult,
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl MockFuncLLM {
        fn new(stream_chunks: Option<Vec<StreamChunk>>, chat_result: LLMResult) -> Self {
            Self {
                stream_chunks,
                chat_result,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockFuncLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            input: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.chat(input, config).await
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockFuncLLM {
        fn model_name(&self) -> &str {
            "mock-func"
        }
        fn get_num_tokens(&self, t: &str) -> usize {
            t.len()
        }
        fn with_temperature(self, _: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _: usize) -> Self {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for MockFuncLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push("chat".to_string());
            Ok(self.chat_result.clone())
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            self.calls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push("stream_chat".to_string());
            match &self.stream_chunks {
                Some(chunks) => {
                    let items: Vec<Result<StreamChunk, ProviderError>> =
                        chunks.iter().cloned().map(Ok).collect();
                    Ok(Box::pin(futures_util::stream::iter(items)))
                }
                None => Err(ProviderError::Assistant(AssistantError::Api(
                    "stream_chat unavailable".to_string(),
                ))),
            }
        }
    }

    fn calculator_call_result() -> LLMResult {
        let call = ToolCall::builder("call_1")
            .name("calculator")
            .arguments(r#"{"expression": "2+3"}"#)
            .build();
        LLMResult {
            content: String::new(),
            model: "mock-func".to_string(),
            token_usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
            tool_calls: Some(vec![call]),
            thinking_content: None,
        }
    }

    fn text_result(content: &str) -> LLMResult {
        LLMResult {
            content: content.to_string(),
            model: "mock-func".to_string(),
            token_usage: Some(TokenUsage {
                prompt_tokens: 8,
                completion_tokens: 6,
                total_tokens: 14,
            }),
            tool_calls: None,
            thinking_content: None,
        }
    }

    fn streaming_agent(llm: MockFuncLLM) -> (FunctionCallingAgent, Arc<MockFuncLLM>) {
        let arc: Arc<MockFuncLLM> = Arc::new(llm);
        let agent = FunctionCallingAgent::from_arc(
            arc.clone() as Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
            vec![],
            None,
        );
        (agent, arc)
    }

    /// S2:最终答案逐 token 流出(Text 事件),并记录流式用量供预算门。
    #[tokio::test]
    async fn test_function_calling_plan_stream_streams_final_answer() {
        let llm = MockFuncLLM::new(
            Some(vec![
                StreamChunk::new("Final "),
                StreamChunk {
                    text: "Answer: 42".to_string(),
                    token_usage: Some(TokenUsage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                    }),
                },
            ]),
            text_result("unused"),
        );
        let (agent, llm) = streaming_agent(llm);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "计算 6 * 7".to_string());
        let mut received = String::new();
        let mut on_token = |text: String| {
            received.push_str(&text);
            Box::pin(async move {}) as Pin<Box<dyn Future<Output = ()> + Send>>
        };

        let output = agent
            .plan_stream(&[], &inputs, &mut on_token)
            .await
            .expect("plan_stream should succeed");

        assert_eq!(received, "Final Answer: 42");
        assert!(matches!(
            output,
            AgentOutput::Finish(f) if f.output() == Some("Final Answer: 42")
        ));
        let usage = agent.last_token_usage().expect("streaming usage recorded");
        assert_eq!(usage.total_tokens, 15);
        // 走的是流式路径:只调了 stream_chat,没有回退 chat。
        assert_eq!(llm.calls(), vec!["stream_chat"]);
    }

    /// S2:工具调用步骤模型不回文本,流式 chunk 拿不到 tool_calls —— `plan_stream`
    /// 回退非流式 `plan()` 拿原生工具调用,不产生"空 Finish"假流式。
    #[tokio::test]
    async fn test_function_calling_plan_stream_falls_back_for_tool_call() {
        // 模拟工具调用步骤:流只回传空文本 + usage 块。
        let llm = MockFuncLLM::new(
            Some(vec![StreamChunk {
                text: String::new(),
                token_usage: Some(TokenUsage {
                    prompt_tokens: 5,
                    completion_tokens: 0,
                    total_tokens: 5,
                }),
            }]),
            calculator_call_result(),
        );
        let (agent, llm) = streaming_agent(llm);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "计算 2 + 3".to_string());
        let mut emitted: Vec<String> = Vec::new();
        let mut on_token = |text: String| {
            emitted.push(text);
            Box::pin(async move {}) as Pin<Box<dyn Future<Output = ()> + Send>>
        };

        let output = agent
            .plan_stream(&[], &inputs, &mut on_token)
            .await
            .expect("plan_stream should succeed");

        assert!(
            matches!(&output, AgentOutput::Action(a) if a.tool == "calculator"),
            "tool-call step must return Action"
        );
        assert!(emitted.is_empty(), "tool-call step emits no free text");
        // 回退路径的用量来自非流式 plan()(预算门仍能拿到真实用量)。
        let usage = agent.last_token_usage().expect("usage via fallback plan");
        assert_eq!(usage.total_tokens, 15);
        // 工具调用步骤:先启动流(空文本),再回退非流式 chat 拿原生 tool_calls。
        assert_eq!(llm.calls(), vec!["stream_chat", "chat"]);
    }

    /// S2:`stream_chat` 立即可用即失败时,`plan_stream` 回退非流式 `plan()`,
    /// 最终答案作为单个 Text 事件转发(与旧非流式路径一致)。
    #[tokio::test]
    async fn test_function_calling_plan_stream_falls_back_on_immediate_error() {
        let llm = MockFuncLLM::new(None, text_result("Final Answer: 42"));
        let (agent, llm) = streaming_agent(llm);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "计算 6 * 7".to_string());
        let mut received = String::new();
        let mut on_token = |text: String| {
            received.push_str(&text);
            Box::pin(async move {}) as Pin<Box<dyn Future<Output = ()> + Send>>
        };

        let output = agent
            .plan_stream(&[], &inputs, &mut on_token)
            .await
            .expect("fallback plan should succeed");

        assert_eq!(received, "Final Answer: 42");
        assert!(matches!(output, AgentOutput::Finish(_)));
        // stream_chat 立即可用即失败 → 回退非流式 chat,整段答案作为单个 Text 事件。
        assert_eq!(llm.calls(), vec!["stream_chat", "chat"]);
    }
}
