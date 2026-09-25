// src/agents/function_calling/agent.rs
//! Function Calling Agent implementation
//!
//! An agent that uses the LLM's native Function Calling, without text parsing.
//! Supports any LLM provider that implements `BaseChatModel`.

use crate::{AgentAction, AgentError, AgentFinish, AgentOutput, AgentStep, BaseAgent, ToolInput};
use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::language_models::{BaseChatModel, LLMResult, TokenUsage};
use lc_core::runnables::RunnableConfig;
use lc_core::tools::{to_tool_definition, BaseTool, ToolCall, ToolDefinition};
use lc_providers::ProviderError;
use lc_schema::Message;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Function Calling Agent
///
/// An agent that uses the LLM's native Function Calling.
/// Does not rely on text parsing; handles `tool_calls` directly.
/// Supports any LLM provider that implements `BaseChatModel`.
pub struct FunctionCallingAgent {
    /// LLM client (with tools bound)
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,

    /// Available tools
    tools: Vec<Arc<dyn BaseTool>>,

    /// Custom system prompt
    system_prompt: Option<String>,

    /// Token usage from the most recent `plan()` call (P1-5).
    last_token_usage: std::sync::Mutex<Option<TokenUsage>>,
}

impl FunctionCallingAgent {
    /// Creates a new Function Calling Agent
    ///
    /// # Parameters
    /// * `llm` - LLM client (any type implementing `BaseChatModel`)
    /// * `tools` - available tools
    /// * `system_prompt` - custom system prompt (optional)
    ///
    /// # Backward compatibility
    /// Legacy code `FunctionCallingAgent::new(openai_chat, tools, None)` still works,
    /// because `OpenAIChat: BaseChatModel` and `OpenAIError: Into<Error>`.
    pub fn new<L>(llm: L, tools: Vec<Arc<dyn BaseTool>>, system_prompt: Option<String>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        // Wrap the LLM first, unifying the error type to ProviderError
        let wrapped = lc_providers::ChatModelWrapper::new(llm);

        let tool_definitions: Vec<ToolDefinition> = tools
            .iter()
            .map(|t| to_tool_definition(t.as_ref()))
            .collect();

        // Prefer the trait bind_tools (returns Box<dyn BaseChatModel<Error = ProviderError>>)
        // Fall back to the wrapped LLM if the provider does not support it
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

    /// Creates an agent from an already-wrapped `Arc<dyn BaseChatModel>`
    ///
    /// For LLM instances already created via `wrap_chat_model()` or `LLMClient`.
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

    /// Returns the number of tools
    pub fn tools_count(&self) -> usize {
        self.tools.len()
    }

    /// Returns the system prompt
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Builds the messages
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
            // B5: pair the ai-side ToolCall id with the tool-result call_id. Uses the
            // action's explicit `tool_call_id` when present; otherwise derives a stable
            // per-action id so the pair stays linked across rebuilds.
            let id = resolve_tool_call_id(&step.action);
            let tool_call = ToolCall::builder(&id)
                .name(&step.action.tool)
                .arguments(match &step.action.tool_input {
                    ToolInput::String { value: s } => s.clone(),
                    ToolInput::Object { value: v } => {
                        serde_json::to_string(v).unwrap_or_else(|_| v.to_string())
                    }
                })
                .build();
            messages.push(Message::ai_with_tool_calls("", vec![tool_call]));
            messages.push(Message::tool(&id, &step.observation));
        }

        messages
    }

    /// Converts complete [`ToolCall`]s from an LLM response into an [`AgentOutput`].
    ///
    /// Shared by the non-streaming [`BaseAgent::plan`] and the streaming
    /// [`BaseAgent::plan_stream`]: both receive the same native `tool_calls`
    /// shape, so they build actions identically. 0.20.0 S3.2: `plan_stream`
    /// now reaches this through the streamed chunks instead of a non-streaming
    /// fallback.
    fn output_from_tool_calls(tool_calls: &[ToolCall]) -> AgentOutput {
        let actions: Vec<AgentAction> = tool_calls
            .iter()
            .map(|call| {
                let tool_input =
                    match serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
                        Ok(v) => ToolInput::Object { value: v },
                        Err(_) => ToolInput::String {
                            value: call.function.arguments.clone(),
                        },
                    };

                AgentAction {
                    tool: call.function.name.clone(),
                    tool_input,
                    log: String::new(),
                    // B5: carry the provider tool-call id on the action instead of
                    // stuffing it into `log`, so rebuilders can pair the ToolCall with
                    // its tool result by real id and the executor can stamp it into
                    // observability without guessing.
                    tool_call_id: Some(call.id.clone()),
                }
            })
            .collect();

        if actions.len() == 1 {
            AgentOutput::Action(actions.into_iter().next().expect("checked len == 1"))
        } else {
            AgentOutput::Actions(actions)
        }
    }
}

/// B5: resolves the id used to pair a `ToolCall` (ai message) with its `tool` result.
///
/// Returns the action's explicit `tool_call_id` when present and non-empty; otherwise
/// derives a deterministic id from `(tool, input)` so the pair stays linked across
/// message rebuilds even for agents that carry no native id.
fn resolve_tool_call_id(action: &AgentAction) -> String {
    if let Some(id) = &action.tool_call_id {
        if !id.is_empty() {
            return id.clone();
        }
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    action.tool.hash(&mut hasher);
    let input_repr = match &action.tool_input {
        ToolInput::String { value: s } => serde_json::to_string(s).unwrap_or_default(),
        ToolInput::Object { value: v } => serde_json::to_string(v).unwrap_or_default(),
    };
    input_repr.hash(&mut hasher);
    format!("call_{:016x}", hasher.finish())
}

#[async_trait]
impl BaseAgent for FunctionCallingAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
        config: Option<&RunnableConfig>,
    ) -> Result<AgentOutput, AgentError> {
        let messages = self.build_messages(inputs, intermediate_steps);

        // A18: forward the executor's config so the provider dispatches the
        // planning-round `on_llm_*` callbacks.
        let result: LLMResult = crate::retry::retry_chat(
            self.llm.as_ref(),
            messages,
            config.cloned(),
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
                return Ok(Self::output_from_tool_calls(tool_calls));
            }
        }

        Ok(AgentOutput::Finish(AgentFinish::new(
            result.content.clone(),
            String::new(),
        )))
    }

    /// Streaming plan (S2 + 0.20.0 S3.2): goes through `stream_chat`, forwarding
    /// model text token by token and accumulating usage **and tool calls**.
    ///
    /// The function-calling agent's **final answer** streams out as text token by
    /// token (typewriter effect). **Tool-call steps** now stream natively too:
    /// providers that support streaming `tool_calls` (OpenAI / Azure and their
    /// delegates) attach the complete tool calls to the terminal `StreamChunk`
    /// (`StreamChunk.tool_calls`), which `plan_stream` accumulates and converts
    /// into [`AgentOutput::Action`]/[`AgentOutput::Actions`] — no non-streaming
    /// fallback needed, so the agent loop does not emit a fake "empty Finish".
    /// **Mixed steps** (text + tool calls) keep both: the text already streamed
    /// via `on_token`, the tool calls preserved here.
    ///
    /// The non-streaming [`BaseAgent::plan`] fallback remains only as a safety net
    /// for: `stream_chat` failing immediately, or a provider that yields neither
    /// text nor `tool_calls` on the stream (e.g. one without streaming tool-call
    /// support when the model makes a tool call).
    async fn plan_stream(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
        on_token: &mut (dyn FnMut(String) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send),
        config: Option<&RunnableConfig>,
    ) -> Result<AgentOutput, AgentError> {
        let messages = self.build_messages(inputs, intermediate_steps);

        // A18: stream planning carries the same config as non-streaming plan.
        let mut stream = match self.llm.stream_chat(messages, config.cloned()).await {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "stream_chat unavailable ({}), falling back to non-streaming plan",
                    e
                );
                let output = self.plan(intermediate_steps, inputs, config).await?;
                if let AgentOutput::Finish(finish) = &output {
                    on_token(finish.output().unwrap_or("").to_string()).await;
                }
                return Ok(output);
            }
        };

        // Token by token: forward non-empty text, accumulate the full text + the
        // last non-None usage, and the last non-empty set of complete tool calls
        // (providers attach them to the terminal chunk).
        let mut full = String::new();
        let mut usage: Option<TokenUsage> = None;
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AgentError::Other(format!("LLM stream error: {}", e)))?;
            if !chunk.text.is_empty() {
                on_token(chunk.text.clone()).await;
            }
            full.push_str(&chunk.text);
            if chunk.token_usage.is_some() {
                usage = chunk.token_usage;
            }
            // 0.20.0 S3.2: take the last non-empty tool_calls set so a tool-call
            // step is returned natively below instead of falling back.
            if let Some(tc) = &chunk.tool_calls {
                if !tc.is_empty() {
                    tool_calls = Some(tc.clone());
                }
            }
        }
        if let Ok(mut guard) = self.last_token_usage.lock() {
            *guard = usage;
        }

        // 0.20.0 S3.2: tool-call steps stream natively — return the accumulated
        // tool_calls as Action/Actions. Mixed steps keep both: the text already
        // streamed via on_token, the tool calls preserved here.
        if let Some(tc) = &tool_calls {
            return Ok(Self::output_from_tool_calls(tc));
        }

        // Neither text nor tool_calls on the stream (a provider without streaming
        // tool-call support on a tool-call step, or an empty reply): fall back to
        // non-streaming plan() so the agent loop does not end early on an empty
        // Finish.
        if full.trim().is_empty() {
            log::debug!(
                "streamed plan produced neither text nor tool_calls, \
                 falling back to non-streaming plan"
            );
            let output = self.plan(intermediate_steps, inputs, config).await?;
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
                tool_call_id: Some("call_123".to_string()),
            },
            "5".to_string(),
        )];

        let messages = agent.build_messages(&inputs, &steps);

        assert_eq!(messages.len(), 4);
        assert!(messages[2].has_tool_calls());
    }

    /// B5 (design point 1): the ai-side `ToolCall.id` must pair with the tool-result
    /// `Message::tool` call_id. An explicit `tool_call_id` on the action is reused
    /// verbatim; when the action carries `None`, the same deterministic id is put on
    /// both sides so the pair stays linked across message rebuilds.
    #[test]
    fn tool_call_id_pairs_explicit_and_synthesized() {
        let config = create_test_config();
        let llm = OpenAIChat::new(config);
        let agent = FunctionCallingAgent::new(llm, vec![], None);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "go".to_string());

        // 1. Explicit id: both the ToolCall id and the tool-result call_id equal it.
        let steps = vec![AgentStep::new(
            AgentAction {
                tool: "calculator".to_string(),
                tool_input: ToolInput::String {
                    value: "1+1".to_string(),
                },
                log: String::new(),
                tool_call_id: Some("call_explicit".to_string()),
            },
            "2".to_string(),
        )];
        let messages = agent.build_messages(&inputs, &steps);
        // [system, human, ai-with-tool-call, tool-result]
        let ai_id = messages[2].get_tool_calls().unwrap()[0].id.clone();
        assert_eq!(
            ai_id, "call_explicit",
            "explicit id must flow to the ToolCall"
        );
        match &messages[3].message_type {
            lc_schema::MessageType::Tool { tool_call_id } => {
                assert_eq!(
                    tool_call_id, &ai_id,
                    "tool-result call_id must equal the ToolCall id"
                );
            }
            other => panic!("expected a Tool message, got {other:?}"),
        }

        // 2. None → non-empty synthesized id, identical on both sides, and
        //    deterministic for the same (tool, input).
        let steps2 = vec![AgentStep::new(
            AgentAction {
                tool: "calculator".to_string(),
                tool_input: ToolInput::String {
                    value: "1+1".to_string(),
                },
                log: String::new(),
                tool_call_id: None,
            },
            "2".to_string(),
        )];
        let m2 = agent.build_messages(&inputs, &steps2);
        let tc2_id = m2[2].get_tool_calls().unwrap()[0].id.clone();
        assert!(
            !tc2_id.is_empty() && tc2_id.starts_with("call_"),
            "synth id must be non-empty and namespaced, got {tc2_id}"
        );
        match &m2[3].message_type {
            lc_schema::MessageType::Tool { tool_call_id } => {
                assert_eq!(tool_call_id, &tc2_id, "synthesized ids must pair");
            }
            other => panic!("expected a Tool message, got {other:?}"),
        }
        let m3 = agent.build_messages(&inputs, &steps2);
        let tc3_id = m3[2].get_tool_calls().unwrap()[0].id.clone();
        assert_eq!(
            tc2_id, tc3_id,
            "same (tool, input) must yield the same synthesized id"
        );
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

    /// S2 streaming mock: configurable `stream_chat` returns (normal chunks /
    /// immediate failure) and `chat` returns (tool call / final answer), and it
    /// records the call sequence, to verify FunctionCallingAgent's `plan_stream`
    /// override: token-by-token forwarding, empty-stream fallback to
    /// non-streaming plan, and immediate-failure fallback.
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

    /// S2: the final answer streams out token by token (Text events), and the streaming usage is recorded for the budget gate.
    #[tokio::test]
    async fn test_function_calling_plan_stream_streams_final_answer() {
        let llm = MockFuncLLM::new(
            Some(vec![
                StreamChunk::new("Final "),
                StreamChunk {
                    thinking_content: None,
                    text: "Answer: 42".to_string(),
                    token_usage: Some(TokenUsage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                    }),
                    tool_calls: None,
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
            .plan_stream(&[], &inputs, &mut on_token, None)
            .await
            .expect("plan_stream should succeed");

        assert_eq!(received, "Final Answer: 42");
        assert!(matches!(
            output,
            AgentOutput::Finish(f) if f.output() == Some("Final Answer: 42")
        ));
        let usage = agent.last_token_usage().expect("streaming usage recorded");
        assert_eq!(usage.total_tokens, 15);
        // Took the streaming path: only stream_chat was called, no chat fallback.
        assert_eq!(llm.calls(), vec!["stream_chat"]);
    }

    /// S2 + 0.20.0 S3.2: a provider **without** streaming tool_calls support
    /// yields only empty text + usage chunks on a tool-call step — `plan_stream`
    /// falls back to non-streaming `plan()` to get the native tool calls, without
    /// a fake "empty Finish" stream. Providers that DO stream tool_calls take the
    /// native path (see `test_..._streams_tool_call_natively`).
    #[tokio::test]
    async fn test_function_calling_plan_stream_falls_back_when_no_streaming_tool_calls() {
        // Simulate a tool-call step: the stream only returns empty text + usage chunks.
        let llm = MockFuncLLM::new(
            Some(vec![StreamChunk {
                thinking_content: None,
                text: String::new(),
                token_usage: Some(TokenUsage {
                    prompt_tokens: 5,
                    completion_tokens: 0,
                    total_tokens: 5,
                }),
                tool_calls: None,
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
            .plan_stream(&[], &inputs, &mut on_token, None)
            .await
            .expect("plan_stream should succeed");

        assert!(
            matches!(&output, AgentOutput::Action(a) if a.tool == "calculator"),
            "tool-call step must return Action"
        );
        assert!(emitted.is_empty(), "tool-call step emits no free text");
        // The fallback path's usage comes from non-streaming plan() (the budget gate still gets real usage).
        let usage = agent.last_token_usage().expect("usage via fallback plan");
        assert_eq!(usage.total_tokens, 15);
        // Tool-call step: start the stream (empty text) first, then fall back to non-streaming chat for native tool_calls.
        assert_eq!(llm.calls(), vec!["stream_chat", "chat"]);
    }

    /// S2: when `stream_chat` fails immediately, `plan_stream` falls back to
    /// non-streaming `plan()`, forwarding the final answer as a single Text event
    /// (consistent with the old non-streaming path).
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
            .plan_stream(&[], &inputs, &mut on_token, None)
            .await
            .expect("fallback plan should succeed");

        assert_eq!(received, "Final Answer: 42");
        assert!(matches!(output, AgentOutput::Finish(_)));
        // stream_chat fails immediately → fall back to non-streaming chat, the whole answer forwarded as a single Text event.
        assert_eq!(llm.calls(), vec!["stream_chat", "chat"]);
    }

    /// 0.20.0 S3.2: a provider WITH streaming tool_calls support lets a pure
    /// tool-call step stream natively — the terminal chunk carries the tool calls,
    /// so `plan_stream` returns Action without a non-streaming fallback (no `chat`
    /// call at all).
    #[tokio::test]
    async fn test_function_calling_plan_stream_streams_tool_call_natively() {
        let tool_chunk = StreamChunk {
            thinking_content: None,
            text: String::new(),
            token_usage: Some(TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            }),
            tool_calls: Some(vec![ToolCall::builder("call_1")
                .name("calculator")
                .arguments(r#"{"expression": "2+3"}"#)
                .build()]),
        };
        let llm = MockFuncLLM::new(Some(vec![tool_chunk]), text_result("unused"));
        let (agent, llm) = streaming_agent(llm);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "计算 2 + 3".to_string());
        let mut emitted: Vec<String> = Vec::new();
        let mut on_token = |text: String| {
            emitted.push(text);
            Box::pin(async move {}) as Pin<Box<dyn Future<Output = ()> + Send>>
        };

        let output = agent
            .plan_stream(&[], &inputs, &mut on_token, None)
            .await
            .expect("plan_stream should succeed");

        assert!(
            matches!(&output, AgentOutput::Action(a) if a.tool == "calculator"),
            "tool-call step must return Action natively"
        );
        assert!(emitted.is_empty(), "no free text on a pure tool-call step");
        // Native path: only stream_chat ran — no non-streaming fallback.
        assert_eq!(llm.calls(), vec!["stream_chat"]);
    }

    /// 0.20.0 S3.2: a mixed step (model emits text AND a tool call in one stream)
    /// keeps both — the text streams out token by token via on_token, and the
    /// tool call is returned as an Action (previously the tool call was silently
    /// dropped).
    #[tokio::test]
    async fn test_function_calling_plan_stream_mixed_step_keeps_text_and_tool_call() {
        let llm = MockFuncLLM::new(
            Some(vec![
                StreamChunk::new("Let me compute"),
                StreamChunk {
                    thinking_content: None,
                    text: String::new(),
                    token_usage: Some(TokenUsage {
                        prompt_tokens: 5,
                        completion_tokens: 0,
                        total_tokens: 5,
                    }),
                    tool_calls: Some(vec![ToolCall::builder("call_1")
                        .name("calculator")
                        .arguments(r#"{"expression": "2+3"}"#)
                        .build()]),
                },
            ]),
            text_result("unused"),
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
            .plan_stream(&[], &inputs, &mut on_token, None)
            .await
            .expect("plan_stream should succeed");

        assert_eq!(emitted, vec!["Let me compute"], "preamble streams out");
        assert!(
            matches!(&output, AgentOutput::Action(a) if a.tool == "calculator"),
            "tool call preserved, not dropped"
        );
        // Native path throughout — no non-streaming fallback.
        assert_eq!(llm.calls(), vec!["stream_chat"]);
    }
}
