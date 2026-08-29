// src/agents/react/agent.rs
//! ReAct Agent implementation
//!
//! Based on the paper "ReAct: Synergizing Reasoning and Acting in Language Models".
//! Supports any LLM provider that implements `BaseChatModel`.

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
/// An agent that uses the ReAct (Reasoning + Acting) pattern:
/// it first thinks, then decides which tool to execute, and finally observes the
/// result. Supports any LLM provider that implements `BaseChatModel`.
pub struct ReActAgent {
    /// LLM client
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,

    /// Available tools
    tools: Vec<Arc<dyn BaseTool>>,

    /// Output parser
    parser: ReActOutputParser,

    /// Custom system prompt (optional)
    system_prompt: Option<String>,

    /// Token usage from the most recent `plan()` call (P1-5).
    last_token_usage: std::sync::Mutex<Option<TokenUsage>>,
}

impl ReActAgent {
    /// Creates a new ReAct Agent
    ///
    /// # Parameters
    /// * `llm` - LLM client (any type implementing `BaseChatModel`)
    /// * `tools` - available tools
    /// * `system_prompt` - custom system prompt (optional)
    ///
    /// # Backward compatibility
    /// Legacy code `ReActAgent::new(openai_chat, tools, None)` still works,
    /// because `OpenAIChat: BaseChatModel` and `OpenAIError: Into<Error>`.
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

    /// Creates an agent from an already-wrapped `Arc<dyn BaseChatModel>`
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

    /// Formats the tool descriptions
    ///
    /// Formats the tool list into the format the ReAct prompt expects.
    fn format_tools(&self) -> String {
        self.tools
            .iter()
            .map(|tool| format!("{}: {}", tool.name(), tool.description()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns the list of tool names
    fn get_tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Builds the ReAct prompt
    ///
    /// # Parameters
    /// * `input` - user question
    /// * `intermediate_steps` - history of executed steps
    /// * `history` - conversation history (optional)
    fn build_prompt(
        &self,
        input: &str,
        intermediate_steps: &[AgentStep],
        history: Option<&str>,
    ) -> String {
        // Format the tool descriptions
        let tools_description = self.format_tools();
        let tool_names = self.get_tool_names();

        // Format the thought history (scratchpad)
        let scratchpad = format_scratchpad(intermediate_steps);

        // Build the base prompt
        let mut prompt = build_react_prompt(&tools_description, &tool_names, input, &scratchpad);

        // Prepend the conversation history if present
        if let Some(h) = history {
            if !h.is_empty() {
                prompt = format!("之前的对话历史:\n{}\n\n{}", h, prompt);
            }
        }

        // Prepend the custom system prompt if present
        if let Some(sys) = &self.system_prompt {
            prompt = format!("{}\n\n{}", sys, prompt);
        }

        prompt
    }
}

#[async_trait]
impl BaseAgent for ReActAgent {
    /// Plans the next action
    ///
    /// # Parameters
    /// * `intermediate_steps` - history of executed steps
    /// * `inputs` - user input
    ///
    /// # Returns
    /// * `AgentOutput::Action` - the action to execute
    /// * `AgentOutput::Finish` - the final answer
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        // Get the user input
        let input = inputs
            .get("input")
            .ok_or_else(|| AgentError::Other("Missing input parameter 'input'".to_string()))?;

        // Get the conversation history (if any)
        let history = inputs.get("history").map(|s| s.as_str());

        // Build the prompt
        let prompt_text = self.build_prompt(input, intermediate_steps, history);

        // Create the message
        let messages = vec![Message::human(prompt_text)];

        // Call the LLM
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

        // Parse the output
        self.parser.parse(&result.content)
    }

    /// Streaming plan (F3): forwards model output token by token, accumulating
    /// the full text before parsing.
    ///
    /// `plan()` goes through non-streaming `chat` (with retry, records token
    /// usage); this goes through `stream_chat`, forwarding each chunk via
    /// `on_token` as a live `Text` event while accumulating the full text for
    /// Action / Final Answer parsing.
    ///
    /// Trade-off: `stream_chat` chunks carry optional `token_usage`; after the
    /// stream ends it is written to `last_token_usage` for the budget gate to
    /// read. When the provider does not report usage (chunk.token_usage is
    /// `None`), the streaming path's metrics usage is filled in by the
    /// non-streaming `invoke` path. If `stream_chat` fails immediately (e.g. the
    /// provider does not implement streaming), it falls back to non-streaming
    /// `plan()` so the agent loop is not interrupted.
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

        // Token by token: forward the chunk live (cloned into an owned String),
        // then append to the full text.
        // Parsing only happens after the stream ends, so the Action / Final
        // Answer decision for a single ReAct step is unaffected.
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
        self.parser.parse(&full)
    }

    /// Returns the allowed tools list
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
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use lc_core::BaseLanguageModel;
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use lc_tools::Calculator;
    use std::pin::Pin;

    /// Creates an OpenAI config for tests
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

    /// S1 streaming mock: returns text chunk by chunk, with the last chunk
    /// carrying `token_usage`. Used to verify that `plan_stream` writes the
    /// streaming usage into `last_token_usage`, which the budget gate of
    /// `AgentExecutor::stream` reads.
    struct UsageStreamingLLM;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for UsageStreamingLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "Final Answer: 42".to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for UsageStreamingLLM {
        fn model_name(&self) -> &str {
            "mock"
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
    impl BaseChatModel for UsageStreamingLLM {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            // First chunk carries no usage, last chunk does — verifies "take the last non-None".
            let chunks = [
                Ok(StreamChunk::new("Final ")),
                Ok(StreamChunk {
                    text: "Answer: 42".to_string(),
                    token_usage: Some(TokenUsage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                    }),
                    tool_calls: None,
                }),
            ];
            Ok(Box::pin(futures_util::stream::iter(chunks)))
        }
    }

    /// S1: `plan_stream` forwards text chunk by chunk and writes the last
    /// non-None token_usage into `last_token_usage` (the streaming path's
    /// budget gate depends on it).
    #[tokio::test]
    async fn test_plan_stream_records_streaming_token_usage() {
        let llm = UsageStreamingLLM;
        let agent = ReActAgent::new(llm, vec![], None);

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "6 * 7".to_string());
        let mut received = String::new();
        let mut on_token = |text: String| {
            received.push_str(&text);
            Box::pin(async move {}) as Pin<Box<dyn Future<Output = ()> + Send>>
        };

        let output = agent
            .plan_stream(&[], &inputs, &mut on_token)
            .await
            .expect("plan_stream should parse to Finish");

        assert_eq!(received, "Final Answer: 42");
        assert!(matches!(output, AgentOutput::Finish(_)));
        let usage = agent.last_token_usage().expect("streaming usage recorded");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }
}
