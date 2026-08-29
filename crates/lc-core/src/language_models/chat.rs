// src/core/language_models/chat.rs
//! Chat model base trait.

use super::BaseLanguageModel;
use crate::tools::ToolDefinition;
use crate::RunnableConfig;
use async_trait::async_trait;
use futures_util::Stream;
use lc_schema::Message;
use lc_shared::tools::ToolCall;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// LLM result containing response content and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LLMResult {
    /// The generated response content.
    #[serde(default)]
    pub content: String,
    /// The model identifier that produced the result.
    #[serde(default)]
    pub model: String,
    /// Token usage statistics, if reported.
    #[serde(default)]
    pub token_usage: Option<TokenUsage>,
    /// Tool calls requested by the model, if any.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Model reasoning/thinking content, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_content: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input token count.
    pub prompt_tokens: usize,

    /// Output token count.
    pub completion_tokens: usize,

    /// Total token count.
    pub total_tokens: usize,
}

/// A single streaming chunk emitted by [`BaseChatModel::stream_chat`].
///
/// Replaces the bare `String` chunk so streaming paths can observe token
/// usage without a separate non-streaming `invoke`. Most providers only
/// populate [`StreamChunk::token_usage`] on the **final** chunk; intermediate
/// chunks carry `text` with `token_usage: None`. Providers that do not report
/// usage (Ollama, local, proxy passthrough) always yield `token_usage: None`.
///
/// 0.20.0 S3.2: [`StreamChunk::tool_calls`] carries the **complete** tool calls
/// requested by the model, accumulated from the provider's streaming
/// `tool_calls` deltas. OpenAI-family providers (OpenAI / Azure / their
/// delegates) attach them to the terminal chunk — the usage chunk, or a
/// dedicated tool-calls chunk when the stream ends without usage; providers
/// without streaming tool-call support always yield `None`. Consumers that only
/// stream text can ignore the field.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// The text delta for this chunk.
    pub text: String,
    /// Token usage for the whole streaming call, typically only on the last
    /// chunk (when the provider reports it).
    pub token_usage: Option<TokenUsage>,
    /// Complete tool calls requested by the model in this streaming call, if
    /// any. 0.20.0 S3.2: filled on the terminal chunk by providers that support
    /// streaming `tool_calls` (OpenAI / Azure and their delegates); `None`
    /// otherwise.
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl StreamChunk {
    /// Creates a text-only chunk with no token usage and no tool calls.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            token_usage: None,
            tool_calls: None,
        }
    }
}

/// Base trait for chat models.
///
/// Extends BaseLanguageModel for chat scenarios.
/// Accepts message list as input, returns AI message.
#[async_trait]
pub trait BaseChatModel: BaseLanguageModel<Vec<Message>, LLMResult> {
    /// Chat with the model.
    ///
    /// # Arguments
    /// * `messages` - Message list.
    /// * `config` - Optional configuration.
    ///
    /// # Returns
    /// LLM result.
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error>;

    /// Stream chat with the model.
    ///
    /// # Arguments
    /// * `messages` - Message list.
    /// * `config` - Optional configuration.
    ///
    /// # Returns
    /// Stream of [`StreamChunk`] items. Chunks carry the text delta; the final
    /// chunk may additionally carry [`StreamChunk::token_usage`] when the
    /// provider reports usage.
    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>;

    /// Chat with system prompt.
    ///
    /// # Arguments
    /// * `system` - System prompt.
    /// * `messages` - Message list.
    ///
    /// # Returns
    /// LLM result.
    async fn chat_with_system(
        &self,
        system: String,
        messages: Vec<Message>,
    ) -> Result<LLMResult, Self::Error> {
        let full_messages = vec![Message::system(system)]
            .into_iter()
            .chain(messages)
            .collect();

        self.chat(full_messages, None).await
    }

    /// Bind tool definitions for function calling.
    ///
    /// Returns `Some(model)` with the tools attached when the provider
    /// supports tool calling; returns `None` when it does not. **The default
    /// returns `None`, signalling a hard capability limit** — callers MUST
    /// treat `None` as "this model cannot call tools" and branch accordingly
    /// (e.g. fall back to text-only prompting). Providers that support
    /// function calling (OpenAI, Ollama) override this.
    ///
    /// This is an explicit result, not a silent degrade: `None` is the honest
    /// answer that tool-calling is unavailable on this model.
    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        None
    }
}

/// Error from [`predict_tools`].
#[derive(Debug, thiserror::Error)]
pub enum PredictToolsError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// The model's `bind_tools` returned `None` — the provider cannot call
    /// tools. Surfaced as an explicit error instead of silently degrading to a
    /// plain-text prompt.
    #[error("model does not support tool calling (bind_tools returned None); use a tool-capable model or call `chat` directly without tools")]
    ToolsUnsupported,

    /// Underlying chat model failure.
    #[error("chat model error: {0}")]
    Chat(#[source] E),
}

/// One-shot tool call: `bind_tools` + `chat` in a single entry point.
///
/// Binds `tools` to `llm`, sends `prompt` as a single human message, and returns
/// the model response — including any `tool_calls`. This is a thin convenience
/// for callers that want one turn with tools and plan to execute the tool calls
/// themselves; it does **not** run an agent loop or auto-execute tools (that is
/// `AgentExecutor`'s job).
///
/// # Errors
///
/// Returns [`PredictToolsError::ToolsUnsupported`] when the model's `bind_tools`
/// returns `None`, instead of silently degrading to a tool-less prompt.
pub async fn predict_tools<M>(
    llm: &M,
    prompt: impl Into<String>,
    tools: Vec<ToolDefinition>,
) -> Result<LLMResult, PredictToolsError<M::Error>>
where
    M: BaseChatModel + ?Sized,
{
    let Some(tool_llm) = llm.bind_tools(tools) else {
        return Err(PredictToolsError::ToolsUnsupported);
    };
    let messages = vec![Message::human(prompt.into())];
    tool_llm
        .chat(messages, None)
        .await
        .map_err(PredictToolsError::Chat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::Runnable;
    use futures_util::Stream;
    use std::pin::Pin;

    /// Mock model whose `bind_tools` attaches tools and whose `chat` echoes
    /// them back as `tool_calls` (tool-capable path).
    #[derive(Debug, Clone)]
    struct ToolCapableMock {
        tools: Option<Vec<ToolDefinition>>,
    }

    impl ToolCapableMock {
        fn new() -> Self {
            Self { tools: None }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for ToolCapableMock {
        type Error = MockError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(self.chat(_input, _config).await?)
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for ToolCapableMock {
        fn model_name(&self) -> &str {
            "mock-tool-capable"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for ToolCapableMock {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let tool_calls = self.tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        ToolCall::builder(format!("call_{i}"))
                            .name(t.function.name.clone())
                            .arguments("{}".to_string())
                            .build()
                    })
                    .collect()
            });
            Ok(LLMResult {
                content: if tool_calls.is_some() {
                    String::new()
                } else {
                    "plain reply".to_string()
                },
                model: "mock-tool-capable".to_string(),
                token_usage: None,
                tool_calls,
                thinking_content: None,
            })
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            unreachable!("stream_chat not exercised in predict_tools tests")
        }

        fn bind_tools(
            &self,
            tools: Vec<ToolDefinition>,
        ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
            Some(Box::new(Self { tools: Some(tools) }))
        }
    }

    /// Tool-capable model whose `chat` always fails (to exercise the `Chat` variant).
    #[derive(Debug, Clone)]
    struct FailingToolModel;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for FailingToolModel {
        type Error = MockError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockError("chat failed".to_string()))
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for FailingToolModel {
        fn model_name(&self) -> &str {
            "mock-failing"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for FailingToolModel {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockError("chat failed".to_string()))
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            unreachable!("stream_chat not exercised in predict_tools tests")
        }

        fn bind_tools(
            &self,
            _tools: Vec<ToolDefinition>,
        ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
            Some(Box::new(Self))
        }
    }

    /// Mock model using the default `bind_tools` (returns `None` — cannot call tools).
    #[derive(Debug)]
    struct ToolIncapableMock;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for ToolIncapableMock {
        type Error = MockError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "plain reply".to_string(),
                model: "mock-tool-incapable".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for ToolIncapableMock {
        fn model_name(&self) -> &str {
            "mock-tool-incapable"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for ToolIncapableMock {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "plain reply".to_string(),
                model: "mock-tool-incapable".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            let stream = futures_util::stream::once(async move { Ok(StreamChunk::new("plain")) });
            Ok(Box::pin(stream))
        }
    }

    #[derive(Debug)]
    struct MockError(String);

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockError: {}", self.0)
        }
    }

    impl std::error::Error for MockError {}

    #[tokio::test]
    async fn predict_tools_binds_tools_and_returns_tool_calls() {
        let llm = ToolCapableMock::new();
        let tools = vec![ToolDefinition::new("get_weather", "Get current weather")];

        let result = predict_tools(&llm, "weather in beijing?", tools)
            .await
            .unwrap();

        let calls = result.tool_calls.expect("tool_calls should be present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name(), "get_weather");
    }

    #[tokio::test]
    async fn predict_tools_returns_clear_error_when_model_cannot_bind() {
        let llm = ToolIncapableMock;
        let tools = vec![ToolDefinition::new("get_weather", "Get current weather")];

        let err = predict_tools(&llm, "weather in beijing?", tools)
            .await
            .unwrap_err();

        assert!(
            matches!(err, PredictToolsError::ToolsUnsupported),
            "expected ToolsUnsupported, got {err:?}"
        );
    }

    #[tokio::test]
    async fn predict_tools_propagates_chat_error() {
        // Tool-capable model failing inside chat: the underlying error is
        // surfaced via the `Chat` variant, not swallowed.
        let llm = FailingToolModel;
        let tools = vec![ToolDefinition::new("get_weather", "Get current weather")];

        let err = predict_tools(&llm, "weather in beijing?", tools)
            .await
            .unwrap_err();

        assert!(
            matches!(err, PredictToolsError::Chat(ref e) if e.0 == "chat failed"),
            "expected Chat(chat failed), got {err:?}"
        );
    }
}
