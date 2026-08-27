//! Test helper: a mock judge that supports `bind_tools` and returns `tool_calls`,
//! used to verify the P0-1 structured-output path (no real network/API dependency).

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{LLMResult, StreamChunk};
use lc_core::tools::{ToolCall, ToolDefinition};
use lc_core::{BaseChatModel, BaseLanguageModel, Runnable, RunnableConfig};
use lc_schema::Message;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Mock judge error
#[derive(Debug, Clone)]
pub(crate) struct JudgeError(String);
impl std::fmt::Display for JudgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for JudgeError {}

/// Mock judge supporting bind_tools: each chat returns one tool_call from the preset `arguments` sequence
/// (taken in call order; after exhausting, falls back to empty arguments).
#[derive(Clone)]
pub(crate) struct ToolJudge {
    /// Arguments returned per chat; with `single` always the first, otherwise in call order
    replies: Vec<String>,
    single: bool,
    calls: Arc<AtomicUsize>,
}

impl ToolJudge {
    /// Returns the same arguments on every call.
    pub(crate) fn new(arguments: impl Into<String>) -> Self {
        Self {
            replies: vec![arguments.into()],
            single: true,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Returns arguments in call order (used when pairwise asks twice with different verdicts, or multiple claims judged one by one).
    pub(crate) fn sequence(replies: Vec<String>) -> Self {
        Self {
            replies,
            single: false,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for ToolJudge {
    type Error = JudgeError;
    async fn invoke(
        &self,
        _input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Err(JudgeError("use chat".into()))
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for ToolJudge {
    fn model_name(&self) -> &str {
        "tool-judge"
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
impl BaseChatModel for ToolJudge {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let idx = self.calls.fetch_add(1, Ordering::SeqCst);
        let slot = if self.single { 0 } else { idx };
        let arguments = self.replies.get(slot).cloned().unwrap_or_default();
        Ok(LLMResult {
            content: String::new(),
            model: "tool-judge".to_string(),
            token_usage: None,
            tool_calls: Some(vec![ToolCall::builder("call_1")
                .name("judge_tool")
                .arguments(arguments)
                .build()]),
            thinking_content: None,
        })
    }
    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        Err(JudgeError("not supported".into()))
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        Some(Box::new(self.clone()))
    }
}
