//! 测试辅助:支持 `bind_tools` 并返回 `tool_calls` 的 mock 裁判,
//! 用于验证 P0-1 结构化输出路径(不依赖真实网络/API)。

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::LLMResult;
use lc_core::tools::{ToolCall, ToolDefinition};
use lc_core::{BaseChatModel, BaseLanguageModel, Runnable, RunnableConfig};
use lc_schema::Message;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// mock 裁判错误
#[derive(Debug, Clone)]
pub(crate) struct JudgeError(String);
impl std::fmt::Display for JudgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for JudgeError {}

/// 支持 bind_tools 的 mock 裁判:每次 chat 返回预设 `arguments` 序列中的一个 tool_call
/// (按调用次序逐条取,取尽后回空参)。
#[derive(Clone)]
pub(crate) struct ToolJudge {
    /// 每次 chat 返回的参数;`single` 时永远取第一份,否则按调用次序取
    replies: Vec<String>,
    single: bool,
    calls: Arc<AtomicUsize>,
}

impl ToolJudge {
    /// 每次调用都返回同一份参数。
    pub(crate) fn new(arguments: impl Into<String>) -> Self {
        Self {
            replies: vec![arguments.into()],
            single: true,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// 按调用次序依次返回参数(用于 pairwise 两次 ask 返回不同判定、多 claim 逐条判定)。
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        Err(JudgeError("not supported".into()))
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        Some(Box::new(self.clone()))
    }
}
