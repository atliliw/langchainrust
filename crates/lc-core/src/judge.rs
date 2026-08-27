//! Shared LLM judge infrastructure: lets an LLM return a verdict via structured arguments.
//!
//! Reused by `lc-evaluation` (scoring / pairwise / faithfulness judges) and `lc-guardrails`
//! (LLM validators): prefer `bind_tools` for structured `tool_calls` arguments; when the model
//! does not support tool binding or still returns plain text, fall back to the caller-provided
//! text parsing. Each crate builds its own prompts and parse rules; what is shared is the common
//! execution path "bind tool → take structured arguments → fall back to text".

use lc_schema::Message;
use serde::de::DeserializeOwned;

use crate::language_models::BaseChatModel;
use crate::tools::ToolDefinition;

/// Structured judge-call errors: distinguishes "LLM call failure" from "structured parse failure",
/// mapped by the caller into its own error domain (e.g. `EvalError::PredictorError` /
/// `EvalError::ParseError`).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StructuredJudgeError {
    /// Underlying LLM call failure (network / rate limit / abnormal reply).
    #[error("LLM call failed: {0}")]
    Call(String),
    /// Tool-call arguments cannot be deserialized into the target type, or tool_calls is empty.
    #[error("structured parse failed: {0}")]
    Parse(String),
}

/// One LLM call, letting the judge return a verdict as structured arguments (T).
///
/// Flow:
/// 1. If the model supports `bind_tools`, bind the verdict tool; parse and return the arguments when the reply has `tool_calls`.
/// 2. Tool bound but still plain text → run the same reply's text through `text_fallback`.
/// 3. Model without `bind_tools` → fall back to text parsing (`text_fallback`) and `log::warn!`.
/// 4. Tool bound but `tool_calls` arguments cannot be parsed → explicit `StructuredJudgeError::Parse`,
///    never a silent default.
///
/// Guarantees at most one LLM round trip per verdict: the bound path does not re-issue a tool-less call.
pub async fn structured_call<M, T, F>(
    judge: &M,
    tool: ToolDefinition,
    messages: Vec<Message>,
    text_fallback: F,
) -> Result<T, StructuredJudgeError>
where
    M: BaseChatModel,
    T: DeserializeOwned,
    F: FnOnce(&str) -> Result<T, StructuredJudgeError>,
{
    if let Some(bound) = judge.bind_tools(vec![tool]) {
        let result = bound
            .chat(messages, None)
            .await
            .map_err(|e| StructuredJudgeError::Call(e.to_string()))?;
        match result.tool_calls {
            Some(calls) => {
                let call = calls.first().ok_or_else(|| {
                    StructuredJudgeError::Parse("judge returned empty tool_calls".to_string())
                })?;
                let parsed = call.parse_arguments::<T>().map_err(|e| {
                    StructuredJudgeError::Parse(format!(
                        "failed to parse judge structured arguments: {}",
                        e
                    ))
                })?;
                Ok(parsed)
            }
            None => {
                log::warn!(
                    "judge model bound tools but returned plain text; falling back to text parsing"
                );
                text_fallback(&result.content)
            }
        }
    } else {
        log::warn!("judge model does not support bind_tools; falling back to text parsing");
        let result = judge
            .chat(messages, None)
            .await
            .map_err(|e| StructuredJudgeError::Call(e.to_string()))?;
        text_fallback(&result.content)
    }
}

/// Truncates long text for error messages, avoiding stuffing a whole LLM reply into an error.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_schema::Message;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::language_models::{LLMResult, StreamChunk};
    use crate::{BaseLanguageModel, Runnable, RunnableConfig};

    #[derive(Debug)]
    struct JudgeError(String);
    impl std::fmt::Display for JudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for JudgeError {}

    /// Mock judge returning preset replies in order, recording received messages for assertions
    struct SeqMockJudge {
        replies: Vec<String>,
        call: Arc<AtomicUsize>,
    }
    impl SeqMockJudge {
        fn new(replies: Vec<String>) -> Self {
            Self {
                replies,
                call: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for SeqMockJudge {
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
    impl BaseLanguageModel<Vec<Message>, LLMResult> for SeqMockJudge {
        fn model_name(&self) -> &str {
            "seq-mock"
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
    impl BaseChatModel for SeqMockJudge {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let idx = self.call.fetch_add(1, Ordering::SeqCst);
            let reply = self.replies.get(idx).cloned().unwrap_or_default();
            Ok(LLMResult {
                content: reply,
                model: "seq-mock".to_string(),
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
            Err(JudgeError("not supported".into()))
        }
    }

    #[derive(serde::Deserialize, Debug)]
    struct MockArgs {
        verdict: String,
    }

    fn mock_tool() -> ToolDefinition {
        ToolDefinition::new("mock_judge", "返回判定。")
    }

    #[tokio::test]
    async fn test_fallback_on_text_only_model() {
        // SeqMockJudge does not implement bind_tools → text fallback; the closure parses plain text.
        let judge = SeqMockJudge::new(vec!["yes".into()]);
        let messages = vec![Message::human("判断")];
        let out = structured_call(&judge, mock_tool(), messages, |raw| {
            Ok(MockArgs {
                verdict: raw.trim().to_string(),
            })
        })
        .await
        .unwrap();
        assert_eq!(out.verdict, "yes");
    }

    #[tokio::test]
    async fn test_parse_error_raised_not_silently_defaulted() {
        // text-fallback parse failure → explicit Parse, no silent default.
        let judge = SeqMockJudge::new(vec!["没法判断".into()]);
        let messages = vec![Message::human("判断")];
        let err = structured_call(
            &judge,
            mock_tool(),
            messages,
            |_raw: &str| -> Result<MockArgs, StructuredJudgeError> {
                Err(StructuredJudgeError::Parse("parse failed".into()))
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, StructuredJudgeError::Parse(_)));
    }
}
