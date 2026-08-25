//! 共享 LLM 裁判基础设施:让 LLM 以结构化参数返回判定。
//!
//! 被 `lc-evaluation`(打分 / 成对 / 忠实度裁判)与 `lc-guardrails`(LLM 校验器)
//! 复用:优先走 `bind_tools` 拿 `tool_calls` 结构化参数,模型不支持工具绑定或
//! 仍返回纯文本时,回落调用方提供的文本解析。两个 crate 各自构造 prompt 与
//! 解析规则,共享的是"绑定工具 → 拿结构化参数 → 回落文本"这条通用执行路径。

use lc_schema::Message;
use serde::de::DeserializeOwned;

use crate::language_models::BaseChatModel;
use crate::tools::ToolDefinition;

/// 结构化裁判调用的错误:区分"LLM 调用失败"与"结构化解析失败",
/// 由调用方映射到自己的错误域(如 `EvalError::PredictorError` /
/// `EvalError::ParseError`)。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StructuredJudgeError {
    /// 底层 LLM 调用失败(网络 / 限流 / 返回异常)。
    #[error("LLM call failed: {0}")]
    Call(String),
    /// 工具调用参数无法按目标类型反序列化,或 tool_calls 为空。
    #[error("structured parse failed: {0}")]
    Parse(String),
}

/// 单次 LLM 调用,让裁判以结构化参数(T)返回判定。
///
/// 流程:
/// 1. 若模型支持 `bind_tools`,绑定判定工具;响应含 `tool_calls` 则解析参数返回。
/// 2. 绑定了工具但仍返回纯文本 → 用同一次响应的文本走 `text_fallback`。
/// 3. 模型不支持 `bind_tools` → 回落文本解析(`text_fallback`),并 `log::warn!`。
/// 4. 绑定了工具但 `tool_calls` 参数无法解析 → 显式 `StructuredJudgeError::Parse`,
///    绝不静默默认。
///
/// 保证每次判定最多一次 LLM 往返:绑定路径不额外重打一次无工具调用。
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

/// 截断长文本用于错误信息,避免把整段 LLM 回复塞进错误。
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

    use crate::language_models::LLMResult;
    use crate::{BaseLanguageModel, Runnable, RunnableConfig};

    #[derive(Debug)]
    struct JudgeError(String);
    impl std::fmt::Display for JudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for JudgeError {}

    /// 依次返回预设回复的 mock 裁判,记录收到的消息供断言。
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
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
        // SeqMockJudge 不实现 bind_tools → 走文本回落,closure 解析纯文本。
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
        // 文本回落解析失败 → 显式 Parse,不静默默认。
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
