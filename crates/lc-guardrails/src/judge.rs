//! Sensitive-leak LLM judge (P2-3)
//!
//! When `SensitiveInfoGuardrail`'s high-false-positive "mention" keywords (e.g. password/token)
//! hit context-sensitively, the LLM judge makes the final call on whether a real leak occurred —
//! real key/credential values are blocked, while normal mentions such as "how to store passwords
//! safely" pass, lowering false positives.
//!
//! Reuses the shared judge infrastructure [`lc_core::judge::structured_call`] (same lineage as
//! lc-evaluation's Faithfulness / Pairwise judges, evaluation P2-6): it prefers `bind_tools`
//! for structured `tool_calls` arguments, falling back to text parsing when the model does not
//! support tool binding or returns plain text.
//!
//! This is also part of the P2-4 dead-dependency handling: `lc-core` had zero references in
//! lc-guardrails' src, and this real connection is the "actually connect" route from the plan
//! (either remove it, or actually connect it).

use async_trait::async_trait;

use crate::guardrail::GuardrailError;

use lc_core::judge::{structured_call, StructuredJudgeError};
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_schema::Message;

/// Sensitive-leak judge: makes the "real leak vs normal mention" second determination for suspected sensitive output.
///
/// `judge` returns `true` = judged a real leak (should block); `false` = normal mention (should pass).
#[async_trait]
pub trait SensitiveJudge: Send + Sync {
    /// The judge's name.
    fn name(&self) -> &str;

    /// Determines whether a text leaks real sensitive information.
    async fn judge(&self, text: &str) -> Result<bool, GuardrailError>;
}

/// Sensitive-leak judge based on the shared LLM judge infrastructure.
///
/// Uses [`structured_call`] to have the judge submit `{"is_leak": bool, "reason": "..."}` as
/// structured arguments; when the model does not support tool binding or returns plain text,
/// falls back to `parse_leak_text` text parsing.
pub struct LlmSensitiveJudge<M: BaseChatModel> {
    judge: M,
}

impl<M: BaseChatModel> LlmSensitiveJudge<M> {
    /// Creates a judge from the given LLM.
    pub fn new(judge: M) -> Self {
        Self { judge }
    }
}

#[async_trait]
impl<M: BaseChatModel> SensitiveJudge for LlmSensitiveJudge<M> {
    fn name(&self) -> &str {
        "llm_sensitive_judge"
    }

    async fn judge(&self, text: &str) -> Result<bool, GuardrailError> {
        let system =
            "你是安全审查员。判断给定文本是否泄露了真实的敏感信息(真实的密码、密钥、凭证值等)。\
                      正常提及、教学或讨论场景(如'如何安全保存密码')不算泄露。\
                      调用 check_leak 工具提交判定。"
                .to_string();
        let user = format!("待审查文本:\n{text}\n\n这段文本是否泄露了真实的敏感信息?");
        let messages = vec![Message::system(system), Message::human(user)];

        let args: LeakArgs = structured_call(&self.judge, leak_tool(), messages, |raw| {
            let is_leak = parse_leak_text(raw).ok_or_else(|| {
                StructuredJudgeError::Parse(format!(
                    "failed to parse leak verdict from judge reply: {}",
                    lc_core::judge::truncate(raw, 200)
                ))
            })?;
            Ok(LeakArgs {
                is_leak,
                reason: String::new(),
            })
        })
        .await
        .map_err(|e| GuardrailError::Judge(e.to_string()))?;
        Ok(args.is_leak)
    }
}

/// Structured judgment arguments (returned via tool_calls).
#[derive(Debug, serde::Deserialize)]
struct LeakArgs {
    #[serde(default)]
    is_leak: bool,
    /// Asks the LLM to attach a brief reason (improves judgment quality); currently unused.
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// Builds the judgment tool: lets the LLM submit the verdict as `{"is_leak": bool, "reason": "..."}`.
fn leak_tool() -> ToolDefinition {
    ToolDefinition::new(
        "check_leak",
        "判断文本是否泄露真实的敏感信息,提交布尔判定。",
    )
    .with_parameters(serde_json::json!({
        "type": "object",
        "properties": {
            "is_leak": { "type": "boolean", "description": "是否真实泄露敏感信息" },
            "reason": { "type": "string", "description": "简短依据" }
        },
        "required": ["is_leak", "reason"]
    }))
}

/// Parses a yes/no leak verdict. Returns `None` (parse failure, reported by the caller) when no
/// yes/no marker is present, rather than silently defaulting — so an off-topic LLM reply is not
/// treated as "no leak".
fn parse_leak_text(raw: &str) -> Option<bool> {
    let lower = raw.to_lowercase();
    // check negatives first (negatives take precedence over positives, so "not"/"cannot" are not misjudged by "yes"/"can").
    if lower.contains("否")
        || lower.contains("no")
        || lower.contains("不能")
        || lower.contains("不是")
        || lower.contains("false")
    {
        return Some(false);
    }
    if lower.contains("是")
        || lower.contains("yes")
        || lower.contains("能")
        || lower.contains("true")
    {
        return Some(true);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::{BaseLanguageModel, Runnable, RunnableConfig};
    use lc_schema::MessageType;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct MockJudgeError(String);
    impl std::fmt::Display for MockJudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for MockJudgeError {}

    /// Mock judge returning preset replies in sequence: it does not implement `bind_tools`,
    /// exercising `structured_call`'s text fallback path (the testable path shared with evaluation, P2-3).
    struct SeqMockJudge {
        replies: Vec<String>,
        call: Arc<AtomicUsize>,
        last_user: Arc<Mutex<Option<String>>>,
    }
    impl SeqMockJudge {
        fn new(replies: Vec<String>) -> Self {
            Self {
                replies,
                call: Arc::new(AtomicUsize::new(0)),
                last_user: Arc::new(Mutex::new(None)),
            }
        }
        fn last_user_content(&self) -> String {
            self.last_user
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
                .unwrap_or_default()
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for SeqMockJudge {
        type Error = MockJudgeError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockJudgeError("use chat".into()))
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
            messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let idx = self.call.fetch_add(1, Ordering::SeqCst);
            let reply = self.replies.get(idx).cloned().unwrap_or_default();
            if let Some(human) = messages
                .iter()
                .find(|m| m.message_type == MessageType::Human)
            {
                *self.last_user.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(human.content.clone());
            }
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
            Err(MockJudgeError("not supported".into()))
        }
    }

    #[tokio::test]
    async fn test_judge_returns_leak_on_yes() {
        let mock = SeqMockJudge::new(vec!["是".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        let result = judge.judge("密码是 abc123456").await.unwrap();
        assert!(result, "裁判判为是 → 应判定为泄露");
    }

    #[tokio::test]
    async fn test_judge_returns_no_leak_on_no() {
        let mock = SeqMockJudge::new(vec!["否".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        let result = judge.judge("如何安全保存密码").await.unwrap();
        assert!(!result, "裁判判为否 → 应判定为正常提及");
    }

    #[tokio::test]
    async fn test_judge_parse_failure_errors() {
        // text-fallback parse failure -> explicit Err, no silent default.
        let mock = SeqMockJudge::new(vec!["无法判断".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        let result = judge.judge("text").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_judge_sends_text_to_model() {
        let mock = SeqMockJudge::new(vec!["是".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        judge.judge("我的 token 是 abc").await.unwrap();
        let sent = judge.judge.last_user_content();
        assert!(
            sent.contains("我的 token 是 abc"),
            "裁判应收到待审查文本, 实际: {sent}"
        );
    }

    #[test]
    fn test_parse_leak_text() {
        assert_eq!(parse_leak_text("是"), Some(true));
        assert_eq!(parse_leak_text("yes"), Some(true));
        assert_eq!(parse_leak_text("是,泄露了"), Some(true));
        assert_eq!(parse_leak_text("否"), Some(false));
        assert_eq!(parse_leak_text("no"), Some(false));
        assert_eq!(parse_leak_text("不是"), Some(false));
        // no yes/no marker = parse failure, must not silently default
        assert_eq!(parse_leak_text("我看不出"), None);
    }
}
