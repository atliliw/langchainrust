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
    /// Fail-closed: `is_leak` has no serde default, so a judge reply missing the field is a
    /// deserialization error (surfaced to the caller) rather than a silent "no leak" (pass).
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
/// English markers are matched as **exact whole tokens**, never as substrings: "no" must not fire
/// on "notable" or "knowledge", and "not" must not fire on "nothing". Chinese has no whitespace
/// word-splitting, so CJK markers keep their substring semantics (=== is a single token), with
/// negatives still taking precedence over positives.
fn parse_leak_text(raw: &str) -> Option<bool> {
    let lower = raw.to_lowercase();
    // Uncertainty markers CHECKED FIRST: a reply expressing inability to decide ("我不能判断这是否泄露",
    // "无法确定", "not sure") must NOT be read as a negative. Returning `None` makes the caller
    // surface a Judge error, fail-closed — matching the module doc: an undecidable reply is not
    // treated as "no leak". (A genuine negative like "这不能算泄露" contains no uncertainty marker,
    // so it still falls through to the `不能` guard below and passes.)
    const UNCERTAIN_SUBSTRINGS: &[&str] = &[
        "无法判断",
        "无法确定",
        "不能判断",
        "不能确定",
        "不确定",
        "无法分辨",
        "不好判断",
        "很难判断",
    ];
    const UNCERTAIN_PHRASES: &[&str] = &[
        "not sure",
        "not certain",
        "cannot tell",
        "can't tell",
        "cannot determine",
        "can't determine",
        "unable to determine",
        "not certain",
    ];
    const UNCERTAIN_TOKENS: &[&str] = &["uncertain", "unclear", "undetermined"];
    if UNCERTAIN_SUBSTRINGS.iter().any(|m| lower.contains(m))
        || UNCERTAIN_PHRASES.iter().any(|p| lower.contains(p))
        || split_tokens(&lower)
            .iter()
            .any(|t| UNCERTAIN_TOKENS.contains(t))
    {
        return None;
    }
    // Chinese negatives: `不能` / `不是` / `否` are full substring markers.
    if lower.contains("不能") || lower.contains("不是") || lower.contains("否") {
        return Some(false);
    }
    // English negatives: whole-word tokens only.
    let tokens: Vec<&str> = split_tokens(&lower);
    if tokens
        .iter()
        .any(|t| *t == "no" || *t == "not" || *t == "false")
    {
        return Some(false);
    }
    // Chinese positives.
    if lower.contains("能") || lower.contains("是") {
        return Some(true);
    }
    // English positives: whole-word tokens only.
    if tokens.iter().any(|t| *t == "yes" || *t == "true") {
        return Some(true);
    }
    None
}

/// Splits text into whole alphabetic/numeric tokens, treating any non-alphanumeric character as a
/// word boundary. "notable" -> ["notable"], never a lone "no".
fn split_tokens(lower: &str) -> Vec<&str> {
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect()
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

    /// M-g2: uncertainty / inability-to-decide must be a parse failure (fail-closed), never a
    /// "no leak" — otherwise "我不能判断这是否泄露" (containing 不能/否) passes unblocked.
    #[test]
    fn test_parse_leak_text_uncertainty_is_fail_closed() {
        // Chinese uncertainty (contains 不能 / 否 substrings that would otherwise read as negative).
        assert_eq!(parse_leak_text("我不能判断这是否泄露"), None);
        assert_eq!(parse_leak_text("无法确定是否泄露"), None);
        assert_eq!(parse_leak_text("不太清楚 不能确定"), None);
        // English multi-word uncertainty (would otherwise read as the negation token "not").
        assert_eq!(parse_leak_text("I am not sure"), None);
        assert_eq!(parse_leak_text("cannot tell if this leaks"), None);
        assert_eq!(parse_leak_text("unclear"), None);
        // Genuine negatives without an uncertainty marker still pass (fail-… as "no leak").
        assert_eq!(parse_leak_text("这不能算泄露"), Some(false));
        assert_eq!(parse_leak_text("不是泄露"), Some(false));
        assert_eq!(parse_leak_text("no leak"), Some(false));
    }

    /// B6: missing `is_leak` is a deserialization error (fail-closed), never a silent allow.
    #[test]
    fn test_leak_args_missing_is_leak_is_deserialization_error() {
        // no `is_leak` field -> hard error, not a default false (pass).
        let missing = serde_json::from_str::<LeakArgs>(r#"{"reason":"r"}"#);
        assert!(
            missing.is_err(),
            "missing is_leak must fail, not default to pass"
        );

        // present field (and reason, which the tool schema marks required) -> ok.
        let ok = serde_json::from_str::<LeakArgs>(r#"{"is_leak":true,"reason":"r"}"#);
        assert!(ok.is_ok());

        // explicit false still deserializes fine.
        let no = serde_json::from_str::<LeakArgs>(r#"{"is_leak":false,"reason":"r"}"#);
        assert!(no.is_ok());
        assert!(!no.unwrap().is_leak);
    }

    /// B6: English verdicts use whole-token match — "no" no longer misfires on notable/knowledge.
    #[test]
    fn test_parse_leak_text_english_token_exact() {
        assert_eq!(parse_leak_text("no"), Some(false));
        assert_eq!(parse_leak_text("No leak."), Some(false));
        assert_eq!(parse_leak_text("not a leak"), Some(false));
        assert_eq!(parse_leak_text("this is not a leak"), Some(false));
        assert_eq!(parse_leak_text("false"), Some(false));

        // "no"/"not" as substrings must NOT trigger.
        assert_eq!(parse_leak_text("notable mention only"), None);
        assert_eq!(parse_leak_text("knowledge is fine to share"), None);
        assert_eq!(parse_leak_text("nothing sensitive here"), None);
        assert_eq!(parse_leak_text("notice the caveat"), None);

        // positives still exact.
        assert_eq!(parse_leak_text("yes"), Some(true));
        assert_eq!(parse_leak_text("true"), Some(true));
        // "yes"/"no" embedded in a larger word must not fire.
        assert_eq!(parse_leak_text("yesterday"), None);
    }
}
