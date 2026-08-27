//! Faithfulness evaluator: detects whether an answer is faithful to the reference context (hallucination detection).
//!
//! The idea comes from Ragas' faithfulness: the answer is split into atomic claims,
//! each judged for whether it can be derived from the reference context; the pass rate is the faithfulness score.
//! Here `reference` acts as the "context / retrieved content" and `prediction` is the answer under test.

use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};
use serde::Deserialize;

use lc_core::judge::{structured_call, truncate, StructuredJudgeError};
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_schema::Message;

use super::{EvalError, Evaluator, Score};

/// P1-5: maximum concurrent claim-verification calls to the judge in a single eval (prevents N paths all dying to rate limits).
const MAX_CONCURRENT_VERIFY: usize = 4;

/// P2-5: character cap for the reference context in a single claim's judge prompt.
/// The full long reference is truncated once and reused by N claims, avoiding re-sending the whole context per claim.
const DEFAULT_MAX_CONTEXT_CHARS: usize = 2000;

/// Faithfulness evaluator (hallucination detection): how faithful an answer is to the reference context.
///
/// Splits `prediction` into atomic claims and asks the judge per claim whether it can be derived from `reference`;
/// pass rate = derivable claims / total claims.
pub struct Faithfulness<M: BaseChatModel> {
    judge: M,
    /// Whether to split claims with the LLM (default false: rule-based split on punctuation)
    llm_split: bool,
    /// Score for an empty prediction (no verifiable claims), default 0.0 (no answer = not faithful).
    empty_score: f64,
    /// Per-claim reference-context transmission cap (chars, default [`DEFAULT_MAX_CONTEXT_CHARS`]).
    max_context_chars: usize,
}

/// Splits an answer into atomic claims (split on period, question mark, exclamation mark, semicolon, newline).
fn split_claims(prediction: &str) -> Vec<String> {
    prediction
        .split(['。', '.', '!', '?', '；', ';', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

impl<M: BaseChatModel> Faithfulness<M> {
    /// Creates a faithfulness evaluator.
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            llm_split: false,
            empty_score: 0.0, // P0-2: empty prediction defaults to 0 (no answer = not faithful)
            max_context_chars: DEFAULT_MAX_CONTEXT_CHARS,
        }
    }

    /// Splits claims with the LLM (default: rule-based on punctuation; LLM split handles comma compound sentences)
    pub fn with_llm_split(mut self, v: bool) -> Self {
        self.llm_split = v;
        self
    }

    /// Score for an empty prediction: default 0.0 (no answer = not faithful); can be set to 1.0 for "not fabricating is faithful"
    pub fn with_empty_score(mut self, score: f64) -> Self {
        self.empty_score = score;
        self
    }

    /// Per-claim reference-context transmission cap (chars). P2-5: default 2000, preventing a long reference from being fully stuffed into the prompt once per claim.
    pub fn with_max_context_chars(mut self, max: usize) -> Self {
        self.max_context_chars = max;
        self
    }

    /// Asks the judge whether a single claim can be derived from the context.
    async fn verify_claim(&self, context: &str, claim: &str) -> Result<bool, EvalError> {
        let system =
            "你是事实核查员。判断给定的陈述能否从参考上下文中推导出来。调用 check_claim 工具提交判定。"
                .to_string();
        let user =
            format!("参考上下文:\n{context}\n\n陈述:\n{claim}\n\n这条陈述能从上下文推导出来吗?");
        let messages = vec![Message::system(system), Message::human(user)];

        // P0-1: prefer structured output (boolean verdict); models without tool binding fall back to text parsing.
        let args: VerdictArgs = structured_call(&self.judge, verdict_tool(), messages, |raw| {
            let verdict = parse_yes_no(raw).ok_or_else(|| {
                StructuredJudgeError::Parse(format!(
                    "failed to parse yes/no from judge reply: {}",
                    truncate(raw, 200)
                ))
            })?;
            Ok(VerdictArgs {
                verdict,
                reason: String::new(),
            })
        })
        .await?;
        Ok(args.verdict)
    }

    /// Splits the answer into atomic claims with the LLM (one per line), handling compound sentences the rule-based split cannot.
    async fn split_claims_llm(&self, prediction: &str) -> Result<Vec<String>, EvalError> {
        let system =
            "你是文本分析助手。把回答拆成原子陈述,每条一行,只输出陈述本身,不要编号不要解释。"
                .to_string();
        let user = format!("回答:\n{prediction}\n\n把它拆成原子陈述,每行一条:");
        let result = self
            .judge
            .chat_with_system(system, vec![Message::human(user)])
            .await
            .map_err(|e| EvalError::PredictorError(e.to_string()))?;
        Ok(result
            .content
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }
}

#[async_trait]
impl<M: BaseChatModel> Evaluator for Faithfulness<M> {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError> {
        let claims = if self.llm_split {
            self.split_claims_llm(prediction).await?
        } else {
            split_claims(prediction)
        };
        if claims.is_empty() {
            return Ok(Score::new(self.empty_score).with_label("no_claims"));
        }
        // P2-5: the reference context is truncated once and reused by all claims (avoiding a full long reference transmitted N times).
        let context = truncate(reference, self.max_context_chars);
        // verify claims concurrently (one LLM call each) but throttle with buffer_unordered:
        // P1-5 — join_all would fire unlimited concurrency at one judge; hitting a rate limit kills all N.
        // `ctx` is a Copy reference so the closure can capture it repeatedly; capturing `context`
        // directly would be moved out claim by claim by async move, and map(FnMut) would not compile.
        let ctx = &context;
        let total = claims.len();
        let results: Vec<Result<bool, EvalError>> = stream::iter(claims)
            .map(|claim| async move { self.verify_claim(ctx, &claim).await })
            .buffer_unordered(MAX_CONCURRENT_VERIFY)
            .collect()
            .await;
        let mut supported = 0usize;
        for r in results {
            if r? {
                supported += 1;
            }
        }
        let value = supported as f64 / total as f64;
        Ok(Score::new(value).with_label("faithfulness"))
    }

    fn name(&self) -> &str {
        "faithfulness"
    }
}

/// Structured verdict arguments (returned via tool_calls).
#[derive(Debug, Deserialize)]
struct VerdictArgs {
    verdict: bool,
    /// Asks the LLM to attach a brief reason (improves judgment quality); currently unused.
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// Builds the verdict tool: lets the LLM submit a verdict as `{"verdict": bool, "reason": "..."}`.
fn verdict_tool() -> ToolDefinition {
    ToolDefinition::new(
        "check_claim",
        "判断陈述能否从参考上下文推导出来,提交布尔判定。",
    )
    .with_parameters(serde_json::json!({
        "type": "object",
        "properties": {
            "verdict": { "type": "boolean", "description": "能否从上下文推导" },
            "reason": { "type": "string", "description": "简短依据" }
        },
        "required": ["verdict", "reason"]
    }))
}

/// Parses "yes/no". With no yes/no marker returns `None` (parse failure, reported by the caller),
/// rather than silently defaulting to false — so an off-topic LLM reply is not read as "unfaithful".
fn parse_yes_no(raw: &str) -> Option<bool> {
    let lower = raw.to_lowercase();
    // check negatives first (so negated phrasings are not caught by the positive keywords; negatives take precedence over positives)
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
    struct JudgeError(String);
    impl std::fmt::Display for JudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for JudgeError {}

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
            Err(JudgeError("not supported".into()))
        }
    }

    #[test]
    fn test_split_claims() {
        let claims = split_claims("巴黎是法国首都。伦敦是英国首都。");
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0], "巴黎是法国首都");
        assert_eq!(claims[1], "伦敦是英国首都");
    }

    #[test]
    fn test_split_claims_empty() {
        assert!(split_claims("").is_empty());
        assert!(split_claims("。。。").is_empty());
    }

    #[tokio::test]
    async fn test_faithfulness_all_supported() {
        let judge = Faithfulness::new(SeqMockJudge::new(vec!["是".into(), "是".into()]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_half_supported() {
        let judge = Faithfulness::new(SeqMockJudge::new(vec!["是".into(), "否".into()]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_none_supported() {
        let judge = Faithfulness::new(SeqMockJudge::new(vec!["否".into(), "否".into()]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_empty_prediction() {
        // P0-2: empty prediction defaults to 0 (no answer = not faithful)
        let judge = Faithfulness::new(SeqMockJudge::new(vec![]));
        let s = judge.eval("", "", "ctx").await.unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
        assert_eq!(s.label.as_deref(), Some("no_claims"));
    }

    #[tokio::test]
    async fn test_faithfulness_empty_score_configurable() {
        // can be explicitly configured to 1.0 (not fabricating is faithful)
        let judge = Faithfulness::new(SeqMockJudge::new(vec![])).with_empty_score(1.0);
        let s = judge.eval("", "", "ctx").await.unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_llm_split() {
        // rule split counts a comma compound as 1 claim; LLM split divides it into 2 and verifies each
        let judge = Faithfulness::new(SeqMockJudge::new(vec![
            "巴黎是法国首都\n伦敦是英国首都".into(),
            "是".into(),
            "是".into(),
        ]))
        .with_llm_split(true);
        let s = judge
            .eval("", "巴黎是法国首都,伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_yes_no() {
        assert_eq!(parse_yes_no("是"), Some(true));
        assert_eq!(parse_yes_no("yes"), Some(true));
        assert_eq!(parse_yes_no("否"), Some(false));
        assert_eq!(parse_yes_no("no"), Some(false));
        assert_eq!(parse_yes_no("不是"), Some(false));
        assert_eq!(parse_yes_no("不能"), Some(false));
        // no yes/no marker = parse failure, must not silently default
        assert_eq!(parse_yes_no("我不会告诉你"), None);
    }

    /// P0-1: models supporting bind_tools use structured output (boolean verdict), no longer relying on text parsing.
    #[tokio::test]
    async fn test_faithfulness_structured_verdict() {
        use crate::test_support::ToolJudge;
        // two claims: one supported, one not -> faithfulness 0.5
        let judge = Faithfulness::new(ToolJudge::sequence(vec![
            r#"{"verdict": true, "reason": "能从上下文推导"}"#.into(),
            r#"{"verdict": false, "reason": "无法推导"}"#.into(),
        ]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "巴黎是法国首都")
            .await
            .unwrap();
        assert!((s.value - 0.5).abs() < 1e-9);
    }

    /// P0-1: all unsupported -> 0 score.
    #[tokio::test]
    async fn test_faithfulness_structured_all_false() {
        use crate::test_support::ToolJudge;
        let judge = Faithfulness::new(ToolJudge::new(
            r#"{"verdict": false, "reason": "均无法推导"}"#,
        ));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "巴黎是法国首都")
            .await
            .unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
    }

    /// P2-5: a long reference context is truncated once and reused by N claims, not re-sent in full.
    #[tokio::test]
    async fn test_faithfulness_reference_truncated_once() {
        let judge = SeqMockJudge::new(vec!["是".into(), "是".into()]);
        let f = Faithfulness::new(judge).with_max_context_chars(10);
        let long_ref =
            "这是一段非常长的参考上下文,远超默认的单条传输上限,里面藏了一个不该被完整发送的尾巴"
                .to_string();
        let s = f
            .eval("", "巴黎是首都。伦敦是首都。", &long_ref)
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
        let sent = f.judge.last_user_content();
        // the reference context is truncated to budget: the head remains, the far-beyond-budget tail is not sent
        assert!(sent.contains("这是一段非常长"), "actual sent: {sent}");
        assert!(
            !sent.contains("不该被完整发送"),
            "full long reference was sent repeatedly"
        );
    }
}
