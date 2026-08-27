//! Pairwise-comparison evaluator: an LLM judge picks the better of two answers (arena mode).
//!
//! Comes with position-bias mitigation: runs twice with A/B swapped, and only a consistent winner counts; otherwise it is a tie.

use async_trait::async_trait;
use futures_util::future;
use serde::Deserialize;

use lc_core::judge::{structured_call, truncate, StructuredJudgeError};
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_schema::Message;

use super::{EvalError, PairwiseEvaluator, Score};

/// Pairwise comparison result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Answer A is better
    AWins,
    /// Answer B is better
    BWins,
    /// Tie
    Tie,
}

/// Which position the judge picked
#[derive(Debug, Clone, PartialEq, Eq)]
enum Pick {
    First,
    Second,
    Tie,
}

/// Pairwise-comparison evaluator (an LLM as the judge, picks one of two).
///
/// P1-1: implements the `PairwiseEvaluator` trait and can join the pointwise `Evaluator`s in `EvalRunner`
/// unified report; calling `compare` directly still yields the fine-grained `Verdict` (A wins / B wins / tie).
pub struct PairwiseJudge<M: BaseChatModel> {
    judge: M,
    rubric: String,
}

const DEFAULT_PAIRWISE_RUBRIC: &str = "\
正确性:回答是否事实准确、是否切题。
完整性:是否完整回答了问题。
清晰性:表达是否清晰、简洁。";

impl<M: BaseChatModel> PairwiseJudge<M> {
    /// Creates a pairwise-comparison evaluator using the default rubric.
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            rubric: DEFAULT_PAIRWISE_RUBRIC.to_string(),
        }
    }

    /// Sets a custom rubric (builder style).
    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = rubric.into();
        self
    }

    /// Compares answers A and B, returning which is better.
    ///
    /// Runs twice with A/B swapped to eliminate position bias: a consistent winner in both counts,
    /// otherwise a tie. P2-4: the two asks are independent and fire concurrently via `future::join`
    /// (eliminating the N+1 serial round-trips).
    pub async fn compare(&self, input: &str, a: &str, b: &str) -> Result<Verdict, EvalError> {
        let (v1, v2) = future::join(self.ask(input, a, b), self.ask(input, b, a)).await;
        let v1 = v1?; // A first
        let v2 = v2?; // swapped, B first

        Ok(match (v1, v2) {
            (Pick::Tie, _) | (_, Pick::Tie) => Verdict::Tie,
            (Pick::First, Pick::Second) => Verdict::AWins, // v1 picks A (first), v2 picks A (second)
            (Pick::Second, Pick::First) => Verdict::BWins, // v1 picks B (second), v2 picks B (first)
            _ => Verdict::Tie, // position bias: both rounds picked the same position but it maps to different answers
        })
    }

    async fn ask(&self, input: &str, first: &str, second: &str) -> Result<Pick, EvalError> {
        let system = format!(
            "你是裁判。根据评分标准,判断两个回答哪个更好。调用 pick_better 工具提交判定。\n\n\
             评分标准:\n{rubric}\n\n\
             判定的 verdict 取三者之一:\"a\"(第一个更好) / \"b\"(第二个更好) / \"tie\"(平局)",
            rubric = self.rubric
        );
        let user =
            format!("题目:\n{input}\n\n第一个回答:\n{first}\n\n第二个回答:\n{second}\n\n哪个更好?");
        let messages = vec![Message::system(system), Message::human(user)];

        // P0-1: prefer structured output (verdict: a/b/tie); models without tool binding fall back to text parsing.
        let args: PickArgs = structured_call(&self.judge, pick_tool(), messages, |raw| {
            let pick = parse_pick(raw).ok_or_else(|| {
                StructuredJudgeError::Parse(format!(
                    "failed to parse winner from judge reply: {}",
                    truncate(raw, 200)
                ))
            })?;
            Ok(PickArgs {
                verdict: pick_to_str(pick).to_string(),
                reason: String::new(),
            })
        })
        .await?;
        str_to_pick(&args.verdict)
    }
}

/// P1-1: enters `EvalRunner` as a `PairwiseEvaluator`, judging the two candidates
/// as (a=prediction, b=reference). Score mapping: 1.0 = A wins,
/// 0.5 = tie, 0.0 = B wins, and the label keeps the verdict meaning (a_wins / tie / b_wins).
#[async_trait]
impl<M: BaseChatModel> PairwiseEvaluator for PairwiseJudge<M> {
    async fn eval_pair(&self, input: &str, a: &str, b: &str) -> Result<Score, EvalError> {
        let (value, label) = match self.compare(input, a, b).await? {
            Verdict::AWins => (1.0, "a_wins"),
            Verdict::Tie => (0.5, "tie"),
            Verdict::BWins => (0.0, "b_wins"),
        };
        Ok(Score::new(value).with_label(label))
    }

    fn name(&self) -> &str {
        "pairwise"
    }
}

/// Structured verdict arguments (returned via tool_calls).
#[derive(Debug, Deserialize)]
struct PickArgs {
    verdict: String, // "a" | "b" | "tie"
    /// Asks the LLM to attach a brief reason (improves judgment quality); currently unused.
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// Builds the pick-one-of-two tool: lets the LLM submit a verdict as `{"verdict": "a"|"b"|"tie", "reason": "..."}`.
fn pick_tool() -> ToolDefinition {
    ToolDefinition::new(
        "pick_better",
        "判断两个回答哪个更好。verdict 取 \"a\"(第一个更好)、\"b\"(第二个更好)、\"tie\"(平局)。",
    )
    .with_parameters(serde_json::json!({
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["a", "b", "tie"],
                "description": "a=第一个更好, b=第二个更好, tie=平局"
            },
            "reason": { "type": "string", "description": "简短理由" }
        },
        "required": ["verdict", "reason"]
    }))
}

fn pick_to_str(pick: Pick) -> &'static str {
    match pick {
        Pick::First => "a",
        Pick::Second => "b",
        Pick::Tie => "tie",
    }
}

/// Maps a structured verdict string back to `Pick`; an invalid value reports a parse error.
fn str_to_pick(verdict: &str) -> Result<Pick, EvalError> {
    match verdict {
        "a" => Ok(Pick::First),
        "b" => Ok(Pick::Second),
        "tie" => Ok(Pick::Tie),
        other => Err(EvalError::ParseError(format!(
            "judge returned invalid verdict: {}",
            other
        ))),
    }
}

/// Parses a judge reply into a Pick. With no valid marker returns `None` (parse failure, reported by the caller),
/// rather than silently defaulting to a tie — so an off-topic LLM reply is not read as "no preference".
fn parse_pick(raw: &str) -> Option<Pick> {
    let lower = raw.to_lowercase();
    if lower.contains("平局") || lower.contains("tie") || lower.contains("一样") {
        return Some(Pick::Tie);
    }
    // "first"/"former" wordings: any phrasing, take the earliest occurrence position
    let first_pos = ["第一个", "first", "前者", "former"]
        .into_iter()
        .filter_map(|kw| lower.find(kw))
        .min();
    // "second"/"latter" wordings
    let second_pos = ["第二个", "second", "后者", "latter"]
        .into_iter()
        .filter_map(|kw| lower.find(kw))
        .min();
    match (first_pos, second_pos) {
        (Some(f), Some(s)) if f < s => Some(Pick::First),
        (Some(_), Some(_)) => Some(Pick::Second),
        (Some(_), None) => Some(Pick::First),
        (None, Some(_)) => Some(Pick::Second),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::{BaseLanguageModel, Runnable, RunnableConfig};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct JudgeError(String);
    impl std::fmt::Display for JudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for JudgeError {}

    /// Mock judge returning preset replies in order
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

    #[tokio::test]
    async fn test_pairwise_a_wins() {
        // first round (A first) picks the first = A; second round (B first) picks the second = A => A wins
        let judge = PairwiseJudge::new(SeqMockJudge::new(vec![
            "第一个更好".into(),
            "第二个更好".into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::AWins);
    }

    #[tokio::test]
    async fn test_pairwise_b_wins() {
        // first round (A first) picks the second = B; second round (B first) picks the first = B => B wins
        let judge = PairwiseJudge::new(SeqMockJudge::new(vec![
            "第二个更好".into(),
            "第一个更好".into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::BWins);
    }

    #[tokio::test]
    async fn test_pairwise_position_bias_tie() {
        // judge always picks the first (position bias): both rounds pick first => maps to different answers => tie
        let judge = PairwiseJudge::new(SeqMockJudge::new(vec![
            "第一个更好".into(),
            "第一个更好".into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::Tie);
    }

    #[tokio::test]
    async fn test_pairwise_explicit_tie() {
        let judge = PairwiseJudge::new(SeqMockJudge::new(vec!["平局".into(), "平局".into()]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::Tie);
    }

    #[test]
    fn test_parse_pick() {
        assert_eq!(parse_pick("第一个更好"), Some(Pick::First));
        assert_eq!(parse_pick("第二个更好"), Some(Pick::Second));
        assert_eq!(parse_pick("平局"), Some(Pick::Tie));
        assert_eq!(parse_pick("两个一样好"), Some(Pick::Tie));
        assert_eq!(parse_pick("第二个比第一个好"), Some(Pick::Second));
        // "former"/"latter" wordings: the LLM may not reply in the "first"/"second" format
        assert_eq!(parse_pick("前者更好"), Some(Pick::First));
        assert_eq!(parse_pick("后者更准确"), Some(Pick::Second));
        assert_eq!(parse_pick("the former is better"), Some(Pick::First));
        assert_eq!(parse_pick("the latter wins"), Some(Pick::Second));
        // no valid marker = parse failure, must not silently default to a tie
        assert_eq!(parse_pick("我无法判断"), None);
    }

    /// P0-1: models supporting bind_tools use structured output (verdict: a/b/tie).
    #[tokio::test]
    async fn test_pairwise_structured_verdict() {
        use crate::test_support::ToolJudge;
        // A wins: round 1 (A first) picks "a" (first = A), round 2 (B first) picks "b" (second = A)
        let judge = PairwiseJudge::new(ToolJudge::sequence(vec![
            r#"{"verdict": "a", "reason": "第一个更完整"}"#.into(),
            r#"{"verdict": "b", "reason": "第二个更完整"}"#.into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::AWins);
    }

    #[tokio::test]
    async fn test_pairwise_structured_verdict_b() {
        use crate::test_support::ToolJudge;
        // B wins: round 1 (A first) picks "b" (second = B), round 2 (B first) picks "a" (first = B)
        let judge = PairwiseJudge::new(ToolJudge::sequence(vec![
            r#"{"verdict": "b", "reason": "第二个更准确"}"#.into(),
            r#"{"verdict": "a", "reason": "第一个更准确"}"#.into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::BWins);
    }

    #[tokio::test]
    async fn test_pairwise_structured_verdict_tie() {
        use crate::test_support::ToolJudge;
        let judge = PairwiseJudge::new(ToolJudge::new(
            r#"{"verdict": "tie", "reason": "难分高下"}"#,
        ));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::Tie);
    }
}
