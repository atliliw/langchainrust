//! 成对比较评测器:让 LLM 裁判在两个回答中二选一(竞技场模式)。
//!
//! 带位置偏差缓解:交换 A/B 顺序跑两次,两次都选同一个才算真赢,否则判平局。

use async_trait::async_trait;
use futures_util::future;
use serde::Deserialize;

use lc_core::judge::{structured_call, truncate, StructuredJudgeError};
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_schema::Message;

use super::{EvalError, PairwiseEvaluator, Score};

/// 成对比较结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// A 回答更优
    AWins,
    /// B 回答更优
    BWins,
    /// 平局
    Tie,
}

/// 裁判选了哪个位置
#[derive(Debug, Clone, PartialEq, Eq)]
enum Pick {
    First,
    Second,
    Tie,
}

/// 成对比较评测器(用 LLM 当裁判,二选一)。
///
/// P1-1: 实现 `PairwiseEvaluator` trait,可与单点 `Evaluator` 一起进 `EvalRunner`
/// 统一报告;直接调用仍可走 `compare` 拿细粒度 `Verdict`(A 赢 / B 赢 / 平局)。
pub struct PairwiseJudge<M: BaseChatModel> {
    judge: M,
    rubric: String,
}

const DEFAULT_PAIRWISE_RUBRIC: &str = "\
正确性:回答是否事实准确、是否切题。
完整性:是否完整回答了问题。
清晰性:表达是否清晰、简洁。";

impl<M: BaseChatModel> PairwiseJudge<M> {
    /// 创建使用默认评分标准的成对比较评测器。
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            rubric: DEFAULT_PAIRWISE_RUBRIC.to_string(),
        }
    }

    /// 设置自定义评分标准(builder 风格)。
    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = rubric.into();
        self
    }

    /// 比较 A、B 两个回答,返回谁更好。
    ///
    /// 交换 A/B 顺序跑两次,消除位置偏差:两次都选同一个才算真赢,否则判平局。
    /// P2-4: 两次 ask 相互独立,用 `future::join` 并发发起(消除 N+1 串行往返)。
    pub async fn compare(&self, input: &str, a: &str, b: &str) -> Result<Verdict, EvalError> {
        let (v1, v2) = future::join(self.ask(input, a, b), self.ask(input, b, a)).await;
        let v1 = v1?; // A 在前
        let v2 = v2?; // 交换,B 在前

        Ok(match (v1, v2) {
            (Pick::Tie, _) | (_, Pick::Tie) => Verdict::Tie,
            (Pick::First, Pick::Second) => Verdict::AWins, // v1 选 A(前),v2 选 A(后)
            (Pick::Second, Pick::First) => Verdict::BWins, // v1 选 B(后),v2 选 B(前)
            _ => Verdict::Tie, // 位置偏差:两次选的位置一致但映射回不同答案
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

        // P0-1: 优先结构化输出(verdict: a/b/tie);不支持工具绑定的模型走文本解析回落。
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

/// P1-1: 作为 `PairwiseEvaluator` 进 `EvalRunner`,judge 以
/// (a=prediction, b=reference) 为两个候选。得分映射:1.0 = A 优、
/// 0.5 = 平局、0.0 = B 优,label 保留裁决含义(a_wins / tie / b_wins)。
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

/// 结构化判定参数(经 tool_calls 返回)。
#[derive(Debug, Deserialize)]
struct PickArgs {
    verdict: String, // "a" | "b" | "tie"
    /// 让 LLM 附上简短理由(改善判定质量),当前不消费。
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// 构建二选一工具:让 LLM 以 `{"verdict": "a"|"b"|"tie", "reason": "..."}` 提交判定。
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

/// 把结构化 verdict 字符串映射回 `Pick`;非法值报解析错误。
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

/// 解析裁判回复为 Pick。无任何有效标记时返回 `None`(解析失败,由调用方报错),
/// 而非静默默认为平局——避免 LLM 跑题回复被当成"无偏好"。
fn parse_pick(raw: &str) -> Option<Pick> {
    let lower = raw.to_lowercase();
    if lower.contains("平局") || lower.contains("tie") || lower.contains("一样") {
        return Some(Pick::Tie);
    }
    // 第一个 / 前者 / former:任一措辞,取最早出现位置
    let first_pos = ["第一个", "first", "前者", "former"]
        .into_iter()
        .filter_map(|kw| lower.find(kw))
        .min();
    // 第二个 / 后者 / latter
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

    /// 依次返回预设回复的 mock 裁判
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
        // 第一次(A在前)选第一个=A;第二次(B在前)选第二个=A => A赢
        let judge = PairwiseJudge::new(SeqMockJudge::new(vec![
            "第一个更好".into(),
            "第二个更好".into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::AWins);
    }

    #[tokio::test]
    async fn test_pairwise_b_wins() {
        // 第一次(A在前)选第二个=B;第二次(B在前)选第一个=B => B赢
        let judge = PairwiseJudge::new(SeqMockJudge::new(vec![
            "第二个更好".into(),
            "第一个更好".into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::BWins);
    }

    #[tokio::test]
    async fn test_pairwise_position_bias_tie() {
        // 裁判总选第一个(位置偏差):两次都选 first => 映射回不同答案 => 平局
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
        // 前者/后者、former/latter:LLM 不一定按"第一个"格式回
        assert_eq!(parse_pick("前者更好"), Some(Pick::First));
        assert_eq!(parse_pick("后者更准确"), Some(Pick::Second));
        assert_eq!(parse_pick("the former is better"), Some(Pick::First));
        assert_eq!(parse_pick("the latter wins"), Some(Pick::Second));
        // 无任何有效标记 = 解析失败,不应静默默认为平局
        assert_eq!(parse_pick("我无法判断"), None);
    }

    /// P0-1: 支持 bind_tools 的模型走结构化输出(verdict: a/b/tie)。
    #[tokio::test]
    async fn test_pairwise_structured_verdict() {
        use crate::test_support::ToolJudge;
        // A 赢:第一轮(A 在前)选 "a"(第一个=A),第二轮(B 在前)选 "b"(第二个=A)
        let judge = PairwiseJudge::new(ToolJudge::sequence(vec![
            r#"{"verdict": "a", "reason": "第一个更完整"}"#.into(),
            r#"{"verdict": "b", "reason": "第二个更完整"}"#.into(),
        ]));
        assert_eq!(judge.compare("q", "A", "B").await.unwrap(), Verdict::AWins);
    }

    #[tokio::test]
    async fn test_pairwise_structured_verdict_b() {
        use crate::test_support::ToolJudge;
        // B 赢:第一轮(A 在前)选 "b"(第二个=B),第二轮(B 在前)选 "a"(第一个=B)
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
