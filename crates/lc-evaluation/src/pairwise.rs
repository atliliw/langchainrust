//! 成对比较评测器:让 LLM 裁判在两个回答中二选一(竞技场模式)。
//!
//! 带位置偏差缓解:交换 A/B 顺序跑两次,两次都选同一个才算真赢,否则判平局。

use lc_core::BaseChatModel;
use lc_schema::Message;

use super::EvalError;

/// 成对比较结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    AWins,
    BWins,
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
/// 注意:成对比较接口与单点 `Evaluator` 不同(需要两个预测),故不实现 `Evaluator` trait,
/// 通过 `compare` 方法调用。
pub struct PairwiseJudge<M: BaseChatModel> {
    judge: M,
    rubric: String,
}

const DEFAULT_PAIRWISE_RUBRIC: &str = "\
正确性:回答是否事实准确、是否切题。
完整性:是否完整回答了问题。
清晰性:表达是否清晰、简洁。";

impl<M: BaseChatModel> PairwiseJudge<M> {
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            rubric: DEFAULT_PAIRWISE_RUBRIC.to_string(),
        }
    }

    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = rubric.into();
        self
    }

    /// 比较 A、B 两个回答,返回谁更好。
    ///
    /// 交换 A/B 顺序跑两次,消除位置偏差:两次都选同一个才算真赢,否则判平局。
    pub async fn compare(&self, input: &str, a: &str, b: &str) -> Result<Verdict, EvalError> {
        let v1 = self.ask(input, a, b).await?; // A 在前
        let v2 = self.ask(input, b, a).await?; // 交换,B 在前

        Ok(match (v1, v2) {
            (Pick::Tie, _) | (_, Pick::Tie) => Verdict::Tie,
            (Pick::First, Pick::Second) => Verdict::AWins, // v1 选 A(前),v2 选 A(后)
            (Pick::Second, Pick::First) => Verdict::BWins, // v1 选 B(后),v2 选 B(前)
            _ => Verdict::Tie, // 位置偏差:两次选的位置一致但映射回不同答案
        })
    }

    async fn ask(&self, input: &str, first: &str, second: &str) -> Result<Pick, EvalError> {
        let system = format!(
            "你是裁判。根据评分标准,判断两个回答哪个更好。\n\n\
             评分标准:\n{rubric}\n\n\
             只输出三者之一:\"第一个更好\" / \"第二个更好\" / \"平局\"",
            rubric = self.rubric
        );
        let user =
            format!("题目:\n{input}\n\n第一个回答:\n{first}\n\n第二个回答:\n{second}\n\n哪个更好?");
        let result = self
            .judge
            .chat_with_system(system, vec![Message::human(user)])
            .await
            .map_err(|e| EvalError::PredictorError(e.to_string()))?;
        Ok(parse_pick(&result.content))
    }
}

/// 解析裁判回复为 Pick。
fn parse_pick(raw: &str) -> Pick {
    let lower = raw.to_lowercase();
    if lower.contains("平局") || lower.contains("tie") || lower.contains("一样") {
        return Pick::Tie;
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
        (Some(f), Some(s)) if f < s => Pick::First,
        (Some(_), Some(_)) => Pick::Second,
        (Some(_), None) => Pick::First,
        (None, Some(_)) => Pick::Second,
        (None, None) => Pick::Tie,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::LLMResult;
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
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
        assert_eq!(parse_pick("第一个更好"), Pick::First);
        assert_eq!(parse_pick("第二个更好"), Pick::Second);
        assert_eq!(parse_pick("平局"), Pick::Tie);
        assert_eq!(parse_pick("两个一样好"), Pick::Tie);
        assert_eq!(parse_pick("第二个比第一个好"), Pick::Second);
        // 前者/后者、former/latter:LLM 不一定按"第一个"格式回
        assert_eq!(parse_pick("前者更好"), Pick::First);
        assert_eq!(parse_pick("后者更准确"), Pick::Second);
        assert_eq!(parse_pick("the former is better"), Pick::First);
        assert_eq!(parse_pick("the latter wins"), Pick::Second);
    }
}
