//! 忠实度评测器:检测回答是否忠于参考上下文(幻觉检测)。
//!
//! 思路来自 Ragas 的 faithfulness:把回答拆成原子陈述,
//! 逐条判断能否从参考上下文推导出来,通过率即为忠实度。
//! 这里 reference 充当"上下文/检索内容",prediction 是待检测的回答。

use async_trait::async_trait;

use lc_core::BaseChatModel;
use lc_schema::Message;

use super::{EvalError, Evaluator, Score};

/// 忠实度评测器(幻觉检测):回答有多忠于参考上下文。
///
/// 把 prediction 拆成原子陈述,逐条问裁判能否从 reference 推导,
/// 通过率 = 能推导的陈述数 / 总陈述数。
pub struct Faithfulness<M: BaseChatModel> {
    judge: M,
    /// 用 LLM 拆原子陈述(默认 false,用规则按标点拆)
    llm_split: bool,
    /// 空预测(无可验证陈述)的得分,默认 1.0(没编造=100% 忠实)
    empty_score: f64,
}

/// 把回答拆成原子陈述(按句号、问号、感叹号、分号、换行切分)。
fn split_claims(prediction: &str) -> Vec<String> {
    prediction
        .split(['。', '.', '!', '?', '；', ';', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

impl<M: BaseChatModel> Faithfulness<M> {
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            llm_split: false,
            empty_score: 1.0,
        }
    }

    /// 用 LLM 拆原子陈述(默认按标点规则拆;LLM 拆能处理逗号复合句)
    pub fn with_llm_split(mut self, v: bool) -> Self {
        self.llm_split = v;
        self
    }

    /// 空预测的得分:默认 1.0(没编造即忠实),可设 0.0 表示"没回答=0"
    pub fn with_empty_score(mut self, score: f64) -> Self {
        self.empty_score = score;
        self
    }

    /// 问裁判:单条陈述能否从上下文推导。
    async fn verify_claim(&self, context: &str, claim: &str) -> Result<bool, EvalError> {
        let system =
            "你是事实核查员。判断给定的陈述能否从参考上下文中推导出来。只输出\"是\"或\"否\"。"
                .to_string();
        let user =
            format!("参考上下文:\n{context}\n\n陈述:\n{claim}\n\n这条陈述能从上下文推导出来吗?");
        let result = self
            .judge
            .chat_with_system(system, vec![Message::human(user)])
            .await
            .map_err(|e| EvalError::PredictorError(e.to_string()))?;
        Ok(parse_yes_no(&result.content))
    }

    /// 用 LLM 把回答拆成原子陈述(每行一条),处理规则拆不动的复合句。
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
        // 并发验证各陈述:每条一次 LLM 调用,串行会变成 N 次顺序往返
        let futs = claims
            .iter()
            .map(|claim| self.verify_claim(reference, claim));
        let results = futures_util::future::join_all(futs).await;
        let mut supported = 0usize;
        for r in results {
            if r? {
                supported += 1;
            }
        }
        let value = supported as f64 / claims.len() as f64;
        Ok(Score::new(value).with_label("faithfulness"))
    }

    fn name(&self) -> &str {
        "faithfulness"
    }
}

/// 解析"是/否"。
fn parse_yes_no(raw: &str) -> bool {
    let lower = raw.to_lowercase();
    // 先判否定(避免"不是""不能"被"是""能"误判)
    if lower.contains("否")
        || lower.contains("no")
        || lower.contains("不能")
        || lower.contains("不是")
        || lower.contains("false")
    {
        return false;
    }
    lower.contains("是") || lower.contains("yes") || lower.contains("能") || lower.contains("true")
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let judge = Faithfulness::new(SeqMockJudge::new(vec![]));
        let s = judge.eval("", "", "ctx").await.unwrap();
        assert!((s.value - 1.0).abs() < 1e-9); // 无陈述视为不撒谎
    }

    #[tokio::test]
    async fn test_faithfulness_empty_score_configurable() {
        // 默认空预测 1.0(没编造);可配置成 0.0(没回答)
        let judge = Faithfulness::new(SeqMockJudge::new(vec![])).with_empty_score(0.0);
        let s = judge.eval("", "", "ctx").await.unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_llm_split() {
        // 逗号复合句规则拆只算 1 条;LLM 拆能拆成 2 条分别验证
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
        assert!(parse_yes_no("是"));
        assert!(parse_yes_no("yes"));
        assert!(!parse_yes_no("否"));
        assert!(!parse_yes_no("no"));
        assert!(!parse_yes_no("不是"));
        assert!(!parse_yes_no("不能"));
    }
}
