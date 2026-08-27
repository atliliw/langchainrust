//! Module-level integration tests: EvalRunner end-to-end, LLMAsJudge with MockJudge.

use super::*;
use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{LLMResult, StreamChunk};
use lc_core::{BaseChatModel, BaseLanguageModel, Runnable, RunnableConfig};
use lc_embeddings::MockEmbeddings;
use lc_schema::Message;
use std::pin::Pin;

struct StaticPredictor(&'static str);
#[async_trait]
impl Predictor for StaticPredictor {
    async fn predict(&self, _input: &str) -> Result<String, EvalError> {
        Ok(self.0.to_string())
    }
}

#[derive(Debug)]
struct JudgeError(String);
impl std::fmt::Display for JudgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "judge error: {}", self.0)
    }
}
impl std::error::Error for JudgeError {}

struct MockJudge {
    reply: String,
}
impl MockJudge {
    fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
        }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockJudge {
    type Error = JudgeError;
    async fn invoke(
        &self,
        _input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Err(JudgeError("invoke not used; judge via chat".into()))
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for MockJudge {
    fn model_name(&self) -> &str {
        "mock-judge"
    }
    fn get_num_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
    fn with_temperature(self, _temp: f32) -> Self {
        self
    }
    fn with_max_tokens(self, _max: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for MockJudge {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Ok(LLMResult {
            content: self.reply.clone(),
            model: "mock-judge".to_string(),
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
        Err(JudgeError("stream_chat not supported in mock".into()))
    }
}

#[tokio::test]
async fn test_runner_summary() {
    let runner = EvalRunner::new(vec![Box::new(ExactMatch)]);
    let dataset = Dataset::new(vec![Example::new("q1", "yes"), Example::new("q2", "no")]);
    let report = runner.run(&dataset, &StaticPredictor("yes")).await.unwrap();
    assert_eq!(report.per_example.len(), 2);
    let summary = report.summary.get("exact_match").unwrap();
    assert!((summary.mean - 0.5).abs() < 1e-9);
    assert_eq!(summary.count, 2);
    assert!(summary.std.is_finite());
}

#[tokio::test]
async fn test_runner_multiple_evaluators() {
    let runner = EvalRunner::new(vec![
        Box::new(ExactMatch),
        Box::new(StringDistance),
        Box::new(EmbeddingSimilarity::new(MockEmbeddings::new(16))),
    ]);
    let dataset = Dataset::new(vec![Example::new("q", "hello")]);
    let report = runner
        .run(&dataset, &StaticPredictor("hello"))
        .await
        .unwrap();
    assert_eq!(report.summary.len(), 3);
    for v in report.summary.values() {
        assert!((v.mean - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_example_dataset_serde() {
    let json = r#"{"input":"q","reference":"a"}"#;
    let ex: Example = serde_json::from_str(json).unwrap();
    assert_eq!(ex.input, "q");
    assert_eq!(ex.reference, "a");
}

#[tokio::test]
async fn test_judge_with_runner() {
    let judge = LLMAsJudge::new(MockJudge::new(r#"{"reason":"ok","score":8}"#));
    let runner = EvalRunner::new(vec![Box::new(judge)]);
    let dataset = Dataset::new(vec![Example::new("q1", "ref1"), Example::new("q2", "ref2")]);
    let report = runner
        .run(&dataset, &StaticPredictor("pred"))
        .await
        .unwrap();
    let avg = report.summary.get("llm_as_judge").unwrap().mean;
    assert!((avg - 0.8).abs() < 1e-9);
}

#[tokio::test]
async fn test_judge_eval_full_score() {
    let judge = LLMAsJudge::new(MockJudge::new(r#"{"reason":"完全正确","score":10}"#));
    let s = judge.eval("法国首都?", "巴黎", "巴黎").await.unwrap();
    assert!((s.value - 1.0).abs() < 1e-9);
    assert_eq!(s.label.as_deref(), Some("llm_judge"));
}

#[tokio::test]
async fn test_judge_eval_half_score() {
    let judge = LLMAsJudge::new(MockJudge::new(r#"{"reason":"部分正确","score":5}"#));
    let s = judge.eval("q", "pred", "ref").await.unwrap();
    assert!((s.value - 0.5).abs() < 1e-9);
}

#[tokio::test]
async fn test_judge_custom_max_score() {
    let judge = LLMAsJudge::new(MockJudge::new(r#"{"reason":"ok","score":4}"#)).with_max_score(5);
    let s = judge.eval("q", "pred", "ref").await.unwrap();
    assert!((s.value - 0.8).abs() < 1e-9);
}

#[tokio::test]
async fn test_judge_eval_text_fallback() {
    let judge = LLMAsJudge::new(MockJudge::new("我觉得分数: 7 分"));
    let s = judge.eval("q", "pred", "ref").await.unwrap();
    assert!((s.value - 0.7).abs() < 1e-9);
}

#[tokio::test]
async fn test_judge_eval_pure_number_fallback() {
    let judge = LLMAsJudge::new(MockJudge::new("评价一般\n7"));
    let s = judge.eval("q", "pred", "ref").await.unwrap();
    assert!((s.value - 0.7).abs() < 1e-9);
}

#[tokio::test]
async fn test_judge_eval_parse_error() {
    let judge = LLMAsJudge::new(MockJudge::new("我不知道怎么评"));
    let err = judge.eval("q", "pred", "ref").await.unwrap_err();
    assert!(matches!(err, EvalError::ParseError(_)));
}

/// P0-1: models supporting bind_tools use structured output (the score tool), no longer relying on text parsing.
#[tokio::test]
async fn test_judge_eval_structured_score() {
    use crate::test_support::ToolJudge;
    let judge = LLMAsJudge::new(ToolJudge::new(
        r#"{"score": 8, "reason": "基本正确,略有遗漏"}"#,
    ));
    let s = judge.eval("q", "pred", "ref").await.unwrap();
    assert!((s.value - 0.8).abs() < 1e-9);
}

/// P0-1: an out-of-range structured score (12 > max 10) should be clamped to 1.0 rather than misparsed as text.
#[tokio::test]
async fn test_judge_eval_structured_score_clamped() {
    use crate::test_support::ToolJudge;
    let judge = LLMAsJudge::new(ToolJudge::new(r#"{"score": 12, "reason": "超满分"}"#));
    let s = judge.eval("q", "pred", "ref").await.unwrap();
    assert!((s.value - 1.0).abs() < 1e-9);
}

#[test]
fn test_judge_name() {
    let judge = LLMAsJudge::new(MockJudge::new(r#"{"score":1}"#));
    assert_eq!(judge.name(), "llm_as_judge");
}

/// P1-1: PairwiseJudge runs as a `PairwiseEvaluator` in EvalRunner with a unified report.
#[tokio::test]
async fn test_runner_with_pairwise_evaluator() {
    use crate::test_support::ToolJudge;
    // compare internally swaps and runs twice: example 0 replies ["a","b"] -> AWins (1.0);
    // example 1 replies ["tie","*"] -> Tie (0.5), mean 0.75.
    let judge = PairwiseJudge::new(ToolJudge::sequence(vec![
        r#"{"verdict": "a", "reason": "预测更好"}"#.into(),
        r#"{"verdict": "b", "reason": "交换后仍预测更好"}"#.into(),
        r#"{"verdict": "tie", "reason": "难分高下"}"#.into(),
        r#"{"verdict": "tie", "reason": "难分高下"}"#.into(),
    ]));
    let runner = EvalRunner::new(vec![]).with_pairwise(vec![Box::new(judge)]);
    let dataset = Dataset::new(vec![Example::new("q1", "R1"), Example::new("q2", "R2")]);
    let report = runner.run(&dataset, &StaticPredictor("P")).await.unwrap();
    let s = report.summary.get("pairwise").unwrap();
    assert_eq!(s.count, 2);
    assert!((s.mean - 0.75).abs() < 1e-9);
    // pairwise scores go into per_example with the original text for traceability
    assert_eq!(report.per_example[0].input, "q1");
    assert_eq!(report.per_example[0].prediction, "P");
    assert_eq!(report.per_example[0].reference, "R1");
}

/// P1-3: a single predict failure is recorded in failures only; other examples still produce results.
#[tokio::test]
async fn test_runner_per_item_predict_failure() {
    struct FlakyPredictor;
    #[async_trait]
    impl Predictor for FlakyPredictor {
        async fn predict(&self, input: &str) -> Result<String, EvalError> {
            if input == "bad" {
                Err(EvalError::PredictorError("predict failed".into()))
            } else {
                Ok("ok".into())
            }
        }
    }
    let runner = EvalRunner::new(vec![Box::new(ExactMatch)]);
    let dataset = Dataset::new(vec![Example::new("good", "ok"), Example::new("bad", "ok")]);
    let report = runner.run(&dataset, &FlakyPredictor).await.unwrap();
    assert_eq!(report.per_example.len(), 1); // only the successful one is counted
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].index, 1);
    assert_eq!(report.failures[0].stage, "predict");
    let s = report.summary.get("exact_match").unwrap();
    assert_eq!(s.count, 1);
    assert!((s.mean - 1.0).abs() < 1e-9);
}

/// P1-3: one evaluator scoring failure is recorded in failures only; other evaluators still score.
#[tokio::test]
async fn test_runner_evaluator_failure_is_tolerated() {
    struct FailingEvaluator;
    #[async_trait]
    impl Evaluator for FailingEvaluator {
        async fn eval(&self, _i: &str, _p: &str, _r: &str) -> Result<Score, EvalError> {
            Err(EvalError::PredictorError("judge failed".into()))
        }
        fn name(&self) -> &str {
            "failing"
        }
    }
    let runner = EvalRunner::new(vec![Box::new(ExactMatch), Box::new(FailingEvaluator)]);
    let dataset = Dataset::new(vec![Example::new("q", "ok")]);
    let report = runner.run(&dataset, &StaticPredictor("ok")).await.unwrap();
    assert_eq!(report.per_example.len(), 1);
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].stage, "failing");
    assert!(report.summary.contains_key("exact_match"));
    assert!(!report.summary.contains_key("failing"));
}

/// P1-4: Report carries the original text and can be deserialized (for post-hoc analysis after persisting).
#[tokio::test]
async fn test_report_serde_roundtrip() {
    let runner = EvalRunner::new(vec![Box::new(ExactMatch)]);
    let dataset = Dataset::new(vec![Example::new("q1", "yes")]);
    let report = runner.run(&dataset, &StaticPredictor("yes")).await.unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let back: Report = serde_json::from_str(&json).unwrap();
    assert_eq!(back.per_example[0].input, "q1");
    assert_eq!(back.per_example[0].reference, "yes");
    assert_eq!(back.per_example[0].prediction, "yes");
    let s = back.summary.get("exact_match").unwrap();
    assert!((s.mean - 1.0).abs() < 1e-9);
    assert_eq!(s.count, 1);
    assert!(back.failures.is_empty());
}
