//! Module-level integration tests: EvalRunner end-to-end, LLMAsJudge with MockJudge.

use super::*;
use crate::embeddings::MockEmbeddings;
use crate::{BaseChatModel, BaseLanguageModel, LLMResult, Message, Runnable, RunnableConfig};
use async_trait::async_trait;
use futures_util::Stream;
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        Err(JudgeError("stream_chat not supported in mock".into()))
    }
}

#[tokio::test]
async fn test_runner_summary() {
    let runner = EvalRunner::new(vec![Box::new(ExactMatch)]);
    let dataset = Dataset::new(vec![Example::new("q1", "yes"), Example::new("q2", "no")]);
    let report = runner.run(&dataset, &StaticPredictor("yes")).await.unwrap();
    assert_eq!(report.per_example.len(), 2);
    let avg = *report.summary.get("exact_match").unwrap();
    assert!((avg - 0.5).abs() < 1e-9);
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
        assert!((v - 1.0).abs() < 1e-6);
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
    let avg = *report.summary.get("llm_as_judge").unwrap();
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

#[test]
fn test_judge_name() {
    let judge = LLMAsJudge::new(MockJudge::new(r#"{"score":1}"#));
    assert_eq!(judge.name(), "llm_as_judge");
}
