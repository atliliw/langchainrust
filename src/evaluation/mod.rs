//! Evaluation 模块 - LLM 应用评测
//!
//! 提供 `Evaluator` trait、内置评测器、数据集加载与批量运行器,
//! 用于量化 prompt/模型改动的效果。
//!
//! # 示例
//! ```ignore
//! use langchainrust::evaluation::{EvalRunner, ExactMatch, StringDistance, Dataset, Example};
//! let dataset = Dataset::new(vec![Example::new("2+2?", "4")]);
//! let runner = EvalRunner::new(vec![Box::new(ExactMatch), Box::new(StringDistance)]);
//! // let report = runner.run(&dataset, &predictor).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::embeddings::{cosine_similarity, Embeddings};
use crate::{BaseChatModel, Message};

mod bleu;
mod faithfulness;
mod pairwise;
mod rules;

pub use bleu::Bleu;
pub use faithfulness::Faithfulness;
pub use pairwise::{PairwiseJudge, Verdict};
pub use rules::{ContainsKeyword, LengthCheck, RegexMatch};

/// 评测错误
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("IO 错误: {0}")]
    IoError(String),
    #[error("解析错误: {0}")]
    ParseError(String),
    #[error("嵌入错误: {0}")]
    EmbeddingError(String),
    #[error("预测错误: {0}")]
    PredictorError(String),
}

/// 评测分数(0.0–1.0,1.0 为最佳)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Score {
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// 评测样例
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub input: String,
    pub reference: String,
}

impl Example {
    pub fn new(input: impl Into<String>, reference: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            reference: reference.into(),
        }
    }
}

/// 数据集
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub examples: Vec<Example>,
}

impl Dataset {
    pub fn new(examples: Vec<Example>) -> Self {
        Self { examples }
    }

    /// 从 JSONL 文件加载(每行一个 `{input, reference}`)
    pub fn from_jsonl(path: &str) -> Result<Self, EvalError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| EvalError::IoError(e.to_string()))?;
        let mut examples = Vec::new();
        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ex: Example = serde_json::from_str(line)
                .map_err(|e| EvalError::ParseError(format!("第 {} 行: {}", i + 1, e)))?;
            examples.push(ex);
        }
        Ok(Self { examples })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// 评测器 trait
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// 对单条预测打分
    ///
    /// # 参数
    /// * `input` - 原始输入
    /// * `prediction` - 模型预测
    /// * `reference` - 参考答案
    async fn eval(
        &self,
        input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError>;

    /// 评测器名称(用于报告汇总)
    fn name(&self) -> &str;
}

/// 预测器 trait(待评测的对象:LLMChain / Agent 等)
#[async_trait]
pub trait Predictor: Send + Sync {
    async fn predict(&self, input: &str) -> Result<String, EvalError>;
}

/// 评测报告
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// 每条样例的各评测器得分
    pub per_example: Vec<HashMap<String, Score>>,
    /// 各评测器的平均分
    pub summary: HashMap<String, f64>,
}

/// 批量运行器
pub struct EvalRunner {
    evaluators: Vec<Box<dyn Evaluator>>,
}

impl EvalRunner {
    pub fn new(evaluators: Vec<Box<dyn Evaluator>>) -> Self {
        Self { evaluators }
    }

    /// 在数据集上运行所有评测器,返回报告
    pub async fn run(
        &self,
        dataset: &Dataset,
        predictor: &dyn Predictor,
    ) -> Result<Report, EvalError> {
        let mut per_example = Vec::with_capacity(dataset.len());
        let mut sums: HashMap<String, (f64, usize)> = HashMap::new();

        for ex in &dataset.examples {
            let prediction = predictor.predict(&ex.input).await?;
            let mut row = HashMap::new();
            for ev in &self.evaluators {
                let score = ev.eval(&ex.input, &prediction, &ex.reference).await?;
                let entry = sums.entry(ev.name().to_string()).or_insert((0.0, 0));
                entry.0 += score.value;
                entry.1 += 1;
                row.insert(ev.name().to_string(), score);
            }
            per_example.push(row);
        }

        let mut summary = HashMap::new();
        for (name, (total, count)) in sums {
            if count > 0 {
                summary.insert(name, total / count as f64);
            }
        }
        Ok(Report {
            per_example,
            summary,
        })
    }
}

// === 内置评测器 ===

/// 精确匹配(归一化后相等得 1.0,否则 0.0)
pub struct ExactMatch;

#[async_trait]
impl Evaluator for ExactMatch {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError> {
        let matched = prediction.trim() == reference.trim();
        let v = if matched { 1.0 } else { 0.0 };
        let label = if matched { "match" } else { "mismatch" };
        Ok(Score::new(v).with_label(label))
    }

    fn name(&self) -> &str {
        "exact_match"
    }
}

/// 字符串距离(归一化 Levenshtein:1.0=完全相同,0.0=完全不同)
pub struct StringDistance;

impl StringDistance {
    fn levenshtein(a: &str, b: &str) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let (m, n) = (a.len(), b.len());
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }
        let mut prev: Vec<usize> = (0..=n).collect();
        let mut curr: Vec<usize> = vec![0; n + 1];
        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[n]
    }
}

#[async_trait]
impl Evaluator for StringDistance {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError> {
        let dist = Self::levenshtein(prediction, reference) as f64;
        let max_len = prediction.chars().count().max(reference.chars().count()) as f64;
        let score = if max_len == 0.0 {
            1.0
        } else {
            1.0 - dist / max_len
        };
        Ok(Score::new(score))
    }

    fn name(&self) -> &str {
        "string_distance"
    }
}

/// 嵌入相似度(余弦相似度,归一化到 0.0–1.0)
pub struct EmbeddingSimilarity<E: Embeddings> {
    embeddings: E,
}

impl<E: Embeddings> EmbeddingSimilarity<E> {
    pub fn new(embeddings: E) -> Self {
        Self { embeddings }
    }
}

#[async_trait]
impl<E: Embeddings> Evaluator for EmbeddingSimilarity<E> {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError> {
        let p = self
            .embeddings
            .embed_query(prediction)
            .await
            .map_err(|e| EvalError::EmbeddingError(e.to_string()))?;
        let r = self
            .embeddings
            .embed_query(reference)
            .await
            .map_err(|e| EvalError::EmbeddingError(e.to_string()))?;
        let sim = cosine_similarity(&p, &r).unwrap_or(0.0);
        // 余弦 -1..1 归一到 0..1
        let v = ((sim + 1.0) / 2.0).clamp(0.0, 1.0);
        Ok(Score::new(v as f64))
    }

    fn name(&self) -> &str {
        "embedding_similarity"
    }
}

/// 用 LLM 当裁判的评测器(单点打分 + 轻量思维链)。
///
/// 把(输入、参考答案、模型预测)打包成裁判 prompt,调用一个
/// `BaseChatModel` 当裁判,从其回复中解析分数并归一到 `0.0–1.0`。
///
/// # 思路来源
/// - LMSYS 的 *LLM-as-a-Judge*(Zheng et al., 2023):用强模型当裁判评估输出。
/// - G-Eval(Liu et al., 2023):要求裁判先写分析(`reason`)再给分(`score`),
///   以提升打分与人类判断的一致性。本实现采用单步 JSON 输出实现轻量 CoT。
///
/// # 示例
/// ```ignore
/// use langchainrust::evaluation::{LLMAsJudge, EvalRunner};
/// // let judge = LLMAsJudge::new(chat_model);
/// // let runner = EvalRunner::new(vec![Box::new(judge)]);
/// // let report = runner.run(&dataset, &predictor).await?;
/// ```
pub struct LLMAsJudge<M: BaseChatModel> {
    judge: M,
    rubric: String,
    max_score: u8,
}

/// 默认评分标准:正确性 / 完整性 / 清晰性。
const DEFAULT_RUBRIC: &str = "\
正确性:回答是否事实准确、是否与参考答案的核心意思一致。
完整性:是否完整回答了输入的问题或指令。
清晰性:表达是否清晰、无歧义、无冗余。";

impl<M: BaseChatModel> LLMAsJudge<M> {
    /// 创建裁判评测器,使用默认评分标准与 0–10 打分。
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            rubric: DEFAULT_RUBRIC.to_string(),
            max_score: 10,
        }
    }

    /// 自定义评分标准(覆盖默认)。
    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = rubric.into();
        self
    }

    /// 自定义满分值(默认 10)。解析后的分数会归一到 `0.0–1.0`。
    pub fn with_max_score(mut self, max_score: u8) -> Self {
        self.max_score = max_score.max(1);
        self
    }

    fn build_prompt(&self, input: &str, prediction: &str, reference: &str) -> (String, String) {
        let system = format!(
            "你是一个严格、公正的评估员。请根据以下评分标准对待评估的回答打分。\n\n\
             评分标准:\n{rubric}\n\n\
             打分范围:0 到 {max}(0 = 完全错误或无关,{max} = 完全正确)。\n\n\
             要求:先在 reason 字段写出简短分析,再在 score 字段给出分数。\n\
             只输出一行 JSON,格式为:{{\"reason\":\"...\",\"score\":N}}",
            rubric = self.rubric,
            max = self.max_score
        );
        let user = format!(
            "输入:\n{input}\n\n参考答案:\n{reference}\n\n待评估的回答:\n{prediction}\n\n请评估。",
            input = input,
            reference = reference,
            prediction = prediction,
        );
        (system, user)
    }
}

#[async_trait]
impl<M: BaseChatModel> Evaluator for LLMAsJudge<M> {
    async fn eval(
        &self,
        input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError> {
        let (system, user) = self.build_prompt(input, prediction, reference);
        let result = self
            .judge
            .chat_with_system(system, vec![Message::human(user)])
            .await
            .map_err(|e| EvalError::PredictorError(e.to_string()))?;

        let raw = result.content;
        let value = parse_score(&raw, self.max_score).ok_or_else(|| {
            EvalError::ParseError(format!("无法从裁判回复解析分数: {}", truncate(&raw, 200)))
        })?;
        Ok(Score::new(value).with_label("llm_judge"))
    }

    fn name(&self) -> &str {
        "llm_as_judge"
    }
}

/// 从裁判回复中解析分数,归一到 `0.0–1.0`。
///
/// 依次尝试:
/// 1. 解析 JSON 中的 `score` 字段(支持整数或浮点)
/// 2. 查找 `score` / `分数` 关键词后的数字
/// 3. 回退到文本中出现的第一个数字
fn parse_score(raw: &str, max_score: u8) -> Option<f64> {
    let max = max_score as f64;
    let n = extract_json_score(raw)
        .or_else(|| find_number_after_keyword(raw, "score"))
        .or_else(|| find_number_after_keyword(raw, "分数"))
        .or_else(|| first_number(raw))?;
    Some((n / max).clamp(0.0, 1.0))
}

/// 从文本中查找第一个 `{...}` JSON 对象,提取 `score` 字段。
fn extract_json_score(raw: &str) -> Option<f64> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end < start {
        return None;
    }
    let val: serde_json::Value = serde_json::from_str(&raw[start..=end]).ok()?;
    val.get("score")?.as_f64()
}

/// 查找关键词之后出现的第一个数字。
fn find_number_after_keyword(raw: &str, keyword: &str) -> Option<f64> {
    let lower_raw = raw.to_lowercase();
    let lower_kw = keyword.to_lowercase();
    let idx = lower_raw.find(lower_kw.as_str())?;
    first_number(&raw[idx + lower_kw.len()..])
}

/// 提取字符串中第一个数字(支持小数)。
fn first_number(s: &str) -> Option<f64> {
    let mut buf = String::new();
    let mut started = false;
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            started = true;
            buf.push(c);
        } else if started {
            break;
        }
    }
    if buf.is_empty() {
        return None;
    }
    // 处理 "1.2.3" 这类异常:取到第一个点之前作为整数
    buf.parse::<f64>().ok().or_else(|| {
        let int_part: String = buf.chars().take_while(|c| c.is_ascii_digit()).collect();
        int_part.parse::<f64>().ok()
    })
}

/// 截断字符串用于错误信息。
fn truncate(s: &str, max: usize) -> String {
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
    use crate::embeddings::{LocalEmbeddings, MockEmbeddings};
    use crate::language_models::openai::{OpenAIChat, OpenAIConfig};
    use crate::{BaseChatModel, BaseLanguageModel, LLMResult, Runnable, RunnableConfig};
    use futures_util::Stream;
    use std::pin::Pin;

    struct StaticPredictor(&'static str);
    #[async_trait]
    impl Predictor for StaticPredictor {
        async fn predict(&self, _input: &str) -> Result<String, EvalError> {
            Ok(self.0.to_string())
        }
    }

    #[tokio::test]
    async fn test_exact_match() {
        let ev = ExactMatch;
        assert_eq!(ev.eval("", "hello", "hello").await.unwrap().value, 1.0);
        assert_eq!(ev.eval("", "hello", "world").await.unwrap().value, 0.0);
        // trim 归一
        assert_eq!(ev.eval("", "  yes  ", "yes").await.unwrap().value, 1.0);
    }

    #[tokio::test]
    async fn test_string_distance() {
        let ev = StringDistance;
        let s = ev.eval("", "hello", "hello").await.unwrap().value;
        assert!((s - 1.0).abs() < 1e-9);

        // kitten -> sitting 距离 3, 长度 7
        let s = ev.eval("", "kitten", "sitting").await.unwrap().value;
        assert!((s - (1.0 - 3.0 / 7.0)).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_embedding_similarity_identical() {
        let ev = EmbeddingSimilarity::new(MockEmbeddings::new(32));
        // 相同文本向量相同,余弦=1,归一=1
        let s = ev.eval("", "hello", "hello").await.unwrap().value;
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_runner_summary() {
        let runner = EvalRunner::new(vec![Box::new(ExactMatch)]);
        let dataset = Dataset::new(vec![Example::new("q1", "yes"), Example::new("q2", "no")]);
        // predictor 总返回 "yes":第1例匹配(1.0),第2例不匹配(0.0),均值 0.5
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
        // 全部匹配,各评测器均应为 1.0
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

    // === LLMAsJudge 测试 ===

    #[derive(Debug)]
    struct JudgeError(String);
    impl std::fmt::Display for JudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "judge error: {}", self.0)
        }
    }
    impl std::error::Error for JudgeError {}

    /// 测试用裁判模型:无论输入什么都返回预设回复。
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            Err(JudgeError("stream_chat not supported in mock".into()))
        }
    }

    #[tokio::test]
    async fn test_judge_eval_full_score() {
        // 裁判给满分 10 -> 归一 1.0
        let judge = LLMAsJudge::new(MockJudge::new(r#"{"reason":"完全正确","score":10}"#));
        let s = judge.eval("法国首都?", "巴黎", "巴黎").await.unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
        assert_eq!(s.label.as_deref(), Some("llm_judge"));
    }

    #[tokio::test]
    async fn test_judge_eval_half_score() {
        // 裁判给 5 / 10 -> 0.5
        let judge = LLMAsJudge::new(MockJudge::new(r#"{"reason":"部分正确","score":5}"#));
        let s = judge.eval("q", "pred", "ref").await.unwrap();
        assert!((s.value - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_judge_custom_max_score() {
        // 满分 5,裁判给 4 -> 0.8
        let judge =
            LLMAsJudge::new(MockJudge::new(r#"{"reason":"ok","score":4}"#)).with_max_score(5);
        let s = judge.eval("q", "pred", "ref").await.unwrap();
        assert!((s.value - 0.8).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_judge_eval_text_fallback() {
        // 裁判没按 JSON,回了 "我觉得分数: 7 分" -> keyword 回退 -> 0.7
        let judge = LLMAsJudge::new(MockJudge::new("我觉得分数: 7 分"));
        let s = judge.eval("q", "pred", "ref").await.unwrap();
        assert!((s.value - 0.7).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_judge_eval_pure_number_fallback() {
        // 裁判回 "评价一般\n7" -> 无 JSON/keyword,回退第一个数字 -> 0.7
        let judge = LLMAsJudge::new(MockJudge::new("评价一般\n7"));
        let s = judge.eval("q", "pred", "ref").await.unwrap();
        assert!((s.value - 0.7).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_judge_eval_parse_error() {
        // 裁判回复无任何数字 -> 解析失败
        let judge = LLMAsJudge::new(MockJudge::new("我不知道怎么评"));
        let err = judge.eval("q", "pred", "ref").await.unwrap_err();
        assert!(matches!(err, EvalError::ParseError(_)));
    }

    #[tokio::test]
    async fn test_judge_with_runner() {
        // 放进 EvalRunner 跑:两条都给 8 -> 平均 0.8
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

    #[test]
    fn test_parse_score_unit() {
        // JSON 整数
        assert!((parse_score(r#"{"score":10}"#, 10).unwrap() - 1.0).abs() < 1e-9);
        // JSON 浮点
        assert!((parse_score(r#"{"score":7.5}"#, 10).unwrap() - 0.75).abs() < 1e-9);
        // 超出满分 -> clamp 到 1.0
        assert!((parse_score(r#"{"score":12}"#, 10).unwrap() - 1.0).abs() < 1e-9);
        // 无数字 -> None
        assert!(parse_score("no number here", 10).is_none());
    }

    #[test]
    fn test_judge_name() {
        let judge = LLMAsJudge::new(MockJudge::new(r#"{"score":1}"#));
        assert_eq!(judge.name(), "llm_as_judge");
    }

    /// 真实 LLM 裁判测试(默认忽略,需联网 + API key)。
    ///
    /// 运行:`cargo test --lib evaluation::tests::test_judge_real_llm -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "需要真实 API key + 网络"]
    async fn test_judge_real_llm() {
        // dashscope 上无 glm5.2;改用 config 默认(maas 专属端点 + glm-5.2)
        let config = OpenAIConfig::default();
        let judge_model = OpenAIChat::new(config);
        let judge = LLMAsJudge::new(judge_model);

        // (标签, 输入, 参考答案, 模型预测)
        let cases: &[(&str, &str, &str, &str)] = &[
            ("正确", "法国首都是哪座城市?", "巴黎", "巴黎"),
            ("错误", "法国首都是哪座城市?", "巴黎", "伦敦"),
            (
                "改写",
                "法国首都是哪座城市?",
                "巴黎",
                "法国的首都是巴黎,位于法国北部。",
            ),
        ];

        let mut results: Vec<(&str, f64)> = Vec::new();
        for (name, input, reference, prediction) in cases {
            match judge.eval(input, prediction, reference).await {
                Ok(s) => {
                    println!("[{}] value={} label={:?}", name, s.value, s.label);
                    results.push((name, s.value));
                }
                Err(e) => {
                    println!("[{}] ERROR: {}", name, e);
                    results.push((name, -1.0));
                }
            }
        }

        // 1. 三条都必须解析成功且分数在 [0,1]
        for (name, v) in &results {
            assert!(*v >= 0.0, "{} 解析失败或分数非法: {}", name, v);
            assert!(*v <= 1.0, "{} 分数超出 [0,1] 范围: {}", name, v);
        }
        // 2. 语义判断:正确回答不应低于错误回答
        let correct = results[0].1;
        let wrong = results[1].1;
        assert!(
            correct >= wrong,
            "正确回答({})不应低于错误回答({})",
            correct,
            wrong
        );
    }

    /// 四个评测器在同一样例上的对比(真实 LLM 裁判)。
    ///
    /// 运行:`cargo test --lib evaluation::tests::test_all_four_evaluators_real -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "需要真实 API key + 网络(LLMAsJudge)"]
    async fn test_all_four_evaluators_real() {
        let config = OpenAIConfig::default();
        let judge_model = OpenAIChat::new(config);

        let evaluators: Vec<Box<dyn Evaluator>> = vec![
            Box::new(ExactMatch),
            Box::new(StringDistance),
            Box::new(EmbeddingSimilarity::new(LocalEmbeddings::default_dim())),
            Box::new(LLMAsJudge::new(judge_model)),
        ];

        // (标签, 输入, 参考答案, 模型预测)
        let cases: &[(&str, &str, &str, &str)] = &[
            ("正确", "法国首都是哪座城市?", "巴黎", "巴黎"),
            ("错误", "法国首都是哪座城市?", "巴黎", "伦敦"),
            (
                "改写",
                "法国首都是哪座城市?",
                "巴黎",
                "法国的首都是巴黎,位于法国北部。",
            ),
        ];

        println!(
            "\n{:<6} | {:<12} | {:<14} | {:<14} | {:<10}",
            "样例", "ExactMatch", "StringDistance", "EmbeddingSim", "LLMAsJudge"
        );
        println!("{}", "-".repeat(70));

        for (label, input, reference, prediction) in cases {
            let mut row = format!("{:<6}", label);
            for ev in &evaluators {
                let s = ev.eval(input, prediction, reference).await.unwrap();
                assert!(
                    s.value >= 0.0 && s.value <= 1.0,
                    "{} / {} 分数越界: {}",
                    label,
                    ev.name(),
                    s.value
                );
                row.push_str(&format!(" | {:<12.4}", s.value));
            }
            println!("{}", row);
        }
        println!();
    }

    /// 四个真实场景 × 四个评测器(真实 LLM 裁判)。
    ///
    /// 运行:`cargo test --lib evaluation::tests::test_four_scenarios_real -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "需要真实 API key + 网络(LLMAsJudge)"]
    async fn test_four_scenarios_real() {
        let config = OpenAIConfig::default();
        let judge_model = OpenAIChat::new(config);
        let evaluators: Vec<Box<dyn Evaluator>> = vec![
            Box::new(ExactMatch),
            Box::new(StringDistance),
            Box::new(EmbeddingSimilarity::new(LocalEmbeddings::default_dim())),
            Box::new(LLMAsJudge::new(judge_model)),
        ];

        // (场景, input, reference, &[(预测标签, 预测内容)])
        let article = "光合作用是植物、藻类和某些细菌利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程。\
                       它主要在叶绿体中进行,依赖叶绿素吸收光能。光合作用分为光反应和暗反应两个阶段:\
                       光反应在类囊体膜上发生,产生 ATP 和 NADPH;暗反应在基质中进行,利用这些产物固定二氧化碳。\
                       光合作用是地球上大多数生命的能量来源,也是大气中氧气的主要来源。";

        #[allow(clippy::type_complexity)] // 测试 fixture: (场景, input, reference, &[(标签, 内容)])
        let scenarios: &[(&str, &str, &str, &[(&str, &str)])] = &[
            (
                "RAG幻觉",
                "公司年假多少天?",
                "年假 15 天",
                &[("A忠实", "员工年假为 15 天"), ("B幻觉", "员工年假为 20 天,可累积")],
            ),
            (
                "翻译",
                "It's raining cats and dogs.",
                "倾盆大雨",
                &[("A意译", "大雨滂沱"), ("B直译", "正在下猫和狗")],
            ),
            (
                "代码",
                "写一个反转字符串的Python函数",
                "s[::-1]",
                &[
                    ("A切片", "return s[::-1]"),
                    ("B循环", "for i in range(len(s)-1,-1,-1): result += s[i]"),
                    ("C错", "s.reverse()"),
                ],
            ),
            (
                "摘要(无参考,原文当reference)",
                "请总结以下文章的要点",
                article,
                &[
                    ("A好摘要", "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程,在叶绿体中进行,分光反应和暗反应两阶段,是地球生命的主要能量来源。"),
                    ("B差摘要", "光合作用是动物利用阳光制造食物的过程,只发生在根部。"),
                ],
            ),
        ];

        for (sname, input, reference, preds) in scenarios {
            println!("\n=== 场景:{} ===", sname);
            println!("输入: {} | 参考: {}", input, reference);
            println!(
                "{:<10} | {:<12} | {:<14} | {:<14} | {:<10}",
                "预测", "ExactMatch", "StringDistance", "EmbeddingSim", "LLMAsJudge"
            );
            println!("{}", "-".repeat(70));
            for (plabel, pred) in *preds {
                let mut row = format!("{:<10}", plabel);
                for ev in &evaluators {
                    let s = ev.eval(input, pred, reference).await.unwrap();
                    row.push_str(&format!(" | {:<12.4}", s.value));
                }
                println!("{}", row);
            }
        }
        println!();
    }

    /// PairwiseJudge + Faithfulness 真实端到端测试。
    ///
    /// 运行:`cargo test --lib evaluation::tests::test_pairwise_and_faithfulness_real -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "需要真实 API key + 网络"]
    async fn test_pairwise_and_faithfulness_real() {
        let model = OpenAIChat::new(OpenAIConfig::default());

        // === PairwiseJudge:忠实回答 vs 幻觉回答,应选忠实 ===
        let pairwise = PairwiseJudge::new(model.clone());
        let v = pairwise
            .compare(
                "公司年假多少天?",
                "员工年假为 15 天",        // A 忠实
                "员工年假为 20 天,可累积", // B 幻觉
            )
            .await
            .unwrap();
        println!("Pairwise(忠实 vs 幻觉): {:?}", v);
        assert_ne!(v, Verdict::BWins, "幻觉回答不应胜出");

        // === Faithfulness:检测幻觉(上下文说 15 天) ===
        let faith = Faithfulness::new(model);
        let s_hallucinated = faith
            .eval(
                "公司年假多少天?",
                "员工年假为 20 天,可累积。",
                "员工年假为 15 天。",
            )
            .await
            .unwrap();
        println!("Faithfulness(幻觉回答): {}", s_hallucinated.value);

        let s_faithful = faith
            .eval(
                "公司年假多少天?",
                "员工年假为 15 天。",
                "员工年假为 15 天。",
            )
            .await
            .unwrap();
        println!("Faithfulness(忠实回答): {}", s_faithful.value);

        assert!(
            s_faithful.value >= s_hallucinated.value,
            "忠实({})应 >= 幻觉({})",
            s_faithful.value,
            s_hallucinated.value
        );
    }
}
