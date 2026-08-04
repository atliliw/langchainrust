//! Built-in evaluators: ExactMatch, StringDistance, EmbeddingSimilarity, LLMAsJudge.

use async_trait::async_trait;

use lc_core::BaseChatModel;
use lc_embeddings::{cosine_similarity, Embeddings};
use lc_schema::Message;

use super::criteria::{EvalError, Evaluator, Score};

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
        let v = ((sim + 1.0) / 2.0).clamp(0.0, 1.0);
        Ok(Score::new(v as f64))
    }
    fn name(&self) -> &str {
        "embedding_similarity"
    }
}

pub struct LLMAsJudge<M: BaseChatModel> {
    judge: M,
    rubric: String,
    max_score: u8,
}

const DEFAULT_RUBRIC: &str = "\
正确性:回答是否事实准确、是否与参考答案的核心意思一致。
完整性:是否完整回答了输入的问题或指令。
清晰性:表达是否清晰、无歧义、无冗余。";

impl<M: BaseChatModel> LLMAsJudge<M> {
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            rubric: DEFAULT_RUBRIC.to_string(),
            max_score: 10,
        }
    }
    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = rubric.into();
        self
    }
    pub fn with_max_score(mut self, max_score: u8) -> Self {
        self.max_score = max_score.max(1);
        self
    }

    fn build_prompt(&self, input: &str, prediction: &str, reference: &str) -> (String, String) {
        let system = format!(
            "你是一个严格、公正的评估员。请根据以下评分标准对待评估的回答打分。\n\n评分标准:\n{rubric}\n\n打分范围:0 到 {max}(0 = 完全错误或无关,{max} = 完全正确)。\n\n要求:先在 reason 字段写出简短分析,再在 score 字段给出分数。\n只输出一行 JSON,格式为:{{\"reason\":\"...\",\"score\":N}}",
            rubric = self.rubric, max = self.max_score
        );
        let user =
            format!(
            "输入:\n{input}\n\n参考答案:\n{reference}\n\n待评估的回答:\n{prediction}\n\n请评估。",
            input = input, reference = reference, prediction = prediction,
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

fn parse_score(raw: &str, max_score: u8) -> Option<f64> {
    let max = max_score as f64;
    let n = extract_json_score(raw)
        .or_else(|| find_number_after_keyword(raw, "score"))
        .or_else(|| find_number_after_keyword(raw, "分数"))
        .or_else(|| first_number(raw))?;
    Some((n / max).clamp(0.0, 1.0))
}

fn extract_json_score(raw: &str) -> Option<f64> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end < start {
        return None;
    }
    let val: serde_json::Value = serde_json::from_str(&raw[start..=end]).ok()?;
    val.get("score")?.as_f64()
}

fn find_number_after_keyword(raw: &str, keyword: &str) -> Option<f64> {
    let lower_raw = raw.to_lowercase();
    let lower_kw = keyword.to_lowercase();
    let idx = lower_raw.find(lower_kw.as_str())?;
    first_number(&raw[idx + lower_kw.len()..])
}

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
    buf.parse::<f64>().ok().or_else(|| {
        let int_part: String = buf.chars().take_while(|c| c.is_ascii_digit()).collect();
        int_part.parse::<f64>().ok()
    })
}

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
    use lc_embeddings::MockEmbeddings;

    #[tokio::test]
    async fn test_exact_match() {
        let ev = ExactMatch;
        assert_eq!(ev.eval("", "hello", "hello").await.unwrap().value, 1.0);
        assert_eq!(ev.eval("", "hello", "world").await.unwrap().value, 0.0);
        assert_eq!(ev.eval("", "  yes  ", "yes").await.unwrap().value, 1.0);
    }

    #[tokio::test]
    async fn test_string_distance() {
        let ev = StringDistance;
        let s = ev.eval("", "hello", "hello").await.unwrap().value;
        assert!((s - 1.0).abs() < 1e-9);
        let s = ev.eval("", "kitten", "sitting").await.unwrap().value;
        assert!((s - (1.0 - 3.0 / 7.0)).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_embedding_similarity_identical() {
        let ev = EmbeddingSimilarity::new(MockEmbeddings::new(32));
        let s = ev.eval("", "hello", "hello").await.unwrap().value;
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_score_unit() {
        assert!((parse_score(r#"{"score":10}"#, 10).unwrap() - 1.0).abs() < 1e-9);
        assert!((parse_score(r#"{"score":7.5}"#, 10).unwrap() - 0.75).abs() < 1e-9);
        assert!((parse_score(r#"{"score":12}"#, 10).unwrap() - 1.0).abs() < 1e-9);
        assert!(parse_score("no number here", 10).is_none());
    }
}
