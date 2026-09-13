//! RAGAS-style RAG evaluators (B9, v0.22.4): answer relevancy, context precision, context recall.
//!
//! All three are LLM-backed and generic over `M: BaseChatModel`; answer relevancy additionally
//! needs an [`lc_embeddings::Embeddings`] model. They share the structured-output-first calling
//! convention of [`crate::Faithfulness`] (`bind_tools` tool call, text fallback) and the same
//! concurrency cap.
//!
//! | Evaluator | RAGAS idea | Inputs used |
//! |---|---|---|
//! | [`AnswerRelevancy`] | generate questions the answer could address; mean cosine similarity of those questions to the actual question | question + answer |
//! | [`ContextPrecision`] | rank-weighted precision of chunks judged relevant to the question | question + ranked contexts |
//! | [`ContextRecall`] | share of reference claims attributable to the retrieved contexts | reference + contexts |
//!
//! [`ContextPrecision`] / [`ContextRecall`] implement [`crate::RagEvaluator`] only (a plain
//! [`crate::Evaluator`] has no contexts slot). [`AnswerRelevancy`] implements both traits, so it
//! also scores inside non-RAG runners (contexts are irrelevant to it).

use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};
use serde::Deserialize;

use lc_core::judge::{structured_call, truncate, StructuredJudgeError};
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_embeddings::{cosine_similarity, Embeddings};
use lc_schema::Message;

use super::criteria::{EvalError, Evaluator, RagEvaluator, Score};
use super::faithfulness::{parse_yes_no, split_claims};

/// Maximum concurrent judge calls in a single metric evaluation (same rationale as faithfulness:
/// avoid N contexts/claims all dying to a judge rate limit).
const MAX_CONCURRENT_JUDGE: usize = 4;

/// Per-context / joined-context character cap sent to the judge.
const DEFAULT_MAX_CONTEXT_CHARS: usize = 2000;

/// Default number of questions generated for answer relevancy.
const DEFAULT_N_QUESTIONS: usize = 3;

/// Structured verdict arguments (returned via tool_calls).
#[derive(Debug, Deserialize)]
struct RagVerdictArgs {
    verdict: bool,
    /// Brief reason (improves judgment quality); not consumed numerically.
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// Text fallback shared by the two boolean judges: an unparseable reply is a parse error,
/// never a silent `false` (an off-topic model must not be read as "irrelevant").
fn parse_verdict_or_error(raw: &str) -> Result<RagVerdictArgs, StructuredJudgeError> {
    let verdict = parse_yes_no(raw).ok_or_else(|| {
        StructuredJudgeError::Parse(format!(
            "failed to parse yes/no from judge reply: {}",
            truncate(raw, 200)
        ))
    })?;
    Ok(RagVerdictArgs {
        verdict,
        reason: String::new(),
    })
}

// =================================================================================================
// Context precision
// =================================================================================================

/// RAGAS context precision: are the retrieved chunks relevant, and are relevant chunks ranked high?
///
/// Each context (in retrieval rank order) is judged relevant to the question; the score is the
/// RAGAS rank-weighted precision:
///
/// ```text
///               K
///             ----
///          1 \
/// CP@K = ----- /   v_k · Precision@k
///        |REL| ----
///             k = 1
/// ```
///
/// where `v_k` is the binary relevance verdict for the chunk at rank k, `Precision@k` the share
/// of relevant chunks in the top k, and `|REL|` the number of relevant chunks. A chunk relevant
/// but buried below irrelevant ones therefore scores lower than the same chunk ranked first.
pub struct ContextPrecision<M: BaseChatModel> {
    judge: M,
    /// Per-context character cap (default 2000).
    max_context_chars: usize,
    /// Score when no contexts were provided (default 0.0).
    empty_score: f64,
}

impl<M: BaseChatModel> ContextPrecision<M> {
    /// Creates a context-precision evaluator.
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            max_context_chars: DEFAULT_MAX_CONTEXT_CHARS,
            empty_score: 0.0,
        }
    }

    /// Per-context character cap sent to the judge.
    pub fn with_max_context_chars(mut self, max: usize) -> Self {
        self.max_context_chars = max;
        self
    }

    /// Score when the example carries no retrieved contexts (default 0.0).
    pub fn with_empty_score(mut self, score: f64) -> Self {
        self.empty_score = score;
        self
    }

    /// Asks the judge whether a single ranked chunk is relevant to the question.
    async fn judge_chunk(&self, input: &str, chunk: &str) -> Result<bool, EvalError> {
        let system = "你是检索质量评估员。判断给定的检索文本块是否包含有助于回答用户问题的信息。调用 judge_context 工具提交判定。"
            .to_string();
        let user =
            format!("用户问题:\n{input}\n\n检索文本块:\n{chunk}\n\n该文本块与回答该问题相关吗?");
        let messages = vec![Message::system(system), Message::human(user)];
        let args: RagVerdictArgs = structured_call(
            &self.judge,
            relevance_tool(),
            messages,
            parse_verdict_or_error,
        )
        .await?;
        Ok(args.verdict)
    }
}

#[async_trait]
impl<M: BaseChatModel> RagEvaluator for ContextPrecision<M> {
    async fn eval_rag(
        &self,
        input: &str,
        _prediction: &str,
        contexts: &[String],
        _reference: &str,
    ) -> Result<Score, EvalError> {
        if contexts.is_empty() {
            return Ok(Score::new(self.empty_score).with_label("no_contexts"));
        }
        // Truncate each chunk once up front; `buffered` (not buffer_unordered) preserves rank
        // order in the returned verdicts — order is the whole point of this metric.
        let chunks: Vec<String> = contexts
            .iter()
            .map(|c| truncate(c, self.max_context_chars).to_string())
            .collect();
        let verdicts: Vec<Result<bool, EvalError>> = stream::iter(chunks)
            .map(|chunk| async move { self.judge_chunk(input, &chunk).await })
            .buffered(MAX_CONCURRENT_JUDGE)
            .collect()
            .await;

        let mut relevant_in_top_k = 0usize;
        let mut total_relevant = 0usize;
        let mut weighted = 0.0;
        for (k, verdict) in verdicts.into_iter().enumerate() {
            let relevant = verdict?;
            if relevant {
                relevant_in_top_k += 1;
                total_relevant += 1;
                let precision_at_k = relevant_in_top_k as f64 / (k + 1) as f64;
                weighted += precision_at_k;
            }
        }
        if total_relevant == 0 {
            // RAGAS: no relevant context at all -> 0, regardless of the empty-context setting.
            return Ok(Score::new(0.0).with_label("no_relevant"));
        }
        Ok(Score::new(weighted / total_relevant as f64).with_label("context_precision"))
    }

    fn name(&self) -> &str {
        "context_precision"
    }
}

fn relevance_tool() -> ToolDefinition {
    ToolDefinition::new(
        "judge_context",
        "判断检索文本块是否与用户问题相关,提交布尔判定。",
    )
    .with_parameters(serde_json::json!({
        "type": "object",
        "properties": {
            "verdict": { "type": "boolean", "description": "文本块是否包含有助于回答问题的信息" },
            "reason": { "type": "string", "description": "简短依据" }
        },
        "required": ["verdict", "reason"]
    }))
}

// =================================================================================================
// Context recall
// =================================================================================================

/// RAGAS context recall: share of the reference answer's claims that the retrieved contexts support.
///
/// The reference is split into atomic claims (same splitter faithfulness uses) and each claim is
/// judged against the union of retrieved contexts; recall = attributable claims / total claims.
pub struct ContextRecall<M: BaseChatModel> {
    judge: M,
    /// Character cap for the joined context block, truncated once (default 2000).
    max_context_chars: usize,
    /// Score when the reference carries no claims or no contexts were given (default 0.0).
    empty_score: f64,
}

impl<M: BaseChatModel> ContextRecall<M> {
    /// Creates a context-recall evaluator.
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            max_context_chars: DEFAULT_MAX_CONTEXT_CHARS,
            empty_score: 0.0,
        }
    }

    /// Character cap for the joined context block sent per claim.
    pub fn with_max_context_chars(mut self, max: usize) -> Self {
        self.max_context_chars = max;
        self
    }

    /// Score when there is nothing to attribute (no claims / no contexts; default 0.0).
    pub fn with_empty_score(mut self, score: f64) -> Self {
        self.empty_score = score;
        self
    }

    /// Asks the judge whether a single reference claim can be derived from the contexts.
    async fn verify_claim(&self, context: &str, claim: &str) -> Result<bool, EvalError> {
        let system = "你是事实核查员。判断参考答案中的陈述能否从任一检索上下文中推导出来。调用 check_claim 工具提交判定。"
            .to_string();
        let user = format!(
            "检索上下文:\n{context}\n\n参考答案陈述:\n{claim}\n\n这条陈述能从检索上下文推导出来吗?"
        );
        let messages = vec![Message::system(system), Message::human(user)];
        let args: RagVerdictArgs =
            structured_call(&self.judge, recall_tool(), messages, parse_verdict_or_error).await?;
        Ok(args.verdict)
    }
}

#[async_trait]
impl<M: BaseChatModel> RagEvaluator for ContextRecall<M> {
    async fn eval_rag(
        &self,
        _input: &str,
        _prediction: &str,
        contexts: &[String],
        reference: &str,
    ) -> Result<Score, EvalError> {
        if contexts.is_empty() {
            return Ok(Score::new(self.empty_score).with_label("no_contexts"));
        }
        let claims = split_claims(reference);
        if claims.is_empty() {
            return Ok(Score::new(self.empty_score).with_label("no_claims"));
        }
        // Join once, truncate the whole block once (same pattern as faithfulness).
        let context = truncate(&contexts.join("\n\n---\n\n"), self.max_context_chars);
        let ctx = &context;
        let total = claims.len();
        let results: Vec<Result<bool, EvalError>> = stream::iter(claims)
            .map(|claim| async move { self.verify_claim(ctx, &claim).await })
            .buffer_unordered(MAX_CONCURRENT_JUDGE)
            .collect()
            .await;
        let mut attributable = 0usize;
        for r in results {
            if r? {
                attributable += 1;
            }
        }
        Ok(Score::new(attributable as f64 / total as f64).with_label("context_recall"))
    }

    fn name(&self) -> &str {
        "context_recall"
    }
}

fn recall_tool() -> ToolDefinition {
    ToolDefinition::new(
        "check_claim",
        "判断参考答案陈述能否从检索上下文推导出来,提交布尔判定。",
    )
    .with_parameters(serde_json::json!({
        "type": "object",
        "properties": {
            "verdict": { "type": "boolean", "description": "能否从任一检索上下文推导" },
            "reason": { "type": "string", "description": "简短依据" }
        },
        "required": ["verdict", "reason"]
    }))
}

// =================================================================================================
// Answer relevancy
// =================================================================================================

/// RAGAS answer relevancy: does the answer actually address the question?
///
/// The generator produces `n` questions that the answer could address (one LLM call); the score
/// is the mean cosine similarity between the original question's embedding and the generated
/// questions' embeddings. An answer full of on-topic-looking but non-answer text generates
/// off-target questions and scores low, without needing a reference answer.
///
/// Uses one chat call (plain-text question list) plus one embedding batch per evaluation.
pub struct AnswerRelevancy<M: BaseChatModel, E: Embeddings> {
    generator: M,
    embeddings: E,
    /// Number of questions to ask the generator for (default 3).
    n_questions: usize,
    /// Score when the prediction is empty (default 0.0).
    empty_score: f64,
}

impl<M: BaseChatModel, E: Embeddings> AnswerRelevancy<M, E> {
    /// Creates an answer-relevancy evaluator.
    pub fn new(generator: M, embeddings: E) -> Self {
        Self {
            generator,
            embeddings,
            n_questions: DEFAULT_N_QUESTIONS,
            empty_score: 0.0,
        }
    }

    /// Sets how many questions the generator should produce (clamped to at least 1).
    pub fn with_n_questions(mut self, n: usize) -> Self {
        self.n_questions = n.max(1);
        self
    }

    /// Score for an empty prediction (default 0.0: no answer is not relevant).
    pub fn with_empty_score(mut self, score: f64) -> Self {
        self.empty_score = score;
        self
    }

    /// Shared scoring core (both trait impls delegate here).
    async fn score(&self, input: &str, prediction: &str) -> Result<Score, EvalError> {
        if prediction.trim().is_empty() {
            return Ok(Score::new(self.empty_score).with_label("no_answer"));
        }
        let questions = self.generate_questions(prediction).await?;
        if questions.is_empty() {
            // Fail fast: a model that returns no questions is a judge failure, not a zero verdict.
            return Err(EvalError::ParseError(
                "answer relevancy generator produced no questions".into(),
            ));
        }

        let original = self
            .embeddings
            .embed_query(input)
            .await
            .map_err(|e| EvalError::EmbeddingError(e.to_string()))?;
        let refs: Vec<&str> = questions.iter().map(String::as_str).collect();
        let generated = self
            .embeddings
            .embed_documents(&refs)
            .await
            .map_err(|e| EvalError::EmbeddingError(e.to_string()))?;
        if generated.len() != questions.len() {
            return Err(EvalError::EmbeddingError(format!(
                "embedding batch mismatch: asked for {}, got {}",
                questions.len(),
                generated.len()
            )));
        }

        let mut sum = 0.0;
        for v in &generated {
            // A dimension mismatch is a data defect (inconsistent embedding space), not a zero.
            let sim = cosine_similarity(&original, v)
                .map_err(|e| EvalError::EmbeddingError(e.to_string()))?
                as f64;
            sum += sim;
        }
        // RAGAS averages raw cosines; Score::new rejects NaN and clamps to 0..=1.
        Ok(Score::new(sum / questions.len() as f64).with_label("answer_relevancy"))
    }

    /// One plain chat call producing `n` questions, one per non-empty line.
    async fn generate_questions(&self, prediction: &str) -> Result<Vec<String>, EvalError> {
        let system = format!(
            "你是问题生成器。仅根据给定回答,生成 {} 个不同的、该回答能够回答的问题。每行一个问题,不要编号、不要解释。",
            self.n_questions
        );
        let user = format!(
            "回答:\n{prediction}\n\n请生成 {} 个问题,每行一个:",
            self.n_questions
        );
        let result = self
            .generator
            .chat_with_system(system, vec![Message::human(user)])
            .await
            .map_err(|e| EvalError::PredictorError(e.to_string()))?;
        Ok(result
            .content
            .lines()
            .map(str::trim)
            .map(|l| {
                // Strip a leading list marker only when digits are followed by a separator
                // ("1. ", "2) ", "3、"); a leading number inside a real question ("2+2=?") stays.
                let bytes = l.as_bytes();
                let mut i = 0;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let ascii_sep = i < bytes.len() && (bytes[i] == b'.' || bytes[i] == b')');
                let ideographic_sep = i < bytes.len() && l[i..].starts_with('、');
                if i > 0 && (ascii_sep || ideographic_sep) {
                    let sep_len = if ascii_sep { 1 } else { '、'.len_utf8() };
                    l[i + sep_len..].trim()
                } else {
                    l
                }
            })
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect())
    }
}

#[async_trait]
impl<M: BaseChatModel, E: Embeddings> RagEvaluator for AnswerRelevancy<M, E> {
    async fn eval_rag(
        &self,
        input: &str,
        prediction: &str,
        _contexts: &[String],
        _reference: &str,
    ) -> Result<Score, EvalError> {
        self.score(input, prediction).await
    }

    fn name(&self) -> &str {
        "answer_relevancy"
    }
}

#[async_trait]
impl<M: BaseChatModel, E: Embeddings> Evaluator for AnswerRelevancy<M, E> {
    async fn eval(
        &self,
        input: &str,
        prediction: &str,
        _reference: &str,
    ) -> Result<Score, EvalError> {
        self.score(input, prediction).await
    }

    fn name(&self) -> &str {
        "answer_relevancy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::{BaseLanguageModel, Runnable, RunnableConfig};
    use lc_embeddings::EmbeddingError;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockError(String);
    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for MockError {}

    /// Plain-text mock chat model (question generator / text-fallback path).
    struct TextMock {
        replies: Vec<String>,
        calls: Arc<AtomicUsize>,
    }
    impl TextMock {
        fn new(replies: Vec<String>) -> Self {
            Self {
                replies,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for TextMock {
        type Error = MockError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockError("use chat".into()))
        }
    }
    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for TextMock {
        fn model_name(&self) -> &str {
            "text-mock"
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
    impl BaseChatModel for TextMock {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let idx = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(LLMResult {
                content: self.replies.get(idx).cloned().unwrap_or_default(),
                model: "text-mock".into(),
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
            Err(MockError("not supported".into()))
        }
    }

    /// Structured tool-call mock reused from the crate test helper.
    use crate::test_support::ToolJudge;

    /// Scripted embeddings: routes a small set of texts to fixed unit vectors.
    struct ScriptedEmbeddings {
        dim: usize,
        map: Vec<(String, Vec<f32>)>,
    }
    impl ScriptedEmbeddings {
        fn new(map: Vec<(&str, Vec<f32>)>) -> Self {
            let dim = map.first().map(|(_, v)| v.len()).unwrap_or(1);
            Self {
                dim,
                map: map.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            }
        }
    }
    #[async_trait]
    impl Embeddings for ScriptedEmbeddings {
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            self.map
                .iter()
                .find(|(k, _)| k == text)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| EmbeddingError::Config(format!("unscripted text: {text}")))
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn model_name(&self) -> &str {
            "scripted"
        }
    }

    // ---- ContextPrecision ---------------------------------------------------------------------

    #[tokio::test]
    async fn context_precision_weights_by_rank() {
        // ranks: relevant, irrelevant, relevant -> (Precision@1 + Precision@3) / 2
        //      = (1 + 2/3) / 2 = 5/6
        let judge = ToolJudge::sequence(vec![
            r#"{"verdict": true, "reason": "r"}"#.into(),
            r#"{"verdict": false, "reason": "r"}"#.into(),
            r#"{"verdict": true, "reason": "r"}"#.into(),
        ]);
        let contexts = vec!["c0".into(), "c1".into(), "c2".into()];
        let s = ContextPrecision::new(judge)
            .eval_rag("q", "a", &contexts, "ref")
            .await
            .unwrap();
        assert!((s.value - 5.0 / 6.0).abs() < 1e-9, "got {}", s.value);
    }

    #[tokio::test]
    async fn context_precision_all_relevant_is_one() {
        let judge = ToolJudge::sequence(vec![
            r#"{"verdict": true}"#.into(),
            r#"{"verdict": true}"#.into(),
        ]);
        let contexts = vec!["c0".into(), "c1".into()];
        let s = ContextPrecision::new(judge)
            .eval_rag("q", "a", &contexts, "ref")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn context_precision_none_relevant_is_zero() {
        let judge = ToolJudge::sequence(vec![
            r#"{"verdict": false}"#.into(),
            r#"{"verdict": false}"#.into(),
        ]);
        let contexts = vec!["c0".into(), "c1".into()];
        let s = ContextPrecision::new(judge)
            .eval_rag("q", "a", &contexts, "ref")
            .await
            .unwrap();
        assert_eq!(s.value, 0.0);
        assert_eq!(s.label.as_deref(), Some("no_relevant"));
    }

    #[tokio::test]
    async fn context_precision_empty_contexts_uses_empty_score() {
        let judge = ToolJudge::new(r#"{"verdict": true}"#);
        let s = ContextPrecision::new(judge)
            .with_empty_score(1.0)
            .eval_rag("q", "a", &[], "ref")
            .await
            .unwrap();
        assert_eq!(s.value, 1.0);
        assert_eq!(s.label.as_deref(), Some("no_contexts"));
    }

    // ---- ContextRecall ------------------------------------------------------------------------

    #[tokio::test]
    async fn context_recall_half_attributable() {
        let judge = ToolJudge::sequence(vec![
            r#"{"verdict": true, "reason": "r"}"#.into(),
            r#"{"verdict": false, "reason": "r"}"#.into(),
        ]);
        let contexts = vec!["ctx".into()];
        let s = ContextRecall::new(judge)
            .eval_rag("q", "a", &contexts, "巴黎是首都。伦敦是首都。")
            .await
            .unwrap();
        assert!((s.value - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn context_recall_no_contexts() {
        let judge = ToolJudge::new(r#"{"verdict": true}"#);
        let s = ContextRecall::new(judge)
            .eval_rag("q", "a", &[], "巴黎是首都。")
            .await
            .unwrap();
        assert_eq!(s.value, 0.0);
        assert_eq!(s.label.as_deref(), Some("no_contexts"));
    }

    #[tokio::test]
    async fn context_recall_no_claims() {
        let judge = ToolJudge::new(r#"{"verdict": true}"#);
        let s = ContextRecall::new(judge)
            .with_empty_score(1.0)
            .eval_rag("q", "a", &["ctx".to_string()], "。。。")
            .await
            .unwrap();
        assert_eq!(s.value, 1.0);
        assert_eq!(s.label.as_deref(), Some("no_claims"));
    }

    // ---- AnswerRelevancy ----------------------------------------------------------------------

    #[tokio::test]
    async fn answer_relevancy_identical_questions_scores_one() {
        let gen = TextMock::new(vec!["q-gen-1\nq-gen-2".into()]);
        let emb = ScriptedEmbeddings::new(vec![
            ("q", vec![1.0, 0.0]),
            ("q-gen-1", vec![1.0, 0.0]),
            ("q-gen-2", vec![1.0, 0.0]),
        ]);
        let s = AnswerRelevancy::new(gen, emb)
            .eval_rag("q", "an answer", &[], "")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-6);
        assert_eq!(s.label.as_deref(), Some("answer_relevancy"));
    }

    #[tokio::test]
    async fn answer_relevancy_averages_cosines() {
        let gen = TextMock::new(vec!["same\northogonal".into()]);
        let emb = ScriptedEmbeddings::new(vec![
            ("q", vec![1.0, 0.0]),
            ("same", vec![1.0, 0.0]),
            ("orthogonal", vec![0.0, 1.0]),
        ]);
        let s = AnswerRelevancy::new(gen, emb)
            .eval("q", "an answer", "")
            .await
            .unwrap();
        assert!((s.value - 0.5).abs() < 1e-6, "got {}", s.value);
    }

    #[tokio::test]
    async fn answer_relevancy_strips_list_numbering() {
        // generators commonly prefix "1. " / "2) " — these must not become part of the text key
        let gen = TextMock::new(vec!["1. same\n2) same".into()]);
        let emb = ScriptedEmbeddings::new(vec![("q", vec![1.0, 0.0]), ("same", vec![1.0, 0.0])]);
        let s = AnswerRelevancy::new(gen, emb)
            .eval("q", "answer", "")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn answer_relevancy_keeps_leading_digits_of_real_questions() {
        // a question that itself starts with a digit must not be treated as a list marker
        let gen = TextMock::new(vec!["2+2等于几?".into()]);
        let emb =
            ScriptedEmbeddings::new(vec![("q", vec![1.0, 0.0]), ("2+2等于几?", vec![1.0, 0.0])]);
        let s = AnswerRelevancy::new(gen, emb)
            .eval("q", "answer", "")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn answer_relevancy_empty_prediction_uses_empty_score() {
        let gen = TextMock::new(vec![]);
        let emb = ScriptedEmbeddings::new(vec![("q", vec![1.0])]);
        let s = AnswerRelevancy::new(gen, emb)
            .with_empty_score(1.0)
            .eval("q", "   ", "")
            .await
            .unwrap();
        assert_eq!(s.value, 1.0);
        assert_eq!(s.label.as_deref(), Some("no_answer"));
    }

    #[tokio::test]
    async fn answer_relevancy_zero_generated_questions_errors() {
        let gen = TextMock::new(vec!["   \n  ".into()]);
        let emb = ScriptedEmbeddings::new(vec![("q", vec![1.0])]);
        let err = AnswerRelevancy::new(gen, emb)
            .eval("q", "answer", "")
            .await
            .unwrap_err();
        assert!(matches!(err, EvalError::ParseError(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn answer_relevancy_dimension_mismatch_errors() {
        let gen = TextMock::new(vec!["g".into()]);
        let emb = ScriptedEmbeddings::new(vec![("q", vec![1.0, 0.0]), ("g", vec![1.0, 0.0, 0.0])]);
        let err = AnswerRelevancy::new(gen, emb)
            .eval("q", "answer", "")
            .await
            .unwrap_err();
        assert!(matches!(err, EvalError::EmbeddingError(_)));
    }
}
