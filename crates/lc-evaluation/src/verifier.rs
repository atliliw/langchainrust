//! Verifiers: fine-grained scoring for candidate selection (test-time scaling).
//!
//! A *judge* answers "is this good?" with a coarse label; a *verifier* answers
//! "how strongly does this satisfy the criteria?" with a score fine-grained
//! enough to compare candidates. The logprob path takes the expectation over
//! the model's score-token distribution instead of trusting a single argmax,
//! which is what makes best-of-N selection meaningful.
//!
//! Two implementations are provided:
//! - [`LogprobVerifier`] — requires a [`LogprobChatModel`] (OpenAI-compatible
//!   providers that expose `logprobs`).
//! - [`JudgeVerifier`] — works with any [`BaseChatModel`], degrades to a single
//!   integer score. Use it when the provider has no logprob support.
//!
//! [`best_of_n`] scores a batch of candidates and returns the winner.

use async_trait::async_trait;
use lc_core::language_models::{
    BaseChatModel, LogprobChatModel, LogprobResult, TokenLogprob,
};
use lc_schema::Message;

use crate::EvalError;

/// Where a [`VerificationScore`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreSource {
    /// Expected value over the model's score-token logprob distribution.
    Logprob,
    /// A single integer parsed from a judge's text output.
    Judge,
}

/// A fine-grained verification score.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationScore {
    /// Expected score in `[0, max_score]`.
    pub score: f64,
    /// Upper bound of the scale.
    pub max_score: f64,
    /// Confidence in `[0, 1]`. For logprob scores this is the normalized
    /// probability mass on the winning score token; for judges it is a fixed
    /// low value because there is no distribution to inspect.
    pub confidence: f64,
    /// How the score was produced.
    pub source: ScoreSource,
}

impl VerificationScore {
    /// Score normalized to `[0, 1]`.
    pub fn normalized(&self) -> f64 {
        if self.max_score <= 0.0 {
            0.0
        } else {
            (self.score / self.max_score).clamp(0.0, 1.0)
        }
    }
}

/// Scores a candidate against a task description.
#[async_trait]
pub trait Verifier: Send + Sync {
    /// Returns a fine-grained score for `candidate` on `task`.
    async fn verify(
        &self,
        task: &str,
        candidate: &str,
    ) -> Result<VerificationScore, EvalError>;
}

/// Verifier backed by a model that exposes token log-probabilities.
///
/// The model is asked to emit a single integer in `0..=max_score`; the
/// verifier expands that token into its top-k distribution, renormalizes it
/// over the valid score tokens, and returns the expectation.
pub struct LogprobVerifier<M> {
    model: M,
    task_prefix: String,
    max_score: u32,
    top_logprobs: usize,
}

impl<M> LogprobVerifier<M> {
    /// Creates a verifier on a `0..=9` scale with 5 alternatives.
    pub fn new(model: M) -> Self {
        Self {
            model,
            task_prefix: DEFAULT_TASK_PREFIX.to_string(),
            max_score: 9,
            top_logprobs: 5,
        }
    }

    /// Sets the maximum score (inclusive).
    pub fn with_max_score(mut self, max_score: u32) -> Self {
        self.max_score = max_score;
        self
    }

    /// Sets how many alternatives to request per token.
    pub fn with_top_logprobs(mut self, n: usize) -> Self {
        self.top_logprobs = n.max(1);
        self
    }

    /// Overrides the instruction placed before the task.
    pub fn with_task_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.task_prefix = prefix.into();
        self
    }

    fn prompt(&self, task: &str, candidate: &str) -> String {
        format!(
            "{}\n\nScore the candidate from 0 to {}.\n\
             Reply with ONE integer only.\n\nTASK:\n{}\n\nCANDIDATE:\n{}",
            self.task_prefix, self.max_score, task, candidate
        )
    }
}

#[async_trait]
impl<M> Verifier for LogprobVerifier<M>
where
    M: LogprobChatModel + Send + Sync,
    M::Error: std::fmt::Display,
{
    async fn verify(
        &self,
        task: &str,
        candidate: &str,
    ) -> Result<VerificationScore, EvalError> {
        let result: LogprobResult = self
            .model
            .chat_with_logprobs(
                vec![Message::human(self.prompt(task, candidate))],
                self.top_logprobs,
                None,
            )
            .await
            .map_err(|e| EvalError::PredictorError(format!("verifier call failed: {e}")))?;

        score_from_logprobs(&result.token_logprobs, self.max_score).ok_or_else(|| {
            EvalError::ParseError(format!(
                "verifier response contained no score token in 0..={}",
                self.max_score
            ))
        })
    }
}

/// Verifier backed by any chat model; asks for one integer and parses it.
///
/// This is the degradation path for providers without logprobs (notably
/// Anthropic). It is usable, but its scores are coarse and its confidence is
/// deliberately reported as low so callers do not over-trust it.
pub struct JudgeVerifier<M> {
    model: M,
    task_prefix: String,
    max_score: u32,
}

impl<M> JudgeVerifier<M> {
    /// Creates a judge verifier on a `0..=9` scale.
    pub fn new(model: M) -> Self {
        Self {
            model,
            task_prefix: DEFAULT_TASK_PREFIX.to_string(),
            max_score: 9,
        }
    }

    /// Sets the maximum score (inclusive).
    pub fn with_max_score(mut self, max_score: u32) -> Self {
        self.max_score = max_score;
        self
    }

    /// Overrides the instruction placed before the task.
    pub fn with_task_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.task_prefix = prefix.into();
        self
    }
}

#[async_trait]
impl<M> Verifier for JudgeVerifier<M>
where
    M: BaseChatModel + Send + Sync,
    M::Error: std::fmt::Display,
{
    async fn verify(
        &self,
        task: &str,
        candidate: &str,
    ) -> Result<VerificationScore, EvalError> {
        let prompt = format!(
            "{}\n\nScore the candidate from 0 to {}.\n\
             Reply with ONE integer only.\n\nTASK:\n{}\n\nCANDIDATE:\n{}",
            self.task_prefix, self.max_score, task, candidate
        );
        let result = self
            .model
            .chat(vec![Message::human(prompt)], None)
            .await
            .map_err(|e| EvalError::PredictorError(format!("verifier call failed: {e}")))?;

        let score = result
            .content
            .trim()
            .chars()
            .find(|c| c.is_ascii_digit())
            .and_then(|c| c.to_digit(10))
            .filter(|s| *s <= self.max_score)
            .ok_or_else(|| {
                EvalError::ParseError(format!(
                    "judge returned no integer in 0..={}: {:?}",
                    self.max_score, result.content
                ))
            })?;

        Ok(VerificationScore {
            score: f64::from(score),
            max_score: f64::from(self.max_score),
            // No distribution to inspect: report low confidence rather than
            // pretending a judge's single integer is certain.
            confidence: 0.25,
            source: ScoreSource::Judge,
        })
    }
}

/// Result of [`best_of_n`].
#[derive(Debug, Clone)]
pub struct BestOfNResult {
    /// Index of the winning candidate.
    pub best_index: usize,
    /// Winning candidate text.
    pub best: String,
    /// One score per candidate, in input order.
    pub scores: Vec<VerificationScore>,
}

/// Scores every candidate and returns the highest-scoring one.
///
/// Ties keep the earliest candidate (deterministic). An empty candidate list
/// is an error rather than a silent `None`.
pub async fn best_of_n<V>(
    verifier: &V,
    task: &str,
    candidates: &[String],
) -> Result<BestOfNResult, EvalError>
where
    V: Verifier + ?Sized,
{
    if candidates.is_empty() {
        return Err(EvalError::PredictorError(
            "best_of_n requires at least one candidate".to_string(),
        ));
    }

    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        scores.push(verifier.verify(task, candidate).await?);
    }

    // Strict `>` keeps the earliest candidate on ties (deterministic).
    let mut best_index = 0usize;
    for (i, score) in scores.iter().enumerate() {
        if score.score > scores[best_index].score {
            best_index = i;
        }
    }

    Ok(BestOfNResult {
        best_index,
        best: candidates[best_index].clone(),
        scores,
    })
}

/// Expands the first score-bearing token into a normalized distribution and
/// returns its expectation.
///
/// Returns `None` when no token/alternative parses as an integer in
/// `0..=max_score`.
pub fn score_from_logprobs(
    token_logprobs: &[TokenLogprob],
    max_score: u32,
) -> Option<VerificationScore> {
    for entry in token_logprobs {
        let mut mass = vec![0.0f64; (max_score + 1) as usize];

        // Include the generated token itself, then the alternatives. A gateway
        // may repeat the chosen token inside `top_logprobs`; summing handles it.
        {
            let mut add = |token: &str, logprob: f64| {
                if let Some(score) = parse_score_token(token, max_score) {
                    mass[score as usize] += logprob.exp();
                }
            };
            // Most providers include the generated token in `top_logprobs`;
            // adding it again would double its mass.
            let chosen_listed = entry
                .top_logprobs
                .iter()
                .any(|alt| alt.token == entry.token);
            if !chosen_listed {
                add(&entry.token, entry.logprob);
            }
            for alt in &entry.top_logprobs {
                add(&alt.token, alt.logprob);
            }
        }

        let total: f64 = mass.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            continue;
        }

        let expected: f64 = mass
            .iter()
            .enumerate()
            .map(|(score, p)| score as f64 * (p / total))
            .sum();
        let confidence = mass.iter().cloned().fold(0.0f64, f64::max) / total;

        return Some(VerificationScore {
            score: expected,
            max_score: f64::from(max_score),
            confidence: confidence.clamp(0.0, 1.0),
            source: ScoreSource::Logprob,
        });
    }
    None
}

fn parse_score_token(token: &str, max_score: u32) -> Option<u32> {
    token
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|score| *score <= max_score)
}

const DEFAULT_TASK_PREFIX: &str = "\
You are a strict verifier. Judge how well the candidate satisfies the task. \
Be calibrated: partial credit is expected, and 9 means fully correct.";

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::{TokenAlternative, TokenLogprob};

    fn token(token: &str, logprob: f64, alts: &[(&str, f64)]) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob,
            top_logprobs: alts
                .iter()
                .map(|(t, lp)| TokenAlternative {
                    token: t.to_string(),
                    logprob: *lp,
                })
                .collect(),
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn logprob_expectation_beats_argmax() {
        // Both candidates would argmax to "9", but the distributions differ.
        let sharp = vec![token("9", 1.0f64.ln(), &[("9", 1.0f64.ln()), ("8", 0.3f64.ln())])];
        let flat = vec![token("9", 1.0f64.ln(), &[("9", 0.2f64.ln()), ("8", 0.55f64.ln())])];

        let s1 = score_from_logprobs(&sharp, 9).unwrap();
        let s2 = score_from_logprobs(&flat, 9).unwrap();
        assert!(s1.score > s2.score, "sharp distribution must score higher");
        assert!(s1.confidence > s2.confidence);
    }

    #[test]
    fn logprob_expectation_value_is_correct() {
        // p(8)=0.5, p(6)=0.5 → expected 7.0
        let entry = token("8", 0.5f64.ln(), &[("8", 0.5f64.ln()), ("6", 0.5f64.ln())]);
        let score = score_from_logprobs(&[entry], 9).unwrap();
        approx(score.score, 7.0);
        approx(score.confidence, 0.5);
        assert_eq!(score.source, ScoreSource::Logprob);
    }

    #[test]
    fn logprob_skips_non_score_tokens() {
        let entries = vec![
            token("The", 0.0f64.ln(), &[]),
            token("7", 0.9f64.ln(), &[("7", 0.9f64.ln())]),
        ];
        let score = score_from_logprobs(&entries, 9).unwrap();
        approx(score.score, 7.0);
        assert!((score.confidence - 1.0).abs() < 1e-9);
    }

    #[test]
    fn logprob_no_score_token_is_none() {
        let entries = vec![token("nope", 0.0f64.ln(), &[])];
        assert!(score_from_logprobs(&entries, 9).is_none());
    }

    #[test]
    fn score_token_must_be_within_range() {
        let entries = vec![token("42", 0.0f64.ln(), &[])];
        assert!(score_from_logprobs(&entries, 9).is_none());
    }

    /// Deterministic verifier used to test `best_of_n` without a model.
    struct FixedVerifier(Vec<f64>);

    #[async_trait]
    impl Verifier for FixedVerifier {
        async fn verify(&self, _task: &str, candidate: &str) -> Result<VerificationScore, EvalError> {
            let idx: usize = candidate.parse().unwrap();
            Ok(VerificationScore {
                score: self.0[idx],
                max_score: 9.0,
                confidence: 1.0,
                source: ScoreSource::Judge,
            })
        }
    }

    #[tokio::test]
    async fn best_of_n_picks_highest_and_keeps_input_order() {
        let verifier = FixedVerifier(vec![3.0, 8.0, 5.0]);
        let candidates = vec!["0".to_string(), "1".to_string(), "2".to_string()];
        let result = best_of_n(&verifier, "task", &candidates).await.unwrap();
        assert_eq!(result.best_index, 1);
        assert_eq!(result.best, "1");
        assert_eq!(result.scores.len(), 3);
        assert_eq!(result.scores[2].score, 5.0);
    }

    #[tokio::test]
    async fn best_of_n_ties_keep_earliest() {
        let verifier = FixedVerifier(vec![8.0, 8.0]);
        let candidates = vec!["0".to_string(), "1".to_string()];
        let result = best_of_n(&verifier, "task", &candidates).await.unwrap();
        assert_eq!(result.best_index, 0);
    }

    #[tokio::test]
    async fn best_of_n_rejects_empty() {
        let verifier = FixedVerifier(vec![]);
        let err = best_of_n(&verifier, "task", &[]).await.unwrap_err();
        assert!(matches!(err, EvalError::PredictorError(_)));
    }
}
