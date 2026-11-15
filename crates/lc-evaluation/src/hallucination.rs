//! Hallucination-span evaluation from the logprobs of a generated answer.
//!
//! [`LogprobVerifier`](crate::LogprobVerifier) scores an answer with a second
//! prompt. When the answer itself was produced through
//! [`LogprobChatModel::chat_with_logprobs`](lc_core::language_models::LogprobChatModel),
//! its [`TokenLogprob`]s already say where the model was unsure.
//! [`HallucinationSpanEvaluator`] turns them into flagged spans and a
//! pass/fail [`Score`] with no extra model call.
//!
//! Detection is done by [`llm_token_visualizer::detect()`]: a word is flagged
//! when one of its tokens has probability below the threshold, and adjacent
//! flagged words are merged into one span that carries the weakest token and
//! the alternatives the model was weighing there.
//!
//! Low probability is a signal, not proof: a model can be confidently wrong,
//! and it can be unsure about phrasing while the facts are right.
//!
//! Requires the `hallucination-spans` feature.

use lc_core::language_models::{LogprobResult, TokenLogprob};
use llm_token_visualizer::{Alternative, LogprobToken};

pub use llm_token_visualizer::{Report as SpanReport, Span as FlaggedSpan, DEFAULT_THRESHOLD};

use crate::{EvalError, Score};

/// Flags low-confidence spans in a generated answer from its token logprobs.
///
/// The answer passes (score `1.0`) when at most `max_spans` spans are flagged
/// (default `0`) and fails (score `0.0`) otherwise; the label names the
/// flagged text.
#[derive(Debug, Clone)]
pub struct HallucinationSpanEvaluator {
    threshold: f64,
    max_spans: usize,
}

impl Default for HallucinationSpanEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl HallucinationSpanEvaluator {
    /// Creates an evaluator with the default probability threshold (`0.5`)
    /// that fails on any flagged span.
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            max_spans: 0,
        }
    }

    /// Sets the probability below which a token counts as low confidence.
    ///
    /// Clamped to `0..=1`; a `NaN` threshold keeps the current value.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        if !threshold.is_nan() {
            self.threshold = threshold.clamp(0.0, 1.0);
        }
        self
    }

    /// Sets how many flagged spans are tolerated before the answer fails.
    pub fn with_max_spans(mut self, max_spans: usize) -> Self {
        self.max_spans = max_spans;
        self
    }

    /// Returns the flagged spans and summary metrics for `token_logprobs`.
    pub fn detect(&self, token_logprobs: &[TokenLogprob]) -> SpanReport {
        llm_token_visualizer::detect(&to_logprob_tokens(token_logprobs), self.threshold)
    }

    /// Scores a [`LogprobResult`]: `1.0` when it passes, `0.0` when too many
    /// spans are flagged.
    ///
    /// Errors when the result carries no token logprobs (the provider ignored
    /// the request), rather than passing an answer that was never inspected.
    pub fn evaluate(&self, result: &LogprobResult) -> Result<Score, EvalError> {
        self.evaluate_tokens(&result.token_logprobs)
    }

    /// Like [`evaluate`](Self::evaluate), for a bare token list.
    pub fn evaluate_tokens(&self, token_logprobs: &[TokenLogprob]) -> Result<Score, EvalError> {
        if token_logprobs.is_empty() {
            return Err(EvalError::ParseError(
                "no token logprobs to inspect; request them with chat_with_logprobs".to_string(),
            ));
        }
        let report = self.detect(token_logprobs);
        if report.spans.len() <= self.max_spans {
            return Ok(Score::new(1.0).with_label("pass"));
        }
        let flagged: Vec<String> = report
            .spans
            .iter()
            .map(|span| format!("{:?}", span.text.trim()))
            .collect();
        Ok(Score::new(0.0).with_label(format!(
            "{} flagged span(s): {}",
            report.spans.len(),
            flagged.join(", ")
        )))
    }
}

fn to_logprob_tokens(token_logprobs: &[TokenLogprob]) -> Vec<LogprobToken> {
    token_logprobs
        .iter()
        .map(|entry| LogprobToken {
            token: entry.token.clone(),
            logprob: entry.logprob,
            top_logprobs: entry
                .top_logprobs
                .iter()
                .map(|alt| Alternative {
                    token: alt.token.clone(),
                    logprob: alt.logprob,
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenAlternative;

    fn token(token: &str, prob: f64, alts: &[(&str, f64)]) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob: prob.ln(),
            top_logprobs: alts
                .iter()
                .map(|(t, p)| TokenAlternative {
                    token: t.to_string(),
                    logprob: p.ln(),
                })
                .collect(),
        }
    }

    fn answer() -> Vec<TokenLogprob> {
        vec![
            token("The", 0.99, &[]),
            token(" capital", 0.98, &[]),
            token(" is", 0.97, &[]),
            token(" Sydney", 0.3, &[(" Sydney", 0.3), (" Canberra", 0.6)]),
            token(".", 0.99, &[]),
        ]
    }

    #[test]
    fn flags_low_probability_span() {
        let report = HallucinationSpanEvaluator::new().detect(&answer());
        assert_eq!(report.token_count, 5);
        assert_eq!(report.spans.len(), 1);
        let span = &report.spans[0];
        assert_eq!(span.text.trim(), "Sydney");
        assert_eq!(span.weakest_token, " Sydney");
        assert_eq!(span.alternatives[1].token, " Canberra");
    }

    #[test]
    fn confident_answer_passes() {
        let tokens = vec![token("Canberra", 0.95, &[]), token(".", 0.99, &[])];
        let score = HallucinationSpanEvaluator::new()
            .evaluate_tokens(&tokens)
            .unwrap();
        assert_eq!(score.value, 1.0);
        assert_eq!(score.label.as_deref(), Some("pass"));
    }

    #[test]
    fn flagged_answer_fails_with_label() {
        let result = LogprobResult {
            content: "The capital is Sydney.".to_string(),
            token_logprobs: answer(),
            ..Default::default()
        };
        let score = HallucinationSpanEvaluator::new().evaluate(&result).unwrap();
        assert_eq!(score.value, 0.0);
        assert_eq!(
            score.label.as_deref(),
            Some("1 flagged span(s): \"Sydney\"")
        );
    }

    #[test]
    fn threshold_and_max_spans_are_respected() {
        let lenient = HallucinationSpanEvaluator::new().with_threshold(0.2);
        assert_eq!(lenient.evaluate_tokens(&answer()).unwrap().value, 1.0);

        let tolerant = HallucinationSpanEvaluator::new().with_max_spans(1);
        assert_eq!(tolerant.evaluate_tokens(&answer()).unwrap().value, 1.0);

        let nan = HallucinationSpanEvaluator::new().with_threshold(f64::NAN);
        assert_eq!(nan.evaluate_tokens(&answer()).unwrap().value, 0.0);
    }

    #[test]
    fn missing_logprobs_is_an_error() {
        let err = HallucinationSpanEvaluator::new()
            .evaluate_tokens(&[])
            .unwrap_err();
        assert!(matches!(err, EvalError::ParseError(_)));
    }
}
