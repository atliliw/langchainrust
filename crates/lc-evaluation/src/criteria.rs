//! Core evaluation types and traits: EvalError, Score, Example, Dataset,
//! plus the Evaluator / Predictor traits.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Evaluation error
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EvalError {
    /// Underlying IO error (e.g. file read failure).
    #[error("IO error: {0}")]
    IoError(String),
    /// Data parse error (e.g. JSON/JSONL parse failure).
    #[error("parse error: {0}")]
    ParseError(String),
    /// Embedding computation error.
    #[error("embedding error: {0}")]
    EmbeddingError(String),
    /// Predictor execution error.
    #[error("prediction error: {0}")]
    PredictorError(String),
    /// Prediction/reference sample counts differ in corpus-level evaluation (one-to-one).
    #[error(
        "length mismatch: {predictions} predictions vs {references} references; \
         sample counts must match"
    )]
    LengthMismatch {
        /// Prediction sample count
        predictions: usize,
        /// Reference sample count
        references: usize,
    },
}

/// P2-6: errors from the shared judge core (lc-core::judge) map into the evaluation error domain,
/// so `structured_call(...).await?` works directly in a `Result<_, EvalError>` context.
impl From<lc_core::judge::StructuredJudgeError> for EvalError {
    fn from(e: lc_core::judge::StructuredJudgeError) -> Self {
        match e {
            lc_core::judge::StructuredJudgeError::Call(s) => EvalError::PredictorError(s),
            lc_core::judge::StructuredJudgeError::Parse(s) => EvalError::ParseError(s),
            // `StructuredJudgeError` is `#[non_exhaustive]`; forward any future
            // variants to the generic predictor-error slot.
            _ => EvalError::PredictorError(e.to_string()),
        }
    }
}

/// Evaluation score (0.0–1.0, 1.0 is best)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    /// Score value (0.0–1.0)
    pub value: f64,
    /// Optional score label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Score {
    /// Constructs a score between 0.0 and 1.0.
    ///
    /// P2-8: Rust's `f64::clamp(0.0, 1.0)` returns NaN for NaN input, polluting
    /// the summary mean/std. A NaN pre-check is done here, treating it as 0.0
    /// (negative/positive infinity are left to `clamp` to converge to the bounds).
    pub fn new(value: f64) -> Self {
        let value = if value.is_nan() {
            log::warn!("Score::new received NaN, treating as 0.0");
            0.0
        } else {
            value
        };
        Self {
            value: value.clamp(0.0, 1.0),
            label: None,
        }
    }

    /// Attaches a score label (builder style).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Evaluation example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Evaluation input
    pub input: String,
    /// Reference answer
    pub reference: String,
}

impl Example {
    /// Constructs an evaluation example.
    pub fn new(input: impl Into<String>, reference: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            reference: reference.into(),
        }
    }
}

/// Dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Evaluation examples in the dataset
    pub examples: Vec<Example>,
}

impl Dataset {
    /// Constructs a dataset.
    pub fn new(examples: Vec<Example>) -> Self {
        Self { examples }
    }

    /// Loads from a JSONL file (one `{input, reference}` per line).
    ///
    /// P2-2: async I/O (`tokio::fs`), avoiding synchronous blocking in the async evaluation pipeline.
    pub async fn from_jsonl(path: &str) -> Result<Self, EvalError> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| EvalError::IoError(e.to_string()))?;
        let mut examples = Vec::new();
        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ex: Example = serde_json::from_str(line)
                .map_err(|e| EvalError::ParseError(format!("line {}: {}", i + 1, e)))?;
            examples.push(ex);
        }
        Ok(Self { examples })
    }

    /// Returns the number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// Evaluator trait
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// Scores a single prediction
    async fn eval(
        &self,
        input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError>;

    /// Evaluator name (used in report summaries)
    fn name(&self) -> &str;
}

/// Pairwise-comparison evaluator trait (arena mode): judges which of two answers (A/B) for the same input is better.
///
/// P1-1: a first-class citizen alongside the pointwise `Evaluator`; `EvalRunner` accepts both,
/// so arena evaluation also enters the unified report. Scoring: 1.0 = A wins, 0.5 = tie, 0.0 = B wins.
#[async_trait]
pub trait PairwiseEvaluator: Send + Sync {
    /// Compares answers A and B, returning a 0-1 score
    /// (1.0 = A wins, 0.5 = tie, 0.0 = B wins).
    async fn eval_pair(&self, input: &str, a: &str, b: &str) -> Result<Score, EvalError>;

    /// Evaluator name (used in report summaries)
    fn name(&self) -> &str;
}

/// Predictor trait (the object under evaluation: LLMChain / Agent, etc.)
#[async_trait]
pub trait Predictor: Send + Sync {
    /// Predicts on a single input, returning the text result.
    async fn predict(&self, input: &str) -> Result<String, EvalError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_new_normal() {
        assert!((Score::new(0.5).value - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_score_new_clamps_overflow() {
        assert_eq!(Score::new(2.0).value, 1.0);
        assert_eq!(Score::new(-1.0).value, 0.0);
        assert_eq!(Score::new(f64::INFINITY).value, 1.0);
        assert_eq!(Score::new(f64::NEG_INFINITY).value, 0.0);
    }

    /// P2-8: NaN no longer passes through `.clamp(0.0, 1.0)` to pollute the summary statistics.
    #[test]
    fn test_score_new_nan_guarded() {
        assert_eq!(Score::new(f64::NAN).value, 0.0);
        // ensure NaN is cleaned up rather than lingering in the statistics
        assert!(Score::new(f64::NAN).value.is_finite());
    }

    /// P2-2: from_jsonl reads the file asynchronously; a per-line parse failure carries the line number.
    #[tokio::test]
    async fn test_from_jsonl_async() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        std::fs::write(
            &path,
            "{\"input\":\"q1\",\"reference\":\"a1\"}\n\n{\"input\":\"q2\",\"reference\":\"a2\"}\n",
        )
        .unwrap();
        let dataset = Dataset::from_jsonl(path.to_str().unwrap()).await.unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.examples[1].input, "q2");
        assert_eq!(dataset.examples[1].reference, "a2");
    }

    #[tokio::test]
    async fn test_from_jsonl_missing_file() {
        let err = Dataset::from_jsonl("不存在-的文件.jsonl")
            .await
            .unwrap_err();
        assert!(matches!(err, EvalError::IoError(_)));
    }

    #[tokio::test]
    async fn test_from_jsonl_bad_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        std::fs::write(&path, "{\"input\":\"q\"}\n").unwrap();
        let err = Dataset::from_jsonl(path.to_str().unwrap())
            .await
            .unwrap_err();
        assert!(matches!(err, EvalError::ParseError(_)));
    }
}
