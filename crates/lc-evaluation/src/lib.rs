#![warn(missing_docs)]
//! Evaluation module — LLM application evaluation
//!
//! Provides the `Evaluator` / `PairwiseEvaluator` traits, built-in evaluators, dataset loading,
//! and a batch runner, for quantifying the effect of prompt/model changes.
//!
//! Core types:
//! - `EvalError` / `Score` / `Example` / `Dataset` / `Evaluator` / `Predictor`
//! - `PairwiseEvaluator` (pairwise comparison, a first-class citizen alongside pointwise, P1-1)
//! - `EvalRunner` and the `Report` (with original text + stddev + failure list)
//! - built-in evaluators: `ExactMatch` / `StringDistance` / `EmbeddingSimilarity` / `LLMAsJudge`
//! - other evaluators: `Bleu` / `Faithfulness` / `PairwiseJudge` / `ContainsKeyword` / `RegexMatch`
//!
//! # Example
//! ```ignore
//! use lc_evaluation::{EvalRunner, ExactMatch, StringDistance, Dataset, Example};
//! let dataset = Dataset::new(vec![Example::new("2+2?", "4")]);
//! let runner = EvalRunner::new(vec![Box::new(ExactMatch), Box::new(StringDistance)]);
//! // let report = runner.run(&dataset, &predictor).await?;
//! ```

mod bleu;
mod criteria;
mod faithfulness;
mod pairwise;
mod results;
mod rules;
mod runner;

#[cfg(test)]
mod test_support;

pub use bleu::Bleu;
pub use criteria::{Dataset, EvalError, Evaluator, Example, PairwiseEvaluator, Predictor, Score};
pub use faithfulness::Faithfulness;
pub use pairwise::{PairwiseJudge, Verdict};
pub use results::{EmbeddingSimilarity, ExactMatch, LLMAsJudge, StringDistance};
pub use rules::{ContainsKeyword, LengthCheck, RegexMatch};
pub use runner::{EvalRunner, Report};

#[cfg(test)]
mod integration_tests;
