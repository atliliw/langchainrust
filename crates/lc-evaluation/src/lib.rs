//! Evaluation 模块 - LLM 应用评测
//!
//! 提供 `Evaluator` trait、内置评测器、数据集加载与批量运行器,
//! 用于量化 prompt/模型改动的效果。
//!
//! 核心类型:
//! - `EvalError` / `Score` / `Example` / `Dataset` / `Evaluator` / `Predictor`
//! - `EvalRunner` 与报告 `Report`
//! - 内置评测器: `ExactMatch` / `StringDistance` / `EmbeddingSimilarity` / `LLMAsJudge`
//! - 其它评测器: `Bleu` / `Faithfulness` / `PairwiseJudge` / `ContainsKeyword` / `RegexMatch`
//!
//! # 示例
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

pub use bleu::Bleu;
pub use criteria::{Dataset, EvalError, Evaluator, Example, Predictor, Score};
pub use faithfulness::Faithfulness;
pub use pairwise::{PairwiseJudge, Verdict};
pub use results::{EmbeddingSimilarity, ExactMatch, LLMAsJudge, StringDistance};
pub use rules::{ContainsKeyword, LengthCheck, RegexMatch};
pub use runner::{EvalRunner, Report};

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod real_llm_tests;
