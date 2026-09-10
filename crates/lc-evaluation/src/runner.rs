//! Batch runner: `Report` and `EvalRunner`.
//!
//! `EvalRunner` calls the `Predictor` per example in the dataset, then scores with the pointwise
//! `Evaluator`s and pairwise `PairwiseEvaluator`s, aggregating into a `Report`.
//!
//! P1-3: per-item tolerance — a failed predict or a failed evaluator score is recorded in
//! `Report::failures`, computed results are kept, and the run does not abort. P1-4: `Report`
//! carries the original text + stddev and implements `Serialize`/`Deserialize` for post-hoc analysis.

use std::collections::{HashMap, HashSet};

use super::criteria::{Dataset, EvalError, Evaluator, PairwiseEvaluator, Predictor, Score};

/// Complete evaluation record for one example (includes the original text, for tracing low scores).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExampleReport {
    /// Example index in the dataset (0-based)
    pub index: usize,
    pub input: String,
    pub reference: String,
    pub prediction: String,
    /// Scores each evaluator assigned to this example (failed or not-run evaluators are absent)
    pub scores: HashMap<String, Score>,
}

/// Summary statistics for one evaluator (mean + population stddev + sample count).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoreSummary {
    pub mean: f64,
    pub std: f64,
    pub count: usize,
}

/// Failure record: a predict or an evaluator score failed for the example at a given index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailureRecord {
    /// Example index in the dataset (0-based)
    pub index: usize,
    /// Failure stage: `"predict"` or an evaluator's `name()`
    pub stage: String,
    pub error: String,
}

/// Evaluation report (with original text, stddev, failure list; deserializable).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Report {
    /// Per-example complete records (including input/reference/prediction originals)
    pub per_example: Vec<ExampleReport>,
    /// Per-evaluator summaries (mean + stddev + sample count)
    pub summary: HashMap<String, ScoreSummary>,
    /// Failure records collected by per-item tolerance (empty = all succeeded)
    pub failures: Vec<FailureRecord>,
    /// Cost ledger for the run (E1). Old reports without this field deserialize to the default.
    #[serde(default)]
    pub cost: crate::OverallCost,
}

/// Batch runner: holds both pointwise and pairwise evaluators.
pub struct EvalRunner {
    evaluators: Vec<Box<dyn Evaluator>>,
    pairwise: Vec<Box<dyn PairwiseEvaluator>>,
    /// USD price book used to turn reported token usage into cost (E1).
    price_book: crate::PriceBook,
}

impl EvalRunner {
    /// Creates a batch runner (pointwise evaluators only).
    pub fn new(evaluators: Vec<Box<dyn Evaluator>>) -> Self {
        Self {
            evaluators,
            pairwise: Vec::new(),
            price_book: crate::PriceBook::default_set(),
        }
    }

    /// Appends pairwise evaluators (P1-1, arena evaluation enters the unified report).
    pub fn with_pairwise(mut self, pairwise: Vec<Box<dyn PairwiseEvaluator>>) -> Self {
        self.pairwise.extend(pairwise);
        self
    }

    /// Overrides the price book used for cost estimation (E1). Takes over the default set.
    pub fn with_price_book(mut self, price_book: crate::PriceBook) -> Self {
        self.price_book = price_book;
        self
    }

    /// Runs all evaluators on the dataset, returning the report.
    ///
    /// P1-3: per-item tolerance — a failed predict records a `"predict"` failure and skips the example;
    /// a failed evaluator score records only that evaluator's failure, others still score.
    /// P1-1: pairwise evaluators participate too, using `(prediction, reference)` as the A/B candidates
    /// (arena usage: put the answer under comparison in the reference slot).
    pub async fn run(
        &self,
        dataset: &Dataset,
        predictor: &dyn Predictor,
    ) -> Result<Report, EvalError> {
        Self::warn_duplicate_names(&self.evaluators, &self.pairwise);

        let mut per_example = Vec::with_capacity(dataset.len());
        let mut failures = Vec::new();
        // accumulate each evaluator's successful sample scores per name, for mean/std computation
        let mut per_name: HashMap<String, Vec<f64>> = HashMap::new();
        // E1: accumulate reported predictor token usage into the report's cost ledger
        let mut cost = crate::OverallCost::default();

        for (i, ex) in dataset.examples.iter().enumerate() {
            let prediction = match predictor.predict(&ex.input).await {
                Ok(p) => p,
                Err(e) => {
                    failures.push(FailureRecord {
                        index: i,
                        stage: "predict".into(),
                        error: e.to_string(),
                    });
                    continue;
                }
            };

            // E1: meter whatever usage the predictor reports (None = no metering, ledger stays zero)
            if let Some(usage) = predictor.report_token_usage().await {
                cost.accumulate(&usage, &self.price_book);
            }

            let mut scores = HashMap::new();
            for ev in &self.evaluators {
                match ev.eval(&ex.input, &prediction, &ex.reference).await {
                    Ok(s) => {
                        per_name
                            .entry(ev.name().to_string())
                            .or_default()
                            .push(s.value);
                        scores.insert(ev.name().to_string(), s);
                    }
                    Err(e) => failures.push(FailureRecord {
                        index: i,
                        stage: ev.name().to_string(),
                        error: e.to_string(),
                    }),
                }
            }
            for ev in &self.pairwise {
                match ev.eval_pair(&ex.input, &prediction, &ex.reference).await {
                    Ok(s) => {
                        per_name
                            .entry(ev.name().to_string())
                            .or_default()
                            .push(s.value);
                        scores.insert(ev.name().to_string(), s);
                    }
                    Err(e) => failures.push(FailureRecord {
                        index: i,
                        stage: ev.name().to_string(),
                        error: e.to_string(),
                    }),
                }
            }

            per_example.push(ExampleReport {
                index: i,
                input: ex.input.clone(),
                reference: ex.reference.clone(),
                prediction,
                scores,
            });
        }

        let mut summary = HashMap::new();
        for (name, values) in per_name {
            let count = values.len();
            let mean = values.iter().sum::<f64>() / count as f64;
            // population stddev: spread/variance reflects evaluator stability better than the mean alone
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
            summary.insert(
                name,
                ScoreSummary {
                    mean,
                    std: variance.sqrt(),
                    count,
                },
            );
        }

        Ok(Report {
            per_example,
            summary,
            failures,
            cost,
        })
    }

    /// P1-4: duplicate-named evaluators silently overwrite each other in the summary/report; at least `log::warn`.
    fn warn_duplicate_names(
        evaluators: &[Box<dyn Evaluator>],
        pairwise: &[Box<dyn PairwiseEvaluator>],
    ) {
        let mut seen = HashSet::new();
        for ev in evaluators {
            if !seen.insert(ev.name()) {
                log::warn!(
                    "EvalRunner: duplicate evaluator name '{}', report data will be overwritten",
                    ev.name()
                );
            }
        }
        for ev in pairwise {
            if !seen.insert(ev.name()) {
                log::warn!(
                    "EvalRunner: duplicate evaluator name '{}', report data will be overwritten",
                    ev.name()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Dummy evaluator that always scores 1.0.
    struct ConstantEvaluator;

    #[async_trait]
    impl Evaluator for ConstantEvaluator {
        async fn eval(
            &self,
            _input: &str,
            _prediction: &str,
            _reference: &str,
        ) -> Result<Score, EvalError> {
            Ok(Score::new(1.0))
        }
        fn name(&self) -> &str {
            "constant"
        }
    }

    /// Predictor that hands back its input padding a "!" and reports a fixed token usage.
    struct UsagePredictor;

    #[async_trait]
    impl Predictor for UsagePredictor {
        async fn predict(&self, input: &str) -> Result<String, EvalError> {
            Ok(format!("{input}!"))
        }
        async fn report_token_usage(&self) -> Option<crate::TokenUsage> {
            Some(crate::TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                model: Some("gpt-4o-mini".into()),
            })
        }
    }

    /// Predictor that performs no token metering (the common case).
    struct UnmeteredPredictor;

    #[async_trait]
    impl Predictor for UnmeteredPredictor {
        async fn predict(&self, input: &str) -> Result<String, EvalError> {
            Ok(input.to_string())
        }
        // report_token_usage defaults to None
    }

    async fn dataset2() -> Dataset {
        Dataset::new(vec![
            crate::Example::new("q1", "a1"),
            crate::Example::new("q2", "a2"),
        ])
    }

    #[tokio::test]
    async fn old_report_without_cost_field_still_deserializes() {
        // E1 compat: a report serialized before the `cost` field existed has no `cost` key.
        let old_json = r#"{
            "per_example": [],
            "summary": {},
            "failures": []
        }"#;
        let report: Report = serde_json::from_str(old_json).unwrap();
        // default ledger: all-zero tokens, no USD
        assert_eq!(report.cost.total_tokens, 0);
        assert!(report.cost.cost_usd.is_none());
    }

    #[tokio::test]
    async fn report_round_trips_with_cost_field() {
        let runner = EvalRunner::new(vec![Box::new(ConstantEvaluator)])
            .with_price_book(crate::PriceBook::default_set());
        let report = runner
            .run(&dataset2().await, &UsagePredictor)
            .await
            .unwrap();

        let json = serde_json::to_string(&report).unwrap();
        let back: Report = serde_json::from_str(&json).unwrap();
        assert_eq!(back.cost.total_tokens, report.cost.total_tokens);
        assert_eq!(back.cost.cost_usd, report.cost.cost_usd);
    }

    #[tokio::test]
    async fn runner_accumulates_priced_usage_across_examples() {
        let runner = EvalRunner::new(vec![Box::new(ConstantEvaluator)])
            .with_price_book(crate::PriceBook::default_set());
        let report = runner
            .run(&dataset2().await, &UsagePredictor)
            .await
            .unwrap();

        // 2 examples x (100 prompt + 50 completion); gpt-4o-mini at $0.15/$0.60 per 1M
        assert_eq!(report.cost.prompt_tokens, 200);
        assert_eq!(report.cost.completion_tokens, 100);
        assert_eq!(report.cost.total_tokens, 300);
        let expected = 200.0 / 1e6 * 0.15 + 100.0 / 1e6 * 0.60; // = 0.00009
        let usd = report.cost.cost_usd.unwrap();
        assert!(
            (usd - expected).abs() < 1e-12,
            "got {usd}, expected {expected}"
        );
    }

    #[tokio::test]
    async fn unmetered_predictor_leaves_cost_at_zero() {
        let runner = EvalRunner::new(vec![Box::new(ConstantEvaluator)]);
        let report = runner
            .run(&dataset2().await, &UnmeteredPredictor)
            .await
            .unwrap();
        // zero behavior change when no metering is wired up
        assert_eq!(report.cost.total_tokens, 0);
        assert!(report.cost.cost_usd.is_none());
    }
}
