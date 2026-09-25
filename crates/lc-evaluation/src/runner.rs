//! Batch runner: `Report` and `EvalRunner`.
//!
//! `EvalRunner` calls the `Predictor` per example in the dataset, then scores with the pointwise
//! `Evaluator`s and pairwise `PairwiseEvaluator`s, aggregating into a `Report`.
//!
//! P1-3: per-item tolerance — a failed predict or a failed evaluator score is recorded in
//! `Report::failures`, computed results are kept, and the run does not abort. P1-4: `Report`
//! carries the original text + stddev and implements `Serialize`/`Deserialize` for post-hoc analysis.

use std::collections::{HashMap, HashSet};

use super::criteria::{
    Dataset, EvalError, Evaluator, PairwiseEvaluator, Predictor, RagEvaluator, Score,
};

/// Complete evaluation record for one example (includes the original text, for tracing low scores).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExampleReport {
    /// Example index in the dataset (0-based)
    pub index: usize,
    /// Original model input (question/prompt) of the example
    pub input: String,
    /// Ground-truth reference answer of the example
    pub reference: String,
    /// What the predictor actually produced
    pub prediction: String,
    /// Retrieved contexts this example was scored against (B9; empty for non-RAG datasets).
    /// Old reports without this field deserialize to an empty vec.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// Scores each evaluator assigned to this example (failed or not-run evaluators are absent)
    pub scores: HashMap<String, Score>,
}

/// Summary statistics for one evaluator (mean + population stddev + sample count).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoreSummary {
    /// Arithmetic mean of the evaluator's scores across the dataset
    pub mean: f64,
    /// Population standard deviation across the dataset
    pub std: f64,
    /// Number of examples this evaluator successfully scored
    pub count: usize,
}

/// Failure record: a predict or an evaluator score failed for the example at a given index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailureRecord {
    /// Example index in the dataset (0-based)
    pub index: usize,
    /// Failure stage: `"predict"` or an evaluator's `name()`
    pub stage: String,
    /// Human-readable error message recorded for the failure
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
    /// Stable identifier of this evaluation run (B9): the join key against traces/spans.
    ///
    /// Set explicitly via [`EvalRunner::with_run_id`] or auto-generated as a UUID v4 when the
    /// runner runs. The predictor receives it through [`Predictor::begin_run`] so a system
    /// under test can stamp it (e.g. into `RunnableConfig.metadata["trace_id"]`, which the agent
    /// executor propagates to callback/OTel spans). Old reports deserialize to an empty id.
    #[serde(default)]
    pub run_id: String,
}

/// Batch runner: holds pointwise, pairwise, and RAG evaluators.
pub struct EvalRunner {
    evaluators: Vec<Box<dyn Evaluator>>,
    pairwise: Vec<Box<dyn PairwiseEvaluator>>,
    /// RAGAS-style evaluators taking the example's retrieved contexts (B9).
    rag: Vec<Box<dyn RagEvaluator>>,
    /// USD price book used to turn reported token usage into cost (E1).
    price_book: crate::PriceBook,
    /// Explicit run id; [`None`] means generate a fresh UUID v4 per [`EvalRunner::run`].
    run_id: Option<String>,
}

impl EvalRunner {
    /// Creates a batch runner (pointwise evaluators only).
    pub fn new(evaluators: Vec<Box<dyn Evaluator>>) -> Self {
        Self {
            evaluators,
            pairwise: Vec::new(),
            rag: Vec::new(),
            price_book: crate::PriceBook::default_set(),
            run_id: None,
        }
    }

    /// Pins the run id stamped into the report and handed to [`Predictor::begin_run`].
    ///
    /// Use this to correlate an evaluation run with an external trace/CI record. Without it,
    /// each `run()` generates a fresh UUID v4.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Appends pairwise evaluators (P1-1, arena evaluation enters the unified report).
    pub fn with_pairwise(mut self, pairwise: Vec<Box<dyn PairwiseEvaluator>>) -> Self {
        self.pairwise.extend(pairwise);
        self
    }

    /// Appends RAG evaluators (B9: context precision/recall, answer relevancy), scored with
    /// each example's retrieved contexts in rank order.
    pub fn with_rag_evaluators(mut self, rag: Vec<Box<dyn RagEvaluator>>) -> Self {
        self.rag.extend(rag);
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
        Self::warn_duplicate_names(&self.evaluators, &self.pairwise, &self.rag);

        // B9: resolve the run id once, tell the predictor, and carry it on the report so
        // evaluation output can be joined to traces/spans produced by the system under test.
        let run_id = self
            .run_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        predictor.begin_run(&run_id).await;

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
            // I3: an LLM judge evaluator spends tokens too; drain its usage into the cost ledger
            // after scoring. Uses a local helper to accumulate each evaluator's report.
            async fn drain_evaluator_cost(
                ev_report: Option<crate::TokenUsage>,
                cost: &mut crate::OverallCost,
                book: &crate::PriceBook,
            ) {
                if let Some(usage) = ev_report {
                    cost.accumulate(&usage, book);
                }
            }
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
                drain_evaluator_cost(ev.report_token_usage().await, &mut cost, &self.price_book)
                    .await;
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
                drain_evaluator_cost(ev.report_token_usage().await, &mut cost, &self.price_book)
                    .await;
            }
            for ev in &self.rag {
                match ev
                    .eval_rag(&ex.input, &prediction, &ex.contexts, &ex.reference)
                    .await
                {
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
                drain_evaluator_cost(ev.report_token_usage().await, &mut cost, &self.price_book)
                    .await;
            }

            per_example.push(ExampleReport {
                index: i,
                input: ex.input.clone(),
                reference: ex.reference.clone(),
                prediction,
                contexts: ex.contexts.clone(),
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
            run_id,
        })
    }

    /// P1-4: duplicate-named evaluators silently overwrite each other in the summary/report; at least `log::warn`.
    fn warn_duplicate_names(
        evaluators: &[Box<dyn Evaluator>],
        pairwise: &[Box<dyn PairwiseEvaluator>],
        rag: &[Box<dyn RagEvaluator>],
    ) {
        let mut seen: HashSet<String> = HashSet::new();
        let mut push = |name: &str| {
            if !seen.insert(name.to_string()) {
                log::warn!(
                    "EvalRunner: duplicate evaluator name '{name}', report data will be overwritten"
                );
            }
        };
        for ev in evaluators {
            push(ev.name());
        }
        for ev in pairwise {
            push(ev.name());
        }
        for ev in rag {
            push(ev.name());
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

    /// RAG evaluator that records how many contexts it received and scores that count > 0.
    struct ContextCountingRag;
    #[async_trait]
    impl RagEvaluator for ContextCountingRag {
        async fn eval_rag(
            &self,
            _input: &str,
            _prediction: &str,
            contexts: &[String],
            _reference: &str,
        ) -> Result<Score, EvalError> {
            Ok(Score::new(if contexts.is_empty() { 0.0 } else { 1.0 })
                .with_label(format!("{} contexts", contexts.len())))
        }
        fn name(&self) -> &str {
            "rag_contexts"
        }
    }

    #[tokio::test]
    async fn rag_evaluators_receive_example_contexts_and_enter_report() {
        let dataset = Dataset::new(vec![
            crate::Example::with_contexts("q1", "a1", vec!["c0".into(), "c1".into()]),
            crate::Example::new("q2", "a2"),
        ]);
        let runner =
            EvalRunner::new(vec![]).with_rag_evaluators(vec![Box::new(ContextCountingRag)]);
        let report = runner.run(&dataset, &UnmeteredPredictor).await.unwrap();

        assert_eq!(report.per_example[0].contexts.len(), 2);
        assert_eq!(
            report.per_example[0].scores["rag_contexts"].value, 1.0,
            "first example carries contexts"
        );
        assert_eq!(
            report.per_example[1].scores["rag_contexts"].value, 0.0,
            "second example has none"
        );
        assert_eq!(report.summary["rag_contexts"].count, 2);
        // report carries a generated run id even when none was pinned
        assert!(!report.run_id.is_empty());
    }

    #[tokio::test]
    async fn rag_evaluator_failure_is_isolated_per_item() {
        struct FailingRag;
        #[async_trait]
        impl RagEvaluator for FailingRag {
            async fn eval_rag(
                &self,
                _input: &str,
                _prediction: &str,
                _contexts: &[String],
                _reference: &str,
            ) -> Result<Score, EvalError> {
                Err(EvalError::ParseError("rag judge broke".into()))
            }
            fn name(&self) -> &str {
                "broken_rag"
            }
        }
        let runner = EvalRunner::new(vec![Box::new(ConstantEvaluator)])
            .with_rag_evaluators(vec![Box::new(FailingRag)]);
        let report = runner
            .run(&dataset2().await, &UnmeteredPredictor)
            .await
            .unwrap();
        // pointwise evaluator still scored; the RAG failure is recorded, run not aborted
        assert_eq!(report.per_example.len(), 2);
        assert!(report.per_example[0].scores.contains_key("constant"));
        assert!(!report.per_example[0].scores.contains_key("broken_rag"));
        assert_eq!(report.failures.len(), 2);
        assert!(report.failures.iter().all(|f| f.stage == "broken_rag"));
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

    /// I3: an LLM-judge evaluator that drains its usage ledger credits its tokens into `Report.cost`.
    struct JudgingEvaluator;
    #[async_trait]
    impl Evaluator for JudgingEvaluator {
        async fn eval(
            &self,
            _input: &str,
            _prediction: &str,
            _reference: &str,
        ) -> Result<Score, EvalError> {
            Ok(Score::new(1.0))
        }
        fn name(&self) -> &str {
            "judging"
        }
        async fn report_token_usage(&self) -> Option<crate::TokenUsage> {
            Some(crate::TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 40,
                model: Some("gpt-4o-mini".into()),
            })
        }
    }

    #[tokio::test]
    async fn runner_accumulates_judge_evaluator_usage_into_cost() {
        let runner = EvalRunner::new(vec![Box::new(JudgingEvaluator)])
            .with_price_book(crate::PriceBook::default_set());
        let report = runner
            .run(&dataset2().await, &UnmeteredPredictor)
            .await
            .unwrap();
        // 2 examples x (100 prompt + 40 completion) judge tokens, credited on top of zero predictor cost
        assert_eq!(report.cost.prompt_tokens, 200);
        assert_eq!(report.cost.completion_tokens, 80);
        let expected = 200.0 / 1e6 * 0.15 + 80.0 / 1e6 * 0.60;
        let usd = report.cost.cost_usd.unwrap();
        assert!(
            (usd - expected).abs() < 1e-12,
            "got {usd}, expected {expected}"
        );
    }
}
