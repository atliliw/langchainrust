//! Offline report export (B9, v0.22.4): JSONL per-example lines and a plain-text table.
//!
//! The JSONL export is one [`crate::ExampleReport`] per line with the report's `run_id` stamped
//! onto every row, so a batch file can be joined back to the trace of the evaluated system
//! without carrying the separate envelope. The table export is a dependency-free human view:
//! per-example score matrix, per-evaluator summary, failures, cost.

use std::path::Path;

use super::{EvalError, Report};

/// Maximum width of the input column in the table view (chars).
const TABLE_INPUT_WIDTH: usize = 40;
/// Width of each score column.
const SCORE_COL_WIDTH: usize = 9;

impl Report {
    /// Renders the report as JSONL: one example record per line, carrying the run id.
    ///
    /// ```json
    /// {"run_id":"...","index":0,"input":"...","reference":"...","prediction":"...",
    ///  "contexts":["..."],"scores":{"exact_match":{"value":1.0}}}
    /// ```
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for ex in &self.per_example {
            let line = serde_json::json!({
                "run_id": self.run_id,
                "index": ex.index,
                "input": ex.input,
                "reference": ex.reference,
                "prediction": ex.prediction,
                "contexts": ex.contexts,
                "scores": ex.scores,
            });
            // compact, single-line JSON; no embedded raw newlines in the record itself
            out.push_str(&line.to_string());
            out.push('\n');
        }
        out
    }

    /// Writes [`Self::to_jsonl`] to `path` (async, truncating/overwriting).
    pub async fn write_jsonl(&self, path: impl AsRef<Path>) -> Result<(), EvalError> {
        tokio::fs::write(path, self.to_jsonl())
            .await
            .map_err(|e| EvalError::IoError(e.to_string()))
    }

    /// Renders a fixed-width, dependency-free plain-text view of the report.
    pub fn to_table(&self) -> String {
        // Deterministic evaluator-column order regardless of HashMap iteration order.
        let mut names: Vec<&str> = self.summary.keys().map(String::as_str).collect();
        names.sort_unstable();

        let mut out = String::new();
        out.push_str(&format!("eval run: {}\n", empty_dash(&self.run_id)));

        // --- per-example score matrix ----------------------------------------------------------
        if !self.per_example.is_empty() {
            let index_width = self
                .per_example
                .last()
                .map(|r| r.index.to_string().len())
                .unwrap_or(1)
                .max(1);
            out.push_str(&format!(
                "{:>width$} | {:<input_width$}",
                "#",
                "input",
                width = index_width,
                input_width = TABLE_INPUT_WIDTH
            ));
            for name in &names {
                out.push_str(&format!("| {:<width$}", name, width = SCORE_COL_WIDTH));
            }
            out.push('\n');

            for row in &self.per_example {
                let input = flatten(&row.input);
                let input: String = input.chars().take(TABLE_INPUT_WIDTH).collect();
                out.push_str(&format!(
                    "{:>width$} | {:<input_width$}",
                    row.index,
                    input,
                    width = index_width,
                    input_width = TABLE_INPUT_WIDTH
                ));
                for name in &names {
                    let cell = row
                        .scores
                        .get(*name)
                        .map(|s| format!("{:.3}", s.value))
                        .unwrap_or_else(|| "-".to_string());
                    out.push_str(&format!("| {:<width$}", cell, width = SCORE_COL_WIDTH));
                }
                out.push('\n');
            }
        }

        // --- summary ---------------------------------------------------------------------------
        out.push_str("\nsummary:\n");
        for name in &names {
            let s = &self.summary[*name];
            out.push_str(&format!(
                "  {name}: mean={:.3} std={:.3} n={}\n",
                s.mean, s.std, s.count
            ));
        }

        // --- failures --------------------------------------------------------------------------
        if !self.failures.is_empty() {
            out.push_str(&format!("\nfailures ({}):\n", self.failures.len()));
            for f in &self.failures {
                out.push_str(&format!(
                    "  [#{} {}] {}\n",
                    f.index,
                    f.stage,
                    one_line(&f.error)
                ));
            }
        }

        // --- cost ------------------------------------------------------------------------------
        if self.cost.total_tokens > 0 || self.cost.cost_usd.is_some() {
            out.push_str(&format!(
                "\ncost: {} tokens ({} prompt / {} completion)",
                self.cost.total_tokens, self.cost.prompt_tokens, self.cost.completion_tokens
            ));
            if let Some(usd) = self.cost.cost_usd {
                out.push_str(&format!(" ${usd:.6}"));
            }
            out.push('\n');
        }

        out
    }
}

fn empty_dash(s: &str) -> &str {
    if s.is_empty() {
        "-"
    } else {
        s
    }
}

fn one_line(s: &str) -> String {
    flatten(s).chars().take(200).collect()
}

fn flatten(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c == '\n' || c == '\r' || c == '\t' {
                ' '
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Dataset, EvalRunner, Example, ExampleReport, FailureRecord, Report, Score, ScoreSummary,
    };
    use async_trait::async_trait;
    use std::collections::HashMap;

    struct EchoPredictor;
    #[async_trait]
    impl crate::Predictor for EchoPredictor {
        async fn predict(&self, input: &str) -> Result<String, EvalError> {
            Ok(input.to_string())
        }
    }

    struct Exactish;
    #[async_trait]
    impl crate::Evaluator for Exactish {
        async fn eval(
            &self,
            _input: &str,
            prediction: &str,
            reference: &str,
        ) -> Result<Score, EvalError> {
            Ok(Score::new(if prediction == reference { 1.0 } else { 0.0 }))
        }
        fn name(&self) -> &str {
            "exactish"
        }
    }

    async fn sample_report() -> Report {
        let runner = EvalRunner::new(vec![Box::new(Exactish)]).with_run_id("run-xyz");
        let ds = Dataset::new(vec![
            Example::with_contexts("a", "a", vec!["ctx-a".into()]),
            Example::new("b", "different"),
        ]);
        runner.run(&ds, &EchoPredictor).await.unwrap()
    }

    #[tokio::test]
    async fn jsonl_carries_run_id_contexts_and_scores_per_line() {
        let report = sample_report().await;
        let jsonl = report.to_jsonl();
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["run_id"], "run-xyz");
            assert!(v["scores"]["exactish"]["value"].is_number());
        }
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["contexts"][0], "ctx-a");
        assert_eq!(first["scores"]["exactish"]["value"], 1.0);
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert!(second["contexts"].as_array().unwrap().is_empty());
        assert_eq!(second["scores"]["exactish"]["value"], 0.0);
    }

    #[tokio::test]
    async fn write_jsonl_round_trips_through_file() {
        let report = sample_report().await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.jsonl");
        report.write_jsonl(&path).await.unwrap();
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(content.lines().count(), 2);
        assert!(content.contains("run-xyz"));
    }

    #[tokio::test]
    async fn table_lists_matrix_summary_and_failures() {
        let mut report = sample_report().await;
        report.failures.push(FailureRecord {
            index: 7,
            stage: "predict".into(),
            error: "boom\nsecond line".into(),
        });
        let table = report.to_table();
        assert!(table.contains("eval run: run-xyz"));
        assert!(table.contains("exactish"));
        assert!(table.contains("mean="));
        assert!(table.starts_with("eval run:"));
        // failure error is flattened to one line
        assert!(table.contains("[#7 predict] boom second line"));
    }

    #[test]
    fn old_report_without_run_id_or_contexts_still_deserializes() {
        let old_json = r#"{
            "per_example": [
                {"index": 0, "input": "q", "reference": "r", "prediction": "p", "scores": {}}
            ],
            "summary": {},
            "failures": []
        }"#;
        let report: Report = serde_json::from_str(old_json).unwrap();
        assert_eq!(report.run_id, "");
        assert!(report.per_example[0].contexts.is_empty());
    }

    #[tokio::test]
    async fn run_id_autogenerates_when_not_pinned() {
        let runner = EvalRunner::new(vec![Box::new(Exactish)]);
        let report = runner
            .run(&Dataset::new(vec![Example::new("a", "a")]), &EchoPredictor)
            .await
            .unwrap();
        // UUID v4 shape: 36 chars with hyphens at the usual positions
        assert_eq!(report.run_id.len(), 36);
        assert_eq!(report.run_id.as_bytes()[14], b'4');
    }

    #[tokio::test]
    async fn begin_run_receives_the_pinned_run_id() {
        use std::sync::Arc;
        use std::sync::Mutex;

        struct CapturingPredictor {
            seen: Arc<Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl crate::Predictor for CapturingPredictor {
            async fn predict(&self, input: &str) -> Result<String, EvalError> {
                Ok(input.to_string())
            }
            async fn begin_run(&self, run_id: &str) {
                self.seen.lock().unwrap().push(run_id.to_string());
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let predictor = CapturingPredictor { seen: seen.clone() };
        let report = EvalRunner::new(vec![Box::new(Exactish)])
            .with_run_id("trace-join-1")
            .run(
                &Dataset::new(vec![Example::new("a", "a"), Example::new("b", "b")]),
                &predictor,
            )
            .await
            .unwrap();
        assert_eq!(report.run_id, "trace-join-1");
        // begin_run fires exactly once for the whole dataset, not once per example
        assert_eq!(seen.lock().unwrap().as_slice(), ["trace-join-1"]);
    }

    #[test]
    fn empty_report_table_renders_without_panicking() {
        let mut scores = HashMap::new();
        scores.insert("x".to_string(), Score::new(1.0));
        let report = Report {
            per_example: vec![ExampleReport {
                index: 0,
                input: "q".into(),
                reference: "r".into(),
                prediction: "p".into(),
                contexts: vec![],
                scores,
            }],
            summary: HashMap::from([(
                "x".to_string(),
                ScoreSummary {
                    mean: 1.0,
                    std: 0.0,
                    count: 1,
                },
            )]),
            failures: vec![],
            cost: Default::default(),
            run_id: String::new(),
        };
        let table = report.to_table();
        assert!(table.contains("eval run: -"));
    }
}
