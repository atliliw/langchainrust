//! N2 (v0.25.0): baseline-vs-candidate report comparison — the verdict half of the
//! eval regression loop.
//!
//! A prompt/model change is scored by running the same golden [`Dataset`](crate::Dataset)
//! twice (the pinned baseline run and the candidate run) and diffing the two
//! [`Report`]s. [`compare_reports`] joins their per-evaluator summaries, reports the
//! mean delta of every evaluator present on both sides, and flags a
//! [`Regression`] whenever the candidate mean falls below the baseline by more
//! than a tolerance. Evaluators present on only one side are listed as
//! `added` / `dropped` rather than silently ignored.
//!
//! The comparison is pure (no I/O, no network): persist both reports as JSON
//! (see [`Report::to_jsonl`](crate::Report::to_jsonl)) and compare in CI even
//! when the two runs happened on different machines.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::runner::Report;

/// Movement of one evaluator's mean score between a baseline run and a candidate run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricDelta {
    /// Baseline mean score.
    pub baseline_mean: f64,
    /// Candidate mean score.
    pub candidate_mean: f64,
    /// `candidate_mean - baseline_mean` (negative = worse).
    pub delta: f64,
    /// Number of examples the evaluator scored in the baseline run.
    pub baseline_count: usize,
    /// Number of examples the evaluator scored in the candidate run.
    pub candidate_count: usize,
}

/// One evaluator whose candidate mean regressed beyond the configured tolerance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Regression {
    /// Evaluator name (`Evaluator::name`).
    pub evaluator: String,
    /// Baseline mean score.
    pub baseline_mean: f64,
    /// Candidate mean score.
    pub candidate_mean: f64,
    /// `candidate_mean - baseline_mean` (negative).
    pub delta: f64,
}

/// Result of diffing two eval runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportComparison {
    /// Run id of the baseline report (empty if the baseline report carried none).
    pub baseline_run_id: String,
    /// Run id of the candidate report (empty if the candidate report carried none).
    pub candidate_run_id: String,
    /// Per-evaluator deltas for evaluators present in **both** reports.
    pub metrics: HashMap<String, MetricDelta>,
    /// Evaluators whose candidate mean dropped by more than `tolerance` (name-sorted).
    pub regressions: Vec<Regression>,
    /// Evaluator names present only in the candidate report (name-sorted).
    pub added: Vec<String>,
    /// Evaluator names present only in the baseline report (name-sorted); not scored.
    pub dropped: Vec<String>,
    /// Shared evaluators whose sample count differs between the two runs (name-sorted).
    /// A count mismatch usually means the run diverged (examples failed to score / were added),
    /// so CI gates on it via [`is_gate_failing`](Self::is_gate_failing).
    #[serde(default)]
    pub count_mismatches: Vec<String>,
    /// Allowed negative movement: a drop strictly larger than this is a regression.
    pub tolerance: f64,
}

impl ReportComparison {
    /// Whether at least one evaluator regressed beyond the tolerance.
    pub fn is_regressed(&self) -> bool {
        !self.regressions.is_empty()
    }

    /// B6: whether the CI gate should fail. Fails when any of:
    /// - at least one evaluator regressed beyond the tolerance ([`is_regressed`](Self::is_regressed)),
    /// - any shared evaluator's sample count differs between baseline and candidate,
    /// - there are **zero** shared evaluators to compare at all (an empty/diverged run is not
    ///   a valid pass).
    pub fn is_gate_failing(&self) -> bool {
        self.is_regressed() || !self.count_mismatches.is_empty() || self.metrics.is_empty()
    }

    /// Renders a dependency-free, name-sorted plain-text view for CI logs.
    pub fn to_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "eval regression: baseline {} -> candidate {}\n",
            dash_if_empty(&self.baseline_run_id),
            dash_if_empty(&self.candidate_run_id)
        ));
        out.push_str(&format!("tolerance: {:.4}\n", self.tolerance));

        let mut names: Vec<&str> = self.metrics.keys().map(String::as_str).collect();
        names.sort_unstable();
        out.push_str(&format!(
            "{:<24} {:>9} {:>9} {:>9}\n",
            "evaluator", "baseline", "candidate", "delta"
        ));
        for name in &names {
            let m = &self.metrics[*name];
            let marker = if m.delta < -self.tolerance {
                "  REGRESSION"
            } else {
                ""
            };
            out.push_str(&format!(
                "{:<24} {:>9.4} {:>9.4} {:>+9.4}{}\n",
                name, m.baseline_mean, m.candidate_mean, m.delta, marker
            ));
        }
        if !self.added.is_empty() {
            out.push_str(&format!("added:     {}\n", self.added.join(", ")));
        }
        if !self.dropped.is_empty() {
            out.push_str(&format!("dropped:   {}\n", self.dropped.join(", ")));
        }
        if !self.count_mismatches.is_empty() {
            out.push_str(&format!(
                "count mismatches: {}\n",
                self.count_mismatches.join(", ")
            ));
        }
        out.push_str(&format!(
            "regressions: {}\n",
            if self.is_regressed() {
                self.regressions.len().to_string()
            } else {
                "none".to_string()
            }
        ));
        out
    }
}

fn dash_if_empty(s: &str) -> &str {
    if s.is_empty() {
        "-"
    } else {
        s
    }
}

/// Diffs two reports. A metric is a [`Regression`] when `candidate_mean < baseline_mean - tolerance`.
///
/// `tolerance` is clamped to `>= 0`: it absorbs run-to-run judge noise (LLM-as-judge
/// scores are not bit-stable), so only meaningful movement trips the gate.
pub fn compare_reports(baseline: &Report, candidate: &Report, tolerance: f64) -> ReportComparison {
    let tolerance = tolerance.max(0.0);

    let mut metrics: HashMap<String, MetricDelta> = HashMap::new();
    let mut regressions: Vec<Regression> = Vec::new();
    let mut added: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();

    for (name, cand) in &candidate.summary {
        match baseline.summary.get(name) {
            Some(base) => {
                let delta = cand.mean - base.mean;
                metrics.insert(
                    name.clone(),
                    MetricDelta {
                        baseline_mean: base.mean,
                        candidate_mean: cand.mean,
                        delta,
                        baseline_count: base.count,
                        candidate_count: cand.count,
                    },
                );
                // Strictly worse than the tolerated band: an equal drop exactly on the
                // boundary passes (the tolerance is inclusive, like an epsilon comparison).
                if delta < -tolerance {
                    regressions.push(Regression {
                        evaluator: name.clone(),
                        baseline_mean: base.mean,
                        candidate_mean: cand.mean,
                        delta,
                    });
                }
            }
            None => added.push(name.clone()),
        }
    }
    for name in baseline.summary.keys() {
        if !candidate.summary.contains_key(name) {
            dropped.push(name.clone());
        }
    }

    // Deterministic output regardless of HashMap iteration order.
    regressions.sort_by(|a, b| a.evaluator.cmp(&b.evaluator));
    added.sort_unstable();
    dropped.sort_unstable();

    // B6: shared evaluators whose sample count drifted between the two runs (name-sorted).
    let mut count_mismatches: Vec<String> = metrics
        .iter()
        .filter(|(_, m)| m.baseline_count != m.candidate_count)
        .map(|(name, _)| name.clone())
        .collect();
    count_mismatches.sort_unstable();

    ReportComparison {
        baseline_run_id: baseline.run_id.clone(),
        candidate_run_id: candidate.run_id.clone(),
        metrics,
        regressions,
        added,
        dropped,
        count_mismatches,
        tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OverallCost, ScoreSummary};

    fn report(run_id: &str, summary: Vec<(&str, f64, usize)>) -> Report {
        Report {
            per_example: Vec::new(),
            summary: summary
                .into_iter()
                .map(|(name, mean, count)| {
                    (
                        name.to_string(),
                        ScoreSummary {
                            mean,
                            std: 0.0,
                            count,
                        },
                    )
                })
                .collect(),
            failures: Vec::new(),
            cost: OverallCost::default(),
            run_id: run_id.to_string(),
        }
    }

    #[test]
    fn identical_reports_have_zero_delta_and_no_regression() {
        let r = report("base", vec![("exact_match", 1.0, 3)]);
        let cmp = compare_reports(&r, &r, 0.01);
        assert!(!cmp.is_regressed());
        assert_eq!(
            cmp.metrics["exact_match"],
            MetricDelta {
                baseline_mean: 1.0,
                candidate_mean: 1.0,
                delta: 0.0,
                baseline_count: 3,
                candidate_count: 3,
            }
        );
        assert!(cmp.added.is_empty());
        assert!(cmp.dropped.is_empty());
    }

    #[test]
    fn drop_beyond_tolerance_is_a_regression() {
        let base = report("b", vec![("exact_match", 1.0, 4)]);
        let cand = report("c", vec![("exact_match", 0.79, 4)]);

        assert!(compare_reports(&base, &cand, 0.10).is_regressed());
        // The same -0.21 drop passes when the tolerance band is wide enough.
        assert!(!compare_reports(&base, &cand, 0.21).is_regressed());
        // Boundary drop equal to the tolerance is inclusive (passes).
        let cand_edge = report("c", vec![("exact_match", 0.8, 4)]);
        assert!(!compare_reports(&base, &cand_edge, 0.20).is_regressed());
    }

    #[test]
    fn improvement_is_never_a_regression() {
        let base = report("b", vec![("judge", 0.5, 10)]);
        let cand = report("c", vec![("judge", 0.9, 10)]);
        let cmp = compare_reports(&base, &cand, 0.0);
        assert!(!cmp.is_regressed());
        assert!(cmp.metrics["judge"].delta > 0.0);
    }

    #[test]
    fn evaluators_on_one_side_are_listed_not_scored() {
        let base = report("b", vec![("a", 1.0, 1), ("gone", 0.4, 1)]);
        let cand = report("c", vec![("a", 1.0, 1), ("new", 0.9, 1)]);
        let cmp = compare_reports(&base, &cand, 0.0);
        assert_eq!(cmp.added, vec!["new".to_string()]);
        assert_eq!(cmp.dropped, vec!["gone".to_string()]);
        assert_eq!(cmp.metrics.len(), 1, "only shared evaluators are diffed");
    }

    #[test]
    fn negative_tolerance_is_clamped_to_zero() {
        let base = report("b", vec![("m", 1.0, 2)]);
        let cand = report("c", vec![("m", 1.0, 2)]);
        let cmp = compare_reports(&base, &cand, -5.0);
        assert!(cmp.tolerance >= 0.0);
        assert!(!cmp.is_regressed());
    }

    #[test]
    fn regressions_are_name_sorted_and_table_carries_the_key_facts() {
        let base = report(
            "b",
            vec![("zeta", 1.0, 1), ("alpha", 1.0, 1), ("same", 0.5, 1)],
        );
        let cand = report(
            "c",
            vec![("zeta", 0.1, 1), ("alpha", 0.2, 1), ("same", 0.5, 1)],
        );
        let cmp = compare_reports(&base, &cand, 0.0);
        assert_eq!(
            cmp.regressions
                .iter()
                .map(|r| r.evaluator.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
        let table = cmp.to_table();
        assert!(table.contains("eval regression: baseline b -> candidate c"));
        assert!(table.contains("REGRESSION"));
        assert!(table.contains("regressions: 2"));
    }

    /// B6: the gate fails on regression, sample-count mismatch, or zero shared evaluators.
    #[test]
    fn is_gate_failing_covers_regression_mismatch_and_zero_shared() {
        // Regression -> failing.
        let base = report("b", vec![("m", 1.0, 4)]);
        let cand = report("c", vec![("m", 0.5, 4)]);
        assert!(compare_reports(&base, &cand, 0.0).is_gate_failing());

        // Sample-count mismatch with no regression -> still failing.
        let cand_mismatch = report("c", vec![("m", 1.0, 2)]);
        let cmp = compare_reports(&base, &cand_mismatch, 0.0);
        assert_eq!(cmp.count_mismatches, vec!["m".to_string()]);
        assert!(!cmp.is_regressed());
        assert!(cmp.is_gate_failing(), "count mismatch must trip the gate");

        // Identical reports -> passing.
        let same = report("c", vec![("m", 1.0, 4)]);
        let ok = compare_reports(&base, &same, 0.0);
        assert!(ok.count_mismatches.is_empty());
        assert!(!ok.is_gate_failing());

        // Zero shared evaluators (both runs effectively empty) -> failing.
        let empty = report("e", vec![]);
        let z = compare_reports(&empty, &empty, 0.0);
        assert!(z.metrics.is_empty());
        assert!(z.is_gate_failing(), "zero samples must trip the gate");
    }

    /// B6: count mismatches are surfaced in the human-readable table too.
    #[test]
    fn to_table_lists_count_mismatches() {
        let base = report("b", vec![("m", 1.0, 4)]);
        let cand = report("c", vec![("m", 1.0, 2)]);
        let cmp = compare_reports(&base, &cand, 0.0);
        let table = cmp.to_table();
        assert!(table.contains("count mismatches: m"));
    }
}
