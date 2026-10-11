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

/// Result of a paired sign test for one evaluator: the candidate's per-example scores versus the
/// baseline's, example by example (joined by [`ExampleReport::index`](crate::ExampleReport::index),
/// the dataset position). Ties carry no direction and are excluded from the binomial count.
///
/// `p_value` is the two-sided exact sign-test p-value: under the null hypothesis of no change
/// (equally likely wins and losses over `n = wins + losses` pairs), it asks how surprising a run
/// of at least `min(wins, losses)` losses is. A large p-value means the mean delta could easily be
/// explained by per-example noise; a small one (`< 0.05`, say) means the movement is unlikely to
/// be chance. It complements the mean-based [`Regression`] gate: a regression is only meaningful
/// when it is also significant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairedSignificance {
    /// Evaluator name (`Evaluator::name`).
    pub evaluator: String,
    /// Paired examples where the candidate scored strictly higher than the baseline.
    pub wins: usize,
    /// Paired examples where the candidate scored strictly lower than the baseline.
    pub losses: usize,
    /// Paired examples where the candidate and baseline scores are equal (carry no binomial signal).
    pub ties: usize,
    /// Paired examples actually used by the sign test: `wins + losses`.
    pub n: usize,
    /// Two-sided exact sign-test p-value (ties excluded); `0.0` when `n` is so large the tiny
    /// binomial mass underflows double precision (in which case it is enormously significant anyway).
    pub p_value: f64,
}

/// Two-sided exact p-value of `k` successes out of `n` fair (p = 0.5) Bernoulli trials — the sign
/// test. Ties must already be excluded (they carry no binomial signal). `n == 0` returns `1.0` (no
/// evidence either way). The tail is summed via the probability recurrence `P(k+1) = P(k)·(n−k)/(k+1)`
/// so binomial coefficients never overflow `u64`; starting from `P(0) = 0.5ⁿ`, which underflows to a
/// true zero once `n ≈ 1073`, after which the whole (astronomically small) tail reports `0.0`.
fn sign_test_p(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    let k = wins.min(losses);
    let mut p_k = 0.5f64.powi(n as i32);
    let mut tail = p_k;
    for i in 0..k {
        p_k *= (n - i) as f64 / (i + 1) as f64;
        tail += p_k;
    }
    (2.0 * tail).min(1.0)
}

/// Computes the paired sign test for one evaluator shared by two runs, joining per-example scores by
/// their dataset index. Returns `None` when no example was scored by that evaluator on both sides.
pub fn paired_significance(
    baseline: &Report,
    candidate: &Report,
    evaluator: &str,
) -> Option<PairedSignificance> {
    let base_by_index: HashMap<usize, &crate::ExampleReport> =
        baseline.per_example.iter().map(|e| (e.index, e)).collect();
    let cand_by_index: HashMap<usize, &crate::ExampleReport> =
        candidate.per_example.iter().map(|e| (e.index, e)).collect();

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut ties = 0usize;
    let mut paired = false;
    for (index, be) in &base_by_index {
        // pair only examples present in both runs at the same dataset position
        let Some(ce) = cand_by_index.get(index) else {
            continue;
        };
        let (Some(bs), Some(cs)) = (be.scores.get(evaluator), ce.scores.get(evaluator)) else {
            continue;
        };
        paired = true;
        if cs.value > bs.value {
            wins += 1;
        } else if cs.value < bs.value {
            losses += 1;
        } else {
            ties += 1;
        }
    }
    if !paired {
        return None;
    }
    let n = wins + losses;
    Some(PairedSignificance {
        evaluator: evaluator.to_string(),
        wins,
        losses,
        ties,
        n,
        p_value: sign_test_p(wins, losses),
    })
}

/// Tuning knobs for [`compare_reports_with`]. Defaults match the semantics of the bare
/// [`compare_reports`] call: `tolerance 0.0`, `gate_on_dropped true` (fail-closed), and a `0.05`
/// significance threshold for annotating p-values in the table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareOptions {
    /// Allowed negative mean movement; a drop strictly larger than this is a [`Regression`].
    /// Clamped to `>= 0` (an equal boundary drop passes, like an epsilon comparison).
    pub tolerance: f64,
    /// When `true` (the default, fail-closed), an evaluator present only in the baseline run
    /// (`dropped`) trips the gate. Set to `false` to allow a candidate to drop an evaluator.
    pub gate_on_dropped: bool,
    /// p-value below which a paired movement is annotated as significant in
    /// [`ReportComparison::to_table`].
    pub significance_level: f64,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            tolerance: 0.0,
            gate_on_dropped: true,
            significance_level: 0.05,
        }
    }
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
    /// Whether [`is_gate_failing`](Self::is_gate_failing) treats evaluators present only in the
    /// baseline (`dropped`) as a failure. Defaults to `true` (fail-closed for old artifacts);
    /// set to `false` via [`CompareOptions`] to allow a candidate to drop an evaluator.
    #[serde(default = "default_gate_on_dropped")]
    pub gate_on_dropped: bool,
    /// Paired sign-test significance per shared evaluator (candidate wins/losses/ties + p-value).
    /// Empty when a run predates per-example scoring or an evaluator scored no paired examples.
    #[serde(default)]
    pub significance: HashMap<String, PairedSignificance>,
    /// p-value threshold at which [`to_table`](Self::to_table) marks a movement significant. Defaults to `0.05`.
    #[serde(default = "default_significance_level")]
    pub significance_level: f64,
}

fn default_gate_on_dropped() -> bool {
    true
}

fn default_significance_level() -> f64 {
    0.05
}

impl ReportComparison {
    /// Whether at least one evaluator regressed beyond the tolerance.
    pub fn is_regressed(&self) -> bool {
        !self.regressions.is_empty()
    }

    /// B6: whether the CI gate should fail. Fails when any of:
    /// - at least one evaluator regressed beyond the tolerance ([`is_regressed`](Self::is_regressed)),
    /// - any shared evaluator's sample count differs between baseline and candidate,
    /// - an evaluator present only in the baseline run (`dropped`) is gated (`gate_on_dropped`,
    ///   the default — the candidate silently dropping an evaluator is usually a real regression),
    /// - there are **zero** shared evaluators to compare at all (an empty/diverged run is not
    ///   a valid pass).
    pub fn is_gate_failing(&self) -> bool {
        let dropped_gated = self.gate_on_dropped && !self.dropped.is_empty();
        self.is_regressed()
            || dropped_gated
            || !self.count_mismatches.is_empty()
            || self.metrics.is_empty()
    }

    /// Whether the paired movement of an evaluator is significant at [`significance_level`](Self::significance_level)
    /// (i.e. its sign-test p-value is strictly below the threshold). Returns `false` when no paired
    /// comparison is available for that evaluator.
    pub fn is_significant(&self, evaluator: &str) -> bool {
        self.significance
            .get(evaluator)
            .is_some_and(|s| s.p_value < self.significance_level)
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
            "{:<24} {:>9} {:>9} {:>9} {:>8}\n",
            "evaluator", "baseline", "candidate", "delta", "p"
        ));
        for name in &names {
            let m = &self.metrics[*name];
            let mut marker = "";
            if m.delta < -self.tolerance {
                marker = "  REGRESSION";
            }
            // paired sign-test p-value; "*" marks significance at the configured level
            let p = match self.significance.get(*name) {
                Some(sig) => {
                    if sig.p_value < self.significance_level {
                        format!("{:.3}*", sig.p_value)
                    } else {
                        format!("{:.3}", sig.p_value)
                    }
                }
                None => "-".to_string(),
            };
            out.push_str(&format!(
                "{:<24} {:>9.4} {:>9.4} {:>+9.4} {:>8}{}\n",
                name, m.baseline_mean, m.candidate_mean, m.delta, p, marker
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
        out.push_str(&format!(
            "gate: {} (regressions={}, dropped={}{}, count_mismatches={}, zero_shared={}, level={:.3})\n",
            if self.is_gate_failing() {
                "FAILING"
            } else {
                "pass"
            },
            self.regressions.len(),
            self.dropped.len(),
            if self.gate_on_dropped {
                ""
            } else {
                "(not gated)"
            },
            self.count_mismatches.len(),
            self.metrics.is_empty(),
            self.significance_level,
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

/// Equivalent to [`compare_reports_with`] with `CompareOptions { tolerance, ..Default::default() }`
/// (i.e. `tolerance` as given, `gate_on_dropped` on, `0.05` significance level).
pub fn compare_reports(baseline: &Report, candidate: &Report, tolerance: f64) -> ReportComparison {
    compare_reports_with(
        baseline,
        candidate,
        CompareOptions {
            tolerance,
            ..CompareOptions::default()
        },
    )
}

/// Diffs two reports with explicit tuning. A metric is a [`Regression`] when
/// `candidate_mean < baseline_mean - tolerance`.
///
/// `tolerance` is clamped to `>= 0`: it absorbs run-to-run judge noise (LLM-as-judge
/// scores are not bit-stable), so only meaningful movement trips the gate. For every shared
/// evaluator the paired sign test is computed from per-example scores and exposed via
/// [`ReportComparison::significance`], letting CI tell a noisy mean wobble from a real change.
pub fn compare_reports_with(
    baseline: &Report,
    candidate: &Report,
    options: CompareOptions,
) -> ReportComparison {
    let tolerance = options.tolerance.max(0.0);

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

    // paired sign-test significance for every evaluator present on both sides
    let mut significance = HashMap::new();
    for name in metrics.keys() {
        if let Some(sig) = paired_significance(baseline, candidate, name) {
            significance.insert(name.clone(), sig);
        }
    }

    ReportComparison {
        baseline_run_id: baseline.run_id.clone(),
        candidate_run_id: candidate.run_id.clone(),
        metrics,
        regressions,
        added,
        dropped,
        count_mismatches,
        tolerance,
        gate_on_dropped: options.gate_on_dropped,
        significance,
        significance_level: options.significance_level,
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

    /// Builds a report carrying per-example scores (indexed by dataset position) for the paired sign test.
    fn report_with_scores(
        run_id: &str,
        summary: Vec<(&str, f64, usize)>,
        per_index: Vec<(usize, Vec<(&str, f64)>)>,
    ) -> Report {
        Report {
            per_example: per_index
                .into_iter()
                .map(|(index, scores)| crate::ExampleReport {
                    index,
                    input: String::new(),
                    reference: String::new(),
                    prediction: String::new(),
                    contexts: Vec::new(),
                    scores: scores
                        .into_iter()
                        .map(|(name, v)| (name.to_string(), crate::Score::new(v)))
                        .collect(),
                })
                .collect(),
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

    /// Gated: a candidate dropping an evaluator is a failing gate by default; `gate_on_dropped: false` permits it.
    #[test]
    fn gate_on_dropped_trips_the_gate_and_can_be_disabled() {
        let base = report("b", vec![("m", 1.0, 2), ("gone", 0.4, 2)]);
        let cand = report("c", vec![("m", 1.0, 2)]);

        let on = compare_reports(&base, &cand, 0.0);
        assert_eq!(on.dropped, vec!["gone".to_string()]);
        assert!(
            on.is_gate_failing(),
            "dropped must trip the gate by default"
        );

        let off = compare_reports_with(
            &base,
            &cand,
            CompareOptions {
                gate_on_dropped: false,
                ..CompareOptions::default()
            },
        );
        assert_eq!(off.dropped, vec!["gone".to_string()]);
        assert!(
            !off.is_gate_failing(),
            "gate_on_dropped=false allows the drop"
        );
    }

    /// The sign test flags a consistent per-example loss that the mean alone might hide.
    #[test]
    fn paired_significance_detects_consistent_direction() {
        // baseline all 1.0, candidate all 0.0: 0 wins / 6 losses -> p = 2 * (1/64) = 0.03125
        let base = report_with_scores(
            "b",
            vec![("exact_match", 1.0, 6)],
            (0..6).map(|i| (i, vec![("exact_match", 1.0)])).collect(),
        );
        let cand = report_with_scores(
            "c",
            vec![("exact_match", 0.0, 6)],
            (0..6).map(|i| (i, vec![("exact_match", 0.0)])).collect(),
        );
        let cmp = compare_reports(&base, &cand, 0.0);
        let sig = &cmp.significance["exact_match"];
        assert_eq!((sig.wins, sig.losses, sig.ties, sig.n), (0, 6, 0, 6));
        assert!((sig.p_value - 0.03125).abs() < 1e-9, "p={}", sig.p_value);
        assert!(
            cmp.is_significant("exact_match"),
            "6/6 losses must be significant at 0.05"
        );

        // balanced wins/losses carry no signal regardless of the mean delta's sign
        let bal_base = report_with_scores(
            "b",
            vec![("j", 0.5, 4)],
            vec![
                (0, vec![("j", 1.0)]),
                (1, vec![("j", 0.0)]),
                (2, vec![("j", 1.0)]),
                (3, vec![("j", 0.0)]),
            ],
        );
        let bal_cand = report_with_scores(
            "c",
            vec![("j", 0.5, 4)],
            vec![
                (0, vec![("j", 0.0)]),
                (1, vec![("j", 1.0)]),
                (2, vec![("j", 0.0)]),
                (3, vec![("j", 1.0)]),
            ],
        );
        let cmp2 = compare_reports(&bal_base, &bal_cand, 0.0);
        assert_eq!(cmp2.significance["j"].wins, 2);
        assert_eq!(cmp2.significance["j"].losses, 2);
        assert_eq!(cmp2.significance["j"].p_value, 1.0);
        assert!(
            !cmp2.is_significant("j"),
            "balanced movement must not be significant"
        );
    }

    /// Phase-two of `.to` — a significance map populated from per-example data surfaces a p column and a gate line.
    #[test]
    fn to_table_exposes_significance_marks_and_gate_state() {
        let base = report_with_scores(
            "b",
            vec![("exact_match", 1.0, 6)],
            (0..6).map(|i| (i, vec![("exact_match", 1.0)])).collect(),
        );
        let cand = report_with_scores(
            "c",
            vec![("exact_match", 0.0, 6)],
            (0..6).map(|i| (i, vec![("exact_match", 0.0)])).collect(),
        );
        let cmp = compare_reports(&base, &cand, 0.0);
        let table = cmp.to_table();
        assert!(table.contains("REGRESSION"), "regression marker present");
        assert!(
            table.contains("0.031*"),
            "significant p-value marked with *"
        );
        assert!(
            table.contains("gate: FAILING"),
            "regression shown as failing"
        );
    }

    /// The comparison is persisted as JSON for CI; the new fields must round-trip for old artifacts too.
    #[test]
    fn comparison_round_trips_and_old_artifacts_default_closed() {
        let base = report("b", vec![("m", 1.0, 2), ("gone", 0.4, 1)]);
        let cand = report("c", vec![("m", 1.0, 2)]);
        let cmp = compare_reports(&base, &cand, 0.0);
        let json = serde_json::to_string(&cmp).unwrap();
        let back: ReportComparison = serde_json::from_str(&json).unwrap();
        assert!(back.gate_on_dropped, "gate_on_dropped round-trips");
        assert_eq!(back.dropped, cmp.dropped);

        // an artifact persisted before these fields existed defaults to fail-closed
        let old_json = r#"{"baseline_run_id":"b","candidate_run_id":"c","metrics":{},"regressions":[],"added":[],"dropped":["gone"],"count_mismatches":[],"tolerance":0.0}"#;
        let old: ReportComparison = serde_json::from_str(old_json).unwrap();
        assert!(
            old.gate_on_dropped,
            "old artifact must default gate_on_dropped to true"
        );
        assert!(old.significance.is_empty());
        assert!((old.significance_level - 0.05).abs() < 1e-9);
    }

    /// sign-test p-value ground truths.
    #[test]
    fn sign_test_p_matches_known_binomial_tails() {
        // no evidence: n=0 -> 1.0; a single tie-free win against a loss is not signal -> 1.0
        assert_eq!(sign_test_p(0, 0), 1.0);
        assert_eq!(sign_test_p(1, 1), 1.0);
        // all 6 in one direction: 2 * P(X=0 | Bin(6, .5)) = 2/64 = 0.03125
        assert!((sign_test_p(6, 0) - 0.03125).abs() < 1e-12);
        assert!((sign_test_p(0, 6) - 0.03125).abs() < 1e-12);
        // symmetric, two-sided -> NOT < 0.05
        assert!((sign_test_p(5, 1) - 0.21875).abs() < 1e-12);
        assert!(sign_test_p(5, 1) > 0.05);
        // large n, all one-sided: hugely significant, double underflow is a true 0
        assert_eq!(sign_test_p(2000, 0), 0.0);
    }
}
