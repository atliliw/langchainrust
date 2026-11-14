//! Reliability accounting for LLM judges.
//!
//! A raw "judge said 8/10" is not a measurement: judges have position bias,
//! self-preference and agreement inflation. This module provides the
//! deterministic pieces that make judge output auditable:
//!
//! - [`order_swap_consistent`] — a winner counts only if it survives A/B and
//!   B/A ordering.
//! - [`cohen_kappa`] — agreement corrected for chance.
//! - [`same_family`] — flags self-preference risk when judge and candidate come
//!   from the same model family.
//! - [`reliability_report`] — aggregates the above over a judged sample.

use crate::pairwise::Verdict;

/// Aggregate reliability of one judged sample.
#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityReport {
    /// Number of pairs considered.
    pub n: usize,
    /// Fraction of pairs with a winner that survived order swap.
    pub agreement: f64,
    /// Cohen's κ on "A wins" (original vs position-corrected swapped).
    /// `None` when the sample is empty or degenerate (all one label).
    pub kappa: Option<f64>,
    /// Number of pairs that became a tie after order swap.
    pub ties: usize,
}

/// Interprets a swapped-round verdict in the original A/B coordinate system.
///
/// In the swapped round the positions are reversed, so `BWins` there means the
/// *original* A won. A pair counts as consistent only when both rounds name the
/// same original answer; everything else is a tie.
pub fn order_swap_consistent(original: Verdict, swapped: Verdict) -> Verdict {
    match (original, swapped) {
        (Verdict::AWins, Verdict::BWins) => Verdict::AWins,
        (Verdict::BWins, Verdict::AWins) => Verdict::BWins,
        _ => Verdict::Tie,
    }
}

/// Cohen's κ for two aligned boolean label vectors.
///
/// Returns `None` for empty inputs, length mismatches, or a degenerate
/// expected-agreement of 1 (all labels identical), where κ is undefined.
pub fn cohen_kappa(a: &[bool], b: &[bool]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let n = a.len() as f64;
    let matches = a.iter().zip(b).filter(|(x, y)| x == y).count() as f64;
    let p_o = matches / n;

    let a_true = a.iter().filter(|x| **x).count() as f64 / n;
    let b_true = b.iter().filter(|x| **x).count() as f64 / n;
    let p_e = a_true * b_true + (1.0 - a_true) * (1.0 - b_true);

    if (1.0 - p_e).abs() < 1e-12 {
        return None;
    }
    Some((p_o - p_e) / (1.0 - p_e))
}

/// Whether two model identifiers belong to the same family.
///
/// The heuristic takes the leading alphabetic prefix (`gpt-5` → `gpt`,
/// `claude-3-5` → `claude`, `qwen2.5-72b` → `qwen`). Used to flag
/// self-preference risk when a judge evaluates its own family.
pub fn same_family(model_a: &str, model_b: &str) -> bool {
    let fa = family(model_a);
    let fb = family(model_b);
    !fa.is_empty() && fa == fb
}

fn family(model: &str) -> String {
    model
        .trim()
        .to_lowercase()
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect()
}

/// Builds a [`ReliabilityReport`] from aligned original/swapped verdicts.
pub fn reliability_report(original: &[Verdict], swapped: &[Verdict]) -> ReliabilityReport {
    if original.is_empty() || original.len() != swapped.len() {
        return ReliabilityReport {
            n: 0,
            agreement: 0.0,
            kappa: None,
            ties: 0,
        };
    }

    let mut ties = 0usize;
    let mut a_original = Vec::with_capacity(original.len());
    let mut a_swapped = Vec::with_capacity(original.len());

    for (o, s) in original.iter().zip(swapped) {
        match order_swap_consistent(*o, *s) {
            Verdict::Tie => ties += 1,
            _ => {}
        }
        a_original.push(matches!(o, Verdict::AWins));
        // Position-corrected: in the swapped round, B winning means A won.
        a_swapped.push(matches!(s, Verdict::BWins));
    }

    let n = original.len();
    ReliabilityReport {
        n,
        agreement: (n - ties) as f64 / n as f64,
        kappa: cohen_kappa(&a_original, &a_swapped),
        ties,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn kappa_perfect_agreement_is_one() {
        let a = [true, false, true, true];
        let b = [true, false, true, true];
        approx(cohen_kappa(&a, &b).unwrap(), 1.0);
    }

    #[test]
    fn kappa_independent_labels_is_zero() {
        // 2x2 balanced with perfect independence: p_o = p_e = 0.5.
        let a = [true, true, false, false];
        let b = [true, false, true, false];
        approx(cohen_kappa(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn kappa_degenerate_sample_is_none() {
        let a = [true, true, true];
        let b = [true, true, true];
        assert!(cohen_kappa(&a, &b).is_none());
    }

    #[test]
    fn kappa_rejects_empty_and_mismatched() {
        assert!(cohen_kappa(&[], &[]).is_none());
        assert!(cohen_kappa(&[true], &[]).is_none());
    }

    #[test]
    fn order_swap_requires_consistent_winner() {
        assert_eq!(
            order_swap_consistent(Verdict::AWins, Verdict::BWins),
            Verdict::AWins
        );
        assert_eq!(
            order_swap_consistent(Verdict::BWins, Verdict::AWins),
            Verdict::BWins
        );
        assert_eq!(
            order_swap_consistent(Verdict::AWins, Verdict::AWins),
            Verdict::Tie
        );
        assert_eq!(
            order_swap_consistent(Verdict::Tie, Verdict::Tie),
            Verdict::Tie
        );
    }

    #[test]
    fn family_detection_groups_versions() {
        assert!(same_family("gpt-5", "gpt-4o"));
        assert!(same_family("qwen2.5-72b", "qwen3-235b"));
        assert!(same_family("claude-3-5-sonnet", "claude-opus-4"));
        assert!(!same_family("gpt-5", "claude-3"));
        assert!(!same_family("deepseek-chat", "qwen3"));
        assert!(!same_family("", "gpt-5"));
    }

    #[test]
    fn report_counts_ties_and_agreement() {
        let original = [Verdict::AWins, Verdict::BWins, Verdict::AWins];
        let swapped = [Verdict::BWins, Verdict::AWins, Verdict::AWins];
        let report = reliability_report(&original, &swapped);
        assert_eq!(report.n, 3);
        assert_eq!(report.ties, 1);
        approx(report.agreement, 2.0 / 3.0);
        assert!(report.kappa.is_some());
    }

    #[test]
    fn report_rejects_mismatched_lengths() {
        let report = reliability_report(&[Verdict::AWins], &[]);
        assert_eq!(report.n, 0);
        assert!(report.kappa.is_none());
    }
}
