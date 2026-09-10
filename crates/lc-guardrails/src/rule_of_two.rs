//! A2 — Rule of Two risk triage (v0.22.1 §S8).
//!
//! A tool that declares all three orthogonal risk properties at once —
//! [`untrusted_input`](lc_core::tools::ToolRiskProfile::untrusted_input) +
//! [`sensitive_access`](lc_core::tools::ToolRiskProfile::sensitive_access) +
//! [`state_changing`](lc_core::tools::ToolRiskProfile::state_changing) — accepts untrusted
//! input to touch something sensitive and change state: the highest-risk shape. The Rule of Two
//! names this shape so a caller (agent executor, MCP gateway, guardrail) can route it to a human
//! or reject it outright.
//!
//! Everything here is a pure function over the tool's declared profile, so it is trivially
//! testable and carries no dependency on how a particular loop decides to enforce the rule.
//!
//! ## Note on placement
//!
//! `lc-agents` cannot depend on `lc-guardrails` (the dependency points the other way), so the
//! executor's own assembly re-implements the threshold against `lc-core`. This module is the
//! **exported, reusable** rule — the single source of truth for the threshold constant.

use lc_core::tools::ToolRiskProfile;

/// Middle marker: three armed properties is the maximum, and under the Rule of Two that means
/// "untrusted input touching a sensitive, state-changing resource" — always route for approval.
pub const RULE_OF_TWO_ARMED_THRESHOLD: usize = 3;

/// Whether a tool profile crosses the Rule-of-Two line (all three properties armed).
pub fn is_high_risk(profile: &ToolRiskProfile) -> bool {
    profile.count_armed() >= RULE_OF_TWO_ARMED_THRESHOLD
}

/// The triage outcome for a tool profile per the Rule of Two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleOfTwoVerdict {
    /// Fewer than three properties armed: ordinary tool call.
    Allowed,
    /// All three properties armed: route for approval or reject.
    HighRisk,
}

/// Classifies a tool profile under the Rule of Two.
pub fn triage(profile: &ToolRiskProfile) -> RuleOfTwoVerdict {
    if is_high_risk(profile) {
        RuleOfTwoVerdict::HighRisk
    } else {
        RuleOfTwoVerdict::Allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(u: bool, s: bool, c: bool) -> ToolRiskProfile {
        ToolRiskProfile {
            untrusted_input: u,
            sensitive_access: s,
            state_changing: c,
        }
    }

    #[test]
    fn all_three_armed_is_high_risk() {
        let p = profile(true, true, true);
        assert!(is_high_risk(&p));
        assert_eq!(triage(&p), RuleOfTwoVerdict::HighRisk);
    }

    #[test]
    fn any_two_or_fewer_is_allowed() {
        // all 7 combinations of 0/1/2 armed properties must be Allowed
        for u in [false, true] {
            for s in [false, true] {
                for c in [false, true] {
                    let p = profile(u, s, c);
                    // skip the single all-true case
                    if u && s && c {
                        continue;
                    }
                    assert_eq!(triage(&p), RuleOfTwoVerdict::Allowed, "{u} {s} {c}");
                    assert_eq!(p.count_armed() < RULE_OF_TWO_ARMED_THRESHOLD, true);
                }
            }
        }
    }

    #[test]
    fn default_profile_is_allowed() {
        // default all-false → never intercepted (zero behavior change)
        assert_eq!(triage(&ToolRiskProfile::default()), RuleOfTwoVerdict::Allowed);
    }
}