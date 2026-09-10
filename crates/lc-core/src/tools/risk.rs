// src/core/tools/risk.rs
//! Risk profile for Rule-of-Two tool approval (A2, v0.22.1).
//!
//! A tool author declares three orthogonal risk properties when implementing a tool.
//! The agent checks these **before executing** a tool call: a call that arms all three
//! properties at once (`count_armed() == 3`) is a complete "untrusted input -> sensitive
//! access -> state change" chain and is refused (see lc-agents' RuleOfTwoCheck).
//!
//! Design note: this lives on the *execution side* (`BaseTool`), NOT in `ToolDefinition`.
//! `ToolDefinition` is serialized into the request body sent to the model provider, so a
//! risk field there would leak into the wire format. `BaseTool` is the framework-internal
//! registry object, so declaring risk here keeps it out of the model request entirely.

/// Declared risk properties of a tool, used by the Rule-of-Two check (A2).
///
/// All three default to `false`, so an undeclared tool is fully low-risk and provokes no
/// interception — behaviour is byte-for-byte identical to a framework without this rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolRiskProfile {
    /// The tool's input originates from an untrusted external source
    /// (web page, retrieved document, raw user text).
    pub untrusted_input: bool,
    /// The tool can touch sensitive resources (read files, open network connections).
    pub sensitive_access: bool,
    /// Calling the tool changes external world state (write DB, send mail, delete, pay).
    pub state_changing: bool,
}

impl ToolRiskProfile {
    /// An all-false profile: no declared risk, never intercepted.
    pub const fn empty() -> Self {
        Self {
            untrusted_input: false,
            sensitive_access: false,
            state_changing: false,
        }
    }

    /// Number of declared (true) properties, in `[0, 3]`.
    ///
    /// The Rule of Two fires when this reaches `3`: `untrusted_input + sensitive_access +
    /// state_changing` all present forms a complete actionable-harm chain.
    pub fn count_armed(&self) -> usize {
        usize::from(self.untrusted_input)
            + usize::from(self.sensitive_access)
            + usize::from(self.state_changing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_profile_arms_nothing() {
        let p = ToolRiskProfile::empty();
        assert_eq!(p, ToolRiskProfile::default());
        assert_eq!(p.count_armed(), 0);
    }

    #[test]
    fn count_armed_covers_all_eight_combinations() {
        // All 3-bit combinations: expected armed count == number of set bits.
        for bits in 0..8u8 {
            let p = ToolRiskProfile {
                untrusted_input: bits & 0b001 != 0,
                sensitive_access: bits & 0b010 != 0,
                state_changing: bits & 0b100 != 0,
            };
            assert_eq!(p.count_armed(), bits.count_ones() as usize);
        }
    }

    #[test]
    fn all_true_is_rule_of_two_trip() {
        let p = ToolRiskProfile {
            untrusted_input: true,
            sensitive_access: true,
            state_changing: true,
        };
        assert_eq!(p.count_armed(), 3);
    }
}