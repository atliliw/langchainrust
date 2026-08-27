// lc-agents/src/policy.rs
//! Tool permission tiering + sandbox gate (P2-9).
//!
//! [`ToolPolicy`] enforces two checks at the `AgentExecutor` tool-execution boundary:
//! 1. **Permission tiering**: tools are classified by risk ([`ToolRisk`]); any tool whose
//!    risk exceeds the executor's permitted tier (`max_permitted`) is rejected outright.
//! 2. **Sandbox gate**: a high-risk tool must be declared as wrapped in a restricted
//!    environment ([`ToolPolicy::sandboxed`]) before it can run — the declaration means the
//!    tool has been wrapped in a restricted backend (e.g. `lc-tools`'s `SandboxTool` /
//!    `LocalSandbox`); undeclared dangerous tools are rejected and cannot run unsandboxed.

use std::collections::{HashMap, HashSet};

use crate::base::AgentError;

/// Tool risk level (from low to high).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ToolRisk {
    /// No side effects or pure computation (calculator / date, etc.).
    Safe,
    /// Has external dependencies but is controlled (web scraping / retrieval).
    Standard,
    /// Can execute arbitrary code / access the filesystem / network, etc.
    /// (code interpreter / file / HTTP).
    Dangerous,
}

impl ToolRisk {
    /// Risk level name.
    pub fn name(&self) -> &'static str {
        match self {
            ToolRisk::Safe => "safe",
            ToolRisk::Standard => "standard",
            ToolRisk::Dangerous => "dangerous",
        }
    }
}

impl std::fmt::Display for ToolRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Tool permission policy: permission tiering + sandbox gate.
///
/// When no policy is configured the executor does not enforce anything; once configured,
/// every tool execution is checked via [`ToolPolicy::check`] before it runs.
///
/// # Example
///
/// ```rust,ignore
/// use lc_agents::{AgentExecutor, ToolPolicy, ToolRisk};
///
/// let policy = ToolPolicy::new()
///     .risk("code_interpreter", ToolRisk::Dangerous)
///     .sandboxed("code_interpreter"); // wrapped in a restricted env, allowed to run
/// let executor = AgentExecutor::new(agent, tools).with_tool_policy(policy);
/// ```
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    /// Tool name -> risk level.
    risks: HashMap<String, ToolRisk>,
    /// Default risk for tools not explicitly declared.
    default_risk: ToolRisk,
    /// Highest risk tier the executor permits (permission tiering).
    max_permitted: ToolRisk,
    /// Tool names wrapped in a restricted environment (sandbox-gate allowlist).
    sandboxed: HashSet<String>,
    /// Explicitly allow unsandboxed dangerous tools (default `false`; opt in per key).
    allow_unrestricted_dangerous: bool,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolPolicy {
    /// Empty policy: every tool defaults to [`ToolRisk::Safe`], the permitted tier is the
    /// highest, and there is no sandbox allowlist.
    pub fn new() -> Self {
        Self {
            risks: HashMap::new(),
            default_risk: ToolRisk::Safe,
            max_permitted: ToolRisk::Dangerous,
            sandboxed: HashSet::new(),
            allow_unrestricted_dangerous: false,
        }
    }

    /// Declare a tool's risk level.
    pub fn risk(mut self, name: impl Into<String>, risk: ToolRisk) -> Self {
        self.risks.insert(name.into(), risk);
        self
    }

    /// Declare that a tool is wrapped in a restricted environment (sandbox-gate allowlist).
    pub fn sandboxed(mut self, name: impl Into<String>) -> Self {
        self.sandboxed.insert(name.into());
        self
    }

    /// Set the default risk level for tools not explicitly declared.
    pub fn with_default_risk(mut self, risk: ToolRisk) -> Self {
        self.default_risk = risk;
        self
    }

    /// Set the highest risk tier the executor permits (permission tiering).
    pub fn with_max_permitted(mut self, risk: ToolRisk) -> Self {
        self.max_permitted = risk;
        self
    }

    /// Explicitly allow unsandboxed dangerous tools (an escape hatch for tools that are
    /// dangerous but must run bare; off by default).
    pub fn allow_unrestricted_dangerous(mut self, allow: bool) -> Self {
        self.allow_unrestricted_dangerous = allow;
        self
    }

    /// Resolve a tool's risk level.
    pub fn risk_of(&self, name: &str) -> ToolRisk {
        self.risks.get(name).copied().unwrap_or(self.default_risk)
    }

    /// Pre-execution gate: returns [`AgentError`] when the tool does not meet the policy,
    /// `Ok(())` when it passes.
    pub fn check(&self, name: &str) -> Result<(), AgentError> {
        let risk = self.risk_of(name);

        // 1. Permission tiering: risk above the permitted tier.
        if risk > self.max_permitted {
            return Err(AgentError::Other(format!(
                "tool '{name}' requires permission tier '{risk}', max permitted is '{}'",
                self.max_permitted
            )));
        }

        // 2. Sandbox gate: a dangerous tool must be wrapped in a restricted environment.
        if risk == ToolRisk::Dangerous
            && !self.sandboxed.contains(name)
            && !self.allow_unrestricted_dangerous
        {
            return Err(AgentError::Other(format!(
                "dangerous tool '{name}' must run in a sandboxed environment \
                 (declare via ToolPolicy::sandboxed(\"{name}\"))"
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_risk_display_and_order() {
        assert_eq!(ToolRisk::Safe.to_string(), "safe");
        assert_eq!(ToolRisk::Standard.to_string(), "standard");
        assert_eq!(ToolRisk::Dangerous.to_string(), "dangerous");
        assert!(ToolRisk::Safe < ToolRisk::Standard);
        assert!(ToolRisk::Standard < ToolRisk::Dangerous);
    }

    #[test]
    fn test_policy_allows_default_safe_tools() {
        // Empty policy: any tool defaults to Safe, nothing is blocked.
        let policy = ToolPolicy::new();
        assert!(policy.check("any_tool").is_ok());
        assert_eq!(policy.risk_of("any_tool"), ToolRisk::Safe);
    }

    #[test]
    fn test_policy_permission_tier_gate() {
        let policy = ToolPolicy::new()
            .risk("calculator", ToolRisk::Dangerous)
            .with_max_permitted(ToolRisk::Standard);
        let err = policy.check("calculator").unwrap_err();
        assert!(err.to_string().contains("permission tier"), "{}", err);
        // Undeclared tools still default to Safe, below the permitted tier.
        assert!(policy.check("other").is_ok());
    }

    #[test]
    fn test_policy_dangerous_requires_sandbox() {
        // Dangerous but not declared sandboxed -> rejected.
        let policy = ToolPolicy::new().risk("code_interpreter", ToolRisk::Dangerous);
        let err = policy.check("code_interpreter").unwrap_err();
        assert!(err.to_string().contains("sandboxed"), "{}", err);

        // Once declared sandboxed it may run (wrapped in a restricted environment).
        let policy = policy.sandboxed("code_interpreter");
        assert!(policy.check("code_interpreter").is_ok());
    }

    #[test]
    fn test_policy_allow_unrestricted_dangerous() {
        let policy = ToolPolicy::new()
            .risk("http", ToolRisk::Dangerous)
            .allow_unrestricted_dangerous(true);
        assert!(policy.check("http").is_ok());
    }

    #[test]
    fn test_policy_sandboxed_but_not_dangerous_still_gated_by_tier() {
        // The sandbox declaration only exempts the sandbox gate, not permission tiering.
        let policy = ToolPolicy::new()
            .risk("calculator", ToolRisk::Dangerous)
            .sandboxed("calculator")
            .with_max_permitted(ToolRisk::Standard);
        let err = policy.check("calculator").unwrap_err();
        assert!(err.to_string().contains("permission tier"), "{}", err);
    }
}
