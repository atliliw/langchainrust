// lc-agents/src/policy.rs
//! 工具权限分级 + 沙箱门禁(P2-9)。
//!
//! [`ToolPolicy`] 在 AgentExecutor 的工具执行边界做两道检查:
//! 1. **权限分级**:工具按风险分级([`ToolRisk`]),风险高于执行器允许档位
//!    (`max_permitted`)的工具直接拒绝执行。
//! 2. **沙箱门禁**:高风险工具必须被声明为已搬进受限环境
//!    ([`ToolPolicy::sandboxed`])才能执行——声明代表该工具已用受限后端
//!    (如 `lc-tools` 的 `SandboxTool` / `LocalSandbox`)包装;未声明的危险
//!    工具被拒绝,不允许以无沙箱状态运行。

use std::collections::{HashMap, HashSet};

use crate::base::AgentError;

/// 工具风险等级(自低到高)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ToolRisk {
    /// 无副作用或纯计算(计算器 / 日期等)。
    Safe,
    /// 有外部依赖但受控(网页抓取 / 检索)。
    Standard,
    /// 可执行任意代码 / 访问文件系统 / 网络等(代码解释器 / 文件 / HTTP)。
    Dangerous,
}

impl ToolRisk {
    /// 风险等级名。
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

/// 工具权限策略:权限分级 + 沙箱门禁。
///
/// 未配置策略时执行器不校验;配置后每次工具执行前经 [`ToolPolicy::check`] 校验。
///
/// # Example
///
/// ```rust,ignore
/// use lc_agents::{AgentExecutor, ToolPolicy, ToolRisk};
///
/// let policy = ToolPolicy::new()
///     .risk("code_interpreter", ToolRisk::Dangerous)
///     .sandboxed("code_interpreter"); // 已搬进受限环境,允许执行
/// let executor = AgentExecutor::new(agent, tools).with_tool_policy(policy);
/// ```
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    /// 工具名 → 风险等级。
    risks: HashMap<String, ToolRisk>,
    /// 未显式声明的工具默认风险。
    default_risk: ToolRisk,
    /// 执行器允许的最高风险档位(权限分级)。
    max_permitted: ToolRisk,
    /// 已搬进受限环境的工具名(沙箱门禁允许清单)。
    sandboxed: HashSet<String>,
    /// 显式放行"未沙箱化危险工具"(默认 false,需逐把钥匙开启)。
    allow_unrestricted_dangerous: bool,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolPolicy {
    /// 空策略:所有工具默认 [`ToolRisk::Safe`],允许档位最高,无沙箱清单。
    pub fn new() -> Self {
        Self {
            risks: HashMap::new(),
            default_risk: ToolRisk::Safe,
            max_permitted: ToolRisk::Dangerous,
            sandboxed: HashSet::new(),
            allow_unrestricted_dangerous: false,
        }
    }

    /// 声明工具风险等级。
    pub fn risk(mut self, name: impl Into<String>, risk: ToolRisk) -> Self {
        self.risks.insert(name.into(), risk);
        self
    }

    /// 声明工具已搬进受限环境(沙箱门禁允许清单)。
    pub fn sandboxed(mut self, name: impl Into<String>) -> Self {
        self.sandboxed.insert(name.into());
        self
    }

    /// 设置未显式声明工具的默认风险等级。
    pub fn with_default_risk(mut self, risk: ToolRisk) -> Self {
        self.default_risk = risk;
        self
    }

    /// 设置执行器允许的最高风险档位(权限分级)。
    pub fn with_max_permitted(mut self, risk: ToolRisk) -> Self {
        self.max_permitted = risk;
        self
    }

    /// 显式放行"未沙箱化危险工具"(危险但必须裸跑的兜底开关,默认关闭)。
    pub fn allow_unrestricted_dangerous(mut self, allow: bool) -> Self {
        self.allow_unrestricted_dangerous = allow;
        self
    }

    /// 解析工具风险等级。
    pub fn risk_of(&self, name: &str) -> ToolRisk {
        self.risks.get(name).copied().unwrap_or(self.default_risk)
    }

    /// 执行前门禁:不达标返回 [`AgentError`],达标返回 `Ok(())`。
    pub fn check(&self, name: &str) -> Result<(), AgentError> {
        let risk = self.risk_of(name);

        // 1. 权限分级:风险高于允许档位。
        if risk > self.max_permitted {
            return Err(AgentError::Other(format!(
                "tool '{name}' requires permission tier '{risk}', max permitted is '{}'",
                self.max_permitted
            )));
        }

        // 2. 沙箱门禁:危险工具必须已搬进受限环境。
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
        // 空策略:任何工具默认 Safe,不拦截。
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
        // 未声明的工具仍按默认 Safe,低于允许档位。
        assert!(policy.check("other").is_ok());
    }

    #[test]
    fn test_policy_dangerous_requires_sandbox() {
        // 危险但未声明沙箱 → 拒绝。
        let policy = ToolPolicy::new().risk("code_interpreter", ToolRisk::Dangerous);
        let err = policy.check("code_interpreter").unwrap_err();
        assert!(err.to_string().contains("sandboxed"), "{}", err);

        // 声明沙箱后允许执行(已搬进受限环境)。
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
        // 沙箱声明只豁免"沙箱门禁",不豁免"权限分级"。
        let policy = ToolPolicy::new()
            .risk("calculator", ToolRisk::Dangerous)
            .sandboxed("calculator")
            .with_max_permitted(ToolRisk::Standard);
        let err = policy.check("calculator").unwrap_err();
        assert!(err.to_string().contains("permission tier"), "{}", err);
    }
}
