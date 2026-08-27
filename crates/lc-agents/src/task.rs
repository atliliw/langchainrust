//! Explicit agent task definition (P2-5)
//!
//! Promotes the "task" in multi-agent dispatch from a bare `String` to an
//! explicit type [`AgentTask`]: objective, expected output, and allowed tools
//! travel together to the child agent, replacing the bare "just give a
//! sentence" input so dispatcher and consumer agree on the task contract.

use serde::{Deserialize, Serialize};

/// Child agent task (P2-5)
///
/// An explicit task dispatched to a child agent, carrying two layers of
/// constraint beyond a bare `String`:
/// - `expected_output`: the expected deliverable, so the child agent aligns its
///   result shape;
/// - `allowed_tools`: the child agent's tool allowlist (actual assembly is the
///   consumer's responsibility).
///
/// Can degrade to a bare objective string via [`From<AgentTask>` for `String`]
/// for `Input=String` orchestrators / executors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    /// Task objective: a one-sentence description of what to do.
    pub objective: String,
    /// Expected output (optional): the result's shape / key points, for the child agent to align to.
    pub expected_output: Option<String>,
    /// Allowed tool allowlist (empty = no restriction).
    pub allowed_tools: Vec<String>,
}

impl AgentTask {
    /// Creates a task with only an objective.
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            expected_output: None,
            allowed_tools: Vec::new(),
        }
    }

    /// Declares the expected output.
    pub fn with_expected_output(mut self, expected_output: impl Into<String>) -> Self {
        self.expected_output = Some(expected_output.into());
        self
    }

    /// Declares the allowed-tool allowlist (overwrite semantics).
    pub fn with_allowed_tools(
        mut self,
        tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_tools = tools.into_iter().map(Into::into).collect();
        self
    }

    /// Task objective.
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// Expected output, if any.
    pub fn expected_output(&self) -> Option<&str> {
        self.expected_output.as_deref()
    }

    /// Allowed-tool allowlist.
    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }

    /// Whether tools are restricted (allowlist non-empty).
    pub fn is_tool_restricted(&self) -> bool {
        !self.allowed_tools.is_empty()
    }
}

/// Degrades to a bare objective string: passes an `AgentTask` to orchestrators / executors that only accept `String`.
impl From<AgentTask> for String {
    fn from(task: AgentTask) -> Self {
        task.objective
    }
}

impl std::fmt::Display for AgentTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.objective)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_task_new_has_no_constraints() {
        let task = AgentTask::new("研究 LangChain");
        assert_eq!(task.objective(), "研究 LangChain");
        assert_eq!(task.expected_output(), None);
        assert!(task.allowed_tools().is_empty());
        assert!(!task.is_tool_restricted());
    }

    #[test]
    fn test_agent_task_with_constraints() {
        let task = AgentTask::new("写周报")
            .with_expected_output("Markdown 一页")
            .with_allowed_tools(["web_search", "calculator"]);
        assert_eq!(task.expected_output(), Some("Markdown 一页"));
        assert_eq!(
            task.allowed_tools(),
            &["web_search".to_string(), "calculator".to_string()]
        );
        assert!(task.is_tool_restricted());
        assert_eq!(task.to_string(), "写周报");
    }

    #[test]
    fn test_agent_task_from_string_loses_constraints() {
        let task = AgentTask::new("查一下天气")
            .with_expected_output("一句话")
            .with_allowed_tools(["weather"]);
        let bare: String = task.into();
        assert_eq!(bare, "查一下天气");
    }

    #[test]
    fn test_agent_task_serialize_roundtrip() {
        let task = AgentTask::new("翻译")
            .with_expected_output("中文")
            .with_allowed_tools(["dict"]);
        let json = serde_json::to_string(&task).unwrap();
        let back: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(back.objective(), "翻译");
        assert_eq!(back.expected_output(), Some("中文"));
        assert_eq!(back.allowed_tools(), &["dict".to_string()]);
    }
}
