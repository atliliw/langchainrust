//! 显式 Agent 任务定义(P2-5)
//!
//! 把多 Agent 派发时的"任务"从裸 `String` 提成显式类型 [`AgentTask`]:
//! 目标(`objective`)、预期输出(`expected_output`)、允许工具(`allowed_tools`)
//! 一起传给子 Agent,替代"只给一句话"的裸输入,让派发方和消费方对齐任务契约。

use serde::{Deserialize, Serialize};

/// 子 Agent 任务(P2-5)
///
/// 派发给子 Agent 的显式任务,比裸 `String` 多携带两层约束:
/// - `expected_output`:预期产出,供子 Agent 对齐结果形态;
/// - `allowed_tools`:子 Agent 可用的工具白名单(实际装配由消费方负责)。
///
/// 可经 [`From<AgentTask>` for `String`] 退化为裸目标文本,供 `Input=String`
/// 的编排器 / 执行器消费。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    /// 任务目标:一句话说清要做什么。
    pub objective: String,
    /// 预期输出(可选):结果的形态 / 要点,供子 Agent 对齐。
    pub expected_output: Option<String>,
    /// 允许工具白名单(空 = 不限制)。
    pub allowed_tools: Vec<String>,
}

impl AgentTask {
    /// 只带目标创建任务。
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            expected_output: None,
            allowed_tools: Vec::new(),
        }
    }

    /// 声明预期输出。
    pub fn with_expected_output(mut self, expected_output: impl Into<String>) -> Self {
        self.expected_output = Some(expected_output.into());
        self
    }

    /// 声明允许工具白名单(覆盖式设置)。
    pub fn with_allowed_tools(
        mut self,
        tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_tools = tools.into_iter().map(Into::into).collect();
        self
    }

    /// 任务目标。
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// 预期输出(若有)。
    pub fn expected_output(&self) -> Option<&str> {
        self.expected_output.as_deref()
    }

    /// 允许工具白名单。
    pub fn allowed_tools(&self) -> &[String] {
        &self.allowed_tools
    }

    /// 是否限制了工具(白名单非空)。
    pub fn is_tool_restricted(&self) -> bool {
        !self.allowed_tools.is_empty()
    }
}

/// 退化为裸目标文本:把 `AgentTask` 交给只认 `String` 的编排器 / 执行器。
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
