// src/agents/react/prompt.rs
//! ReAct prompt templates
//!
//! Provides the prompt templates used by the ReAct Agent.

/// ReAct prompt prefix
///
/// Describes the available tools and the expected format
pub const REACT_PREFIX: &str = r#"回答以下问题，你可以使用以下工具：

{tools}

使用以下格式：

Question: 需要回答的问题
Thought: 你应该思考要做什么
Action: 要执行的动作，应该是 [{tool_names}] 之一
Action Input: 动作的输入
Observation: 动作的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 原始问题的最终答案

开始！

Question: {input}
Thought:{agent_scratchpad}"#;

/// Builds the ReAct prompt
///
/// # Parameters
/// * `tools_description` - the tool descriptions string
/// * `tool_names` - the tool name list
/// * `input` - the user question
/// * `scratchpad` - the agent's thought history
///
/// # Returns
/// The complete prompt string
pub fn build_react_prompt(
    tools_description: &str,
    tool_names: &[&str],
    input: &str,
    scratchpad: &str,
) -> String {
    REACT_PREFIX
        .replace("{tools}", tools_description)
        .replace("{tool_names}", &tool_names.join(", "))
        .replace("{input}", input)
        .replace("{agent_scratchpad}", scratchpad)
}

/// Formats `intermediate_steps` into a scratchpad
///
/// # Parameters
/// * `steps` - the list of executed steps
///
/// # Returns
/// The formatted thought-history string
pub fn format_scratchpad(steps: &[crate::types::AgentStep]) -> String {
    let mut scratchpad = String::new();

    for step in steps {
        scratchpad.push_str(&format!(
            " {}\nAction: {}\nAction Input: {}\nObservation: {}\n",
            step.action.log.lines().next().unwrap_or(""),
            step.action.tool,
            step.action.tool_input,
            step.observation
        ));
    }

    scratchpad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_react_prompt() {
        let prompt = build_react_prompt(
            "calculator: 计算数学表达式",
            &["calculator"],
            "计算 2 + 2",
            "",
        );

        assert!(prompt.contains("calculator: 计算数学表达式"));
        assert!(prompt.contains("计算 2 + 2"));
        assert!(prompt.contains("[calculator]"));
    }

    #[test]
    fn test_format_scratchpad() {
        use crate::{AgentAction, AgentStep, ToolInput};

        let steps = vec![AgentStep::new(
            AgentAction {
                tool: "calculator".to_string(),
                tool_input: ToolInput::String {
                    value: "2 + 2".to_string(),
                },
                log: "我需要计算".to_string(),
            },
            "结果: 4".to_string(),
        )];

        let scratchpad = format_scratchpad(&steps);

        assert!(scratchpad.contains("calculator"));
        assert!(scratchpad.contains("结果: 4"));
    }
}
