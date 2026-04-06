// src/agents/react/prompt.rs
//! ReAct Prompt 模板
//!
//! 提供 ReAct Agent 使用的 prompt 模板。

/// ReAct Prompt 前缀
///
/// 描述可用工具和使用格式
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

/// 构建 ReAct prompt
///
/// # 参数
/// * `tools_description` - 工具描述字符串
/// * `tool_names` - 工具名称列表
/// * `input` - 用户问题
/// * `scratchpad` - Agent 的思考历史
///
/// # 返回
/// 完整的 prompt 字符串
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

/// 格式化 intermediate_steps 为 scratchpad
///
/// # 参数
/// * `steps` - 已执行的步骤列表
///
/// # 返回
/// 格式化的思考历史字符串
pub fn format_scratchpad(steps: &[crate::agents::AgentStep]) -> String {
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
        use crate::agents::{AgentAction, AgentStep, ToolInput};

        let steps = vec![AgentStep::new(
            AgentAction {
                tool: "calculator".to_string(),
                tool_input: ToolInput::String("2 + 2".to_string()),
                log: "我需要计算".to_string(),
            },
            "结果: 4".to_string(),
        )];

        let scratchpad = format_scratchpad(&steps);

        assert!(scratchpad.contains("calculator"));
        assert!(scratchpad.contains("结果: 4"));
    }
}
