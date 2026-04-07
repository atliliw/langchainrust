// src/agents/react/parser.rs
//! ReAct 输出解析器
//!
//! 解析 LLM 的 ReAct 格式输出。

use crate::agents::{AgentAction, AgentError, AgentFinish, AgentOutput, ToolInput};
use regex::Regex;

/// ReAct 输出解析器
///
/// 解析格式：
/// ```text
/// Thought: 思考内容
/// Action: 工具名称
/// Action Input: 工具输入
/// ```
/// 或
/// ```text
/// Thought: 思考内容
/// Final Answer: 最终答案
/// ```
pub struct ReActOutputParser {
    /// Action 正则表达式
    action_regex: Regex,
    /// Final Answer 标记
    final_answer_marker: &'static str,
}

impl ReActOutputParser {
    /// 创建新的解析器
    pub fn new() -> Self {
        Self {
            // 匹配: Action: xxx\nAction Input: yyy
            action_regex: Regex::new(r"Action\s*:\s*(.*?)\s*\nAction\s*Input\s*:\s*(.*?)(?:\n|$)")
                .expect("Invalid regex"),
            final_answer_marker: "Final Answer:",
        }
    }

    /// 解析 LLM 输出
    ///
    /// # 参数
    /// * `text` - LLM 的输出文本
    ///
    /// # 返回
    /// * `AgentOutput::Action` - 需要执行动作
    /// * `AgentOutput::Finish` - 最终答案
    pub fn parse(&self, text: &str) -> Result<AgentOutput, AgentError> {
        let text = text.trim();

        // 检查是否包含 Final Answer
        if text.contains(self.final_answer_marker) {
            return self.parse_final_answer(text);
        }

        // 尝试解析 Action
        if let Some(action) = self.parse_action(text)? {
            return Ok(AgentOutput::Action(action));
        }

        // 无法解析
        Err(AgentError::OutputParsingError(format!(
            "无法解析输出。请使用以下格式:\n\
             Thought: 你的思考\n\
             Action: 工具名称\n\
             Action Input: 工具输入\n\n\
             或\n\n\
             Thought: 你的思考\n\
             Final Answer: 最终答案\n\n\
             实际输出: {}",
            text
        )))
    }

    /// 解析 Final Answer
    fn parse_final_answer(&self, text: &str) -> Result<AgentOutput, AgentError> {
        let parts: Vec<&str> = text.split(self.final_answer_marker).collect();

        if parts.len() < 2 {
            return Err(AgentError::OutputParsingError(
                "Final Answer 后缺少内容".to_string(),
            ));
        }

        let answer = parts[1].trim().to_string();

        Ok(AgentOutput::Finish(AgentFinish::new(
            answer,
            text.to_string(),
        )))
    }

    /// 解析 Action
    fn parse_action(&self, text: &str) -> Result<Option<AgentAction>, AgentError> {
        if let Some(caps) = self.action_regex.captures(text) {
            let tool = caps
                .get(1)
                .map(|m| m.as_str().trim().to_string())
                .ok_or_else(|| AgentError::OutputParsingError("缺少 Action".to_string()))?;

            let tool_input_str = caps
                .get(2)
                .map(|m| m.as_str().trim().to_string())
                .ok_or_else(|| AgentError::OutputParsingError("缺少 Action Input".to_string()))?;

            // 解析工具输入
            let tool_input = self.parse_tool_input(&tool_input_str);

            return Ok(Some(AgentAction {
                tool,
                tool_input,
                log: text.to_string(),
            }));
        }

        Ok(None)
    }

    /// 解析工具输入
    fn parse_tool_input(&self, input: &str) -> ToolInput {
        let input = input.trim();

        // 尝试解析为 JSON
        if input.starts_with('{') || input.starts_with('[') {
            if let Ok(value) = serde_json::from_str(input) {
                return ToolInput::Object(value);
            }
        }

        // 移除引号
        let cleaned = input.trim_matches('"').trim_matches('\'');

        ToolInput::String(cleaned.to_string())
    }
}

impl Default for ReActOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_action() {
        let parser = ReActOutputParser::new();

        let text = r#"Thought: 我需要计算这个表达式
Action: calculator
Action Input: {"expression": "2 + 3"}"#;

        let result = parser.parse(text).unwrap();

        match result {
            AgentOutput::Action(action) => {
                assert_eq!(action.tool, "calculator");
            }
            _ => panic!("期望 Action"),
        }
    }

    #[test]
    fn test_parse_final_answer() {
        let parser = ReActOutputParser::new();

        let text = r#"Thought: 我已经知道答案了
Final Answer: 答案是 42"#;

        let result = parser.parse(text).unwrap();

        match result {
            AgentOutput::Finish(finish) => {
                assert_eq!(finish.output(), Some("答案是 42"));
            }
            _ => panic!("期望 Finish"),
        }
    }

    #[test]
    fn test_parse_string_input() {
        let parser = ReActOutputParser::new();

        let text = r#"Thought: 需要查询天气
Action: weather
Action Input: 北京"#;

        let result = parser.parse(text).unwrap();

        match result {
            AgentOutput::Action(action) => {
                assert_eq!(action.tool, "weather");
                match action.tool_input {
                    ToolInput::String(s) => assert_eq!(s, "北京"),
                    _ => panic!("期望 String 输入"),
                }
            }
            _ => panic!("期望 Action"),
        }
    }

    #[test]
    fn test_parse_error() {
        let parser = ReActOutputParser::new();

        let text = "这是无效的输出";

        let result = parser.parse(text);
        assert!(result.is_err());
    }
}
