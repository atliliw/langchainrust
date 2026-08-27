// src/agents/react/parser.rs
//! ReAct output parser
//!
//! Parses the LLM's ReAct-format output.

use crate::{AgentAction, AgentError, AgentFinish, AgentOutput, ToolInput};
use regex::Regex;

/// ReAct output parser
///
/// Parsed format:
/// ```text
/// Thought: thought content
/// Action: tool name
/// Action Input: tool input
/// ```
/// or
/// ```text
/// Thought: thought content
/// Final Answer: final answer
/// ```
pub struct ReActOutputParser {
    /// Action regex
    action_regex: Regex,
    /// Final Answer marker
    final_answer_marker: &'static str,
}

impl ReActOutputParser {
    /// Creates a new parser
    pub fn new() -> Self {
        Self {
            // Matches: Action: xxx\nAction Input: yyy
            action_regex: Regex::new(r"Action\s*:\s*(.*?)\s*\nAction\s*Input\s*:\s*(.*?)(?:\n|$)")
                .expect("Invalid regex"),
            final_answer_marker: "Final Answer:",
        }
    }

    /// Parses LLM output
    ///
    /// # Parameters
    /// * `text` - the LLM's output text
    ///
    /// # Returns
    /// * `AgentOutput::Action` - the action to execute (Action takes priority over Final Answer)
    /// * `AgentOutput::Finish` - the final answer (content after the last occurrence)
    pub fn parse(&self, text: &str) -> Result<AgentOutput, AgentError> {
        let text = text.trim();

        // F6: try Action first — if there is an Action, it wins. The model may
        // mention "Final Answer:" in its Thought (explaining the format / giving
        // an example) but then actually call a tool; the old logic treated any
        // `contains` hit as the end and would skip the following Action.
        if let Some(action) = self.parse_action(text)? {
            return Ok(AgentOutput::Action(action));
        }

        // No Action: check Final Answer, take the content after the last occurrence.
        if text.contains(self.final_answer_marker) {
            return self.parse_final_answer(text);
        }

        // Unparseable
        Err(AgentError::OutputParsingError(format!(
            "failed to parse output. Use one of the following formats:\n\
             Thought: <your reasoning>\n\
             Action: <tool name>\n\
             Action Input: <tool input>\n\n\
             or\n\n\
             Thought: <your reasoning>\n\
             Final Answer: <final answer>\n\n\
             Actual output: {}",
            text
        )))
    }

    /// Parses the Final Answer
    fn parse_final_answer(&self, text: &str) -> Result<AgentOutput, AgentError> {
        let parts: Vec<&str> = text.split(self.final_answer_marker).collect();

        if parts.len() < 2 {
            return Err(AgentError::OutputParsingError(
                "missing content after Final Answer".to_string(),
            ));
        }

        // F6: take the content after the last occurrence, not the first (the
        // model may reference the marker several times mid-output; the real
        // answer is at the end).
        let answer = parts.last().unwrap_or(&"").trim().to_string();

        Ok(AgentOutput::Finish(AgentFinish::new(
            answer,
            text.to_string(),
        )))
    }

    /// Parses an Action
    fn parse_action(&self, text: &str) -> Result<Option<AgentAction>, AgentError> {
        if let Some(caps) = self.action_regex.captures(text) {
            let tool = caps
                .get(1)
                .map(|m| m.as_str().trim().to_string())
                .ok_or_else(|| AgentError::OutputParsingError("missing Action".to_string()))?;

            let tool_input_str = caps
                .get(2)
                .map(|m| m.as_str().trim().to_string())
                .ok_or_else(|| {
                    AgentError::OutputParsingError("missing Action Input".to_string())
                })?;

            // Parse the tool input
            let tool_input = self.parse_tool_input(&tool_input_str);

            return Ok(Some(AgentAction {
                tool,
                tool_input,
                log: text.to_string(),
            }));
        }

        Ok(None)
    }

    /// Parses a tool input
    fn parse_tool_input(&self, input: &str) -> ToolInput {
        let input = input.trim();

        // Try to parse as JSON
        if input.starts_with('{') || input.starts_with('[') {
            if let Ok(value) = serde_json::from_str(input) {
                return ToolInput::Object { value };
            }
        }

        // Strip surrounding quotes
        let cleaned = input.trim_matches('"').trim_matches('\'');

        ToolInput::String {
            value: cleaned.to_string(),
        }
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
                    ToolInput::String { value: s } => assert_eq!(s, "北京"),
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

    #[test]
    fn test_action_preferred_when_final_answer_mentioned_in_thought() {
        // F6: the Thought mentions "Final Answer:" (explaining the format) but a
        // real Action follows — must parse as Action, not misjudge it as the end.
        let parser = ReActOutputParser::new();

        let text = r#"Thought: 用户要算数,不能用 Final Answer: 直接回答,需要调工具
Action: calculator
Action Input: {"expression": "2 + 3"}"#;

        let result = parser.parse(text).unwrap();

        match result {
            AgentOutput::Action(action) => assert_eq!(action.tool, "calculator"),
            _ => panic!("期望 Action,而不是被 'Final Answer:' 字样误判收尾"),
        }
    }

    #[test]
    fn test_final_answer_takes_last_occurrence() {
        // F6: multiple Final Answer occurrences → take the content after the last one.
        let parser = ReActOutputParser::new();

        let text = r#"Thought: 先给个草稿
Final Answer: 草稿答案
Final Answer: 正式答案是 42"#;

        let result = parser.parse(text).unwrap();

        match result {
            AgentOutput::Finish(finish) => {
                assert_eq!(finish.output(), Some("正式答案是 42"));
            }
            _ => panic!("期望 Finish"),
        }
    }
}
