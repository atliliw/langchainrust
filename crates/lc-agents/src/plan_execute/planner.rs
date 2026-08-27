//! Planner - generates / replans execution plans with the LLM

use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_providers::ProviderError;
use lc_schema::Message;
use serde_json::{json, Value};

use super::plan::Plan;
use crate::AgentError;

use std::sync::Arc;

/// JSON Schema for the planning tool: forces the LLM to emit a structured steps array (P1-3).
fn plan_tool() -> ToolDefinition {
    ToolDefinition::new(
        "generate_plan",
        "为给定目标生成执行计划,返回按顺序执行的步骤描述数组",
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": { "type": "string" },
                "description": "按顺序执行的步骤描述"
            }
        },
        "required": ["steps"]
    }))
}

/// Extracts the steps array from the tool_call args, serializing it back to `["a", "b"]` for parse_plan to reuse.
fn steps_to_json_string(args: &Value) -> String {
    args.get("steps")
        .and_then(|v| serde_json::to_string(v).ok())
        .unwrap_or_default()
}

/// Planner: calls the LLM to generate a step list
pub struct Planner {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
}

impl Planner {
    /// Creates a new planner.
    pub fn new(llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>) -> Self {
        Self { llm }
    }

    /// Generates an execution plan
    pub async fn plan(&self, objective: &str) -> Result<Plan, AgentError> {
        let prompt = format!(
            "为以下目标制定执行计划,输出 JSON 字符串数组,每项是一个步骤描述。\n\
             目标: {}\n\
             输出格式: [\"步骤1\", \"步骤2\", ...]\n\
             只输出 JSON,不要任何其他内容。",
            objective
        );
        let messages = vec![
            Message::system("你是规划助手,只输出 JSON。"),
            Message::human(prompt),
        ];
        let structured = crate::structured::chat_structured(
            self.llm.as_ref(),
            Some(plan_tool()),
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AgentError::Other(format!("LLM error: {:?}", e)))?;
        let content = match &structured.tool_args {
            Some(args) => steps_to_json_string(args),
            None => structured.content,
        };
        self.parse_plan(objective, &content)
    }

    /// Replans (when a step fails)
    pub async fn replan(
        &self,
        objective: &str,
        failed_step: &str,
        reason: &str,
    ) -> Result<Plan, AgentError> {
        let prompt = format!(
            "原目标: {}\n之前步骤 '{}' 失败: {}\n请重新制定完整计划。输出 JSON 字符串数组 [\"步骤\", ...],只输出 JSON。",
            objective, failed_step, reason
        );
        let messages = vec![
            Message::system("你是规划助手,只输出 JSON。"),
            Message::human(prompt),
        ];
        let structured = crate::structured::chat_structured(
            self.llm.as_ref(),
            Some(plan_tool()),
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AgentError::Other(format!("LLM error: {:?}", e)))?;
        let content = match &structured.tool_args {
            Some(args) => steps_to_json_string(args),
            None => structured.content,
        };
        self.parse_plan(objective, &content)
    }

    fn parse_plan(&self, objective: &str, content: &str) -> Result<Plan, AgentError> {
        let json_str = extract_json_array(content);
        let descs: Vec<String> = serde_json::from_str(&json_str).map_err(|e| {
            AgentError::OutputParsingError(format!(
                "failed to parse plan: {} | raw: {}",
                e, content
            ))
        })?;
        Ok(Plan::from_descriptions(objective, descs))
    }
}

/// Extracts a JSON array from LLM output (tolerates markdown code fences)
fn extract_json_array(content: &str) -> String {
    let trimmed = content.trim();
    // Strip markdown ```json ... ```
    let stripped = if trimmed.starts_with("```") {
        trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed)
            .strip_suffix("```")
            .unwrap_or(trimmed)
            .trim()
    } else {
        trimmed
    };
    // Take from the first [ to the last ]
    if let Some(start) = stripped.find('[') {
        if let Some(end) = stripped.rfind(']') {
            if end > start {
                return stripped[start..=end].to_string();
            }
        }
    }
    stripped.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_plain_json() {
        let s = r#"["步骤1", "步骤2"]"#;
        assert_eq!(extract_json_array(s), s);
    }

    #[test]
    fn test_extract_markdown_json() {
        let s = "```json\n[\"a\", \"b\"]\n```";
        assert_eq!(extract_json_array(s), r#"["a", "b"]"#);
    }

    #[test]
    fn test_extract_json_with_surrounding_text() {
        let s = r#"结果如下: ["x", "y"] 完成"#;
        assert_eq!(extract_json_array(s), r#"["x", "y"]"#);
    }

    #[test]
    fn test_parse_plan() {
        // No LLM needed: test the parse logic directly (through extract)
        let content = r#"["搜索资料", "总结"]"#;
        let json = extract_json_array(content);
        let descs: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(descs, vec!["搜索资料", "总结"]);
    }

    #[test]
    fn test_steps_to_json_string() {
        // P1-3: the tool_call's steps array serializes back to a JSON array that parse_plan can consume.
        let args = serde_json::json!({"steps": ["a", "b"]});
        assert_eq!(steps_to_json_string(&args), r#"["a","b"]"#);
    }

    #[test]
    fn test_steps_to_json_string_missing_steps() {
        let args = serde_json::json!({"other": 1});
        assert_eq!(steps_to_json_string(&args), "");
    }

    #[test]
    fn test_plan_tool_schema() {
        let tool = plan_tool();
        assert_eq!(tool.function.name, "generate_plan");
        assert!(tool.function.parameters.is_some());
    }
}
