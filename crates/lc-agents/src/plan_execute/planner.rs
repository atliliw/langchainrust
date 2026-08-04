//! Planner - 用 LLM 生成 / 重规划执行计划

use lc_core::language_models::BaseChatModel;
use lc_providers::ProviderError;
use lc_schema::Message;

use super::plan::Plan;

use std::sync::Arc;

/// 规划器:调用 LLM 生成步骤列表
pub struct Planner {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
}

impl Planner {
    pub fn new(llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>) -> Self {
        Self { llm }
    }

    /// 生成执行计划
    pub async fn plan(&self, objective: &str) -> Result<Plan, String> {
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
        let response = self
            .llm
            .chat(messages, None)
            .await
            .map_err(|e| format!("LLM 错误: {:?}", e))?;
        self.parse_plan(objective, &response.content)
    }

    /// 重新规划(当步骤失败时)
    pub async fn replan(
        &self,
        objective: &str,
        failed_step: &str,
        reason: &str,
    ) -> Result<Plan, String> {
        let prompt = format!(
            "原目标: {}\n之前步骤 '{}' 失败: {}\n请重新制定完整计划。输出 JSON 字符串数组 [\"步骤\", ...],只输出 JSON。",
            objective, failed_step, reason
        );
        let messages = vec![
            Message::system("你是规划助手,只输出 JSON。"),
            Message::human(prompt),
        ];
        let response = self
            .llm
            .chat(messages, None)
            .await
            .map_err(|e| format!("LLM 错误: {:?}", e))?;
        self.parse_plan(objective, &response.content)
    }

    fn parse_plan(&self, objective: &str, content: &str) -> Result<Plan, String> {
        let json_str = extract_json_array(content);
        let descs: Vec<String> = serde_json::from_str(&json_str)
            .map_err(|e| format!("解析计划失败: {} | 原文: {}", e, content))?;
        Ok(Plan::from_descriptions(objective, descs))
    }
}

/// 从 LLM 输出提取 JSON 数组(容忍 markdown 代码块)
fn extract_json_array(content: &str) -> String {
    let trimmed = content.trim();
    // 去除 markdown ```json ... ```
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
    // 截取第一个 [ 到最后一个 ]
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
        // 不依赖 LLM,直接测 parse 逻辑(通过 extract)
        let content = r#"["搜索资料", "总结"]"#;
        let json = extract_json_array(content);
        let descs: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(descs, vec!["搜索资料", "总结"]);
    }
}
