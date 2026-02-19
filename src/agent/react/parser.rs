use crate::agent::{AgentAction, AgentError};
use std::collections::HashMap;

/// 解析模型响应，提取工具调用或最终答案
pub fn parse_response(response: &str) -> Result<AgentAction, AgentError> {
    let response = response.trim();

    // 检查是否有工具调用标记 [TOOL: tool_name key=value ...]
    if response.contains("[TOOL:") {
        for line in response.lines() {
            if line.contains("[TOOL:")
                && let Some(start) = line.find("[TOOL:")
                && let Some(end) = line.find("]")
            {
                let content = &line[start + 6..end].trim();
                let parts: Vec<&str> = content.split_whitespace().collect();

                if parts.is_empty() {
                    continue;
                }

                let tool_name = parts[0].to_string();
                let mut params = HashMap::new();

                for part in &parts[1..] {
                    if let Some((k, v)) = part.split_once('=') {
                        params.insert(k.to_string(), v.to_string());
                    }
                }

                return Ok(AgentAction::ToolCall(tool_name, params));
            }
        }
    }

    // 兼容旧格式 "行为：tool_name key=value"
    for line in response.lines() {
        if line.starts_with("行为：") {
            let rest = line.trim_start_matches("行为：").trim();
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();

            if parts.is_empty() {
                continue;
            }

            let tool_name = parts[0].to_string();
            let mut params = HashMap::new();

            if parts.len() == 2 {
                for pair in parts[1].split_whitespace() {
                    if let Some((k, v)) = pair.split_once('=') {
                        params.insert(k.to_string(), v.to_string());
                    }
                }
            }

            return Ok(AgentAction::ToolCall(tool_name, params));
        }
    }

    Ok(AgentAction::FinalAnswer(response.to_string()))
}

/// 生成工具描述文本
pub fn tool_descriptions(tools: &[std::sync::Arc<dyn crate::tools::Tool>]) -> String {
    tools
        .iter()
        .map(|t| {
            let params = t.parameters();
            let param_str = if params.is_empty() {
                "无参数".to_string()
            } else {
                params
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!("{} - {} (参数: {})", t.name(), t.description(), param_str)
        })
        .collect::<Vec<_>>()
        .join("\n")
}
