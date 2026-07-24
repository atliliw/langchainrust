//! MCP Prompts - 提示词模板类型与 `prompts/list`、`prompts/get` 处理
//!
//! MCP Prompts 允许 Server 暴露可复用的提示词模板,Client 可带参数获取。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 提示词参数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// 提示词描述(来自 `prompts/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub arguments: Vec<PromptArgument>,
}

/// 提示词内容(内联枚举)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PromptContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { uri: String, name: String },
}

/// 提示词消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: PromptContent,
}

/// `prompts/list` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsResult {
    pub prompts: Vec<Prompt>,
}

/// `prompts/get` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptParams {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// `prompts/get` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub messages: Vec<PromptMessage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_serialization() {
        let prompt = Prompt {
            name: "code_review".to_string(),
            description: Some("Review code for issues".to_string()),
            arguments: vec![PromptArgument {
                name: "language".to_string(),
                description: Some("Programming language".to_string()),
                required: true,
            }],
        };
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(json.contains("\"name\":\"code_review\""));
        assert!(json.contains("\"required\":true"));
    }

    #[test]
    fn test_prompt_optional_fields() {
        let prompt = Prompt {
            name: "simple".to_string(),
            description: None,
            arguments: vec![],
        };
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(!json.contains("description"));
        assert!(json.contains("\"arguments\":[]"));
    }

    #[test]
    fn test_prompt_content_text() {
        let content = PromptContent::Text {
            text: "Review this code".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Review this code\""));
    }

    #[test]
    fn test_prompt_content_image() {
        let content = PromptContent::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"image\""));
    }

    #[test]
    fn test_prompt_content_resource() {
        let content = PromptContent::Resource {
            uri: "file:///code.rs".to_string(),
            name: "code.rs".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"resource\""));
    }

    #[test]
    fn test_prompt_message() {
        let msg = PromptMessage {
            role: "user".to_string(),
            content: PromptContent::Text {
                text: "Hello".to_string(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
    }

    #[test]
    fn test_list_prompts_result_deserialization() {
        let json = r#"{"prompts":[{"name":"greet","arguments":[]}]}"#;
        let result: ListPromptsResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.prompts.len(), 1);
        assert_eq!(result.prompts[0].name, "greet");
    }

    #[test]
    fn test_get_prompt_params() {
        let params = GetPromptParams {
            name: "code_review".to_string(),
            arguments: Some(serde_json::json!({"language": "rust"})),
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"name\":\"code_review\""));
        assert!(json.contains("language"));
    }

    #[test]
    fn test_get_prompt_result_deserialization() {
        let json = r#"{"description":"A greeting","messages":[{"role":"user","content":{"type":"text","text":"Hello"}}]}"#;
        let result: GetPromptResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.description.as_deref(), Some("A greeting"));
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
    }
}
