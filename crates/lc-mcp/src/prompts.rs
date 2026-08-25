//! MCP Prompts - 提示词模板类型与 `prompts/list`、`prompts/get` 处理
//!
//! MCP Prompts 允许 Server 暴露可复用的提示词模板,Client 可带参数获取。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 提示词参数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    /// 参数名
    pub name: String,
    /// 参数的可选描述
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 是否为必填参数
    #[serde(default)]
    pub required: bool,
}

/// 提示词描述(来自 `prompts/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    /// 提示词名称
    pub name: String,
    /// 提示词的可选描述
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 提示词参数定义列表
    #[serde(default)]
    pub arguments: Vec<PromptArgument>,
}

/// 提示词内容(内联枚举)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PromptContent {
    /// 文本内容
    #[serde(rename = "text")]
    Text {
        /// 文本数据
        text: String,
    },
    /// 图片内容
    #[serde(rename = "image")]
    Image {
        /// 图片数据(base64 编码)
        data: String,
        /// 图片 MIME 类型
        mime_type: String,
    },
    /// 资源引用内容
    #[serde(rename = "resource")]
    Resource {
        /// 资源 URI
        uri: String,
        /// 资源名称
        name: String,
    },
}

/// 提示词消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    /// 消息角色(如 `user` 或 `assistant`)
    pub role: String,
    /// 消息内容
    pub content: PromptContent,
}

/// `prompts/list` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsResult {
    /// 提示词列表
    pub prompts: Vec<Prompt>,
}

/// `prompts/get` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptParams {
    /// 要获取的提示词名称
    pub name: String,
    /// 提示词参数(JSON 对象,可选)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// `prompts/get` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    /// 提示词的可选描述
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 提示词消息列表
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
