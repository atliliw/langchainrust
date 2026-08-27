//! MCP Prompts - prompt-template types and `prompts/list` / `prompts/get` handling
//!
//! MCP Prompts lets a Server expose reusable prompt templates that a Client can fetch with arguments.

use crate::protocol::MCPError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A prompt-argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    /// Argument name
    pub name: String,
    /// Optional description of the argument
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the argument is required
    #[serde(default)]
    pub required: bool,
}

/// Prompt description (from `prompts/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    /// Prompt name
    pub name: String,
    /// Optional description of the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// List of prompt-argument definitions
    #[serde(default)]
    pub arguments: Vec<PromptArgument>,
}

/// Prompt content (inline enum)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PromptContent {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// Text data
        text: String,
    },
    /// Image content
    #[serde(rename = "image")]
    Image {
        /// Image data (base64-encoded)
        data: String,
        /// Image MIME type
        mime_type: String,
    },
    /// Resource-reference content
    #[serde(rename = "resource")]
    Resource {
        /// Resource URI
        uri: String,
        /// Resource name
        name: String,
    },
}

/// A prompt message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    /// Message role (e.g. `user` or `assistant`)
    pub role: String,
    /// Message content
    pub content: PromptContent,
}

/// `prompts/list` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsResult {
    /// List of prompts
    pub prompts: Vec<Prompt>,
}

/// `prompts/get` request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptParams {
    /// Name of the prompt to fetch
    pub name: String,
    /// Prompt arguments (JSON object, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// `prompts/get` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    /// Optional description of the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// List of prompt messages
    pub messages: Vec<PromptMessage>,
}

/// Prompt provider: once a server registers, `prompts/list` / `prompts/get` have a data source.
///
/// When nothing is registered the methods still return `method_not_found` (an honest boundary, no pretending to support it).
#[async_trait]
pub trait PromptProvider: Send + Sync {
    /// Returns the full list of prompts.
    async fn list_prompts(&self) -> Result<Vec<Prompt>, MCPError>;
    /// Builds the prompt messages by name + optional arguments.
    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<&Value>,
    ) -> Result<GetPromptResult, MCPError>;
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
