//! MCP Sampling - 采样类型与 `sampling/createMessage` 处理
//!
//! MCP Sampling 允许 Server 请求 Host(即 LLM Client)执行 LLM 推理,
//! Server 可借此利用 Host 的模型能力完成子任务。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 采样消息内容(内联枚举)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SamplingContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
}

/// 采样消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingMessage {
    pub role: SamplingRole,
    pub content: SamplingContent,
}

/// 消息角色
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplingRole {
    User,
    Assistant,
}

/// 模型偏好提示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreferences {
    #[serde(rename = "costPriority", skip_serializing_if = "Option::is_none")]
    pub cost_priority: Option<f64>,
    #[serde(rename = "speedPriority", skip_serializing_if = "Option::is_none")]
    pub speed_priority: Option<f64>,
    #[serde(rename = "intelligencePriority", skip_serializing_if = "Option::is_none")]
    pub intelligence_priority: Option<f64>,
    #[serde(rename = "hints", skip_serializing_if = "Option::is_none")]
    pub hints: Option<Vec<ModelHint>>,
}

/// 模型提示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// `sampling/createMessage` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingRequest {
    pub messages: Vec<SamplingMessage>,
    #[serde(rename = "maxTokens")]
    pub max_tokens: usize,
    #[serde(rename = "systemPrompt", skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(rename = "modelPreferences", skip_serializing_if = "Option::is_none")]
    pub model_preferences: Option<ModelPreferences>,
    #[serde(rename = "temperature", skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(rename = "includeContext", skip_serializing_if = "Option::is_none")]
    pub include_context: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// `sampling/createMessage` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingResult {
    pub role: SamplingRole,
    pub content: SamplingContent,
    #[serde(rename = "model", skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_content_text() {
        let content = SamplingContent::Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_sampling_content_image() {
        let content = SamplingContent::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"image\""));
    }

    #[test]
    fn test_sampling_message_user() {
        let msg = SamplingMessage {
            role: SamplingRole::User,
            content: SamplingContent::Text {
                text: "What is Rust?".to_string(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
    }

    #[test]
    fn test_sampling_message_assistant() {
        let msg = SamplingMessage {
            role: SamplingRole::Assistant,
            content: SamplingContent::Text {
                text: "Rust is a systems language".to_string(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
    }

    #[test]
    fn test_sampling_request_minimal() {
        let req = SamplingRequest {
            messages: vec![SamplingMessage {
                role: SamplingRole::User,
                content: SamplingContent::Text {
                    text: "Hello".to_string(),
                },
            }],
            max_tokens: 100,
            system_prompt: None,
            model_preferences: None,
            temperature: None,
            stop_sequences: None,
            include_context: None,
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"maxTokens\":100"));
        assert!(!json.contains("systemPrompt"));
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn test_sampling_request_full() {
        let req = SamplingRequest {
            messages: vec![SamplingMessage {
                role: SamplingRole::User,
                content: SamplingContent::Text {
                    text: "Hello".to_string(),
                },
            }],
            max_tokens: 100,
            system_prompt: Some("You are helpful".to_string()),
            model_preferences: Some(ModelPreferences {
                cost_priority: Some(0.5),
                speed_priority: None,
                intelligence_priority: None,
                hints: Some(vec![ModelHint {
                    name: Some("claude-3".to_string()),
                }]),
            }),
            temperature: Some(0.7),
            stop_sequences: Some(vec!["\n".to_string()]),
            include_context: None,
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"systemPrompt\":\"You are helpful\""));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"costPriority\":0.5"));
    }

    #[test]
    fn test_sampling_result_deserialization() {
        let json = r#"{"role":"assistant","content":{"type":"text","text":"Hi there"},"model":"claude-3","stopReason":"endTurn"}"#;
        let result: SamplingResult = serde_json::from_str(json).unwrap();
        assert!(matches!(result.role, SamplingRole::Assistant));
        assert_eq!(result.model.as_deref(), Some("claude-3"));
        assert_eq!(result.stop_reason.as_deref(), Some("endTurn"));
    }

    #[test]
    fn test_model_preferences_serialization() {
        let prefs = ModelPreferences {
            cost_priority: Some(0.3),
            speed_priority: Some(0.7),
            intelligence_priority: None,
            hints: None,
        };
        let json = serde_json::to_string(&prefs).unwrap();
        assert!(json.contains("\"costPriority\":0.3"));
        assert!(json.contains("\"speedPriority\":0.7"));
        assert!(!json.contains("intelligencePriority"));
    }
}
