//! MCP Elicitation - 交互请求类型与 `elicitation/create` 处理
//!
//! MCP Elicitation 允许 Server 向用户请求信息(如确认、输入等),
//! 通过 Host 的 UI 向用户展示表单并收集响应。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `elicitation/create` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitationRequest {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
}

/// `elicitation/create` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitationResponse {
    pub action: ElicitationAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
}

/// 用户响应动作
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ElicitationAction {
    Accept,
    Decline,
    Cancel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_elicitation_request_serialization() {
        let req = ElicitationRequest {
            message: "Do you want to proceed?".to_string(),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "confirm": { "type": "boolean" }
                }
            })),
        };
        let json_str = serde_json::to_string(&req).unwrap();
        assert!(json_str.contains("\"message\":\"Do you want to proceed?\""));
        assert!(json_str.contains("\"schema\""));
    }

    #[test]
    fn test_elicitation_request_no_schema() {
        let req = ElicitationRequest {
            message: "Simple question".to_string(),
            schema: None,
        };
        let json_str = serde_json::to_string(&req).unwrap();
        assert!(!json_str.contains("schema"));
    }

    #[test]
    fn test_elicitation_response_accept() {
        let resp = ElicitationResponse {
            action: ElicitationAction::Accept,
            content: Some(json!({"confirm": true})),
        };
        let json_str = serde_json::to_string(&resp).unwrap();
        assert!(json_str.contains("\"action\":\"accept\""));
        assert!(json_str.contains("\"content\""));
    }

    #[test]
    fn test_elicitation_response_decline() {
        let resp = ElicitationResponse {
            action: ElicitationAction::Decline,
            content: None,
        };
        let json_str = serde_json::to_string(&resp).unwrap();
        assert!(json_str.contains("\"action\":\"decline\""));
        assert!(!json_str.contains("content"));
    }

    #[test]
    fn test_elicitation_response_cancel() {
        let resp = ElicitationResponse {
            action: ElicitationAction::Cancel,
            content: None,
        };
        let json_str = serde_json::to_string(&resp).unwrap();
        assert!(json_str.contains("\"action\":\"cancel\""));
    }

    #[test]
    fn test_elicitation_response_deserialization() {
        let json = r#"{"action":"accept","content":{"confirm":true}}"#;
        let resp: ElicitationResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(resp.action, ElicitationAction::Accept));
        assert!(resp.content.is_some());
    }
}
