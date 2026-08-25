//! MCP Elicitation - 交互请求类型与 `elicitation/create` 处理
//!
//! MCP Elicitation 允许 Server 向用户请求信息(如确认、输入等),
//! 通过 Host 的 UI 向用户展示表单并收集响应。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `elicitation/create` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitationRequest {
    /// 展示给用户的消息文本
    pub message: String,
    /// 期望用户填写的可选 JSON Schema 表单
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
}

/// `elicitation/create` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitationResponse {
    /// 用户做出的响应动作
    pub action: ElicitationAction,
    /// 用户提交的响应内容(可选)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
}

/// 用户响应动作
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ElicitationAction {
    /// 用户接受请求
    Accept,
    /// 用户拒绝请求
    Decline,
    /// 用户取消请求
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
