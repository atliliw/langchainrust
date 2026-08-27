//! MCP Elicitation - interaction-request types and `elicitation/create` handling
//!
//! MCP Elicitation lets a Server request information from the user (confirmation, input, etc.),
//! displaying a form to the user through the Host's UI and collecting the response.

use crate::protocol::MCPError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `elicitation/create` request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitationRequest {
    /// Message text shown to the user
    pub message: String,
    /// Optional JSON Schema form the user is expected to fill in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
}

/// `elicitation/create` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitationResponse {
    /// The action the user took in response
    pub action: ElicitationAction,
    /// Content the user submitted (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
}

/// User response action
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ElicitationAction {
    /// The user accepted the request
    Accept,
    /// The user rejected the request
    Decline,
    /// The user cancelled the request
    Cancel,
}

/// Interaction-request handler (server→host direction).
///
/// Per MCP semantics, `elicitation/create` is initiated by the Server and the Host collects user
/// input through its UI. The framework layer doesn't attach to a specific transport; the host injects
/// this callback, which is responsible for delivering the request to the Host and fetching the response.
/// When not injected, [`crate::MCPServer::create_elicitation`] returns a clear error.
#[async_trait]
pub trait ElicitationHandler: Send + Sync {
    /// Initiates an interaction request, returning the user's response.
    async fn create(&self, request: &ElicitationRequest) -> Result<ElicitationResponse, MCPError>;
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
