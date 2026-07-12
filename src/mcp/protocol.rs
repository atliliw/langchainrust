//! MCP 协议定义(JSON-RPC 2.0)
//!
//! MCP 基于 JSON-RPC 2.0,本模块定义请求/响应/错误类型。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP 协议版本
pub const MCP_VERSION: &str = "2024-11-05";

/// JSON-RPC 请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl MCPRequest {
    pub fn new(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<MCPError>,
}

impl MCPResponse {
    /// 是否为错误响应
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// 取 result(错误时返回 MCPError)
    pub fn into_result(self) -> Result<Value, MCPError> {
        if let Some(err) = self.error {
            return Err(err);
        }
        Ok(self.result.unwrap_or(Value::Null))
    }
}

/// JSON-RPC 错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl MCPError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// 标准错误:方法不存在
    pub fn method_not_found() -> Self {
        Self::new(-32601, "Method not found")
    }

    /// 标准错误:无效参数
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::new(-32602, msg)
    }
}

impl std::fmt::Display for MCPError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MCP Error [{}]: {}", self.code, self.message)
    }
}

impl std::error::Error for MCPError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization_skips_none_params() {
        let req = MCPRequest::new(1, "tools/list", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"method\":\"tools/list\""));
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(!json.contains("params"));
    }

    #[test]
    fn test_request_with_params() {
        let params = serde_json::json!({"name": "test"});
        let req = MCPRequest::new(2, "tools/call", Some(params));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("params"));
        assert!(json.contains("\"name\":\"test\""));
    }

    #[test]
    fn test_response_deserialization_success() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let resp: MCPResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        assert!(!resp.is_error());
    }

    #[test]
    fn test_response_deserialization_error() {
        let json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: MCPResponse = serde_json::from_str(json).unwrap();
        assert!(resp.is_error());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
    }

    #[test]
    fn test_into_result_ok() {
        let resp = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: 1,
            result: Some(Value::Bool(true)),
            error: None,
        };
        assert!(resp.into_result().is_ok());
    }

    #[test]
    fn test_into_result_err() {
        let resp = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: 1,
            result: None,
            error: Some(MCPError::method_not_found()),
        };
        assert!(resp.into_result().is_err());
    }

    #[test]
    fn test_error_display() {
        let err = MCPError::new(-1, "boom");
        assert_eq!(format!("{}", err), "MCP Error [-1]: boom");
    }
}
