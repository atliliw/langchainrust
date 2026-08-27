//! MCP protocol definitions (JSON-RPC 2.0)
//!
//! MCP is built on JSON-RPC 2.0; this module defines the request/response/error types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP protocol version (the version this library currently implements, sent as the requested version at `initialize`).
pub const MCP_VERSION: &str = "2024-11-05";

/// Protocol versions supported by this library (P2-10).
///
/// Recognized in order during the handshake; the first entry is the currently implemented version. New versions
/// are appended as the protocol evolves while old ones are kept for compatibility with older servers (degradation);
/// versions not in the list are handled by [`VersionPolicy`] — degrade or reject.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &[MCP_VERSION];

/// Protocol version negotiation policy (P2-10): what to do when a server declares a version outside the support list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VersionPolicy {
    /// Degrade to the library's implemented version and keep going (compatible with servers declaring a newer/older protocol).
    #[default]
    Degrade,
    /// Strict mode: an unsupported version fails the handshake and rejects the connection.
    Reject,
}

/// The version negotiation result of one handshake (P2-10).
///
/// Locked by the client once the handshake completes (the version is pinned after connecting); `protocol_info()` can read it at any time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolInfo {
    /// Version declared by the client in the `initialize` request.
    pub requested: String,
    /// Version declared by the server in the `initialize` response.
    pub server_version: String,
    /// The version that actually takes effect after negotiation (pinned after connecting; the library's implemented version when degraded).
    pub negotiated: String,
    /// Whether the version the server declared is inside this library's support list.
    pub supported: bool,
}

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    /// JSON-RPC version identifier (fixed `"2.0"`)
    pub jsonrpc: String,
    /// Request ID (used to match responses)
    pub id: u64,
    /// Method name
    pub method: String,
    /// Optional request parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl MCPRequest {
    /// Builds a new JSON-RPC request.
    pub fn new(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    /// JSON-RPC version identifier (fixed `"2.0"`)
    pub jsonrpc: String,
    /// Per JSON-RPC 2.0 spec, `id` is `null` when the request could not be parsed.
    pub id: Option<u64>,
    /// The result on success (`None` on error responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error info (`None` on success responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<MCPError>,
}

impl MCPResponse {
    /// Whether this is an error response
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Extracts `result` (returns the `MCPError` on error)
    pub fn into_result(self) -> Result<Value, MCPError> {
        if let Some(err) = self.error {
            return Err(err);
        }
        Ok(self.result.unwrap_or(Value::Null))
    }
}

/// JSON-RPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    /// JSON-RPC error code
    pub code: i32,
    /// Error description message
    pub message: String,
    /// Optional extra error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl MCPError {
    /// Builds a JSON-RPC error.
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Standard error: method not found
    pub fn method_not_found() -> Self {
        Self::new(-32601, "Method not found")
    }

    /// Standard error: invalid params
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::new(-32602, msg)
    }

    /// Transport connection dropped (child process exited / SSE long connection broken).
    ///
    /// Layers above receive this error and should trigger the reconnect flow and re-handshake.
    pub fn connection_lost() -> Self {
        Self::new(-32000, "MCP connection lost")
    }

    /// Whether this is a connection-dropped error.
    pub fn is_connection_lost(&self) -> bool {
        self.code == -32000
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
        assert_eq!(resp.id, Some(1));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        assert!(!resp.is_error());
    }

    #[test]
    fn test_response_deserialization_error() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: MCPResponse = serde_json::from_str(json).unwrap();
        assert!(resp.is_error());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
    }

    #[test]
    fn test_response_deserialization_null_id() {
        // JSON-RPC 2.0: parse error responses should have id: null
        let json = r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}"#;
        let resp: MCPResponse = serde_json::from_str(json).unwrap();
        assert!(resp.id.is_none());
        assert!(resp.is_error());
    }

    #[test]
    fn test_into_result_ok() {
        let resp = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(1),
            result: Some(Value::Bool(true)),
            error: None,
        };
        assert!(resp.into_result().is_ok());
    }

    #[test]
    fn test_into_result_err() {
        let resp = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(1),
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
