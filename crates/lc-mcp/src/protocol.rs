//! MCP 协议定义(JSON-RPC 2.0)
//!
//! MCP 基于 JSON-RPC 2.0,本模块定义请求/响应/错误类型。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP 协议版本(本库当前实现的版本,`initialize` 时作为请求版本)。
pub const MCP_VERSION: &str = "2024-11-05";

/// 本库支持的协议版本列表(P2-10)。
///
/// 握手时按序识别;列表首项为当前实现版本。随协议演进追加新版本,
/// 保留旧版本以便与旧 Server 兼容(降级);不在列表内的版本由
/// [`VersionPolicy`] 决定降级还是拒绝。
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &[MCP_VERSION];

/// 协议版本协商策略(P2-10):Server 声明版本不在支持列表时怎么处理。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VersionPolicy {
    /// 降级到本库实现版本继续用(兼容声明了新 / 旧协议的 Server)。
    #[default]
    Degrade,
    /// 严格模式:版本不受支持则握手失败、拒绝连接。
    Reject,
}

/// 一次握手的版本协商结果(P2-10)。
///
/// 握手完成后由客户端锁定(连接后锁版本),`protocol_info()` 可随时读取。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolInfo {
    /// 客户端在 `initialize` 请求中声明的版本。
    pub requested: String,
    /// Server 在 `initialize` 响应中声明的版本。
    pub server_version: String,
    /// 实际协商生效的版本(连接后锁定;降级时为本库实现版本)。
    pub negotiated: String,
    /// Server 声明版本是否在本库支持列表内。
    pub supported: bool,
}

/// JSON-RPC 请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    /// JSON-RPC 版本标识(固定 `"2.0"`)
    pub jsonrpc: String,
    /// 请求 ID(用于匹配响应)
    pub id: u64,
    /// 方法名
    pub method: String,
    /// 可选的请求参数
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl MCPRequest {
    /// 构造新的 JSON-RPC 请求。
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
    /// JSON-RPC 版本标识(固定 `"2.0"`)
    pub jsonrpc: String,
    /// Per JSON-RPC 2.0 spec, `id` is `null` when the request could not be parsed.
    pub id: Option<u64>,
    /// 成功时的结果(错误响应为 `None`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// 错误信息(成功响应为 `None`)
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
    /// JSON-RPC 错误码
    pub code: i32,
    /// 错误描述消息
    pub message: String,
    /// 可选的附加错误数据
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl MCPError {
    /// 构造 JSON-RPC 错误。
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

    /// 传输层连接断开(子进程退出 / SSE 长连接中断)。
    ///
    /// 上层收到此错误后应触发重连流程并重新握手。
    pub fn connection_lost() -> Self {
        Self::new(-32000, "MCP connection lost")
    }

    /// 是否为连接断开错误。
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
