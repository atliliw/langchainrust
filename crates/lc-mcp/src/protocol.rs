//! MCP protocol definitions (JSON-RPC 2.0)
//!
//! MCP is built on JSON-RPC 2.0; this module defines the request/response/error types.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// MCP protocol version (the version this library currently implements, sent as the requested version at `initialize`).
pub const MCP_VERSION: &str = "2024-11-05";

/// Stateless MCP spec version (2026-07-28 fifth edition): no handshake, no
/// session ids — every request is self-contained via `_meta` (0.22.0 S2).
pub const MCP_VERSION_STATELESS: &str = "2026-07-28";

/// Protocol versions supported by this library (P2-10).
///
/// Recognized in order during the handshake; the first entry is the currently implemented version. New versions
/// are appended as the protocol evolves while old ones are kept for compatibility with older servers (degradation);
/// versions not in the list are handled by [`VersionPolicy`] — degrade or reject.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &[MCP_VERSION_STATELESS, MCP_VERSION];

/// HTTP header carrying the JSON-RPC method name on stateless POSTs
/// (2026-07-28 SEP-2243): gateways / WAFs / rate limiters can route and
/// throttle without parsing the body.
pub const MCP_METHOD_HEADER: &str = "Mcp-Method";

/// HTTP header carrying the MCP protocol namespace on stateless POSTs.
pub const MCP_NAME_HEADER: &str = "Mcp-Name";

/// JSON-RPC error code for authorization failures on the stateless track
/// (maps to HTTP 401 at the transport boundary).
pub const MCP_ERROR_UNAUTHORIZED: i32 = -32001;

/// JSON-RPC error code for a request that timed out before the peer answered
/// (no response was received; the call may or may not have run server-side).
pub const MCP_ERROR_REQUEST_TIMEOUT: i32 = -32004;

/// JSON-RPC error code for a protocol-version negotiation rejection
/// (strict policy; the server declared an unsupported `protocolVersion`).
pub const MCP_ERROR_VERSION_UNSUPPORTED: i32 = -32005;

/// JSON-RPC error code for a Streamable HTTP session the server no longer
/// recognizes (HTTP 404 / bad `Mcp-Session-Id`): the client must run a new
/// `initialize` handshake to obtain a fresh session before retrying.
pub const MCP_ERROR_SESSION_LOST: i32 = -32006;

/// Self-contained per-request metadata for the stateless track: replaces the
/// deleted `initialize` handshake / session id. Serialized as the JSON-RPC
/// `_meta` member of `params`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestMeta {
    /// Protocol version the client speaks (e.g. [`MCP_VERSION_STATELESS`]).
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Client identity (replaces the handshake `clientInfo`).
    #[serde(rename = "clientInfo")]
    pub client_info: ClientIdentity,
    /// Client capabilities the server may rely on for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Value>,
    /// Opaque MRTR continuation token (`requestState`): present only on
    /// resent requests after an `input_required` round trip.
    #[serde(rename = "requestState", skip_serializing_if = "Option::is_none")]
    pub request_state: Option<String>,
}

/// Client identity carried in [`RequestMeta`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClientIdentity {
    /// Client name (e.g. `"langchainrust-mcp-client"`).
    pub name: String,
    /// Client version.
    pub version: String,
}

impl RequestMeta {
    /// Builds the default meta for this library.
    pub fn default_for(version: impl Into<String>) -> Self {
        Self {
            protocol_version: version.into(),
            client_info: ClientIdentity {
                name: "langchainrust-mcp-client".to_string(),
                version: "0.22.0".to_string(),
            },
            capabilities: None,
            request_state: None,
        }
    }

    /// Sets the MRTR continuation token.
    pub fn with_request_state(mut self, request_state: impl Into<String>) -> Self {
        self.request_state = Some(request_state.into());
        self
    }
}

/// One question the server asks during a multi-round-trip (MRTR) exchange.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MrtrQuestion {
    /// Question id the answer must reference.
    pub id: String,
    /// What the server needs from the client.
    pub prompt: String,
}

/// The `input_required` shape a stateless server returns when it needs
/// client input before it can finish a request (MRTR).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputRequired {
    /// Opaque continuation token; echoed back on the resent request.
    #[serde(rename = "requestState")]
    pub request_state: String,
    /// What the server needs.
    pub questions: Vec<MrtrQuestion>,
}

impl InputRequired {
    /// Extracts an `input_required` payload from a successful result, if the
    /// result is one.
    pub fn from_result(result: &Value) -> Option<Self> {
        let ir = result.get("input_required")?;
        serde_json::from_value(ir.clone()).ok()
    }
}

/// The client's answers to one MRTR round, sent alongside the resent request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MrtrAnswer {
    /// Question id being answered.
    pub id: String,
    /// The collected answer.
    pub value: String,
}

/// Protocol version negotiation policy (P2-10): what to do when a server declares a version outside the support list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VersionPolicy {
    /// Degrade to the library's implemented version and keep going (compatible with servers declaring a newer/older protocol).
    #[default]
    Degrade,
    /// Strict mode: an unsupported version fails the handshake and rejects the connection.
    Reject,
}

/// Applies the version policy to the server-declared protocol version.
///
/// Shared by every handshake track (stdio, Streamable HTTP). Returns
/// `(negotiated_version, supported)`: when the server version is in
/// [`SUPPORTED_PROTOCOL_VERSIONS`] it is pinned as-is; otherwise
/// [`VersionPolicy::Degrade`] pins the library's [`MCP_VERSION`] while
/// [`VersionPolicy::Reject`] fails with [`MCP_ERROR_VERSION_UNSUPPORTED`].
pub fn negotiate_protocol_version(
    server_version: &str,
    policy: VersionPolicy,
) -> Result<(String, bool), MCPError> {
    if SUPPORTED_PROTOCOL_VERSIONS.contains(&server_version) {
        return Ok((server_version.to_string(), true));
    }
    match policy {
        VersionPolicy::Degrade => Ok((MCP_VERSION.to_string(), false)),
        VersionPolicy::Reject => Err(MCPError::new(
            MCP_ERROR_VERSION_UNSUPPORTED,
            format!(
                "MCP server negotiated unsupported protocol version '{server_version}' \
                 (supported: {SUPPORTED_PROTOCOL_VERSIONS:?})"
            ),
        )),
    }
}

/// Builds a JSON-RPC notification envelope (no `id`, no response expected).
///
/// Shared by every transport track for `notifications/initialized` and peers.
pub fn notification_message(method: &str, params: Option<Value>) -> Value {
    let mut message = json!({"jsonrpc": "2.0", "method": method});
    if let Some(params) = params {
        message["params"] = params;
    }
    message
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

/// JSON-RPC request id (B8): a request/response id may be a number or a
/// string per JSON-RPC 2.0 §4.2 (untagged, bidirectional — `Number(u64)` and
/// `String(String)` serde as themselves in both directions). Notifications
/// carry no id and do not use this type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum JsonRpcId {
    /// Numeric id.
    Number(u64),
    /// String id.
    String(String),
}

impl JsonRpcId {
    /// Returns the numeric id, if this is a [`JsonRpcId::Number`].
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonRpcId::Number(n) => Some(*n),
            JsonRpcId::String(_) => None,
        }
    }

    /// Returns the string id, if this is a [`JsonRpcId::String`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonRpcId::String(s) => Some(s),
            JsonRpcId::Number(_) => None,
        }
    }
}

impl From<u64> for JsonRpcId {
    fn from(n: u64) -> Self {
        JsonRpcId::Number(n)
    }
}

impl From<i32> for JsonRpcId {
    /// Enables `MCPRequest::new(1, …)` integer literals.
    fn from(n: i32) -> Self {
        JsonRpcId::Number(n as u64)
    }
}

impl From<usize> for JsonRpcId {
    fn from(n: usize) -> Self {
        JsonRpcId::Number(n as u64)
    }
}

impl From<String> for JsonRpcId {
    fn from(s: String) -> Self {
        JsonRpcId::String(s)
    }
}

impl From<&str> for JsonRpcId {
    fn from(s: &str) -> Self {
        JsonRpcId::String(s.to_string())
    }
}

impl std::fmt::Display for JsonRpcId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonRpcId::Number(n) => write!(f, "{n}"),
            JsonRpcId::String(s) => write!(f, "{s}"),
        }
    }
}

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    /// JSON-RPC version identifier (fixed `"2.0"`)
    pub jsonrpc: String,
    /// Request ID (used to match responses)
    pub id: JsonRpcId,
    /// Method name
    pub method: String,
    /// Optional request parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    /// Stateless-track per-request metadata, serialized as the top-level
    /// `_meta` member of the JSON-RPC message (self-contained request; no
    /// handshake / session id). `None` on the legacy handshake track.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<RequestMeta>,
}

impl MCPRequest {
    /// Builds a new JSON-RPC request. The id may be a number or a string
    /// (anything convertible to [`JsonRpcId`]).
    pub fn new(id: impl Into<JsonRpcId>, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params,
            meta: None,
        }
    }

    /// Builds a stateless-track request with self-contained `_meta`.
    pub fn new_stateless(
        id: impl Into<JsonRpcId>,
        method: impl Into<String>,
        params: Option<Value>,
        meta: RequestMeta,
    ) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params,
            meta: Some(meta),
        }
    }
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    /// JSON-RPC version identifier (fixed `"2.0"`)
    pub jsonrpc: String,
    /// Per JSON-RPC 2.0 spec, `id` is `null` when the request could not be parsed.
    pub id: Option<JsonRpcId>,
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

    /// The Streamable HTTP session is gone server-side (HTTP 404 / unknown
    /// `Mcp-Session-Id`); a fresh `initialize` handshake is required.
    pub fn session_lost() -> Self {
        Self::new(
            MCP_ERROR_SESSION_LOST,
            "MCP Streamable HTTP session lost: re-initialize required",
        )
    }

    /// Whether this is a session-lost error.
    pub fn is_session_lost(&self) -> bool {
        self.code == MCP_ERROR_SESSION_LOST
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
        assert_eq!(resp.id, Some(JsonRpcId::Number(1)));
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
            id: Some(JsonRpcId::Number(1)),
            result: Some(Value::Bool(true)),
            error: None,
        };
        assert!(resp.into_result().is_ok());
    }

    #[test]
    fn test_into_result_err() {
        let resp = MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(JsonRpcId::Number(1)),
            result: None,
            error: Some(MCPError::method_not_found()),
        };
        assert!(resp.into_result().is_err());
    }

    /// B8: request/response ids round-trip as numbers, strings, and implement
    /// Display for diagnostics (JSON-RPC 2.0 §4.2 allows either).
    #[test]
    fn json_rpc_id_number_and_string_roundtrip() {
        assert_eq!(
            serde_json::to_value(JsonRpcId::Number(7)).unwrap(),
            json!(7)
        );
        assert_eq!(
            serde_json::to_value(JsonRpcId::String("req-42".into())).unwrap(),
            json!("req-42")
        );
        let n: JsonRpcId = serde_json::from_value(json!(7)).unwrap();
        assert_eq!(n, JsonRpcId::Number(7));
        let s: JsonRpcId = serde_json::from_value(json!("req-42")).unwrap();
        assert_eq!(s, JsonRpcId::String("req-42".into()));
        assert_eq!(JsonRpcId::Number(7).to_string(), "7");
        assert_eq!(JsonRpcId::String("x".into()).to_string(), "x");
        assert_eq!(JsonRpcId::Number(7).as_u64(), Some(7));
        assert_eq!(JsonRpcId::String("x".into()).as_str(), Some("x"));
        // A mixed/non-string, non-number id must not silently deserialize.
        assert!(serde_json::from_value::<JsonRpcId>(json!([1])).is_err());
    }

    #[test]
    fn test_error_display() {
        let err = MCPError::new(-1, "boom");
        assert_eq!(format!("{}", err), "MCP Error [-1]: boom");
    }

    #[test]
    fn negotiate_supported_version_is_pinned() {
        let (negotiated, supported) =
            negotiate_protocol_version(MCP_VERSION, VersionPolicy::Reject).unwrap();
        assert_eq!(negotiated, MCP_VERSION);
        assert!(supported);
    }

    #[test]
    fn negotiate_unknown_version_degrades() {
        let (negotiated, supported) =
            negotiate_protocol_version("1999-01-01", VersionPolicy::Degrade).unwrap();
        assert_eq!(negotiated, MCP_VERSION);
        assert!(!supported);
    }

    #[test]
    fn negotiate_unknown_version_rejects() {
        let err = negotiate_protocol_version("1999-01-01", VersionPolicy::Reject).unwrap_err();
        assert_eq!(err.code, MCP_ERROR_VERSION_UNSUPPORTED);
    }

    #[test]
    fn session_lost_error_class() {
        assert!(MCPError::session_lost().is_session_lost());
        assert!(!MCPError::connection_lost().is_session_lost());
    }

    #[test]
    fn notification_envelope_shape() {
        let bare = notification_message("notifications/initialized", None);
        assert_eq!(bare["jsonrpc"], "2.0");
        assert_eq!(bare["method"], "notifications/initialized");
        assert!(bare.get("id").is_none());
        assert!(bare.get("params").is_none());

        let with_params = notification_message("notifications/x", Some(json!({"a": 1})));
        assert_eq!(with_params["params"]["a"], 1);
    }
}
