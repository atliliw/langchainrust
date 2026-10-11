//! MCP `logging` capability (0.27.0 #79 item ④) — the low-priority half of
//! "logging / resource-subscribed 通知".
//!
//! Adds the two protocol shapes the spec defines for telemetry:
//! - client→server [`LOGGING_SETLEVEL`] (`logging/setLevel`): the client asks
//!   the server to emit only messages at or above a [`LogLevel`] threshold, for
//!   an optional named logger.
//! - server→client [`NOTIFICATIONS_MESSAGE`] (`notifications/message`): one
//!   structured log/telemetry frame carrying `level`, `logger`, and `data`.
//!
//! A [`LogSink`] (registered on [`crate::MCPServer`] via `with_log_sink`)
//! physically delivers the envelope — the same split as [`crate::ResourcePushSink`]
//! for resource notifications: this module is pure type/state logic (the level
//! model and the envelope shape), fully unit-testable offline.

use crate::protocol::{notification_message, MCPError};
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};

/// `logging/setLevel` JSON-RPC method name (client→server).
pub const LOGGING_SETLEVEL: &str = "logging/setLevel";
/// `notifications/message` notification method name (server→client).
pub const NOTIFICATIONS_MESSAGE: &str = "notifications/message";

/// MCP logging severity, ordered low→high (0..=8, syslog-like). A server emits
/// a frame when its configured `logging/setLevel` threshold is `<=` the frame's
/// level (a lower threshold passes more). `None` threshold = no filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    /// Most verbose: fine-grained diagnostic detail.
    Trace = 0,
    /// Normal diagnostic messages for the developer/operator.
    Debug = 1,
    /// Generally useful operational messages.
    Info = 2,
    /// Notable events that may need attention.
    Notice = 3,
    /// Unexpected but recoverable conditions.
    Warning = 4,
    /// A condition that produced an error result.
    Error = 5,
    /// A serious failure within the process.
    Critical = 6,
    /// A condition requiring immediate operator attention.
    Alert = 7,
    /// The process cannot continue; an emergency condition.
    Emergency = 8,
}

impl LogLevel {
    /// Resolves a numeric severity. Returns `None` outside 0..=8.
    pub fn from_u8(n: u8) -> Option<Self> {
        Some(match n {
            0 => Self::Trace,
            1 => Self::Debug,
            2 => Self::Info,
            3 => Self::Notice,
            4 => Self::Warning,
            5 => Self::Error,
            6 => Self::Critical,
            7 => Self::Alert,
            8 => Self::Emergency,
            _ => return None,
        })
    }

    /// Resolves the spec's lowercase level name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "trace" => Self::Trace,
            "debug" => Self::Debug,
            "info" => Self::Info,
            "notice" => Self::Notice,
            "warning" => Self::Warning,
            "error" => Self::Error,
            "critical" => Self::Critical,
            "alert" => Self::Alert,
            "emergency" => Self::Emergency,
            _ => return None,
        })
    }

    /// Canonical lowercase spec name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Notice => "notice",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
            Self::Alert => "alert",
            Self::Emergency => "emergency",
        }
    }

    /// Whether a frame at `self` clears the `threshold` filter (`None` = no
    /// filter → always passes).
    pub fn passes(self, threshold: Option<LogLevel>) -> bool {
        match threshold {
            None => true,
            Some(t) => self >= t,
        }
    }
}

/// `logging/setLevel` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetLevelParams {
    /// Minimum severity the server should emit. Accepts a name (`"debug"`) or
    /// an integer (`1`).
    pub level: LogLevel,
    /// Optional per-logger override (when omitted, applies to the default
    /// unnamed logger).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logger_name: Option<String>,
}

/// Receives a `notifications/message` envelope for the server's log sink.
///
/// The transport implements this to physically deliver the envelope (SSE
/// write, WebSocket frame, channel send, …). Delivery failures are the sink's
/// choice; the server treats `emit` as best-effort.
pub trait LogSink: Send + Sync {
    /// Deliver one `notifications/message` envelope.
    fn emit(&self, message: &Value);
}

/// Builds the `notifications/message` envelope for `level`/`data`. `logger` is
/// an optional namespace the sink can use for routing.
pub fn message_envelope(level: LogLevel, logger: Option<&str>, data: Value) -> Value {
    let mut params = json!({ "level": level, "data": data });
    if let Some(logger) = logger {
        params["logger"] = Value::String(logger.to_string());
    }
    notification_message(NOTIFICATIONS_MESSAGE, Some(params))
}

/// `logging/setLevel` coerces its `level` from either the string name or the
/// integer severity on the wire; `MessageLevel`-style clients in the official
/// SDK send the number, but both must land on the same `LogLevel`.
impl Serialize for LogLevel {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u8(*self as u8)
    }
}
impl<'de> Deserialize<'de> for LogLevel {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl de::Visitor<'_> for V {
            type Value = LogLevel;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a logging level name (`warning`, `error`, …) or integer 0..=8")
            }
            fn visit_u64<E: de::Error>(self, n: u64) -> Result<LogLevel, E> {
                LogLevel::from_u8(n as u8)
                    .ok_or_else(|| E::custom(format!("invalid log level number {n}")))
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<LogLevel, E> {
                LogLevel::from_name(s).ok_or_else(|| E::custom(format!("invalid log level '{s}'")))
            }
        }
        d.deserialize_any(V)
    }
}

/// Convenience so a log level mismatch surfaces as a typed error instead of a
/// silent skip (used by tests and client callers that want strictness).
pub fn parse_level(value: &Value) -> Result<LogLevel, MCPError> {
    serde_json::from_value(value.clone())
        .map_err(|e| MCPError::invalid_params(format!("invalid logging level: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_constants() {
        assert_eq!(LOGGING_SETLEVEL, "logging/setLevel");
        assert_eq!(NOTIFICATIONS_MESSAGE, "notifications/message");
    }

    #[test]
    fn level_names_roundtrip() {
        assert_eq!(LogLevel::from_name("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_name("emergency"), Some(LogLevel::Emergency));
        assert_eq!(LogLevel::from_name("bogus"), None);
        assert_eq!(LogLevel::from_u8(5), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_u8(99), None);
        assert_eq!(LogLevel::Info.as_str(), "info");
    }

    #[test]
    fn level_ordering_for_threshold_filter() {
        // A `warning` threshold passes warning/error/critical… and drops info.
        assert!(LogLevel::Warning.passes(Some(LogLevel::Warning)));
        assert!(LogLevel::Error.passes(Some(LogLevel::Warning)));
        assert!(!LogLevel::Info.passes(Some(LogLevel::Warning)));
        // No threshold = all levels pass.
        assert!(LogLevel::Trace.passes(None));
    }

    #[test]
    fn level_deserializes_from_name_or_number() {
        let num: LogLevel = serde_json::from_str("4").unwrap();
        assert_eq!(num, LogLevel::Warning);
        let name: LogLevel = serde_json::from_str("\"error\"").unwrap();
        assert_eq!(name, LogLevel::Error);
        assert!(serde_json::from_str::<LogLevel>("9").is_err());
        assert!(serde_json::from_str::<LogLevel>("\"nope\"").is_err());
    }

    #[test]
    fn setlevel_params_roundtrip() {
        let p = SetLevelParams {
            level: LogLevel::Warning,
            logger_name: Some("tool".to_string()),
        };
        let json = serde_json::to_value(&p).unwrap();
        assert_eq!(json["level"], 4, "level serializes as its integer severity");
        assert_eq!(json["logger_name"], "tool");
    }

    #[test]
    fn message_envelope_shape() {
        let e = message_envelope(LogLevel::Error, Some("tools"), json!({"detail": "boom"}));
        assert_eq!(e["method"], NOTIFICATIONS_MESSAGE);
        assert_eq!(e["params"]["level"], 5);
        assert_eq!(e["params"]["logger"], "tools");
        assert_eq!(e["params"]["data"]["detail"], "boom");
        assert!(e.get("id").is_none(), "notifications carry no id");
    }
}