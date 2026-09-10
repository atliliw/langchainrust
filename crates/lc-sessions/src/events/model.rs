//! Event data model: append-only session events (0.22.0 S4.1).
//!
//! The atomic unit is a *turn*: one user message plus everything the assistant
//! did after it (messages, tool calls, tool results). Events are immutable —
//! compaction appends a [`EventPayload::Snapshot`] instead of rewriting history.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An immutable, append-only session event.
///
/// `id` is monotonically increasing *within* a `(session_id, branch)` key —
/// replay order is id order. `turn_index` groups events into turns. `branch`
/// enables forks: the main line is `"main"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionEvent {
    /// Monotonically increasing sequence number within `(session_id, branch)`.
    pub id: u64,
    /// Owning session id.
    pub session_id: String,
    /// Atomic turn index this event belongs to (0-based).
    pub turn_index: u64,
    /// Wall-clock timestamp (Unix milliseconds).
    pub ts: i64,
    /// Branch tag; `"main"` is the trunk, forks write to new branch names.
    pub branch: String,
    /// The event payload.
    pub payload: EventPayload,
}

impl SessionEvent {
    /// Creates an event stamped with the current time.
    pub fn now(
        id: u64,
        session_id: impl Into<String>,
        turn_index: u64,
        branch: impl Into<String>,
        payload: EventPayload,
    ) -> Self {
        Self {
            id,
            session_id: session_id.into(),
            turn_index,
            ts: Utc::now().timestamp_millis(),
            branch: branch.into(),
            payload,
        }
    }

    /// Timestamp as a chrono UTC datetime.
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.ts).unwrap_or_default()
    }
}

/// The payload of a session event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventPayload {
    /// A user message (opens a turn).
    UserMessage {
        /// Message text.
        content: String,
    },
    /// An assistant reply.
    AssistantMessage {
        /// Reply text.
        content: String,
    },
    /// A tool invocation (paired with its result via `call_id`).
    ToolCall {
        /// Tool name.
        tool: String,
        /// Tool input as JSON (string inputs are wrapped as `{"value": ...}`).
        tool_input: Value,
        /// Pairing key for the matching [`EventPayload::ToolResult`].
        call_id: String,
    },
    /// A tool observation (paired with its call via `call_id`).
    ToolResult {
        /// Pairing key of the originating [`EventPayload::ToolCall`].
        call_id: String,
        /// Observation text.
        observation: String,
    },
    /// A compaction/consolidation product: a synthetic summary standing in for
    /// dropped turns. Appended — never rewrites history.
    Snapshot {
        /// Synthetic summary of the compacted prefix.
        summary: String,
        /// Number of original turns represented by this snapshot.
        compacted_turns: u64,
    },
    /// Free-form session metadata.
    Metadata {
        /// Metadata key.
        key: String,
        /// Metadata value.
        value: Value,
    },
}

/// A turn: the events of one `turn_index`, in id order. A turn is complete
/// when its `ToolResult`s all pair with a `ToolCall` in the same turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// Turn index (== `turn_index` of every contained event).
    pub index: u64,
    /// Events of this turn, id-ascending.
    pub events: Vec<SessionEvent>,
}

impl Turn {
    /// Returns the user message that opened this turn, if any.
    pub fn user_message(&self) -> Option<&str> {
        self.events.iter().find_map(|e| match &e.payload {
            EventPayload::UserMessage { content } => Some(content.as_str()),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_now_stamps_time_and_fields() {
        let e = SessionEvent::now(
            1,
            "s1",
            0,
            "main",
            EventPayload::UserMessage {
                content: "hi".into(),
            },
        );
        assert_eq!(e.id, 1);
        assert_eq!(e.session_id, "s1");
        assert_eq!(e.turn_index, 0);
        assert_eq!(e.branch, "main");
        assert!(e.ts > 0);
    }

    #[test]
    fn payload_serde_tagged_roundtrip() {
        let e = EventPayload::ToolCall {
            tool: "search".into(),
            tool_input: serde_json::json!({"q": "rust"}),
            call_id: "c1".into(),
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"type\":\"tool_call\""), "{json}");
        let back: EventPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn snapshot_payload_roundtrip() {
        let e = EventPayload::Snapshot {
            summary: "earlier turns".into(),
            compacted_turns: 3,
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: EventPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn turn_user_message_finds_opener() {
        let t = Turn {
            index: 0,
            events: vec![
                SessionEvent::now(
                    0,
                    "s",
                    0,
                    "main",
                    EventPayload::UserMessage {
                        content: "q".into(),
                    },
                ),
                SessionEvent::now(
                    1,
                    "s",
                    0,
                    "main",
                    EventPayload::AssistantMessage {
                        content: "a".into(),
                    },
                ),
            ],
        };
        assert_eq!(t.user_message(), Some("q"));
    }
}
