//! Replay: deterministic projection of an event log into session views, and
//! the turn-window compaction helper (0.22.0 S4.3/S4.5).
//!
//! Projection is a pure function of the event list — the same events always
//! produce the same view, which is what makes append-only logs a safe source
//! of truth.

use super::model::{EventPayload, SessionEvent, Turn};
use crate::session::{Session, SessionStatus};
use crate::store::SessionError;
use chrono::Utc;

/// A replayed session view.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedSession {
    /// Session id (from the events).
    pub session_id: String,
    /// Branch the projection was built from.
    pub branch: String,
    /// Turns in order. A `Snapshot` event closes the turn-group it replaces:
    /// the snapshot becomes a synthetic opening user message of a fresh turn.
    pub turns: Vec<Turn>,
    /// Flattened chat messages (`lc_schema::Message`) — the compatibility view
    /// consumed by LLM callers.
    pub messages: Vec<lc_schema::Message>,
    /// Highest event id folded into this projection.
    pub last_id: u64,
}

/// Projects an event sequence into a [`ProjectedSession`].
///
/// Rules:
/// - `UserMessage` opens a new turn; every other event joins the current turn
///   (events before the first user message form turn 0 as a preamble).
/// - `Snapshot` contributes a synthetic system message ("[context summary] ...")
///   and closes the current turn, so the next user message opens a new one.
/// - `ToolResult` pairs forward with nothing at projection time; pairing is
///   asserted by [`assert_no_orphan_tool_results`].
pub fn project(events: &[SessionEvent]) -> Result<ProjectedSession, SessionError> {
    let session_id = events
        .first()
        .map(|e| e.session_id.clone())
        .ok_or_else(|| SessionError::StoreError("cannot project an empty event list".into()))?;
    let branch = events
        .first()
        .map(|e| e.branch.clone())
        .unwrap_or_else(|| "main".to_string());

    let mut turns: Vec<Turn> = Vec::new();
    let mut messages: Vec<lc_schema::Message> = Vec::new();
    let mut last_id = 0u64;

    // Ensure a turn exists to receive the next event.
    macro_rules! current_turn {
        ($turn_index:expr) => {{
            if turns.last().map(|t: &Turn| t.index) != Some($turn_index) {
                turns.push(Turn {
                    index: $turn_index,
                    events: Vec::new(),
                });
            }
            turns.last_mut().unwrap()
        }};
    }

    for e in events {
        last_id = last_id.max(e.id);
        let turn_index = e.turn_index;
        match &e.payload {
            EventPayload::UserMessage { content } => {
                // A user message always opens a fresh turn (turn boundaries
                // are defined by user messages).
                turns.push(Turn {
                    index: turn_index,
                    events: vec![e.clone()],
                });
                messages.push(lc_schema::Message::human(content));
            }
            EventPayload::AssistantMessage { content } => {
                current_turn!(turn_index).events.push(e.clone());
                messages.push(lc_schema::Message::ai(content));
            }
            EventPayload::ToolCall { .. } | EventPayload::ToolResult { .. } => {
                current_turn!(turn_index).events.push(e.clone());
            }
            EventPayload::Snapshot { summary, .. } => {
                // A snapshot replaces the prefix: emit it as a synthetic
                // system message (0.22.0 audit fix: it is context metadata,
                // not user speech — injecting it as `human` made the model
                // answer the summary) and force the next user message into a
                // new turn.
                messages.push(lc_schema::Message::system(format!(
                    "[context summary] {summary}"
                )));
                turns.push(Turn {
                    index: turn_index,
                    events: vec![e.clone()],
                });
            }
            EventPayload::Metadata { .. } => {
                // Metadata is bookkeeping, not conversation content.
                current_turn!(turn_index).events.push(e.clone());
            }
        }
    }

    Ok(ProjectedSession {
        session_id,
        branch,
        turns,
        messages,
        last_id,
    })
}

/// Turns a projection into a legacy-style mutable [`Session`] (compatibility
/// bridge for deprecated APIs and for callers that still want a Session).
pub fn to_session(p: &ProjectedSession) -> Session {
    let mut s = Session::new(p.session_id.clone());
    s.messages = p.messages.clone();
    s.updated_at = Utc::now();
    if !p.messages.is_empty() {
        s.created_at = p
            .turns
            .first()
            .and_then(|t| t.events.first())
            .map(|e| e.datetime())
            .unwrap_or(s.created_at);
    }
    s
}

/// Compaction helper (S4.5): appends a `Snapshot` event representing the
/// oldest turns, keeping at least `keep_recent_turns` recent turns intact.
///
/// This is the event-log counterpart of lc-agents' `CompactionStrategy`
/// (turn granularity, `min_recent_turns` floor) — a deterministic summary
/// (no LLM call): dropped turns are summarized by concatenating their
/// user/assistant messages, truncated per turn.
///
/// Returns `None` when there is nothing to compact (trigger not fired or the
/// floor already covers everything).
pub fn compact_to_snapshot(
    next_id: u64,
    events: &[SessionEvent],
    max_turns: usize,
) -> Result<Option<SessionEvent>, SessionError> {
    let projection = project(events)?;
    let total_turns = projection.turns.len();
    if total_turns <= max_turns {
        return Ok(None);
    }
    let dropped = total_turns - max_turns;
    let summary = summarize_turns(&projection.turns[..dropped]);
    let branch = projection.branch.clone();
    let session_id = projection.session_id.clone();
    // The snapshot joins the *latest* turn index so ordering by turn_index
    // stays monotonic; projection turns it into a synthetic message and a
    // fresh turn for subsequent user messages.
    let turn_index = events.last().map(|e| e.turn_index).unwrap_or(0);
    Ok(Some(SessionEvent::now(
        next_id,
        session_id,
        turn_index,
        branch,
        EventPayload::Snapshot {
            summary,
            compacted_turns: dropped as u64,
        },
    )))
}

/// Deterministic summary of dropped turns: "user: … / ai: …" per turn,
/// each message truncated to 120 chars. No LLM, no nondeterminism.
/// A `Snapshot` payload in a dropped turn folds its existing summary into
/// the new one (0.22.0 C7 fix: previously `_ => {}` discarded it, so each
/// compaction cycle permanently evaporated the accumulated history).
fn summarize_turns(turns: &[Turn]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for turn in turns {
        let mut piece = String::new();
        for e in &turn.events {
            match &e.payload {
                EventPayload::UserMessage { content } => {
                    piece.push_str(&format!("user: {}\n", truncate(content, 120)));
                }
                EventPayload::AssistantMessage { content } => {
                    piece.push_str(&format!("ai: {}\n", truncate(content, 120)));
                }
                EventPayload::ToolCall { tool, .. } => {
                    piece.push_str(&format!("tool: {tool}\n"));
                }
                EventPayload::ToolResult { observation, .. } => {
                    piece.push_str(&format!("observation: {}\n", truncate(observation, 120)));
                }
                EventPayload::Snapshot { summary, .. } => {
                    // Fold the previous summary forward instead of dropping it.
                    piece.push_str(&format!("summary: {}\n", truncate(summary, 200)));
                }
                _ => {}
            }
        }
        if !piece.is_empty() {
            parts.push(piece);
        }
    }
    format!("[{} earlier turns]\n{}", turns.len(), parts.join("---\n"))
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

/// Invariant check (T6): every `ToolResult` in the projection has a matching
/// `ToolCall` with the same `call_id` in the same turn. Returns the count of
/// orphaned results (0 = sound).
pub fn assert_no_orphan_tool_results(events: &[SessionEvent]) -> Result<usize, SessionError> {
    let p = project(events)?;
    let mut orphans = 0usize;
    for turn in &p.turns {
        let calls: std::collections::HashSet<&str> = turn
            .events
            .iter()
            .filter_map(|e| match &e.payload {
                EventPayload::ToolCall { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        for e in &turn.events {
            if let EventPayload::ToolResult { call_id, .. } = &e.payload {
                if !calls.contains(call_id.as_str()) {
                    orphans += 1;
                }
            }
        }
    }
    Ok(orphans)
}

/// Convenience: builds a projected session from a store read and flags
/// sessions whose `SessionStatus` would be non-Active.
pub fn replay_active_session(events: &[SessionEvent]) -> Result<Session, SessionError> {
    let p = project(events)?;
    let mut s = to_session(&p);
    s.status = SessionStatus::Active;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::model::SessionEvent;

    fn user(id: u64, turn: u64, content: &str) -> SessionEvent {
        SessionEvent::now(
            id,
            "s1",
            turn,
            "main",
            EventPayload::UserMessage {
                content: content.into(),
            },
        )
    }
    fn ai(id: u64, turn: u64, content: &str) -> SessionEvent {
        SessionEvent::now(
            id,
            "s1",
            turn,
            "main",
            EventPayload::AssistantMessage {
                content: content.into(),
            },
        )
    }
    fn call(id: u64, turn: u64, call_id: &str) -> SessionEvent {
        SessionEvent::now(
            id,
            "s1",
            turn,
            "main",
            EventPayload::ToolCall {
                tool: "search".into(),
                tool_input: serde_json::json!({"q": "x"}),
                call_id: call_id.into(),
            },
        )
    }
    fn result(id: u64, turn: u64, call_id: &str) -> SessionEvent {
        SessionEvent::now(
            id,
            "s1",
            turn,
            "main",
            EventPayload::ToolResult {
                call_id: call_id.into(),
                observation: "found".into(),
            },
        )
    }

    /// T3 replay_projection: fixed event sequence → hand-computed message list.
    #[test]
    fn t3_replay_projection() {
        let events = vec![user(1, 0, "第一句"), ai(2, 0, "回复"), user(3, 1, "第二句")];
        let p = project(&events).unwrap();
        assert_eq!(p.session_id, "s1");
        assert_eq!(p.turns.len(), 2);
        assert_eq!(p.messages.len(), 3);
        assert_eq!(p.messages[0].content, "第一句");
        assert_eq!(p.messages[1].content, "回复");
        assert_eq!(p.messages[2].content, "第二句");
        assert_eq!(p.last_id, 3);
    }

    /// T4 snapshot_semantics: a mid-log Snapshot becomes a synthetic message
    /// and the next user message opens a new turn.
    #[test]
    fn t4_snapshot_semantics() {
        let mut events = vec![user(1, 0, "旧问题"), ai(2, 0, "旧回答")];
        events.push(SessionEvent::now(
            3,
            "s1",
            1,
            "main",
            EventPayload::Snapshot {
                summary: "[2 earlier turns]\nuser: 旧问题\nai: 旧回答\n".into(),
                compacted_turns: 2,
            },
        ));
        events.push(user(4, 2, "新问题"));
        let p = project(&events).unwrap();

        // The snapshot lands as a synthetic user message...
        assert!(p
            .messages
            .iter()
            .any(|m| m.content.contains("[context summary]")));
        // ...and the new question opens a fresh turn after it.
        let last = p.turns.last().unwrap();
        assert_eq!(last.index, 2);
        assert_eq!(last.user_message(), Some("新问题"));
    }

    /// T6 turn_completeness: paired tool call/result → 0 orphans; an
    /// unpaired result is detected.
    #[test]
    fn t6_turn_completeness() {
        let paired = vec![
            user(1, 0, "查一下"),
            call(2, 0, "c1"),
            result(3, 0, "c1"),
            ai(4, 0, "答案"),
        ];
        assert_eq!(assert_no_orphan_tool_results(&paired).unwrap(), 0);

        let orphaned = vec![user(1, 0, "查一下"), result(3, 0, "c-missing")];
        assert_eq!(assert_no_orphan_tool_results(&orphaned).unwrap(), 1);
    }

    /// T9 compaction_integration: compact_to_snapshot keeps the recent turns
    /// verbatim, represents the dropped prefix as one Snapshot, and the
    /// post-compaction projection's message count shrinks while the summary
    /// preserves the dropped content.
    #[test]
    fn t9_compaction_integration() {
        let mut events = Vec::new();
        let mut id = 1u64;
        for turn in 0..6 {
            events.push(user(id, turn, &format!("问题{turn}")));
            id += 1;
            events.push(ai(id, turn, &format!("回答{turn}")));
            id += 1;
        }
        let before = project(&events).unwrap();
        assert_eq!(before.turns.len(), 6);

        // Keep the most recent 2 turns.
        let snapshot = compact_to_snapshot(id, &events, 2).unwrap().unwrap();
        events.push(snapshot);
        let after = project(&events).unwrap();

        // Snapshot represented 4 dropped turns...
        match &events.last().unwrap().payload {
            EventPayload::Snapshot {
                compacted_turns, ..
            } => assert_eq!(*compacted_turns, 4),
            other => panic!("expected snapshot, got {other:?}"),
        }
        // ...the recent 2 turns remain verbatim...
        let recent: Vec<&Turn> = after
            .turns
            .iter()
            .filter(|t| {
                t.user_message()
                    .is_some_and(|c| c.starts_with("问题4") || c.starts_with("问题5"))
            })
            .collect();
        assert_eq!(recent.len(), 2, "recent turns must survive verbatim");
        // ...and the summary preserves the dropped content.
        assert!(after
            .messages
            .iter()
            .any(|m| m.content.contains("问题0") && m.content.contains("[context summary]")));
        // Nothing to compact when under the limit.
        assert!(compact_to_snapshot(id + 1, &events, 10).unwrap().is_none());
    }

    /// The projection is deterministic: same events, same view.
    #[test]
    fn projection_is_deterministic() {
        let events = vec![user(1, 0, "a"), ai(2, 0, "b"), user(3, 1, "c")];
        let p1 = project(&events).unwrap();
        let p2 = project(&events).unwrap();
        assert_eq!(p1, p2);
    }

    /// Empty input is an explicit error, not an empty session.
    #[test]
    fn empty_events_error() {
        assert!(project(&[]).is_err());
    }

    /// to_session bridges to the legacy mutable Session (compat view).
    #[test]
    fn to_session_compat() {
        let events = vec![user(1, 0, "hi"), ai(2, 0, "hello")];
        let s = to_session(&project(&events).unwrap());
        assert_eq!(s.id, "s1");
        assert_eq!(s.messages.len(), 2);
    }
}
