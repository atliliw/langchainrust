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
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A replayed session view.
///
/// `Serialize`/`Deserialize` so a projection can be persisted by a
/// [`SessionCheckpoint`](super::checkpoint::SessionCheckpoint) and merged with a
/// replayed tail after a restart ([`merge_projections`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// How much of a tool observation is echoed into the replayed context.
///
/// Same budget as [`summarize_turns`]: enough to recognize what the tool
/// returned, not enough to flood the window with stale payloads.
const TOOL_OBSERVATION_PREVIEW_CHARS: usize = 120;

/// Renders one event as a tool-activity context line, or `None` for non-tool
/// events.
///
/// # Why a labeled context line instead of a real tool message
///
/// `MessageType::Tool` is a *protocol* message: providers require it to answer a
/// preceding assistant message that carries the matching `tool_calls`. A replayed
/// log has no such message — the original assistant tool-call turn is not stored
/// as an assistant message — so emitting a raw tool message makes OpenAI answer
/// HTTP 400 ("messages with role 'tool' must be a response to a preceding message
/// with 'tool_calls'"). Tool activity is therefore replayed as a clearly labeled
/// system line, the same convention the projector already uses for
/// `[context summary]`.
///
/// The point is that the *tool name survives* (T2): a windowed replay used to
/// drop tool events entirely, so a session where an agent ran `search_docs`
/// looked to the model like nothing had ever been called.
pub(crate) fn tool_context_message(payload: &EventPayload) -> Option<lc_schema::Message> {
    match payload {
        EventPayload::ToolCall { tool, .. } => {
            Some(lc_schema::Message::system(format!("[tool call] {tool}")))
        }
        EventPayload::ToolResult { observation, .. } => Some(lc_schema::Message::system(format!(
            "[tool result] {}",
            truncate(observation, TOOL_OBSERVATION_PREVIEW_CHARS)
        ))),
        // L253: in-place tool edits and revocations surface as labeled context
        // lines so a replay sees the correction, not just the original call.
        EventPayload::ToolEdit {
            tool,
            superseded_call_id,
            ..
        } => Some(lc_schema::Message::system(format!(
            "[tool edit] {tool} (replaces {superseded_call_id})"
        ))),
        EventPayload::ToolUndo { undo_call_id, .. } => Some(lc_schema::Message::system(format!(
            "[tool undo] {undo_call_id}"
        ))),
        _ => None,
    }
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
            // INVARIANT: 上面刚 push(或 turns 本就非空),last_mut 必为 Some。
            turns
                .last_mut()
                .expect("a turn was just pushed or already present")
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
            EventPayload::ToolCall { .. }
            | EventPayload::ToolResult { .. }
            | EventPayload::ToolEdit { .. }
            | EventPayload::ToolUndo { .. } => {
                current_turn!(turn_index).events.push(e.clone());
                // T2: keep the tool name (and a truncated observation) in the
                // message view instead of dropping tool activity silently.
                // L253: edits and undos reuse the same labeled context line.
                if let Some(msg) = tool_context_message(&e.payload) {
                    messages.push(msg);
                }
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

/// Returns the id of the first event that must be kept after a compaction,
/// i.e. the opening event of the first *kept* turn (kept = the trailing
/// `max_turns` turns). Everything with `id < boundary` was summarized into the
/// just-created Snapshot and can be physically pruned with
/// [`super::store::EventStore::delete_before`] (mirrors `compact_to_snapshot`'s
/// own `turns[..dropped]` arithmetic, so the boundary always matches what the
/// Snapshot replaced).
///
/// Returns `None` when compaction would drop nothing (total turns <= `max_turns`).
pub fn compaction_prune_boundary(
    events: &[SessionEvent],
    max_turns: usize,
) -> Result<Option<u64>, SessionError> {
    let projection = project(events)?;
    let total = projection.turns.len();
    if total <= max_turns {
        return Ok(None);
    }
    let dropped = total - max_turns;
    Ok(projection
        .turns
        .get(dropped)
        .and_then(|t| t.events.first())
        .map(|e| e.id))
}

/// T2 (time-based compaction/GC): tightens a turn budget by age.
///
/// `max_turns` alone never reclaims an *idle* session: a conversation with three
/// turns from two years ago sits below every turn-count threshold, so its events
/// stay in the log forever. This returns the effective budget at `now_ms` —
/// `max_turns`, tightened to the turns that ended at or after `now_ms - max_age`.
///
/// A turn straddling the cutoff is counted as recent (it ended after it), and the
/// result never drops below 1: compaction always leaves a live turn behind.
pub fn age_bounded_turns(
    turns: &[Turn],
    max_turns: usize,
    max_age: Duration,
    now_ms: i64,
) -> usize {
    let max_age_ms = i64::try_from(max_age.as_millis()).unwrap_or(i64::MAX);
    let cutoff = now_ms.saturating_sub(max_age_ms);
    let recent = turns
        .iter()
        .filter(|t| {
            // A turn's age is the age of its newest event; a turn with no events
            // cannot be dated and is treated as recent (never pruned blindly).
            t.events.last().map(|e| e.ts >= cutoff).unwrap_or(true)
        })
        .count();
    max_turns.min(recent).max(1)
}

/// T2: folds a replayed tail into a checkpointed projection.
///
/// `base` is the projection materialized in a [`super::checkpoint::SessionCheckpoint`];
/// `tail` is the projection of the events appended since (`id > up_to_id`). The
/// result is identical to projecting the whole log at once — the invariant the
/// tests assert — because a tail turn that shares `base`'s last turn index is
/// *extended* rather than appended, exactly as [`project`] merges same-index
/// events into one turn.
///
/// The tail must not be empty: [`project`] rejects an empty event list, so a caller
/// with nothing to replay should keep `base` unchanged instead of calling this.
pub fn merge_projections(mut base: ProjectedSession, tail: ProjectedSession) -> ProjectedSession {
    for turn in tail.turns {
        match base.turns.last_mut() {
            Some(last) if last.index == turn.index => last.events.extend(turn.events),
            _ => base.turns.push(turn),
        }
    }
    base.messages.extend(tail.messages);
    base.last_id = base.last_id.max(tail.last_id);
    base
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

    /// H-bounded: compaction_prune_boundary returns the id of the first kept
    /// turn's opening event — everything below it is summary-prefix (§3.1).
    #[test]
    fn compaction_prune_boundary_matches_compact_to_snapshot() {
        let mut events = Vec::new();
        let mut id = 1u64;
        for turn in 0..4 {
            events.push(user(id, turn, &format!("q{turn}")));
            id += 1;
            events.push(ai(id, turn, &format!("a{turn}")));
            id += 1;
        }
        // ids: turn0 (1,2), turn1 (3,4), turn2 (5,6), turn3 (7,8).
        // Keeping 2 turns keeps turn2/3; the first kept event is user id 5.
        let boundary = compaction_prune_boundary(&events, 2).unwrap().unwrap();
        assert_eq!(boundary, 5);

        // And the snapshot compact_to_snapshot would emit summarizes ids < 5.
        let snapshot = compact_to_snapshot(id, &events, 2).unwrap().unwrap();
        let snapshot_id = snapshot.id;
        assert!(snapshot_id > boundary);

        // Under the limit there is nothing to prune.
        assert!(compaction_prune_boundary(&events, 10).unwrap().is_none());
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

    // ---- T2: tool activity survives the projection -------------------------

    /// T2: tool activity is replayed as labeled context lines, so the model can
    /// see that a tool ran and which one — previously tool events were dropped
    /// from the message view entirely.
    #[test]
    fn tool_activity_is_replayed_as_context_lines() {
        let events = vec![
            user(1, 0, "查一下"),
            call(2, 0, "c1"),
            result(3, 0, "c1"),
            ai(4, 0, "答案"),
        ];
        let p = project(&events).unwrap();
        let contents: Vec<&str> = p.messages.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(
            contents,
            vec![
                "查一下",
                "[tool call] search",
                "[tool result] found",
                "答案"
            ]
        );
    }

    /// A raw `MessageType::Tool` would be rejected by providers (no preceding
    /// assistant `tool_calls` in a replayed log), so tool activity must arrive as
    /// a system line — never as a protocol tool message.
    #[test]
    fn tool_activity_is_not_a_protocol_tool_message() {
        let events = vec![user(1, 0, "go"), call(2, 0, "c1"), result(3, 0, "c1")];
        let p = project(&events).unwrap();
        assert!(
            p.messages.iter().all(|m| m.message_type
                != lc_schema::MessageType::Tool {
                    tool_call_id: "c1".into()
                }),
            "a replayed log must not emit protocol tool messages"
        );
    }

    /// A long observation is previewed, not echoed whole — the window must not
    /// be flooded by a stale tool payload.
    #[test]
    fn long_tool_observation_is_truncated() {
        let long = "x".repeat(TOOL_OBSERVATION_PREVIEW_CHARS + 500);
        let events = vec![
            user(1, 0, "go"),
            SessionEvent::now(
                2,
                "s1",
                0,
                "main",
                EventPayload::ToolResult {
                    call_id: "c1".into(),
                    observation: long,
                },
            ),
        ];
        let p = project(&events).unwrap();
        let line = p
            .messages
            .iter()
            .find(|m| m.content.starts_with("[tool result]"))
            .expect("tool result line present");
        assert_eq!(
            line.content.len(),
            "[tool result] ".len() + TOOL_OBSERVATION_PREVIEW_CHARS
        );
    }

    // ---- T2: age-bounded compaction ---------------------------------------

    /// Stamps an event with an explicit timestamp (age-bounded tests need
    /// controlled time; `now()` always uses the wall clock).
    fn at(mut e: SessionEvent, ts: i64) -> SessionEvent {
        e.ts = ts;
        e
    }

    /// Builds `n` one-event turns timestamped `ts_step` apart, newest last.
    fn timed_turns(n: u64, ts_step: i64) -> Vec<Turn> {
        project(
            &(0..n)
                .map(|t| {
                    at(
                        user(t * 2 + 1, t, &format!("q{t}")),
                        (t as i64 + 1) * ts_step,
                    )
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .turns
    }

    /// T2: the age bound tightens the turn budget — turns that ended before the
    /// cutoff are no longer protected, so an idle session can be reclaimed.
    #[test]
    fn age_bound_tightens_the_turn_budget() {
        // 4 turns at ts 1000/2000/3000/4000; now = 4000.
        let turns = timed_turns(4, 1000);
        // 2500ms window → cutoff 1500 → only ts 2000/3000/4000 are recent.
        assert_eq!(
            age_bounded_turns(&turns, 10, Duration::from_millis(2500), 4000),
            3
        );
        // A wide window protects everything...
        assert_eq!(
            age_bounded_turns(&turns, 10, Duration::from_secs(3600), 4000),
            4
        );
        // ...and the count bound still caps it when it is the tighter one.
        assert_eq!(
            age_bounded_turns(&turns, 2, Duration::from_secs(3600), 4000),
            2
        );
    }

    /// The floor: however old the log, compaction always leaves one live turn
    /// behind (dropping every turn would leave nothing to project).
    #[test]
    fn age_bound_never_drops_below_one_turn() {
        let turns = timed_turns(3, 1000);
        assert_eq!(
            age_bounded_turns(&turns, 10, Duration::from_millis(1), 100_000),
            1
        );
    }

    /// A turn is as recent as its newest event, so a turn straddling the cutoff
    /// is kept; an undatable turn (no events) is kept too, never pruned blindly.
    #[test]
    fn age_bound_keeps_straddling_and_undated_turns() {
        let mut turns = timed_turns(1, 1000); // one turn ending at ts 1000
        turns.push(Turn {
            index: 1,
            events: Vec::new(),
        });
        // cutoff = 1000 (now 1000, age 0) → the dated turn is recent (>= cutoff).
        assert_eq!(
            age_bounded_turns(&turns, 10, Duration::from_millis(0), 1000),
            2
        );
        // cutoff = 2000 → the dated turn is old, the undated one still counts.
        assert_eq!(
            age_bounded_turns(&turns, 10, Duration::from_millis(1000), 3000),
            1
        );
    }

    // ---- T2: checkpoint recovery merge ------------------------------------

    /// The recovery invariant: merging a checkpointed prefix with a replayed
    /// tail equals projecting the whole log — including when the split falls
    /// *inside* a turn (the case that makes a naive append wrong).
    #[test]
    fn merge_of_prefix_and_tail_equals_full_projection() {
        let full = vec![
            user(1, 0, "查"),
            call(2, 0, "c1"),
            result(3, 0, "c1"),
            ai(4, 0, "答"),
            user(5, 1, "再查"),
        ];
        // Split inside turn 0 (after the ToolCall): the tail opens mid-turn.
        let merged = merge_projections(project(&full[..3]).unwrap(), project(&full[3..]).unwrap());
        assert_eq!(merged, project(&full).unwrap());

        // And on a turn boundary.
        let merged = merge_projections(project(&full[..4]).unwrap(), project(&full[4..]).unwrap());
        assert_eq!(merged, project(&full).unwrap());
    }

    /// The merged `last_id` is the tail's — the caller resumes from there.
    #[test]
    fn merge_takes_the_higher_last_id() {
        let full = [
            user(1, 0, "a"),
            ai(2, 0, "b"),
            user(3, 1, "c"),
            ai(4, 1, "d"),
        ];
        let merged = merge_projections(project(&full[..2]).unwrap(), project(&full[2..]).unwrap());
        assert_eq!(merged.last_id, 4);
        assert_eq!(merged.turns.len(), 2);
        assert_eq!(merged.messages.len(), 4);
    }

    /// A tail that opens a new turn is appended, not merged into the previous
    /// turn — turn identity is the index, not "whatever came last".
    #[test]
    fn merge_appends_a_tail_that_opens_a_new_turn() {
        let full = [user(1, 0, "a"), ai(2, 0, "b"), user(3, 1, "c")];
        let merged = merge_projections(project(&full[..2]).unwrap(), project(&full[2..]).unwrap());
        assert_eq!(merged.turns.len(), 2);
        assert_eq!(merged.turns[0].index, 0);
        assert_eq!(merged.turns[1].index, 1);
        assert_eq!(merged.turns[1].events.len(), 1);
    }
}
