// lc-memory/src/mongo_memory.rs
//! MongoDB Persistent Memory Implementation

use async_trait::async_trait;
use mongodb::{
    bson::{doc, oid::ObjectId},
    options::ClientOptions,
    Client, Collection, Database,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock as StdRwLock;
use tokio::sync::RwLock;

use super::base::{BaseMemory, MemoryError};
use super::persistent::{MemoryData, PersistenceConfig, PersistentMemory};
use super::summary_buffer::ConversationSummaryBufferMemory;
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::Runnable;
use lc_schema::{Message, MessageType};

/// MongoDB-stored memory document
#[derive(Debug, Serialize, Deserialize)]
struct MongoMemoryDoc {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    id: Option<ObjectId>,
    session_id: String,
    messages: Vec<Message>,
    summary: Option<String>,
    metadata: HashMap<String, String>,
    created_at: String,
    updated_at: String,
    /// P2-3: optimistic-lock version number, incremented by 1 on each save. Defaults to 0 on old documents.
    #[serde(default)]
    version: u64,
}

impl From<MemoryData> for MongoMemoryDoc {
    fn from(data: MemoryData) -> Self {
        Self {
            id: None,
            session_id: data.session_id,
            messages: data.messages,
            summary: data.summary,
            metadata: data.metadata,
            created_at: data.created_at,
            updated_at: data.updated_at,
            version: data.version,
        }
    }
}

impl From<MongoMemoryDoc> for MemoryData {
    fn from(doc: MongoMemoryDoc) -> Self {
        Self {
            session_id: doc.session_id,
            messages: doc.messages,
            summary: doc.summary,
            metadata: doc.metadata,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
            version: doc.version,
        }
    }
}

/// P2-3: pure merge function for optimistic-lock conflicts (no DB dependency, unit-testable).
///
/// When two concurrent writers each summarize from an old version, `remote` (already persisted)
/// is the base:
/// - messages: append local-snapshot messages the remote lacks (deduped by role + content),
///   preserving relative order, losing nothing;
/// - summary: take the longer of the two, preventing summary regression.
///
/// M13: the message merge is keyed by per-message identity. Two **real**
/// messages that are byte-identical (e.g. the user literally asked the same
/// thing twice) therefore stay separate — a (role, content) count canvas would
/// silently swallow one concurrent append. A local message whose stamped id is
/// already in the remote document is the shared snapshot and is not duplicated.
fn merge_memory_data(
    remote: &MemoryData,
    local_messages: &[Message],
    local_summary: &str,
) -> (Vec<Message>, String) {
    // M13: PER-MESSAGE IDENTITY is the only sound dedup. Two concurrent writers
    // each genuinely appending an identical ask ("hi") MUST both survive, but a
    // (role, content) count canvas equates them with the shared snapshot and
    // silently swallows one. A stamped `Message::id` distinguishes "same logical
    // message re-read by both writers" (dedup) from "two distinct messages with
    // same bytes" (keep both). Ids already round-trip through MongoDB serde and
    // never reach a provider body (requests are built field-by-field).
    //
    // Primary rule: a local message whose id the remote carries is part of the
    // shared snapshot -> skip; one whose id the remote never saw is genuine ->
    // append unconditionally (content is irrelevant to the decision).
    //
    // Fallback (legacy docs / manually-built messages with no id anywhere):
    // the pre-M13 occurrence heuristic, unchanged, so pre-lineage documents are
    // still handled with the old semantics rather than newly duplicated or lost.
    let remote_ids: HashSet<&str> = remote
        .messages
        .iter()
        .filter_map(|m| m.id.as_deref())
        .collect();

    // Legacy fallback uses the *static* remote occurrence counts (counted once),
    // exactly matching the pre-M13 behavior for id-less documents.
    let mut remote_occurrences: HashMap<(String, String), u32> = HashMap::new();
    for m in &remote.messages {
        *remote_occurrences.entry(message_key(m)).or_insert(0) += 1;
    }

    let mut messages = remote.messages.clone();
    let mut local_occurrences: HashMap<(String, String), u32> = HashMap::new();
    for msg in local_messages {
        // Identity path: exact, content-independent dedup.
        if let Some(id) = msg.id.as_deref() {
            if remote_ids.contains(id) {
                // Same logical message both writers already hold -> shared snapshot.
                continue;
            }
            messages.push(msg.clone());
            continue;
        }

        // Legacy / id-less path: fall back to the occurrence-count heuristic.
        let base = message_key(msg);
        let occ = local_occurrences.entry(base.clone()).or_insert(0);
        let this_occ = *occ;
        *occ += 1;
        if remote_occurrences.get(&base).copied().unwrap_or(0) > this_occ {
            continue;
        }
        messages.push(msg.clone());
    }

    let summary = match remote.summary.as_deref() {
        Some(remote_summary)
            if !remote_summary.trim().is_empty()
                && (local_summary.is_empty() || remote_summary.len() >= local_summary.len()) =>
        {
            remote_summary.to_string()
        }
        _ => local_summary.to_string(),
    };

    (messages, summary)
}

/// D5/M-me4: whether a MongoDB error is the duplicate-key (E11000) rejection of
/// the `session_id` unique index — the signature of two concurrent first writes
/// to the same brand-new session.
fn is_duplicate_key_error(err: &mongodb::error::Error) -> bool {
    matches!(
        err.kind.as_ref(),
        mongodb::error::ErrorKind::Write(mongodb::error::WriteFailure::WriteError(we))
            if we.code == 11000
    )
}

/// Message dedup base key: role + content. `MessageType` does not implement `Hash`/`Eq`, so a stable string representation is used.
fn message_key(msg: &Message) -> (String, String) {
    let role = match msg.message_type {
        MessageType::System => "system",
        MessageType::Human => "human",
        MessageType::AI => "ai",
        MessageType::Tool { .. } => "tool",
    };
    (role.to_string(), msg.content.clone())
}

/// M13: stamp a stable lineage id onto every message that lacks one, in place.
///
/// This is the identity that lets the optimistic-lock merge dedup by message
/// rather than by (role, content) count — without it, two concurrent writers
/// both appending an identical ask would collapse into one. Called from
/// `save_context` (holding the write lock on the inner history) so the ids are
/// assigned the moment a message enters the session and persist through the
/// Mongo round-trip. Existing ids are preserved verbatim.
fn stamp_missing_ids(messages: &mut [Message]) {
    for m in messages.iter_mut() {
        if m.id.is_none() {
            m.id = Some(uuid::Uuid::new_v4().to_string());
        }
    }
}

/// B6: whether the assistant message at `idx` has at least one `Tool` result message later in the
/// stream. An `AI` message with `tool_calls` but no following tool result is a dangling pair that
/// OpenAI/Anthropic reject with a 400.
fn has_tool_result(messages: &[Message], idx: usize) -> bool {
    messages[idx + 1..]
        .iter()
        .any(|m| matches!(m.message_type, MessageType::Tool { .. }))
}

/// B6: faithfully reconstructs a persisted message stream, restoring **all** message kinds —
/// including `Tool` results and `AI` messages carrying `tool_calls` — in original order. The only
/// drop point is an *orphaned* `AI.tool_calls` whose matching `Tool` result (followed as the next
/// consecutive `Tool` turn) is absent from the persisted stream; dropping it prevents a 400. Tool
/// results and normal tool_calls-dense turns are preserved verbatim. Previously the restore
/// discarded every Tool message and every AI.tool_calls, losing the "tools used" half of a turn.
fn restore_messages(messages: &[Message]) -> Vec<Message> {
    let mut restored: Vec<Message> = Vec::with_capacity(messages.len());
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        let is_ai_with_tool_calls =
            matches!(msg.message_type, MessageType::AI) && msg.tool_calls.is_some();
        if is_ai_with_tool_calls {
            if has_tool_result(messages, i) {
                // Paired turn: keep the assistant tool_calls and every following Tool result.
                restored.push(msg.clone());
                i += 1;
                while i < messages.len()
                    && matches!(messages[i].message_type, MessageType::Tool { .. })
                {
                    restored.push(messages[i].clone());
                    i += 1;
                }
            } else {
                // Orphaned tool_calls (no tool result anywhere after) -> drop to avoid a 400.
                i += 1;
            }
        } else if matches!(msg.message_type, MessageType::Tool { .. }) {
            // D8: a Tool landing here is orphaned — the preceding assistant
            // tool_calls pairing was already consumed by the paired branch, or
            // never existed (e.g. a leading Tool whose AI was cut outside the
            // persisted window). A standalone tool message is malformed for
            // OpenAI/Anthropic → 400, so drop it, mirroring the truncate /
            // summarize cleanup.
            i += 1;
        } else {
            restored.push(msg.clone());
            i += 1;
        }
    }
    restored
}

/// B6: reconciles the inner memory's local chat state against a loaded (or absent) document.
///
/// Local state is always cleared first. `Some(data)` restores its messages (via
/// [`restore_messages`]) and summary; `None` (a session with no persisted document) leaves the
/// inner memory empty so switching to that session never inherits a stale snapshot. Pure and
/// sync, so the cross-session-leak and tool-restore behaviors are unit-testable without a DB.
fn reconcile_inner<M: BaseChatModel>(
    inner: &mut ConversationSummaryBufferMemory<M>,
    data: Option<&MemoryData>,
) {
    // Always start from a fresh local state: both message history and the summary buffer are
    // cleared, so a `None` document (absent session) — or a present one with no summary — can
    // never inherit the previous session's snapshot.
    inner.chat_memory_mut().clear();
    inner.set_summary(String::new());
    if let Some(data) = data {
        for msg in restore_messages(&data.messages) {
            inner.chat_memory_mut().add_message(msg);
        }
        if let Some(summary) = &data.summary {
            if !summary.trim().is_empty() {
                inner.set_summary(summary.clone());
            }
        }
    }
}

/// MongoDB Persistent Memory
///
/// P0-3: generic over `M: BaseChatModel` instead of hard-coding `OpenAIChat`, so any LLM
/// implementing `BaseChatModel` (DeepSeek / Qwen / Ollama, etc.) can be used.
pub struct MongoPersistentMemory<M: BaseChatModel> {
    inner: RwLock<ConversationSummaryBufferMemory<M>>,
    database: Database,
    collection_name: String,
    session_id: StdRwLock<Option<String>>,
    config: RwLock<PersistenceConfig>,
}

impl<M: BaseChatModel> MongoPersistentMemory<M> {
    /// Creates a new MongoDB-backed persistent memory connected to the given
    /// database and collection, using `llm` and `token_limit` for the summary buffer.
    pub async fn new(
        mongo_uri: &str,
        database_name: &str,
        collection_name: &str,
        llm: M,
        token_limit: usize,
    ) -> Result<Self, MemoryError> {
        let client_options = ClientOptions::parse(mongo_uri)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB connection failed: {}", e)))?;

        let client = Client::with_options(client_options)
            .map_err(|e| MemoryError::LoadError(format!("MongoDB client error: {}", e)))?;

        let database = client.database(database_name);

        let inner = ConversationSummaryBufferMemory::<M>::new(llm, token_limit);

        // P1-3: single source of truth for token_limit — the constructor arg and config stay in sync; `with_config` overrides later.
        let config = PersistenceConfig::default().with_token_limit(token_limit);

        Ok(Self {
            inner: RwLock::new(inner),
            database,
            collection_name: collection_name.to_string(),
            session_id: StdRwLock::new(None),
            config: RwLock::new(config),
        })
    }

    /// Replaces the persistence config, keeping its token limit in sync with the inner memory.
    pub async fn with_config(mut self, config: PersistenceConfig) -> Self {
        // P1-3: config's token_limit lands on the inner memory's budget, so "config changed but inner did not" can no longer happen.
        self.inner.get_mut().set_max_token_limit(config.token_limit);
        *self.config.write().await = config;
        self
    }

    fn collection(&self) -> Collection<MongoMemoryDoc> {
        self.database.collection(&self.collection_name)
    }

    /// Creates a unique index on `session_id` to prevent duplicate documents.
    pub async fn create_indexes(&self) -> Result<(), MemoryError> {
        let collection = self.collection();

        // P2-3: unique index on session_id — the last line of defense against duplicate
        // documents when concurrent upserts create brand-new sessions (both inserting with
        // version=0). A dedicated index name avoids clashing with older non-unique indexes.
        collection
            .create_index(
                mongodb::IndexModel::builder()
                    .keys(doc! { "session_id": 1 })
                    .options(
                        mongodb::options::IndexOptions::builder()
                            .name("session_id_unique".to_string())
                            .unique(true)
                            .build(),
                    )
                    .build(),
                None,
            )
            .await
            .map_err(|e| MemoryError::SaveError(format!("Index creation failed: {}", e)))?;

        Ok(())
    }

    /// Returns the current session ID, if any.
    pub async fn get_session_id(&self) -> Option<String> {
        self.session_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Sets the session ID used for subsequent persistence operations.
    pub async fn set_session_id_async(&self, session_id: impl Into<String>) {
        *self.session_id.write().unwrap_or_else(|e| e.into_inner()) = Some(session_id.into());
    }

    async fn do_load_from_store(&self, session_id: &str) -> Result<(), MemoryError> {
        let collection = self.collection();

        let filter = doc! { "session_id": session_id };
        let result = collection
            .find_one(filter, None)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB find failed: {}", e)))?;

        let data: Option<MemoryData> = result.map(Into::into);

        // B6: local state is always reconciled against the loaded document — including the
        // `None` (absent session) branch, which clears the previous session's snapshot. Without
        // this clear, switching `session_id` below would let a subsequent upsert write the old
        // session's snapshot into the brand-new session (cross-session leak).
        let mut inner = self.inner.write().await;
        reconcile_inner(&mut inner, data.as_ref());

        *self.session_id.write().unwrap_or_else(|e| e.into_inner()) = Some(session_id.to_string());

        Ok(())
    }

    /// P2-3: Mongo optimistic-lock save.
    ///
    /// Every write carries `version` and replaces on `{session_id, version: expected}`;
    /// a miss means a concurrent writer landed first -> read the newest document, merge with the
    /// local snapshot, and retry. Brand-new sessions (version 0) use upsert, with the
    /// `session_id` unique index as the duplicate guard.
    async fn do_save_to_store(&self, session_id: &str) -> Result<(), MemoryError> {
        const MAX_ATTEMPTS: u32 = 3;
        let collection = self.collection();

        for _attempt in 0..MAX_ATTEMPTS {
            // snapshot the current in-memory state (re-snapshotted on every retry; merging is based on the newest local state).
            let (local_messages, local_summary) = {
                let inner = self.inner.read().await;
                (
                    inner.chat_memory().messages().to_vec(),
                    inner.buffer().await,
                )
            };

            // read the newest remote document: the version baseline + the merge source on conflict.
            let existing = collection
                .find_one(doc! { "session_id": session_id }, None)
                .await
                .map_err(|e| MemoryError::SaveError(format!("MongoDB find failed: {}", e)))?;

            let expected_version = existing.as_ref().map(|d| d.version).unwrap_or(0);
            let existing_created_at = existing.as_ref().map(|d| d.created_at.clone());

            let (mut messages, summary) = match existing {
                Some(remote) => {
                    let remote_data: MemoryData = remote.into();
                    merge_memory_data(&remote_data, &local_messages, &local_summary)
                }
                None => (local_messages, local_summary),
            };
            // M13: converge the *persisted* document to full lineage even when the
            // merge carried forward id-less remote messages (legacy docs). Stamping
            // here keeps every stored message id-carrying, so the next merge is
            // identity-exact rather than count-heuristic.
            stamp_missing_ids(&mut messages);

            let now = chrono::Utc::now().to_rfc3339();
            let data = MemoryData {
                session_id: session_id.to_string(),
                messages,
                summary: Some(summary),
                metadata: HashMap::new(),
                created_at: existing_created_at.unwrap_or_else(|| now.clone()),
                updated_at: now,
                version: expected_version + 1,
            };
            let new_doc: MongoMemoryDoc = data.into();

            let opts = mongodb::options::ReplaceOptions::builder()
                .upsert(expected_version == 0)
                .build();
            let res = match collection
                .replace_one(
                    doc! { "session_id": session_id, "version": expected_version as i64 },
                    new_doc,
                    opts,
                )
                .await
            {
                Ok(res) => res,
                // M-me4/D5: a brand-new session (`version == 0`) races a concurrent
                // first write. One upsert wins; the other is rejected by the
                // `session_id` unique index (error 11000). Retry the loop to
                // re-read the winner and merge — not a durable failure.
                Err(e) if is_duplicate_key_error(&e) && expected_version == 0 => {
                    continue;
                }
                Err(e) => {
                    return Err(MemoryError::SaveError(format!(
                        "MongoDB save failed: {}",
                        e
                    )))
                }
            };

            // a match means the save landed; a miss means the version advanced (concurrent write) — retry the merge loop.
            if res.matched_count > 0 || res.upserted_id.is_some() {
                return Ok(());
            }
        }

        Err(MemoryError::SaveError(format!(
            "MongoDB optimistic lock conflict: session {} kept being concurrently written within {} retries",
            session_id, MAX_ATTEMPTS
        )))
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseMemory for MongoPersistentMemory<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    fn memory_variables(&self) -> Vec<&str> {
        vec!["history"]
    }

    async fn load_memory_variables(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
        let config = self.config.read().await;
        if config.auto_load {
            let session_id = self
                .session_id
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            if let Some(sid) = session_id {
                let inner = self.inner.read().await;
                if inner.chat_memory().is_empty() {
                    drop(inner);
                    drop(config);
                    self.do_load_from_store(&sid).await?;
                }
            }
        }

        let inner = self.inner.read().await;
        inner.load_memory_variables(inputs).await
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        {
            let mut inner = self.inner.write().await;
            inner.save_context(inputs, outputs).await?;
            // M13: stamp lineage ids on the messages just added (and any legacy
            // id-less ones) so the merge dedups by identity. New ids persist
            // through the Mongo round-trip; surviving messages keep their ids on
            // later saves, giving the "shared snapshot re-read by both writers"
            // dedup a stable hook.
            let live = inner.chat_memory_mut();
            stamp_missing_ids(live.messages_mut());
        }

        let config = self.config.read().await;
        if config.auto_save {
            let session_id = self
                .session_id
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            if let Some(sid) = session_id {
                drop(config);
                self.do_save_to_store(&sid).await?;
            }
        }

        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        {
            let mut inner = self.inner.write().await;
            inner.clear().await?;
        }

        let session_id = self
            .session_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(sid) = session_id {
            self.do_delete_session(&sid).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> PersistentMemory for MongoPersistentMemory<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    async fn load_from_store(&mut self, session_id: &str) -> Result<(), MemoryError> {
        self.do_load_from_store(session_id).await
    }

    async fn save_to_store(&mut self, session_id: &str) -> Result<(), MemoryError> {
        self.do_save_to_store(session_id).await
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        self.do_delete_session(session_id).await
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError> {
        let collection = self.collection();
        let filter = doc! { "session_id": session_id };
        let result = collection
            .find_one(filter, None)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB find failed: {}", e)))?;

        Ok(result.is_some())
    }

    /// P0-2: reads the real session ID from the inner field (symmetric with `set_session_id` reads/writes).
    fn current_session_id(&self) -> Option<String> {
        self.session_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set_session_id(&mut self, session_id: String) {
        // C17: Using std::sync::RwLock instead of tokio::sync::RwLock::blocking_write
        // to avoid potential deadlock in async runtime.
        *self.session_id.write().unwrap_or_else(|e| e.into_inner()) = Some(session_id);
    }
}

impl<M: BaseChatModel> MongoPersistentMemory<M> {
    async fn do_delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        let collection = self.collection();

        collection
            .delete_one(doc! { "session_id": session_id }, None)
            .await
            .map_err(|e| MemoryError::ClearError(format!("MongoDB delete failed: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::MockLlm;

    #[test]
    fn test_mongo_memory_doc_from_memory_data() {
        let data = MemoryData::new("session_123".to_string());
        let doc: MongoMemoryDoc = data.into();

        assert_eq!(doc.session_id, "session_123");
        assert!(doc.messages.is_empty());
        assert_eq!(doc.version, 0);
    }

    #[test]
    fn test_memory_data_from_mongo_doc() {
        let doc = MongoMemoryDoc {
            id: None,
            session_id: "session_456".to_string(),
            messages: vec![Message::human("Hello")],
            summary: Some("Test summary".to_string()),
            metadata: HashMap::new(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            version: 7,
        };

        let data: MemoryData = doc.into();
        assert_eq!(data.session_id, "session_456");
        assert_eq!(data.messages.len(), 1);
        assert_eq!(data.version, 7);
    }

    /// P2-3: the MemoryData <-> MongoMemoryDoc roundtrip preserves the version number.
    #[test]
    fn test_version_roundtrip() {
        let mut data = MemoryData::new("s".to_string());
        data.version = 42;
        let doc: MongoMemoryDoc = data.clone().into();
        let back: MemoryData = doc.into();
        assert_eq!(back.version, 42);
    }

    /// H-5 regression: after the first save persists a window, a later summary
    /// rebuild + save must NOT re-append that same window. The summary-buffer fix
    /// keeps each surviving message's id through the rebuild (see
    /// `retain_to_window`), so `stamp_missing_ids` has nothing new to stamp and
    /// the optimistic-lock merge dedups the re-read snapshot by identity. Without
    /// the fix, the persisted window is treated as new each save and the Mongo
    /// document grows unboundedly (M13 violation).
    #[test]
    fn test_persisted_window_does_not_regrow_on_repeated_save() {
        // A window already persisted to Mongo: every message carries lineage ids.
        let window = vec![
            Message::human("h1").with_id("id-1"),
            Message::ai("a1").with_id("id-2"),
            Message::human("h2").with_id("id-3"),
        ];
        let remote = MemoryData::new("s".to_string()).with_messages(window.clone());

        // Second (and later) saves: the local snapshot preserves the same ids
        // (H-5 keeps them through the summary rebuild), so stamping adds nothing
        // and the merge dedups by identity — the doc must not regrow.
        let mut local = window.clone();
        stamp_missing_ids(&mut local);
        let (merged, _) = merge_memory_data(&remote, &local, "");
        assert_eq!(
            merged.len(),
            window.len(),
            "an already-persisted window must NOT be re-appended on a second save (H-5): {}",
            merged.len()
        );

        // Repeat the save many times: history length must stay flat (the old bug
        // regrew on every summary).
        let mut persisted = MemoryData::new("s".to_string()).with_messages(merged);
        for _ in 0..10 {
            let mut local = window.clone();
            stamp_missing_ids(&mut local);
            let (merged, _) = merge_memory_data(&persisted, &local, "");
            assert_eq!(merged.len(), 3, "history must stay flat across repeated saves");
            persisted = MemoryData::new("s".to_string()).with_messages(merged);
        }
    }

    /// M13: when both sides carry stable `Message::id`s, the merge must preserve
    /// *identity*, not collapse by count. Two writers genuinely asking "hi" twice
    /// are distinct messages that each survive; the byte-identical content must
    /// not be treated as one shared-snapshot occurrence. The old count-based
    /// heuristic (`remote_occ > this_occ`) walked only the *local* consecutive
    /// occurrence index, so a remote copy of the same bytes silently swallowed a
    /// genuine duplicate. With ids, dedup is exact: a local message with an id
    /// remote never saw is unconditionally kept.
    #[test]
    fn test_merge_preserves_genuine_duplicates_by_id() {
        // Remote (another writer) has one "hi" and one "ai" with their own ids.
        let remote = MemoryData::new("s".to_string()).with_messages(vec![
            Message::human("hi").with_id("remote-hi"),
            Message::ai("ok").with_id("remote-ok"),
        ]);
        // Local (this writer) asked "hi" twice — two *genuine* duplicates, each
        // freshly stamped. Ids are globally unique, so neither matches remote.
        let local = vec![
            Message::human("hi").with_id("local-hi-1"),
            Message::human("hi").with_id("local-hi-2"),
        ];

        let (messages, _) = merge_memory_data(&remote, &local, "");
        let hi_count = messages.iter().filter(|m| m.content == "hi").count();
        // All three "hi" are real messages from the two writers; the old
        // count heuristic dropped local occurrence index 0 because remote had 1.
        assert_eq!(
            hi_count, 3,
            "two writers' genuine 'hi' messages must all survive, got {hi_count}"
        );
        assert_eq!(messages.len(), 4);
    }

    /// M13: a local message whose id the remote already carries is part of the
    /// shared snapshot (the same logical message, re-read by both writers) and
    /// must NOT be re-appended — identity makes the overlap dedup exact.
    #[test]
    fn test_merge_dedups_shared_snapshot_by_id() {
        let remote = MemoryData::new("s".to_string()).with_messages(vec![
            Message::human("hi").with_id("shared-hi"),
            Message::ai("ok").with_id("shared-ok"),
        ]);
        // Local re-read the same snapshot (ids match remote) and then added one new turn.
        let local = vec![
            Message::human("hi").with_id("shared-hi"),
            Message::ai("ok").with_id("shared-ok"),
            Message::human("new").with_id("local-new"),
        ];

        let (messages, _) = merge_memory_data(&remote, &local, "");
        assert_eq!(messages.len(), 3, "shared snapshot must not be re-appended");
        assert!(messages.iter().any(|m| m.content == "new"));
    }

    /// M13: id-less messages (legacy docs, pre-lineage) fall back to the count
    /// heuristic unchanged — backward compatible behavior for docs that never
    /// carried identity.
    #[test]
    fn test_merge_falls_back_to_counts_without_ids() {
        // No ids present anywhere: must behave exactly as before (dedup by
        // role+content occurrence so a shared snapshot isn't duplicated).
        let remote = MemoryData::new("s".to_string())
            .with_messages(vec![Message::human("h1"), Message::ai("a1")]);
        let local = vec![
            Message::human("h1"),
            Message::ai("a1"),
            Message::human("h2"),
            Message::ai("a2"),
        ];

        let (messages, _) = merge_memory_data(&remote, &local, "");
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].content, "h2");
        assert_eq!(messages[3].content, "a2");
    }

    /// M13: the merge stamps a unique id onto messages that lack one, so a
    /// mixed id-ful/id-less history converges to full lineage on the next save.
    #[test]
    fn test_merge_stamps_missing_local_ids() {
        let remote = MemoryData::new("s".to_string());
        let local = vec![
            Message::human("h1"),
            Message::human("h1").with_id("kept-id"),
        ];

        let (mut messages, _) = merge_memory_data(&remote, &local, "");
        // Production save path converges the merged history to full lineage
        // (stamping lives at the persistence boundary so it also covers the
        // absent-session branch, not just the merge).
        stamp_missing_ids(&mut messages);
        // h1 had no id -> fresh uuid; kept-id preserves its own id.
        let h1 = messages.iter().find(|m| m.id.as_deref() == Some("kept-id"));
        assert!(h1.is_some(), "existing id must be preserved");
        // Every message now carries a unique id (dedup is identity-ready).
        let ids: Vec<_> = messages.iter().filter_map(|m| m.id.clone()).collect();
        assert_eq!(ids.len(), 2, "both messages must be stamped");
        let uniq: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(uniq.len(), 2, "stamped ids must be unique across messages");
    }

    /// P2-3: merging appends new local messages (deduped by type + content), preserving order.
    #[test]
    fn test_merge_appends_new_messages() {
        let remote = MemoryData::new("s".to_string())
            .with_messages(vec![Message::human("h1"), Message::ai("a1")]);
        let local = vec![
            Message::human("h1"),
            Message::ai("a1"),
            Message::human("h2"),
            Message::ai("a2"),
        ];

        let (messages, summary) = merge_memory_data(&remote, &local, "");
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].content, "h2");
        assert_eq!(messages[3].content, "a2");
        assert_eq!(summary, "");
    }

    /// P2-3: the summary takes the longer of the two, avoiding summary regression from concurrent writes.
    #[test]
    fn test_merge_prefers_longer_summary() {
        let remote = MemoryData::new("s".to_string()).with_summary("shorter".to_string());
        let (_, summary) = merge_memory_data(&remote, &[], "a much longer local summary");
        assert_eq!(summary, "a much longer local summary");

        let remote =
            MemoryData::new("s".to_string()).with_summary("longer remote summary".to_string());
        let (_, summary) = merge_memory_data(&remote, &[], "short");
        assert_eq!(summary, "longer remote summary");
    }

    /// P2-3: falls back to the local summary when the remote has none.
    #[test]
    fn test_merge_uses_local_summary_when_remote_none() {
        let remote = MemoryData::new("s".to_string());
        let (_, summary) = merge_memory_data(&remote, &[], "local-only");
        assert_eq!(summary, "local-only");
    }

    /// B6: the restore keeps Tool results and paired AI.tool_calls verbatim, and drops only an
    /// orphaned AI.tool_calls whose Tool result is absent (preventing a 400).
    #[test]
    fn test_restore_messages_keeps_tool_results_drops_orphan_tool_calls() {
        let msgs = vec![
            Message::human("q1"),
            // paired: AI carrying tool_calls followed by its Tool result -> both kept
            Message::ai_with_tool_calls("planning", vec![]),
            Message::tool("c1", "tool result 1"),
            Message::ai("done with tool"),
            // orphaned: AI.tool_calls with no Tool result after -> dropped
            Message::ai_with_tool_calls("orphan", vec![]),
            Message::human("q2"),
        ];
        let restored = restore_messages(&msgs);
        let contents: Vec<&str> = restored.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(
            restored.len(),
            5,
            "should drop only the orphaned tool_calls, got {contents:?}"
        );
        // tool result and its paired assistant.tool_calls are preserved verbatim.
        assert!(restored.iter().any(|m| m.content == "tool result 1"));
        assert!(restored.iter().any(|m| m.content == "planning"));
        // orphaned assistant.tool_calls is gone.
        assert!(!restored.iter().any(|m| m.content == "orphan"));
        // order preserved for the surviving messages.
        assert_eq!(restored[0].content, "q1");
        assert_eq!(restored[4].content, "q2");
    }

    /// B6: an absent session must not inherit a stale snapshot (cross-session leak), and a
    /// present document restores messages + summary faithfully.
    #[tokio::test]
    async fn test_reconcile_inner_clears_stale_snapshot_on_absent_session() {
        let llm = MockLlm::new(vec![]);
        let mut inner = ConversationSummaryBufferMemory::new(llm, 1000);

        // stale state from a previously-loaded session.
        inner
            .chat_memory_mut()
            .add_user_message("old session message");
        inner.set_summary("old summary".to_string());

        // switching to a brand-new (absent) session must clear the inner snapshot —
        // otherwise a later upsert would write it into the new session.
        reconcile_inner(&mut inner, None);
        assert_eq!(
            inner.chat_memory().len(),
            0,
            "absent session must not inherit a stale snapshot"
        );
        assert!(
            inner.buffer().await.is_empty(),
            "absent session must not inherit a stale summary"
        );

        // a present document restores messages (incl. tool turns) + the summary.
        let data = MemoryData::new("s".to_string())
            .with_messages(vec![
                Message::human("h1"),
                Message::ai_with_tool_calls("tc", vec![]),
                Message::tool("t1", "tr"),
            ])
            .with_summary("restored summary".to_string());
        reconcile_inner(&mut inner, Some(&data));
        assert_eq!(inner.chat_memory().len(), 3);
        assert_eq!(inner.chat_memory().messages()[0].content, "h1");
        assert_eq!(inner.buffer().await, "restored summary");
    }
}
