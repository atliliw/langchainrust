# Sessions

Sessions provide multi-turn conversation lifecycle management: create, retrieve, archive, and chat with automatic history maintenance.

> **⚠️ v0.22.0**: the event-sourcing path (`EventSessionManager`) is the recommended API. The legacy `SessionManager` is `#[deprecated]` (kept until 0.23.0). See [Event Sourcing (v0.22.0)](#event-sourcing-v0220) below.

## Core Types

| Type | Description |
|------|-------------|
| `SessionManager` | Legacy API (deprecated in 0.22.0, removed in 0.23.0) |
| `EventSessionManager` | **Recommended** event-sourced manager (v0.22.0) |
| `Session` | A conversation with messages, metadata, and status |
| `SessionStore` | Trait for pluggable storage backends (legacy) |
| `EventStore` | Append-only event log trait (v0.22.0) |
| `MemorySessionStore` / `MemoryEventStore` | In-memory implementations |
| `SessionStatus` | `Active`, `Archived`, `Deleted` |

## Basic Usage

```rust
use langchainrust::{SessionManager, MemorySessionStore};
use std::sync::Arc;

let manager = SessionManager::new(Arc::new(MemorySessionStore::new()));

// Create a new session
let session_id = manager.create_session().await?;

// Create a session for a specific user
let session_id = manager.create_session_for("user_123").await?;

// Chat (auto-maintains history)
let reply = manager.chat(&session_id, &llm, "What is Rust?".to_string()).await?;
let reply2 = manager.chat(&session_id, &llm, "Tell me more about ownership.".to_string()).await?;

// Get conversation history
let messages = manager.history(&session_id).await?;

// Archive or clear
manager.archive(&session_id).await?;
manager.clear(&session_id).await?;

// List sessions for a user
let sessions = manager.list_by_user("user_123").await?;
```

## Session Struct

```rust
pub struct Session {
    pub id: String,
    pub user_id: Option<String>,
    pub messages: Vec<Message>,
    pub metadata: HashMap<String, Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: SessionStatus,
}
```

Methods: `add_message()`, `recent_messages(n)`, `clear()`, `archive()`

## Custom SessionStore

```rust
use langchainrust::{SessionStore, Session, SessionError};

struct RedisSessionStore { /* ... */ }

#[async_trait]
impl SessionStore for RedisSessionStore {
    async fn create(&self, session: Session) -> Result<String, SessionError>;
    async fn get(&self, id: &str) -> Result<Option<Session>, SessionError>;
    async fn update(&self, session: &Session) -> Result<(), SessionError>;
    async fn delete(&self, id: &str) -> Result<(), SessionError>;
    async fn list_by_user(&self, user_id: &str) -> Result<Vec<Session>, SessionError>;
}
```

## SessionManager Methods

| Method | Description |
|--------|-------------|
| `create_session()` | Create an anonymous session |
| `create_session_for(user_id)` | Create a session for a user |
| `get_session(id)` | Get session by ID |
| `chat(id, llm, message)` | Send message and get reply (auto-persists) |
| `history(id)` | Get conversation messages |
| `clear(id)` | Clear messages |
| `archive(id)` | Archive session |
| `list_by_user(user_id)` | List sessions for a user |

## Event Sourcing (v0.22.0)

The legacy manager is "rewrite-style": each turn reads the whole `Session`, mutates it, and writes it back — a crash mid-way loses messages, and there is no way to fork or time-travel. 0.22.0 rewrites sessions as **event sourcing**: the session is an append-only event log; history is a projection (replay) of the log.

```rust
use langchainrust::sessions::{EventSessionManager, MemoryEventStore, AutoCompaction};
use std::sync::Arc;

let manager = EventSessionManager::new(Arc::new(MemoryEventStore::new()))
    .with_max_context_turns(10)                      // turn window (default: full history)
    .with_auto_compaction(AutoCompaction::new(20)?); // snapshot beyond 20 turns

let id = manager.create_session().await?;            // uuid v7

let reply = manager.chat(&id, &llm, "What is Rust?".to_string()).await?;

let history = manager.history(&id).await?;   // projected Vec<Message>
// clear / archive / delete: append Metadata events to the EventStore directly
```

### Old API → New API Mapping

| Old (0.21.x) | New (0.22.0) | Note |
|---|---|---|
| `SessionManager::new(Arc<dyn SessionStore>)` | `EventSessionManager::new(Arc<dyn EventStore>)` | storage becomes an append-only log |
| `create_session_for(user)` | `create_session()` (uuid v7) | ownership via `Metadata` events |
| `chat(&id, &llm, msg)` | same signature | persistence changes from rewrite to append — crash-safe |
| `history(&id)` | `history(&id)` / `replay_session(&id)` | `replay_session` yields the old mutable `Session` view |
| — | `fork_session(&id, branch, until)` | **new**: branch from any point in the log |
| `max_context_messages(n)` | `with_max_context_turns(n)` | turn window; user/ai always paired |
| — | `with_auto_compaction(AutoCompaction)` | **new**: deterministic snapshot past N turns (no LLM call) |
| `clear / archive / delete_session` | append `Metadata` events to the `EventStore` directly (wrappers in 0.23.0) | the log is immutable; cleanup is a state event |
| `SessionStore` | `EventStore` | `append / append_batch / read / fork`; idempotency key `(session, branch, id)` |

### Fork

```rust
// Copy the prefix up to some event sequence number into a new branch;
// the trunk is unaffected. Returns a manager pinned to the branch.
let mut experiment = manager.fork_session(&id, "experiment", None).await?;
experiment.chat(&id, &llm, "branch message".to_string()).await?;
// Re-forking the same (session, branch) is idempotent
```

### Crash Safety

- `append` is idempotent per `(session_id, branch, id)` — replaying a half-written batch after a crash never duplicates events.
- `project()` / `to_session()` rebuild state from the log; orphan tool results are detected and rejected, so the history fed to the LLM is always properly paired.
- `SessionCheckpoint` trait + `NoopCheckpoint` are the placeholder for durable checkpoints (0.23.0).

### Event Types

| `EventPayload` variant | Meaning |
|---|---|
| `SessionCreated` | session established (uuid v7) |
| `UserMessage` / `AssistantMessage` | one Turn per round |
| `ToolCall` / `ToolResult` | appended as a pair |
| `Snapshot` | deterministic compaction snapshot |
| `Metadata` | lifecycle / ownership key-value state |
