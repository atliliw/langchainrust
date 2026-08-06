# Sessions

Sessions provide multi-turn conversation lifecycle management: create, retrieve, archive, and chat with automatic history maintenance.

## Core Types

| Type | Description |
|------|-------------|
| `SessionManager` | Main API for managing sessions |
| `Session` | A conversation with messages, metadata, and status |
| `SessionStore` | Trait for pluggable storage backends |
| `MemorySessionStore` | In-memory implementation |
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
