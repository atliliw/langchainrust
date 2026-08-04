//! Sessions 会话管理
//!
//! 提供多轮对话的会话生命周期管理:创建/获取/归档会话,
//! 在会话中对话(自动维护历史),支持可插拔存储。
//!
//! # 示例
//! ```no_run
//! use lc_sessions::{SessionManager, MemorySessionStore};
//! use lc_core::BaseChatModel;
//! use std::sync::Arc;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
//! let id = manager.create_session().await?;
//! // let llm = ...; // any type implementing BaseChatModel
//! // let reply = manager.chat(&id, &llm, "你好".to_string()).await?;
//! # Ok(())
//! # }
//! ```

pub mod manager;
pub mod memory_store;
pub mod session;
pub mod store;

pub use manager::SessionManager;
pub use memory_store::MemorySessionStore;
pub use session::{Session, SessionStatus};
pub use store::{SessionError, SessionStore};
