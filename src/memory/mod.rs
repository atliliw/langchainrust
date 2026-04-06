// src/memory/mod.rs
//! Memory 系统
//!
//! 提供对话记忆管理功能。
//!
//! # 核心概念
//!
//! - **BaseMemory**: Memory 的基础 trait
//! - **ConversationBufferMemory**: 简单的对话缓冲区
//! - **ConversationBufferWindowMemory**: 带窗口的对话缓冲区
//!
//! # 示例
//!
//! ```ignore
//! use langchainrust::{ConversationBufferMemory, BaseMemory};
//! use std::collections::HashMap;
//!
//! let mut memory = ConversationBufferMemory::new();
//!
//! // 保存对话
//! let inputs = HashMap::from([("input".to_string(), "你好".to_string())]);
//! let outputs = HashMap::from([("output".to_string(), "你好！".to_string())]);
//! memory.save_context(&inputs, &outputs).await?;
//!
//! // 加载记忆
//! let vars = memory.load_memory_variables(&HashMap::new()).await?;
//! println!("{:?}", vars.get("history"));
//! ```

pub mod base;
pub mod buffer;
pub mod window;

pub use base::{BaseMemory, MemoryError, ChatMessageHistory};
pub use buffer::ConversationBufferMemory;
pub use window::ConversationBufferWindowMemory;