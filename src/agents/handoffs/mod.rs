//! Handoffs 多 Agent 交接
//!
//! 参考 OpenAI Agents SDK,实现 Agent 间任务委托:
//! 主 Agent 可通过 HandoffTool 将任务交给专业 Agent。
//!
//! # 示例
//! ```no_run
//! use langchainrust::agents::handoffs::HandoffManager;
//! use std::sync::Arc;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mgr = HandoffManager::new();
//! // mgr.register_agent("writer", executor).await?;
//! // mgr.set_primary("writer")?;
//! // let result = mgr.run("写文章".to_string()).await?;
//! # Ok(())
//! # }
//! ```

pub mod handoff;
pub mod manager;

pub use handoff::{Handoff, HandoffContext, HandoffError, HandoffRecord, HandoffResult};
pub use manager::{HandoffManager, HandoffTool};
