//! Handoffs multi-agent handoff
//!
//! Modeled on the OpenAI Agents SDK, it implements task delegation between agents:
//! the primary agent can hand tasks to specialist agents via `HandoffTool`.
//!
//! # Example
//! ```no_run
//! use lc_agents::handoffs::HandoffManager;
//! use std::sync::Arc;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mgr = HandoffManager::new();
//! // mgr.register_agent("writer", executor).await?;
//! // mgr.set_primary("writer")?;
//! // let result = mgr.run("write an article".to_string()).await?;
//! # Ok(())
//! # }
//! ```

pub mod handoff;
pub mod manager;

pub use handoff::{Handoff, HandoffContext, HandoffError, HandoffRecord, HandoffResult};
pub use manager::{HandoffManager, HandoffTool};
