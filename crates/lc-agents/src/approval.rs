// lc-agents/src/approval.rs
//! Approval gate (§4.2): **asynchronous** approval gate before tool execution.
//!
//! Coexists with the synchronous hook system (`hooks::ApprovalHook` /
//! `ToolCallAction`) without conflict: the sync hook runs inside
//! `execute_tool`, and the approval gate also runs inside `execute_tool`, after
//! the sync hook and **before** actual execution — the order is
//! `budget gate → execute_tool (sync hook → approval gate → tool execution)`.
//!
//! The framework only provides the gate; the approval strategy is implemented
//! by the caller via [`ApprovalHandler`]. [`AllowAll`] is a reference
//! implementation for testing / demos.

use async_trait::async_trait;
use serde_json::Value;

use crate::hooks::ToolCallContext;

/// Approval decision before tool execution.
#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    /// Allow: execute as-is.
    Allow,
    /// Deny: do not execute the tool; feed the reason back to the loop as an
    /// observation so the next round replans.
    Deny {
        /// Denial reason (goes into the observation).
        reason: String,
    },
    /// Modify arguments then execute: replaces the original arguments with `arguments`.
    Modify {
        /// Replacement tool arguments.
        arguments: Value,
        /// Modification note (for logging).
        note: String,
    },
}

/// Approval-gate interface. Implemented by the caller and injected via
/// `AgentExecutor::with_approval`.
///
/// `approve` is async: implementations may `await` an approval signal (CLI
/// interaction / webhook / messaging channel). Same-process resume works
/// naturally through async/await — the future suspends waiting for the signal
/// and continues from the same line when it arrives, no serialization /
/// Checkpointer needed.
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    /// Called before tool execution; returns the approval decision.
    async fn approve(&self, ctx: &ToolCallContext) -> ApprovalDecision;
}

/// Reference implementation: allows everything. For tests / demos.
#[derive(Debug, Default)]
pub struct AllowAll;

#[async_trait]
impl ApprovalHandler for AllowAll {
    async fn approve(&self, _ctx: &ToolCallContext) -> ApprovalDecision {
        ApprovalDecision::Allow
    }
}
