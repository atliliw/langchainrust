// src/agents/mod.rs
//! Agent system for building autonomous LLM applications.
//!
//! Provides core abstractions and implementations for agents.
//!
//! # Core Concepts
//!
//! - **Agent**: Responsible for planning, deciding what action to execute next.
//! - **AgentExecutor**: Responsible for execution loop (plan → act → observe).
//! - **Tool**: Callable tools that agents can invoke.
//!
//! # Execution Flow
//!
//! ```text
//! Input question
//!     ↓
//! Agent.plan() → AgentAction or AgentFinish
//!     ↓
//! If Action: execute tool → get observation
//!     ↓
//! Add to intermediate_steps
//!     ↓
//! Loop until AgentFinish returned
//! ```

pub mod types;
pub mod base;
pub mod react;
pub mod function_calling;

pub use types::{AgentAction, AgentFinish, AgentStep, AgentOutput, ToolInput};
pub use base::{BaseAgent, AgentExecutor, AgentError};
pub use react::ReActAgent;
pub use function_calling::FunctionCallingAgent;