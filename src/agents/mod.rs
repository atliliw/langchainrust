// src/agents/mod.rs
//! Agent 系统
//!
//! 提供构建 Agent 的核心抽象和实现。
//!
//! # 核心概念
//!
//! - **Agent**: 负责决策（plan），决定下一步执行什么动作
//! - **AgentExecutor**: 负责执行循环（plan → act → observe）
//! - **Tool**: 可被 Agent 调用的工具
//!
//! # 执行流程
//!
//! ```text
//! 输入问题
//!     ↓
//! Agent.plan() → AgentAction 或 AgentFinish
//!     ↓
//! 如果是 Action: 执行工具 → 得到 observation
//!     ↓
//! 添加到 intermediate_steps
//!     ↓
//! 循环直到返回 AgentFinish
//! ```

pub mod types;
pub mod base;
pub mod react;
pub mod function_calling;

pub use types::{AgentAction, AgentFinish, AgentStep, AgentOutput, ToolInput};
pub use base::{BaseAgent, AgentExecutor, AgentError};
pub use react::ReActAgent;
pub use function_calling::FunctionCallingAgent;