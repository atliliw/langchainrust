// src/agents/react/mod.rs
//! ReAct Agent 实现
//!
//! 基于 "ReAct: Synergizing Reasoning and Acting in Language Models" 论文。

pub mod parser;
pub mod prompt;
pub mod agent;

pub use parser::ReActOutputParser;
pub use agent::ReActAgent;