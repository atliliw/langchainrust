// src/agents/react/mod.rs
//! ReAct Agent 实现
//!
//! 基于 "ReAct: Synergizing Reasoning and Acting in Language Models" 论文。

pub mod agent;
pub mod parser;
pub mod prompt;

pub use agent::ReActAgent;
pub use parser::ReActOutputParser;
