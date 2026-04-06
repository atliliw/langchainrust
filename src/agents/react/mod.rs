// src/agents/react/mod.rs
//! ReAct Agent 实现
//!
//! 基于 "ReAct: Synergizing Reasoning and Acting in Language Models" 论文。
//! 参考 Python 版本: langchain/libs/langchain/langchain_classic/agents/react/agent.py

pub mod parser;
pub mod prompt;
pub mod agent;

pub use parser::ReActOutputParser;
pub use agent::ReActAgent;