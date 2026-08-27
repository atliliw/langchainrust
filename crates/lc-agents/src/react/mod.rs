// src/agents/react/mod.rs
//! ReAct Agent implementation
//!
//! Based on the "ReAct: Synergizing Reasoning and Acting in Language Models" paper.

pub mod agent;
pub mod parser;
pub mod prompt;

pub use agent::ReActAgent;
pub use parser::ReActOutputParser;
