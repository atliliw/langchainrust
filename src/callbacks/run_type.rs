// src/callbacks/run_type.rs
//! Run type enumeration for tracing

use serde::{Deserialize, Serialize};

/// Run type for tracing
///
/// Each run in a trace has a type that indicates what kind of operation it represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunType {
    /// LLM call (e.g., OpenAI chat completion)
    Llm,
    /// Chain execution (e.g., LLMChain, SequentialChain)
    Chain,
    /// Tool invocation (e.g., Calculator, DateTime)
    Tool,
    /// Retriever query (e.g., vector search)
    Retriever,
    /// Embedding generation
    Embedding,
    /// Prompt template formatting
    Prompt,
    /// Output parsing
    Parser,
}

impl RunType {
    /// Get string representation for API
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Chain => "chain",
            Self::Tool => "tool",
            Self::Retriever => "retriever",
            Self::Embedding => "embedding",
            Self::Prompt => "prompt",
            Self::Parser => "parser",
        }
    }

    /// Get display emoji for console output
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Llm => "🤖",
            Self::Chain => "🔗",
            Self::Tool => "🔧",
            Self::Retriever => "📚",
            Self::Embedding => "📊",
            Self::Prompt => "📝",
            Self::Parser => "📄",
        }
    }
}

impl std::fmt::Display for RunType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
