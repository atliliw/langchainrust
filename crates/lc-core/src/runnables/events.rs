// lc-core/src/runnables/events.rs
//! Stream events for fine-grained LCEL pipeline observability.
//!
//! `astream_events` produces a stream of `StreamEvent` values that
//! describe what's happening inside a pipeline at each step. This is
//! useful for building UIs that show real-time progress.
//!
//! v0.9.0 implements 5 core event types. Full v2 event coverage
//! is planned for v0.10.0.

use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Fine-grained event emitted during LCEL pipeline execution.
///
/// v0.9.0 supports 5 core event types:
/// - `OnLlmStart` / `OnLlmStream` / `OnLlmEnd`: LLM lifecycle
/// - `OnToolEnd`: Tool execution completion
/// - `OnChainEnd`: Chain execution completion
///
/// v0.10.0 will add: OnRetrieverStart/End, OnPromptStart/End,
/// OnToolStart, OnChainStart, OnToolError, OnChainError, etc.
///
/// Note: Named `LcelStreamEvent` to avoid collision with
/// `lc_langgraph::StreamEvent` (graph execution events).
#[derive(Debug, Clone)]
pub enum LcelStreamEvent {
    /// LLM invocation started.
    OnLlmStart {
        /// Unique run identifier.
        run_id: Uuid,
        /// Name of the LLM (e.g. "gpt-4o", "claude-3-opus").
        name: String,
        /// Additional metadata.
        metadata: HashMap<String, Value>,
    },

    /// LLM streaming token.
    OnLlmStream {
        /// Unique run identifier.
        run_id: Uuid,
        /// Name of the LLM.
        name: String,
        /// The token text.
        token: String,
    },

    /// LLM invocation completed.
    OnLlmEnd {
        /// Unique run identifier.
        run_id: Uuid,
        /// Name of the LLM.
        name: String,
        /// The full LLM output.
        output: Value,
    },

    /// Tool execution completed.
    OnToolEnd {
        /// Unique run identifier.
        run_id: Uuid,
        /// Name of the tool.
        name: String,
        /// The tool output.
        output: String,
    },

    /// Chain execution completed.
    OnChainEnd {
        /// Unique run identifier.
        run_id: Uuid,
        /// Name of the chain.
        name: String,
        /// The chain output.
        output: Value,
    },
}

impl LcelStreamEvent {
    /// Get the run_id for this event.
    pub fn run_id(&self) -> &Uuid {
        match self {
            LcelStreamEvent::OnLlmStart { run_id, .. } => run_id,
            LcelStreamEvent::OnLlmStream { run_id, .. } => run_id,
            LcelStreamEvent::OnLlmEnd { run_id, .. } => run_id,
            LcelStreamEvent::OnToolEnd { run_id, .. } => run_id,
            LcelStreamEvent::OnChainEnd { run_id, .. } => run_id,
        }
    }

    /// Get the name associated with this event.
    pub fn name(&self) -> &str {
        match self {
            LcelStreamEvent::OnLlmStart { name, .. } => name,
            LcelStreamEvent::OnLlmStream { name, .. } => name,
            LcelStreamEvent::OnLlmEnd { name, .. } => name,
            LcelStreamEvent::OnToolEnd { name, .. } => name,
            LcelStreamEvent::OnChainEnd { name, .. } => name,
        }
    }

    /// Get the event kind as a string.
    pub fn kind(&self) -> &str {
        match self {
            LcelStreamEvent::OnLlmStart { .. } => "on_llm_start",
            LcelStreamEvent::OnLlmStream { .. } => "on_llm_stream",
            LcelStreamEvent::OnLlmEnd { .. } => "on_llm_end",
            LcelStreamEvent::OnToolEnd { .. } => "on_tool_end",
            LcelStreamEvent::OnChainEnd { .. } => "on_chain_end",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_event_kind() {
        let event = LcelStreamEvent::OnLlmStart {
            run_id: Uuid::new_v4(),
            name: "gpt-4o".to_string(),
            metadata: HashMap::new(),
        };
        assert_eq!(event.kind(), "on_llm_start");
        assert_eq!(event.name(), "gpt-4o");
    }

    #[test]
    fn stream_event_variants() {
        let id = Uuid::new_v4();
        let event = LcelStreamEvent::OnLlmStream {
            run_id: id,
            name: "claude".to_string(),
            token: "Hello".to_string(),
        };
        assert_eq!(event.kind(), "on_llm_stream");
        assert_eq!(*event.run_id(), id);
    }
}
