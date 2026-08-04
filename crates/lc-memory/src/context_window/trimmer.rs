// lc-memory/src/context_window/trimmer.rs
//! Strategy for fitting messages within a token limit.

use std::sync::Arc;

use lc_core::language_models::BaseChatModel;

/// Default summary prompt for the Summarize strategy.
pub(crate) const DEFAULT_SUMMARY_PROMPT: &str = "\
Summarize the following conversation concisely, preserving key facts, \
decisions, and context. Write the summary in the same language as the conversation.

Conversation:
{conversation}

Summary:";

/// Strategy for fitting messages within a token limit.
#[derive(Debug)]
pub enum Strategy<M: BaseChatModel> {
    /// Drop oldest messages to fit within the token limit.
    /// System messages are always preserved.
    Truncate,

    /// Use an LLM to compress old messages into a summary system message.
    Summarize {
        /// The LLM used to generate summaries.
        llm: Arc<M>,
        /// Custom summary prompt. Must contain `{conversation}` placeholder.
        summary_prompt: String,
    },
}

impl<M: BaseChatModel> Strategy<M> {
    /// Creates a new Summarize strategy with the given LLM and default prompt.
    pub fn summarize(llm: M) -> Self {
        Strategy::Summarize {
            llm: Arc::new(llm),
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
        }
    }

    /// Creates a new Summarize strategy with a custom prompt.
    ///
    /// The prompt must contain the `{conversation}` placeholder.
    pub fn summarize_with_prompt(llm: M, prompt: impl Into<String>) -> Self {
        Strategy::Summarize {
            llm: Arc::new(llm),
            summary_prompt: prompt.into(),
        }
    }
}
