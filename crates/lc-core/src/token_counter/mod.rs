//! Token counter and cost tracking
//!
//! Provides token counting (tiktoken), usage statistics, cost estimation,
//! and a `TokenTrackingLLM` wrapper.

pub mod counter;
pub mod tiktoken;
pub mod tracker;

pub use counter::{CharRatioCounter, TokenCounter, TokenUsage};
pub use tiktoken::TiktokenCounter;
pub use tracker::{ModelPricing, TokenTrackingLLM};

use std::sync::LazyLock;
use tiktoken_rs::CoreBPE;

/// Global cached tiktoken encoder (cl100k_base, for GPT-3.5/4/4o).
static GLOBAL_ENCODER: LazyLock<CoreBPE> = LazyLock::new(|| {
    tiktoken_rs::cl100k_base().expect("Failed to load tiktoken cl100k_base encoder")
});

/// Count tokens in text using the global tiktoken encoder.
///
/// This is a convenience function that uses a lazily-initialized
/// cl100k_base encoder (suitable for GPT-3.5/4/4o models).
///
/// # Examples
/// ```no_run
/// use lc_core::token_counter::count_tokens;
///
/// let n = count_tokens("Hello, world!");
/// assert!(n > 0);
/// ```
pub fn count_tokens(text: &str) -> usize {
    GLOBAL_ENCODER.encode_with_special_tokens(text).len()
}
