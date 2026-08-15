//! Token counter and cost tracking
//!
//! Provides token counting (tiktoken), usage statistics, cost estimation,
//! and a `TokenTrackingLLM` wrapper.

pub mod counter;
pub mod tiktoken;
pub mod tracker;

pub use counter::{CharRatioCounter, TokenCounter, TrackerTokenUsage};
pub use tiktoken::TiktokenCounter;
pub use tracker::{ModelPricing, TokenTrackingLLM};

use std::sync::LazyLock;
use tiktoken_rs::CoreBPE;

/// Global cached tiktoken encoder (cl100k_base, for GPT-3.5/4/4o).
///
/// Holds a `Result` so encoder-load failure surfaces as a `count_tokens`
/// error instead of a process-wide `expect` panic on first use (Q9).
static GLOBAL_ENCODER: LazyLock<Result<CoreBPE, String>> = LazyLock::new(|| {
    tiktoken_rs::cl100k_base()
        .map_err(|e| format!("Failed to load tiktoken cl100k_base encoder: {e}"))
});

/// Count tokens in text using the global tiktoken encoder.
///
/// This is a convenience function that uses a lazily-initialized
/// cl100k_base encoder (suitable for GPT-3.5/4/4o models).
///
/// Returns an error if the tiktoken encoder could not be loaded (e.g. the
/// vendored BPE file is missing), rather than panicking.
///
/// # Examples
/// ```no_run
/// use lc_core::token_counter::count_tokens;
///
/// let n = count_tokens("Hello, world!").unwrap();
/// assert!(n > 0);
/// ```
pub fn count_tokens(text: &str) -> Result<usize, String> {
    let encoder = GLOBAL_ENCODER.as_ref().map_err(|e| e.clone())?;
    Ok(encoder.encode_with_special_tokens(text).len())
}
