//! lc-testkit error types and the bridge to `ProviderError`.

use lc_providers::ProviderError;

/// Unified lc-testkit error.
///
/// Bridges via [`From<TestkitError> for ProviderError`], so the record/replay provider
/// can be fed directly into generic entry points like chains that require
/// `L::Error: Into<ProviderError>`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TestkitError {
    /// IO error during recording/replaying (reading/writing files, etc.).
    #[error("io error while recording/replaying: {0}")]
    Io(#[from] std::io::Error),
    /// Replay queue exhausted: more requests than recorded exchanges.
    #[error("replay queue exhausted (requested {requested} messages, no recording left)")]
    ReplayExhausted { requested: usize },
    /// No recording matches the request message signature under `ReplayStrategy::Exact`
    /// (explicit error, no silent FIFO fallback); `left` is the remaining queue length,
    /// useful for debugging field drift between the recording and the request.
    #[error("replay has no recording matching request messages (strategy=Exact, {left} exchange(s) left)")]
    ReplayNoMatch { left: usize },
    /// Inner model error, passed through losslessly from the real provider error.
    #[error("inner model error: {0}")]
    Inner(#[from] ProviderError),
}

impl From<TestkitError> for ProviderError {
    fn from(e: TestkitError) -> Self {
        match e {
            // Pass the real provider error through losslessly.
            TestkitError::Inner(p) => p,
            // Other errors are testkit's own; they land in `ProviderError::Testkit` via `From<String>`.
            other => other.to_string().into(),
        }
    }
}
