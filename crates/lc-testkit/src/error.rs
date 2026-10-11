//! lc-testkit error types and the bridge to `ProviderError`.

use lc_providers::ProviderError;

use crate::golden::GoldenError;
use crate::replay::{MatchDiff, ReplayStrategy};

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
    /// (explicit error, no silent FIFO fallback). `left` is the remaining queue length and
    /// `closest` (when the queue is non-empty) names the nearest recorded exchange plus the
    /// field-level divergence — so a runtime `Message.id` drift is debuggable instead of a
    /// bare "no match" (the usual fix is [`crate::ReplayProvider::ignore_ids`]).
    #[error("replay has no recording matching request messages (strategy=Exact, {left} exchange(s) left)")]
    ReplayNoMatch {
        left: usize,
        closest: Option<MatchDiff>,
    },
    /// A recording line deserialized but failed the minimal replay-qualification check
    /// (replay relies on every tool call carrying a non-empty `id` and `name`), so it
    /// cannot be replayed faithfully.
    #[error("recording exchange[{index}] fails replay qualification: {detail}")]
    InvalidRecording { index: usize, detail: String },
    /// Inner model error, passed through losslessly from the real provider error.
    #[error("inner model error: {0}")]
    Inner(#[from] ProviderError),
    /// `ReplayStrategy::Exact` (`ReplayProvider`) matches the request against the recorded exchange
    /// by a full-message signature that must be element-wise equal. But [`crate::ReplayPredictor`] sends a
    /// fresh single human message per eval input, so it can never satisfy `Exact` (or `ByToolName`,
    /// which silently degrades to FIFO) — the golden-replay predictor is only meaningful under
    /// [`ReplayStrategy::Fifo`]. Non-Fifo strategies are therefore a footgun and rejected here.
    #[error("golden replay predictor only supports ReplayStrategy::Fifo (got {strategy:?}); use Fifo to keep example-N ↔ recording-N alignment")]
    GoldenReplayStrategyMismatch { strategy: ReplayStrategy },
    /// A recording could not be turned into a golden evaluation example (N2).
    #[error("golden dataset build failed: {0}")]
    Golden(#[from] GoldenError),
}

impl From<TestkitError> for ProviderError {
    fn from(e: TestkitError) -> Self {
        match e {
            // Pass the real provider error through losslessly.
            TestkitError::Inner(p) => p,
            // Other errors are testkit's own. `ProviderError` is `#[non_exhaustive]` and
            // `ProviderError::Testkit` only carries a `String`, so a typed cross-crate bridge is
            // impossible here; embedding the full `Debug` form (variant name + every field) keeps
            // the original variant detail instead of flattening it to the Display-only message.
            other => format!("{other:?}").into(),
        }
    }
}
