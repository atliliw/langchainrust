//! N2 (v0.24.0): recorded traces → golden eval dataset → offline scoring.
//!
//! Closes the regression loop between this crate's record/replay harness and
//! `lc-evaluation`:
//!
//! 1. **Record** real traffic with [`RecordingProvider`](crate::RecordingProvider) — one
//!    [`RecordedExchange`] per model call lands in a JSONL file.
//! 2. **Sink** the recording into a golden [`Dataset`] via [`golden_dataset`] /
//!    [`golden_dataset_from_file`], optionally checking it in as evaluation JSONL with
//!    [`write_golden_jsonl`]. Each exchange maps to one [`Example`]:
//!    - `input`  — the **last** human message in the recorded request history (the actual
//!      user turn; agent-loop re-requests still carry it in history),
//!    - `reference` — the recorded response text,
//!    - `contexts` — non-empty tool-result messages in order, so RAGAS-style
//!      [`RagEvaluator`]s can score retrieval too.
//! 3. **Score** prompt/model changes against the golden set. Two ways:
//!    - fully offline smoke/baseline run: [`replay_golden_from_file`] hands back the
//!      dataset plus a [`ReplayPredictor`] backed by the same file — zero network, and a
//!      pinned model answers exactly its recorded reference;
//!    - real regression run: load the checked-in JSONL with `Dataset::from_jsonl` and run
//!      it through `EvalRunner` with a predictor wrapping the **new** prompt/model.
//! 4. **Gate** with `lc_evaluation::compare_reports(&baseline, &candidate, tolerance)`:
//!    any evaluator mean dropping beyond the tolerance fails the gate.
//!
//! Agent-loop traces record several exchanges per user turn (one per model call); the
//! intermediate ones are often pure tool-call responses with empty text. Filter with
//! [`is_scoring_candidate`] before [`golden_dataset`] when converting such a file.

use std::io::{BufWriter, Write};
use std::path::Path;

use async_trait::async_trait;
use lc_core::language_models::BaseChatModel;
use lc_evaluation::{Dataset, EvalError, Example, Predictor};
use lc_schema::{Message, MessageType};

use crate::error::TestkitError;
use crate::recording::{read_exchanges, RecordedExchange};
use crate::replay::{ReplayProvider, ReplayStrategy};

/// Why an exchange cannot become a golden example.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GoldenError {
    /// The recorded request carries no human message, so there is no eval input.
    #[error("recording exchange at index {0} has no human message to use as eval input")]
    NoUserMessage(usize),
    /// The recorded response text is empty (typically a pure tool-call response),
    /// so there is no reference answer.
    #[error(
        "recording exchange at index {0} has empty assistant text to use as the reference answer"
    )]
    EmptyReference(usize),
}

/// Whether an exchange can become a scored golden example: it has a human prompt
/// and non-empty response text.
///
/// Use this to filter agent-loop recordings, whose intermediate model calls may be pure
/// tool-call requests (`content == ""`).
pub fn is_scoring_candidate(exchange: &RecordedExchange) -> bool {
    let has_prompt = exchange
        .messages
        .iter()
        .any(|m| matches!(m.message_type, MessageType::Human));
    has_prompt && !exchange.response.content.trim().is_empty()
}

/// Converts one exchange into a golden example; `index` anchors the error position.
fn convert_one(index: usize, exchange: &RecordedExchange) -> Result<Example, GoldenError> {
    // The last human message is the actual user turn of this request.
    let input = exchange
        .messages
        .iter()
        .rev()
        .find(|m| matches!(m.message_type, MessageType::Human))
        .map(|m| m.content.trim().to_string())
        .ok_or(GoldenError::NoUserMessage(index))?;

    let reference = exchange.response.content.trim().to_string();
    if reference.is_empty() {
        return Err(GoldenError::EmptyReference(index));
    }

    // Tool observations stand in for retrieved documents in RAG traces; order is
    // rank/occurrence order, which RagEvaluator context precision relies on.
    let contexts: Vec<String> = exchange
        .messages
        .iter()
        .filter_map(|m| match &m.message_type {
            MessageType::Tool { .. } if !m.content.trim().is_empty() => {
                Some(m.content.trim().to_string())
            }
            _ => None,
        })
        .collect();

    Ok(Example::with_contexts(input, reference, contexts))
}

/// Strictly converts recorded exchanges into a golden dataset.
///
/// Fails fast on the first unscorable exchange (no human prompt / empty reference);
/// filter with [`is_scoring_candidate`] first when mixing in agent-loop traces.
pub fn golden_dataset(exchanges: &[RecordedExchange]) -> Result<Dataset, GoldenError> {
    exchanges
        .iter()
        .enumerate()
        .map(|(i, ex)| convert_one(i, ex))
        .collect::<Result<Vec<_>, _>>()
        .map(Dataset::new)
}

/// Reads a recording file and builds the golden dataset from it.
pub fn golden_dataset_from_file(path: impl AsRef<Path>) -> Result<Dataset, TestkitError> {
    let exchanges = read_exchanges(path)?;
    Ok(golden_dataset(&exchanges)?)
}

/// Writes the golden dataset as evaluation JSONL (`{input, reference, contexts}` per
/// line), the format `Dataset::from_jsonl` reads back without lc-testkit at eval time.
///
/// Truncates/overwrites `path`.
pub fn write_golden_jsonl(dataset: &Dataset, path: impl AsRef<Path>) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for example in &dataset.examples {
        serde_json::to_writer(&mut writer, example)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()
}

/// A [`Predictor`] driven by a [`ReplayProvider`]: each eval input is sent as a single
/// human message and the popped recording's response text is the prediction.
///
/// This is the offline half of the N2 loop — scoring a prompt/agent build needs no API
/// key and never hits the network. Under [`ReplayStrategy::Fifo`] (the default and the
/// intended mode here), example N pops recording N, so feeding the predictor the dataset
/// built from the same file reproduces every recorded reference exactly. The queue is
/// shared: clones of the underlying [`ReplayProvider`] draw from the same recordings.
pub struct ReplayPredictor {
    provider: ReplayProvider,
}

impl ReplayPredictor {
    /// Wraps a replay provider as an evaluation predictor.
    pub fn new(provider: ReplayProvider) -> Self {
        Self { provider }
    }

    /// Number of recordings left in the shared replay queue.
    pub fn remaining(&self) -> usize {
        self.provider.len()
    }
}

#[async_trait]
impl Predictor for ReplayPredictor {
    async fn predict(&self, input: &str) -> Result<String, EvalError> {
        let messages = vec![Message::human(input.to_string())];
        self.provider
            .chat(messages, None)
            .await
            .map(|result| result.content)
            .map_err(|e| EvalError::PredictorError(e.to_string()))
    }
}

/// One-click offline N2 run setup: reads a recording once and returns both the golden
/// dataset and a FIFO [`ReplayPredictor`] over the same exchanges.
///
/// Pass them to `EvalRunner::run` for a zero-network regression smoke run (every
/// reference should reproduce under `Fifo`). For real-model regression runs, export the
/// dataset with [`write_golden_jsonl`] and load it back with `Dataset::from_jsonl`
/// against a predictor wrapping the new prompt/model.
pub fn replay_golden_from_file(
    path: impl AsRef<Path>,
    strategy: ReplayStrategy,
) -> Result<(Dataset, ReplayPredictor), TestkitError> {
    let exchanges = read_exchanges(path)?;
    let dataset = golden_dataset(&exchanges)?;
    let provider = ReplayProvider::from_exchanges(exchanges).with_strategy(strategy);
    Ok((dataset, ReplayPredictor::new(provider)))
}
