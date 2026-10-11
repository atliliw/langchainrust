// lc-memory/src/context_window/manager.rs
//! Context window manager for fitting messages within a token budget.

use std::sync::Arc;

use lc_core::language_models::BaseChatModel;
use lc_core::token_counter::{TiktokenCounter, TokenCounter};
use lc_schema::Message;

use crate::base::MemoryError;

use super::trimmer::Strategy;

/// Default share of `max_tokens` reserved for the summary message under
/// [`Strategy::Summarize`].
///
/// The reservation is what decides how many recent messages survive next to the
/// summary, so it must reflect the *real* size of a summary rather than the size
/// of a probe string (T2).
pub const DEFAULT_SUMMARY_RESERVE_RATIO: f64 = 0.25;

/// Hard ceiling on the reservation: the recent window always keeps at least half
/// the budget, otherwise a large summary would squeeze out the live conversation.
const MAX_SUMMARY_RESERVE_RATIO: f64 = 0.5;

/// Drop leading `Tool` messages from a kept window.
///
/// A window that opens on a tool result whose matching assistant `tool_calls`
/// fell outside the window is malformed for OpenAI/Anthropic (400). Shared by
/// the truncate path (0.22.0 H-M2) and both summarize paths (D8).
fn strip_leading_tool(mut messages: Vec<Message>) -> Vec<Message> {
    while messages
        .first()
        .is_some_and(|m| matches!(m.message_type, lc_schema::MessageType::Tool { .. }))
    {
        messages.remove(0);
    }
    messages
}

/// Context window manager for fitting messages within a token budget.
///
/// Counts tokens using a `TokenCounter` (defaults to `TiktokenCounter`),
/// and applies a `Strategy` when messages exceed `max_tokens`.
pub struct ContextWindow<M: BaseChatModel> {
    /// Maximum token count allowed.
    max_tokens: usize,
    /// Token counter implementation.
    counter: Arc<dyn TokenCounter>,
    /// Strategy for reducing messages when over the limit.
    strategy: Strategy<M>,
    /// Share of `max_tokens` reserved for the summary message (T2).
    summary_reserve_ratio: f64,
    /// Explicit reservation in tokens, overriding `summary_reserve_ratio`.
    summary_reserve_override: Option<usize>,
}

impl<M: BaseChatModel> ContextWindow<M> {
    /// Creates a new ContextWindow with the Truncate strategy and default TiktokenCounter.
    ///
    /// P1-4: returns `Result` instead of panicking — a tiktoken model download/load failure
    /// (offline / missing model) returns [`MemoryError`], so the library constructor no longer
    /// crashes on a bad local environment.
    pub fn new(max_tokens: usize) -> Result<Self, MemoryError> {
        Self::build(max_tokens, Strategy::Truncate)
    }

    /// Creates a new ContextWindow with a specific strategy and default TiktokenCounter.
    pub fn with_strategy(max_tokens: usize, strategy: Strategy<M>) -> Result<Self, MemoryError> {
        Self::build(max_tokens, strategy)
    }

    /// Creates a new ContextWindow with a custom max token limit and default counter/strategy.
    pub fn with_max_tokens(max_tokens: usize) -> Result<Self, MemoryError> {
        Self::new(max_tokens)
    }

    /// Shared constructor: loads the TiktokenCounter once, propagating failures.
    fn build(max_tokens: usize, strategy: Strategy<M>) -> Result<Self, MemoryError> {
        let counter = TiktokenCounter::new()
            .map_err(|e| MemoryError::Other(format!("Failed to load tiktoken encoder: {}", e)))?;
        Ok(Self {
            max_tokens,
            counter: Arc::new(counter),
            strategy,
            summary_reserve_ratio: DEFAULT_SUMMARY_RESERVE_RATIO,
            summary_reserve_override: None,
        })
    }

    /// Sets a custom token counter.
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = counter;
        self
    }

    /// Sets an explicit token reservation for the summary message
    /// ([`Strategy::Summarize`] only), overriding the default ratio.
    ///
    /// The reservation is capped at half of `max_tokens` so the live conversation
    /// always keeps room. This is the *initial estimate*; the real summary is
    /// measured after generation and the window is rebuilt from that count, so an
    /// inaccurate reservation costs an extra trim, never a budget overrun.
    pub fn with_summary_reserve_tokens(mut self, tokens: usize) -> Self {
        self.summary_reserve_override = Some(tokens);
        self
    }

    /// Sets the share of `max_tokens` reserved for the summary message
    /// ([`Strategy::Summarize`] only). Clamped to `[0, 0.5]`.
    pub fn with_summary_reserve_ratio(mut self, ratio: f64) -> Self {
        self.summary_reserve_ratio = ratio.clamp(0.0, MAX_SUMMARY_RESERVE_RATIO);
        self
    }

    /// Returns the max token limit.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Tokens reserved for the summary message: the explicit override when set,
    /// otherwise `max_tokens * summary_reserve_ratio`.
    ///
    /// Always capped at half the budget (see `MAX_SUMMARY_RESERVE_RATIO`).
    pub fn summary_reserve(&self) -> usize {
        let raw = self
            .summary_reserve_override
            .unwrap_or((self.max_tokens as f64 * self.summary_reserve_ratio) as usize);
        raw.min(self.max_tokens / 2)
    }

    /// Fits messages within the token limit by applying the configured strategy.
    ///
    /// - If total tokens are within `max_tokens`, returns messages as-is.
    /// - If over, applies the `Strategy` (truncate or summarize).
    ///
    /// # Budget semantics (P1-4 contract)
    ///
    /// **`Strategy::Truncate`**: System messages are always kept and do **not** consume budget
    /// (M7); if the System messages alone exceed `max_tokens`, they are returned as-is (the
    /// result may exceed the budget). Even when the budget is too small for a single
    /// conversation message, at least the newest non-System message is kept — history is never
    /// silently emptied (H7). Callers must not assume `fit`'s result is always within budget.
    ///
    /// **`Strategy::Summarize`**: a *reservation* for the summary message counts toward the budget
    /// when the recent window is chosen (see [`Self::summary_reserve`]), and once the summary has
    /// been generated its **real token count** is measured and the window is rebuilt from that
    /// number — so the returned list honours `max_tokens` rather than a guess. System messages are
    /// charged to the summarize budget (unlike Truncate, where they are free); the intentionally
    /// inconsistent handling of System messages between the two is deliberate — each strategy
    /// defines its own.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages to fit.
    ///
    /// # Returns
    /// A vector of messages that fits within the token budget.
    pub async fn fit(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        let total_tokens = self.counter.count_messages(&messages) as usize;

        if total_tokens <= self.max_tokens {
            return Ok(messages);
        }

        match &self.strategy {
            Strategy::Truncate => self.truncate(messages),
            Strategy::Summarize {
                llm,
                summary_prompt,
            } => self.summarize(messages, llm, summary_prompt).await,
        }
    }

    /// Truncates messages by removing the oldest non-system messages
    /// until the total fits within `max_tokens`.
    ///
    /// System messages are always preserved and placed at the beginning,
    /// and they do **not** count toward the budget (P1-4 contract): if the system
    /// messages alone exceed `max_tokens`, they are returned as-is and the
    /// result may exceed the budget.
    ///
    /// M7: system messages do not consume budget (base starts at 0), so they no longer crowd
    /// out usable context.
    ///
    /// H7: even when the budget is too small for a single conversation message, at least the
    /// newest non-System message is kept — history is never silently emptied (the result may
    /// exceed the budget, which the contract allows).
    ///
    /// M10: Optimized from O(n^2) to O(n) by computing token counts
    /// incrementally instead of rebuilding and recounting the full candidate
    /// list on every iteration.
    fn truncate(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        // Separate system messages from the rest.
        let mut system_messages: Vec<Message> = Vec::new();
        let mut other_messages: Vec<Message> = Vec::new();

        for msg in messages {
            if matches!(msg.message_type, lc_schema::MessageType::System) {
                system_messages.push(msg);
            } else {
                other_messages.push(msg);
            }
        }

        // M7: System messages are always kept and do **not** consume budget — base starts at 0.
        // The old implementation counted `count_messages(&system_messages)` into base, crowding
        // out usable context.
        //
        // The running total starts at the shared `+2` conversation-boundary marker
        // (`count_messages` adds it once for the whole list) so the kept-suffix cost is
        // accounted exactly. The old code started at 0 and undercounted every window by the
        // boundary — the loop could keep a suffix that is really 2 tokens over budget. This
        // matches `summarize`'s `2 + system_cost + ...` treatment of the same boundary.
        let mut running_tokens: usize = 2;

        // Pre-compute per-message incremental cost.
        // For a single message, count_messages returns (4 + content_tokens + 2).
        // The incremental cost of adding a message to an existing list is (4 + content_tokens).
        // We subtract the boundary overhead (2) from single-message counts to get incremental cost.
        // The one shared boundary marker is charged separately above (running_tokens init).
        let msg_incremental_costs: Vec<usize> = other_messages
            .iter()
            .map(|m| {
                let single_count = self.counter.count_messages(std::slice::from_ref(m)) as usize;
                // single_count = 4 + content_tokens + 2, incremental = 4 + content_tokens
                single_count.saturating_sub(2)
            })
            .collect();

        // Walk from the end (newest) and accumulate until we exceed the budget.
        let mut kept: Vec<Message> = Vec::new();

        for (msg, cost) in other_messages
            .into_iter()
            .rev()
            .zip(msg_incremental_costs.into_iter().rev())
        {
            if running_tokens + cost <= self.max_tokens {
                running_tokens += cost;
                kept.push(msg);
            } else if kept.is_empty() {
                // H7: when even the newest message cannot fit, still keep the newest one — the
                // result may exceed the budget (contract allows), but history is never silently
                // emptied. L8: *also* charge its cost to `running_tokens` so it now reads as over
                // budget and the next (older) message falls into the `break` below — otherwise we
                // would keep appending older messages on top of an already-over-budget window.
                running_tokens += cost;
                kept.push(msg);
            } else {
                // This message would push us over; stop adding more.
                break;
            }
        }

        kept.reverse();

        // 0.22.0 H-M2: the kept suffix must not open on an orphaned Tool
        // message (a tool result whose matching assistant tool_calls fell
        // outside the window). A standalone tool message is malformed for
        // OpenAI/Anthropic → 400. Drop leading Tool messages until the window
        // opens on a non-Tool message.
        let kept = strip_leading_tool(kept);

        let mut result = system_messages;
        result.extend(kept);
        Ok(result)
    }

    /// Token cost of one message as an *incremental* addition to a list: the
    /// per-message overhead plus its content, excluding the list's shared `+2`
    /// conversation-boundary marker.
    fn message_cost(&self, message: &Message) -> usize {
        (self.counter.count_messages(std::slice::from_ref(message)) as usize).saturating_sub(2)
    }

    /// Start index of the largest recent suffix that fits alongside `reserve`
    /// tokens of summary budget.
    ///
    /// `msg_cost[i]` is the incremental cost of `other_messages[i]`; the kept
    /// window is `[idx, ..)` and `[..idx)` is summarized. Returns `None` when even
    /// an empty recent window cannot fit the reservation — the caller then
    /// degrades to [`Self::truncate`] (H7).
    ///
    /// L8: `count_messages` is additive per message plus one shared boundary, so
    /// walking the index forward with a running total is O(n) — the old version
    /// rebuilt and re-counted the whole candidate on every step (O(n²)).
    fn split_for(&self, msg_cost: &[usize], system_cost: usize, reserve: usize) -> Option<usize> {
        // +2 once for the whole concatenated list's boundary marker.
        let mut total = 2 + system_cost + reserve + msg_cost.iter().sum::<usize>();
        for (i, cost) in msg_cost.iter().enumerate() {
            if total <= self.max_tokens {
                return Some(i);
            }
            // Drop the front message to shrink the retained suffix.
            total = total.saturating_sub(*cost);
        }
        // Every message moved into the summary: does the summary alone fit?
        (total <= self.max_tokens).then_some(msg_cost.len())
    }

    /// Renders the conversation, asks the LLM for a summary, and wraps the answer
    /// in the `[Conversation Summary]` system message.
    async fn generate_summary(
        &self,
        to_summarize: &[Message],
        llm: &Arc<M>,
        summary_prompt: &str,
    ) -> Result<Message, MemoryError> {
        let conversation_text = to_summarize
            .iter()
            .map(|msg| {
                let role = match msg.message_type {
                    lc_schema::MessageType::Human => "Human",
                    lc_schema::MessageType::AI => "AI",
                    lc_schema::MessageType::System => "System",
                    lc_schema::MessageType::Tool { .. } => "Tool",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = summary_prompt.replace("{conversation}", &conversation_text);

        let result = llm
            .invoke(vec![Message::human(&prompt)], None)
            .await
            .map_err(|e| MemoryError::SaveError(format!("LLM summarization failed: {}", e)))?;

        Ok(Message::system(format!(
            "[Conversation Summary] {}",
            result.content
        )))
    }

    /// Summarizes old messages using the LLM, replacing them with a
    /// single system message containing the summary.
    ///
    /// System messages are preserved. The oldest non-system messages
    /// are summarized until the remaining messages fit within the budget.
    ///
    /// # Budget (T2)
    ///
    /// Two token counts decide the result, in this order:
    ///
    /// 1. **Reservation** — [`Self::summary_reserve`] tokens of the budget are held
    ///    back for the summary while the recent window is chosen. This is an
    ///    estimate, and it is the only thing the estimate is allowed to influence.
    /// 2. **Measurement** — once the LLM has answered, the summary's *real* token
    ///    count is taken and the window is rebuilt from it, so a summary that came
    ///    out larger than reserved shrinks the recent window instead of pushing the
    ///    result over `max_tokens`.
    ///
    /// The previous implementation reserved the token cost of the literal string
    /// `"summary placeholder"` — a handful of tokens — and then, on overflow,
    /// called `truncate(final_messages)`. Because the summary is a `System`
    /// message and `truncate` charges System messages **zero** budget, that
    /// fallback kept the over-budget summary *and* all recent messages: the
    /// returned list could exceed `max_tokens` by the whole length of the summary.
    /// The trim below is applied to the recent window only, so the summary's cost
    /// is always paid.
    async fn summarize(
        &self,
        messages: Vec<Message>,
        llm: &Arc<M>,
        summary_prompt: &str,
    ) -> Result<Vec<Message>, MemoryError> {
        // Separate system messages from the rest.
        let mut system_messages: Vec<Message> = Vec::new();
        let mut other_messages: Vec<Message> = Vec::new();

        for msg in messages {
            if matches!(msg.message_type, lc_schema::MessageType::System) {
                system_messages.push(msg);
            } else {
                other_messages.push(msg);
            }
        }

        if other_messages.is_empty() {
            // Only system messages; nothing to summarize.
            return Ok(system_messages);
        }

        // L8: one incremental cost per message (shared `+2` boundary excluded), so
        // the window search is O(n) rather than a rebuild-and-recount per step.
        let msg_cost: Vec<usize> = other_messages
            .iter()
            .map(|m| self.message_cost(m))
            .collect();
        let system_cost: usize = system_messages.iter().map(|m| self.message_cost(m)).sum();

        // Step 1: pick the recent window using the reserved summary budget.
        let reserve = self.summary_reserve();
        let Some(keep_from) = self.split_for(&msg_cost, system_cost, reserve) else {
            // H7: not even an empty recent window fits the reservation → degrade to
            // truncation, which guarantees at least the newest message survives.
            let mut all = system_messages;
            all.extend(other_messages);
            return self.truncate(all);
        };

        let mut to_keep = strip_leading_tool(other_messages[keep_from..].to_vec());

        if keep_from == 0 {
            // Everything fits next to the reservation — nothing to summarize.
            let mut result = system_messages;
            result.extend(to_keep);
            return Ok(result);
        }

        // Step 2: generate the summary and measure it for real.
        let summary_message = self
            .generate_summary(&other_messages[..keep_from], llm, summary_prompt)
            .await?;
        let measured = self.message_cost(&summary_message);

        if measured > reserve {
            // The real summary is bigger than reserved, so fewer recent messages
            // fit. Re-derive the window from the MEASURED count; if the summary
            // plus the system messages alone already exceed the budget, there is
            // nothing left to trade away and truncation is the only honest answer.
            let Some(rebuilt) = self.split_for(&msg_cost, system_cost, measured) else {
                let mut all = system_messages;
                all.extend(other_messages);
                return self.truncate(all);
            };
            if rebuilt > keep_from {
                to_keep = strip_leading_tool(other_messages[rebuilt..].to_vec());
            }
        }

        // Assemble: system + summary + recent window. The system messages are cloned
        // rather than moved so the fallback below can still rebuild the original
        // conversation instead of truncating an already-emptied candidate.
        let mut candidate: Vec<Message> =
            Vec::with_capacity(system_messages.len() + 1 + to_keep.len());
        candidate.extend(system_messages.iter().cloned());
        candidate.push(summary_message);
        let tail_start = candidate.len();
        candidate.extend(to_keep);

        // Belt and braces: the rebuild above is exact whenever the split moved, but
        // the Tool-message cleanup can also have removed entries. Trim the OLDEST
        // recent messages (the ones nearest the summarized region) until the list
        // fits; never `truncate(candidate)`, which would treat the summary as a free
        // System message and blow the budget — the very hole this change closes.
        while candidate.len() > tail_start
            && self.counter.count_messages(&candidate) as usize > self.max_tokens
        {
            candidate.remove(tail_start);
        }

        if self.counter.count_messages(&candidate) as usize > self.max_tokens {
            // Even system + summary alone do not fit: the summary is unusable, so drop
            // it and fall back to plain truncation of the ORIGINAL conversation, which
            // guarantees the newest message survives (H7).
            let mut all = system_messages;
            all.extend(other_messages);
            return self.truncate(all);
        }

        Ok(candidate)
    }
}
