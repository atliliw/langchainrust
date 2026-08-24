// lc-memory/src/context_window/manager.rs
//! Context window manager for fitting messages within a token budget.

use std::sync::Arc;

use lc_core::language_models::BaseChatModel;
use lc_core::token_counter::{TiktokenCounter, TokenCounter};
use lc_schema::Message;

use crate::base::MemoryError;

use super::trimmer::Strategy;

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
}

impl<M: BaseChatModel> ContextWindow<M> {
    /// Creates a new ContextWindow with the Truncate strategy and default TiktokenCounter.
    ///
    /// P1-4: 返回 `Result` 而非 panic——tiktoken 模型下载/加载失败(离线/缺模型)
    /// 时返回 [`MemoryError`],库构造器不再因本地环境崩溃。
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
        })
    }

    /// Sets a custom token counter.
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = counter;
        self
    }

    /// Returns the max token limit.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Fits messages within the token limit by applying the configured strategy.
    ///
    /// - If total tokens are within `max_tokens`, returns messages as-is.
    /// - If over, applies the `Strategy` (truncate or summarize).
    ///
    /// # Budget semantics (P1-4 契约)
    ///
    /// **`Strategy::Truncate`**: System 消息恒保留且**不占预算**(M7);若 System
    /// 消息自身超过 `max_tokens`,原样返回(结果可能超预算)。即使预算小到一条
    /// 对话都放不下,也至少保留最新一条非 System 消息,不静默清空历史(H7)。
    /// 调用方不应假设 `fit` 的返回结果一定在预算内。
    ///
    /// **`Strategy::Summarize`**: 摘要占位计入预算,但 LLM 实际产出的摘要 token
    /// 数未知——预算语义与 Truncate 不同,两者在 System 消息上口径不一致是有意为之,
    /// 各策略自行定义。
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
    /// and they do **not** count toward the budget (P1-4 契约): if the system
    /// messages alone exceed `max_tokens`, they are returned as-is and the
    /// result may exceed the budget.
    ///
    /// M7: system messages 不计入预算(base 从 0 起算),不再挤占可用上下文。
    ///
    /// H7: 即使预算小到一条对话都放不下,也至少保留最新一条非 System 消息,
    /// 绝不静默丢光全部历史(结果可能超预算,契约允许)。
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

        // M7: System 消息恒保留且**不占预算**,base 从 0 起算——旧实现把
        // `count_messages(&system_messages)` 计入 base,挤占可用上下文。
        let mut running_tokens: usize = 0;

        // Pre-compute per-message incremental cost.
        // For a single message, count_messages returns (4 + content_tokens + 2).
        // The incremental cost of adding a message to an existing list is (4 + content_tokens).
        // We subtract the boundary overhead (2) from single-message counts to get incremental cost.
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
                // H7: 预算连最新一条都放不下时,仍保留最新一条——结果可能超预算
                // (契约允许),但绝不静默丢光全部对话历史。
                kept.push(msg);
            } else {
                // This message would push us over; stop adding more.
                break;
            }
        }

        kept.reverse();

        let mut result = system_messages;
        result.extend(kept);
        Ok(result)
    }

    /// Summarizes old messages using the LLM, replacing them with a
    /// single system message containing the summary.
    ///
    /// System messages are preserved. The oldest non-system messages
    /// are summarized until the remaining messages fit within the budget.
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

        // Find how many recent messages we can keep within the budget,
        // reserving some space for the summary message.
        // We try keeping the newest messages and summarizing the rest.
        // Iterate from the smallest window (fewest recent messages) to the largest,
        // keeping track of the best (smallest i = most messages kept) that fits.
        let mut keep_from_idx = other_messages.len(); // default: keep all (no summarization)

        for i in 0..other_messages.len() {
            let recent = &other_messages[i..];
            let mut candidate = system_messages.clone();
            // Reserve space for a summary message (estimate ~100 tokens).
            candidate.push(Message::system("summary placeholder"));
            candidate.extend(recent.iter().cloned());

            let tokens = self.counter.count_messages(&candidate) as usize;
            if tokens <= self.max_tokens {
                keep_from_idx = i;
                break;
            }
        }

        // If we can't even fit the recent messages with a summary placeholder,
        // fall back to truncation for the recent portion.
        if keep_from_idx >= other_messages.len() {
            // H7: 预算小到连摘要占位都放不下时,退化为截断——截断保证至少保留
            // 最新一条;旧实现 `truncate(system_messages)` 会静默丢光全部历史。
            let mut all = system_messages;
            all.extend(other_messages);
            return self.truncate(all);
        }

        let to_summarize = &other_messages[..keep_from_idx];
        let to_keep = &other_messages[keep_from_idx..];

        if to_summarize.is_empty() {
            // All messages fit with the summary placeholder; no need to summarize.
            let mut result = system_messages;
            result.extend(to_keep.to_vec());
            return Ok(result);
        }

        // Format the conversation for summarization.
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

        let summary_messages = vec![Message::human(&prompt)];

        let result = llm
            .invoke(summary_messages, None)
            .await
            .map_err(|e| MemoryError::SaveError(format!("LLM summarization failed: {}", e)))?;

        let summary_message = Message::system(format!("[Conversation Summary] {}", result.content));

        // Build final message list: system + summary + recent.
        let mut final_messages = system_messages;
        final_messages.push(summary_message);
        final_messages.extend(to_keep.to_vec());

        // Verify the final result fits; if not, truncate the recent portion.
        let final_tokens = self.counter.count_messages(&final_messages) as usize;
        if final_tokens > self.max_tokens {
            return self.truncate(final_messages);
        }

        Ok(final_messages)
    }
}
