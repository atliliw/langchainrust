// lc-memory/src/context_manager.rs
//! Composable context management layer (`ContextManager`).
//!
//! T2-1 (v0.26.0): the historical lc-memory design forces you to **pick one** memory
//! type — a `ConversationBufferMemory`, a `ConversationSummaryMemory`, a window, a
//! semantic store. This module flips that to a **composable stack**: the *read*, *write*,
//! *compress* and *retrieve* concerns are expressed as independent [`ReadStrategy`] /
//! [`WriteStrategy`] / [`CompressStrategy`] / [`RetrieveStrategy`] implementations that
//! can be **combined** on a single [`ContextManager`] via [`ContextManagerBuilder`], and
//! each step is optional with a sensible default:
//!
//! - read defaults to a [`ConversationBufferMemory`]-backed strategy;
//! - write defaults to the same buffer-backed strategy (so save→load roundtrips);
//! - compress defaults to a **passthrough** (no compression);
//! - retrieve defaults to **`None`** (no retrieval injection).
//!
//! Existing single components are reused as concrete strategy implementations instead of
//! being rewritten: Buffer (read/write defaults),
//! [`ContextWindow`] (an opt-in
//! [`ContextWindowCompressStrategy`]), the semantic [`MemoryStore`] (retrieve + indexing
//! write strategies). Nothing in this module mutates the existing components.
//!
//! # Order of operations (deterministic)
//!
//! `load_memory_variables` runs **read → compress → merge-retrieve**, in that exact,
//! fixed order, so a combination is reproducible bit-for-bit: the same inputs always
//! yield the same variables (covered by the determinism test). `save_context` routes
//! entirely through the write strategy.
//!
//! `lc-memory`'s existing `BaseMemory` / `BaseChatMemory` signatures are untouched;
//! `ContextManager` additionally implements `BaseMemory`, so it is drop-in usable
//! anywhere a single memory previously was.
//!
//! # Builder
//!
//! ```
//! use lc_memory::context_manager::ContextManagerBuilder;
//!
//! let manager = ContextManagerBuilder::new()
//!     // .read(...).write(...).compress(...).retrieve(...) — all optional
//!     .build();
//! ```

use async_trait::async_trait;
use lc_core::token_counter::{CharRatioCounter, TokenCounter};
use lc_schema::{Message, MessageType};
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::base::{serialize_message_to_prompt, BaseMemory, MemoryError};
use crate::buffer::ConversationBufferMemory;
use crate::context_window::ContextWindow;
use crate::semantic::{MemoryItem, MemoryQuery, MemoryStore};
use lc_core::language_models::BaseChatModel;

// ───────────────────────────── strategy traits ─────────────────────────────

/// Reads raw conversation history from some backing store.
///
/// The manager calls [`ReadStrategy::load`] first in `load_memory_variables`, then
/// hands the returned messages to the compress strategy.
#[async_trait]
pub trait ReadStrategy: Send + Sync {
    /// Read the stored history as a message list.
    async fn load(&self, inputs: &HashMap<String, String>) -> Result<Vec<Message>, MemoryError>;

    /// Clear the state this strategy owns. Stateless strategies are a no-op by
    /// default; stateful ones (e.g. the buffer store) clear their buffer. Called
    /// by [`ContextManager::clear`].
    async fn clear(&self) -> Result<(), MemoryError> {
        Ok(())
    }
}

/// Persists a completed turn into some backing store.
///
/// The manager routes `save_context` entirely through [`WriteStrategy::save`].
#[async_trait]
pub trait WriteStrategy: Send + Sync {
    /// Persist the input/output turn.
    async fn save(
        &self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError>;

    /// Clear the state this strategy owns. Stateless strategies are a no-op by
    /// default; stateful ones (e.g. the buffer store) clear their buffer. Called
    /// by [`ContextManager::clear`].
    async fn clear(&self) -> Result<(), MemoryError> {
        Ok(())
    }
}

/// Compresses an over-budget history into one that fits a budget.
///
/// Applied after [`ReadStrategy::load`] and before retrieval merge. Stateless and pure so
/// the composition stays byte-for-byte deterministic.
#[async_trait]
pub trait CompressStrategy: Send + Sync {
    /// Reduce the history to within the strategy's budget.
    async fn compress(&self, history: Vec<Message>) -> Result<Vec<Message>, MemoryError>;
}

/// Optionally injects external, meaning-based context on load.
///
/// Query derives from the current `inputs`; results are appended to the compressed
/// history. A `None` strategy on the manager means no injection.
#[async_trait]
pub trait RetrieveStrategy: Send + Sync {
    /// Produce extra context messages (empty when nothing matches).
    async fn retrieve(&self, inputs: &HashMap<String, String>)
        -> Result<Vec<Message>, MemoryError>;
}

// ─────────────────────────── concrete strategies ───────────────────────────

/// Buffer-backed [`ReadStrategy`] reusing a [`ConversationBufferMemory`] over a shared
/// store. Works with a [`BufferWriteStrategy`] sharing the same `Arc` to roundtrip.
#[derive(Debug, Clone)]
pub struct BufferReadStrategy {
    store: Arc<Mutex<ConversationBufferMemory>>,
}

impl BufferReadStrategy {
    /// Wrap an existing shared store.
    pub fn new(store: Arc<Mutex<ConversationBufferMemory>>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl ReadStrategy for BufferReadStrategy {
    async fn load(&self, _inputs: &HashMap<String, String>) -> Result<Vec<Message>, MemoryError> {
        let mem = self.store.lock().await;
        Ok(mem.chat_memory().messages().to_vec())
    }
}

/// Buffer-backed [`WriteStrategy`] reusing a [`ConversationBufferMemory`] over a shared
/// store.
#[derive(Debug, Clone)]
pub struct BufferWriteStrategy {
    store: Arc<Mutex<ConversationBufferMemory>>,
}

impl BufferWriteStrategy {
    /// Wrap an existing shared store.
    pub fn new(store: Arc<Mutex<ConversationBufferMemory>>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl WriteStrategy for BufferWriteStrategy {
    async fn save(
        &self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        let mut mem = self.store.lock().await;
        mem.save_context(inputs, outputs).await
    }
}

/// Default [`CompressStrategy`]: leaves the history untouched (no compression).
#[derive(Debug, Clone, Copy, Default)]
pub struct PassthroughCompressStrategy;

#[async_trait]
impl CompressStrategy for PassthroughCompressStrategy {
    async fn compress(&self, history: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        Ok(history)
    }
}

/// Deterministic, offline-safe [`CompressStrategy`] that drops the oldest non-system
/// messages until the remainder fits a token budget.
///
/// Reuses lc-core's [`TokenCounter`] (default [`CharRatioCounter`], no tiktoken model, so
/// usable in tests and on air-gapped machines). System messages are always preserved;
/// like the `ContextWindow` truncation contract, the newest message is always kept even
/// if a single message alone exceeds the budget (history is never silently emptied).
#[derive(Clone)]
pub struct TokenBudgetCompressStrategy {
    budget_tokens: usize,
    counter: Arc<dyn TokenCounter>,
}

impl TokenBudgetCompressStrategy {
    /// Create with an explicit token budget and counter.
    pub fn new(budget_tokens: usize, counter: Box<dyn TokenCounter>) -> Self {
        Self {
            budget_tokens,
            counter: counter.into(),
        }
    }

    /// Override the injected token counter (defaults to a char-ratio estimate).
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = counter;
        self
    }
}

impl Default for TokenBudgetCompressStrategy {
    fn default() -> Self {
        Self {
            budget_tokens: 1024,
            counter: Arc::new(CharRatioCounter::new(4)),
        }
    }
}

#[async_trait]
impl CompressStrategy for TokenBudgetCompressStrategy {
    async fn compress(&self, history: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        let mut system = Vec::new();
        let mut others = Vec::new();
        for msg in history {
            if matches!(msg.message_type, MessageType::System) {
                system.push(msg);
            } else {
                others.push(msg);
            }
        }

        // Drop the oldest non-system messages until the newest tail fits the budget.
        let mut candidate = others.clone();
        while !candidate.is_empty()
            && self.counter.count_messages(&candidate) as usize > self.budget_tokens
        {
            candidate.remove(0);
        }

        // H7-style guard: a single message that overflows the budget still survives.
        if candidate.is_empty() && !others.is_empty() {
            candidate.push(
                others
                    .into_iter()
                    .next_back()
                    .unwrap_or_else(|| Message::system(String::new())),
            );
        }

        let mut result = system;
        result.extend(candidate);
        Ok(result)
    }
}

/// [`CompressStrategy`] that delegates to a pre-built [`ContextWindow`], reusing its
/// truncate/summarize machinery (and its injected token counter). Constructing the
/// `ContextWindow` up front lets the caller supply a tiktoken-free counter or an LLM
/// summarization strategy.
pub struct ContextWindowCompressStrategy<M: BaseChatModel> {
    window: ContextWindow<M>,
}

impl<M: BaseChatModel> ContextWindowCompressStrategy<M> {
    /// Wrap an existing context window.
    pub fn new(window: ContextWindow<M>) -> Self {
        Self { window }
    }
}

#[async_trait]
impl<M: BaseChatModel> CompressStrategy for ContextWindowCompressStrategy<M> {
    async fn compress(&self, history: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        self.window.fit(history).await
    }
}

// ───────────────────────── Semantic-backed strategies ─────────────────────────

/// [`WriteStrategy`] that indexes a completed turn into a shared semantic [`MemoryStore`]
/// so a [`SemanticRetrieveStrategy`] over the same store can later recall it.
#[derive(Clone)]
pub struct SemanticWriteStrategy {
    store: Arc<dyn MemoryStore>,
    namespace: String,
    input_key: String,
    output_key: String,
}

impl SemanticWriteStrategy {
    /// Create against a shared semantic store and namespace.
    pub fn new(store: Arc<dyn MemoryStore>, namespace: impl Into<String>) -> Self {
        Self {
            store,
            namespace: namespace.into(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
        }
    }

    /// Set the input/output key names used to read a turn.
    pub fn with_keys(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.input_key = input.into();
        self.output_key = output.into();
        self
    }
}

#[async_trait]
impl WriteStrategy for SemanticWriteStrategy {
    async fn save(
        &self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        let input = inputs.get(&self.input_key).ok_or_else(|| {
            MemoryError::SaveError(format!("Missing input key '{}'", self.input_key))
        })?;
        let output = outputs.get(&self.output_key).ok_or_else(|| {
            MemoryError::SaveError(format!("Missing output key '{}'", self.output_key))
        })?;

        let content = format!("Human: {input}\n AI: {output}");
        // 0.26.0 §3.1: the previous key `turn-<input.len()>` made any two turns
        // whose inputs shared a byte length collide — the older turn was
        // silently dropped from the semantic store (semantic context went
        // missing). Key by a content hash of the whole turn instead: different
        // turns (differing input or output) map to distinct keys and are all
        // kept; only an identical repeated turn dedupes onto one entry. Retrieval
        // is embedding-based, so the key value itself is irrelevant to reads.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&content, &mut hasher);
        let key = format!("turn-{:016x}", hasher.finish());
        let item = MemoryItem::new(key, content);
        self.store.put(&self.namespace, item).await
    }
}

/// [`RetrieveStrategy`] that recalls meaning-based hits from a shared semantic
/// [`MemoryStore`], surfaced as `System` messages merged into the final context.
#[derive(Clone)]
pub struct SemanticRetrieveStrategy {
    store: Arc<dyn MemoryStore>,
    namespace: String,
    query_key: String,
    k: usize,
}

impl SemanticRetrieveStrategy {
    /// Create against a shared semantic store and namespace.
    pub fn new(store: Arc<dyn MemoryStore>, namespace: impl Into<String>) -> Self {
        Self {
            store,
            namespace: namespace.into(),
            query_key: "input".to_string(),
            k: 3,
        }
    }

    /// Set which `inputs` key supplies the query text.
    pub fn with_query_key(mut self, key: impl Into<String>) -> Self {
        self.query_key = key.into();
        self
    }

    /// Set the maximum number of hits to inject (default 3).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.max(1);
        self
    }
}

#[async_trait]
impl RetrieveStrategy for SemanticRetrieveStrategy {
    async fn retrieve(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<Vec<Message>, MemoryError> {
        let query = inputs.get(&self.query_key).cloned().unwrap_or_default();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let q = MemoryQuery::new(&self.namespace, &query).k(self.k);
        let hits = self.store.search(&q).await?;
        let messages = hits
            .into_iter()
            .map(|hit| Message::system(format!("[Memory] {}", hit.text)))
            .collect();
        Ok(messages)
    }
}

// ──────────────────────────────── manager ────────────────────────────────

/// A composable, strategy-driven context manager.
///
/// Holds an optional implementation of each concern. `save_context` routes through the
/// write strategy; `load_memory_variables` runs **read → compress → merge-retrieve** in
/// that fixed order. Implements [`BaseMemory`] so it can replace a single memory anywhere.
pub struct ContextManager {
    read: Box<dyn ReadStrategy>,
    write: Box<dyn WriteStrategy>,
    compress: Box<dyn CompressStrategy>,
    retrieve: Option<Box<dyn RetrieveStrategy>>,
    memory_key: String,
    return_messages: bool,
}

impl ContextManager {
    /// Start building a composable manager.
    pub fn builder() -> ContextManagerBuilder {
        ContextManagerBuilder::new()
    }

    /// Assemble the memory variable map from a processed message list.
    fn build_variables(
        &self,
        history: Vec<Message>,
    ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
        let mut result = HashMap::new();
        if self.return_messages {
            let messages: Vec<serde_json::Value> = history
                .iter()
                .map(serialize_message_to_prompt)
                .collect::<Result<Vec<_>, MemoryError>>()?;
            result.insert(self.memory_key.clone(), serde_json::Value::Array(messages));
        } else {
            result.insert(
                self.memory_key.clone(),
                serde_json::Value::String(format_history(history)),
            );
        }
        Ok(result)
    }
}

/// Format a message list as a readable transcript (mirrors `ChatMessageHistory` display).
fn format_history(messages: Vec<Message>) -> String {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.message_type {
                MessageType::Human => "Human",
                MessageType::AI => "AI",
                MessageType::System => "System",
                MessageType::Tool { .. } => "Tool",
            };
            format!("{}: {}", role, msg.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl BaseMemory for ContextManager {
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }

    async fn load_memory_variables(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
        // Fixed, deterministic pipeline: read → compress → merge-retrieve.
        let mut history = self.read.load(inputs).await?;
        history = self.compress.compress(history).await?;
        if let Some(retrieve) = &self.retrieve {
            let extra = retrieve.retrieve(inputs).await?;
            history.extend(extra);
        }
        self.build_variables(history)
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        self.write.save(inputs, outputs).await
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        // 0.28 (M-series audit): this was a silent no-op, violating `BaseMemory::clear`
        // — a caller that wiped its context got stale history back. The composable
        // strategies own their stores and expose `clear`; delegate to both read and
        // write so any loaded or saved history is actually emptied. Compress/retrieve
        // strategies are stateless/pure (no owned store), so they need no clearing.
        self.read.clear().await?;
        self.write.clear().await?;
        Ok(())
    }
}

// ──────────────────────────────── builder ────────────────────────────────

/// Builder for [`ContextManager`] that composes optional strategies.
///
/// Defaults: read + write share a fresh [`ConversationBufferMemory`]; compress is a
/// [`PassthroughCompressStrategy`]; retrieve is `None`.
#[derive(Default)]
pub struct ContextManagerBuilder {
    read: Option<Box<dyn ReadStrategy>>,
    write: Option<Box<dyn WriteStrategy>>,
    compress: Option<Box<dyn CompressStrategy>>,
    retrieve: Option<Box<dyn RetrieveStrategy>>,
    memory_key: String,
    return_messages: bool,
}

impl ContextManagerBuilder {
    /// Start with all defaults.
    pub fn new() -> Self {
        Self {
            read: None,
            write: None,
            compress: None,
            retrieve: None,
            memory_key: "history".to_string(),
            return_messages: true,
        }
    }

    /// Set the read strategy.
    pub fn read(mut self, read: impl ReadStrategy + 'static) -> Self {
        self.read = Some(Box::new(read));
        self
    }

    /// Set the write strategy.
    pub fn write(mut self, write: impl WriteStrategy + 'static) -> Self {
        self.write = Some(Box::new(write));
        self
    }

    /// Set the compress strategy.
    pub fn compress(mut self, compress: impl CompressStrategy + 'static) -> Self {
        self.compress = Some(Box::new(compress));
        self
    }

    /// Set the retrieve strategy.
    pub fn retrieve(mut self, retrieve: impl RetrieveStrategy + 'static) -> Self {
        self.retrieve = Some(Box::new(retrieve));
        self
    }

    /// Override the memory variable key (default `"history"`).
    pub fn memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Surface history as message objects (default `true`) or a transcript string.
    pub fn return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    /// Compose the manager, applying defaults for any omitted strategy.
    pub fn build(self) -> ContextManager {
        // When neither read nor write is supplied, they must share one buffer store so
        // save→load roundtrips. When only one side is customized, a fresh store suffices.
        let (read, write) = match (self.read, self.write) {
            (None, None) => {
                let store = Arc::new(Mutex::new(ConversationBufferMemory::new()));
                (
                    Box::new(BufferReadStrategy::new(store.clone())) as Box<dyn ReadStrategy>,
                    Box::new(BufferWriteStrategy::new(store)) as Box<dyn WriteStrategy>,
                )
            }
            (Some(read), None) => (
                read,
                Box::new(BufferWriteStrategy::new(Arc::new(Mutex::new(
                    ConversationBufferMemory::new(),
                )))) as Box<dyn WriteStrategy>,
            ),
            (None, Some(write)) => (
                Box::new(BufferReadStrategy::new(Arc::new(Mutex::new(
                    ConversationBufferMemory::new(),
                )))) as Box<dyn ReadStrategy>,
                write,
            ),
            (Some(read), Some(write)) => (read, write),
        };

        ContextManager {
            read,
            write,
            compress: self
                .compress
                .unwrap_or_else(|| Box::new(PassthroughCompressStrategy)),
            retrieve: self.retrieve,
            memory_key: self.memory_key,
            return_messages: self.return_messages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::memory_variables_to_messages;
    use crate::semantic::LongTermMemory;
    use lc_core::token_counter::{CharRatioCounter, TokenCounter};

    fn turn(input: &str, output: &str) -> (HashMap<String, String>, HashMap<String, String>) {
        (
            HashMap::from([("input".to_string(), input.to_string())]),
            HashMap::from([("output".to_string(), output.to_string())]),
        )
    }

    /// Test 1 — default (read-only) manager roundtrips save → load.
    #[tokio::test]
    async fn default_read_write_roundtrips() {
        let mut manager = ContextManagerBuilder::new().build();
        let (inputs, outputs) = turn("Hello", "Hi there!");
        manager.save_context(&inputs, &outputs).await.unwrap();

        let vars = manager
            .load_memory_variables(&HashMap::new())
            .await
            .unwrap();
        let messages = memory_variables_to_messages(&vars);
        assert_eq!(messages.len(), 2);
        assert!(messages.iter().any(|m| m.content.contains("Hello")));
        assert!(messages.iter().any(|m| m.content.contains("Hi there!")));
    }

    /// Test 2 — stacking a compress strategy pulls an over-budget history into budget.
    #[tokio::test]
    async fn compress_brings_history_within_budget() {
        let budget = 100usize;
        let mut manager = ContextManagerBuilder::new()
            .compress(TokenBudgetCompressStrategy::new(
                budget,
                Box::new(CharRatioCounter::new(4)),
            ))
            .build();

        for i in 0..10 {
            let (inputs, outputs) = turn(
                &format!("question number {i} here"),
                &format!("answer number {i} here that is quite long text"),
            );
            manager.save_context(&inputs, &outputs).await.unwrap();
        }

        let vars = manager
            .load_memory_variables(&HashMap::new())
            .await
            .unwrap();
        let messages = memory_variables_to_messages(&vars);
        let counter = CharRatioCounter::new(4);
        let used = counter.count_messages(&messages) as usize;
        assert!(
            used <= budget,
            "compressed history must fit budget, used {used} > {budget}"
        );
    }

    /// Test 3 — a retrieve strategy injects meaning-based context into the load.
    #[tokio::test]
    async fn retrieve_injects_memory_on_load() {
        let store: Arc<dyn MemoryStore> = Arc::new(LongTermMemory::new());
        let mut manager = ContextManagerBuilder::new()
            .write(SemanticWriteStrategy::new(store.clone(), "ns"))
            .compress(PassthroughCompressStrategy)
            .retrieve(SemanticRetrieveStrategy::new(store, "ns"))
            .build();

        let (inputs, outputs) = turn("I love Rust programming", "Great choice!");
        manager.save_context(&inputs, &outputs).await.unwrap();

        let vars = manager
            .load_memory_variables(&HashMap::from([(
                "input".to_string(),
                "tell me about Rust".to_string(),
            )]))
            .await
            .unwrap();
        let messages = memory_variables_to_messages(&vars);
        let joined: String = messages
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            joined.contains("Rust"),
            "retrieved memory must be injected, got: {joined}"
        );
    }

    /// Test 4 — read + compress ordering is deterministic across identical inputs.
    #[tokio::test]
    async fn read_compress_is_deterministic() {
        let mut manager = ContextManagerBuilder::new()
            .compress(TokenBudgetCompressStrategy::new(
                80,
                Box::new(CharRatioCounter::new(4)),
            ))
            .build();
        for i in 0..6 {
            let (inputs, outputs) = turn(&format!("turn number {i}"), &format!("reply number {i}"));
            manager.save_context(&inputs, &outputs).await.unwrap();
        }

        let first = manager
            .load_memory_variables(&HashMap::new())
            .await
            .unwrap();
        let second = manager
            .load_memory_variables(&HashMap::new())
            .await
            .unwrap();
        assert_eq!(
            first, second,
            "identical inputs must yield identical output"
        );
    }

    /// Test 5 — what the write strategy saved is visible to the retrieve strategy.
    #[tokio::test]
    async fn write_is_visible_to_retrieve() {
        let store: Arc<dyn MemoryStore> = Arc::new(LongTermMemory::new());
        let mut manager = ContextManagerBuilder::new()
            .write(SemanticWriteStrategy::new(store.clone(), "facts"))
            .compress(PassthroughCompressStrategy)
            .retrieve(SemanticRetrieveStrategy::new(store, "facts"))
            .build();

        let (inputs, outputs) = turn("The user likes swimming", "Noted.");
        manager.save_context(&inputs, &outputs).await.unwrap();

        let vars = manager
            .load_memory_variables(&HashMap::from([("input".to_string(), "sport".to_string())]))
            .await
            .unwrap();
        let messages = memory_variables_to_messages(&vars);
        let joined: String = messages
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            joined.contains("swimming"),
            "saved knowledge must be retrievable, got: {joined}"
        );
    }

    /// 0.26.0 §3.1 — two turns whose inputs share a byte length must both
    /// persist. The old `turn-<len>` key collided and the earlier turn was
    /// silently dropped (semantic context went missing); a content-hash key
    /// gives each distinct turn its own entry.
    #[tokio::test]
    async fn semantic_write_same_length_different_content_keeps_both() {
        let store: Arc<dyn MemoryStore> = Arc::new(LongTermMemory::new());
        let strategy = SemanticWriteStrategy::new(store.clone(), "ns");

        // Two equal-byte-length but different inputs: "abcd" and "wxyp" are
        // both 4 bytes. Under the old key both collided on `turn-4`.
        let (i1, o1) = turn("abcd", "first answer");
        let (i2, o2) = turn("wxyp", "second answer");
        strategy.save(&i1, &o1).await.unwrap();
        strategy.save(&i2, &o2).await.unwrap();

        assert_eq!(
            store.len_namespace("ns").await.unwrap(),
            2,
            "same-length different-content turns must both be stored, not overwritten"
        );

        // Both turns are retrievable — neither was dropped.
        let hits = store.search(&MemoryQuery::new("ns", "abcd")).await.unwrap();
        let joined: String = hits
            .iter()
            .map(|h| h.text.clone())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            joined.contains("abcd") && joined.contains("wxyp"),
            "both turns must be retrievable, got: {joined}"
        );
    }

    /// Extra — all four concerns stack on one manager with a deterministic order.
    #[tokio::test]
    async fn all_strategies_stack_on_one_manager() {
        let store: Arc<dyn MemoryStore> = Arc::new(LongTermMemory::new());
        let mut manager = ContextManagerBuilder::new()
            .write(SemanticWriteStrategy::new(store.clone(), "stack"))
            .compress(TokenBudgetCompressStrategy::new(
                80,
                Box::new(CharRatioCounter::new(4)),
            ))
            .retrieve(SemanticRetrieveStrategy::new(store, "stack"))
            .build();

        // Buffer read stays empty (default), semantic write indexes a fact, retrieve injects.
        let (inputs, outputs) = turn("My favourite fruit is mango", "Tasty.");
        manager.save_context(&inputs, &outputs).await.unwrap();

        let vars = manager
            .load_memory_variables(&HashMap::from([("input".to_string(), "mango".to_string())]))
            .await
            .unwrap();
        let messages = memory_variables_to_messages(&vars);
        let joined: String = messages
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(joined.contains("mango"));
    }
}
