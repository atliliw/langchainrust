// src/core/router_llm/mod.rs
//! Model routing, fallback and load balancing.
//!
//! `RouterLLM` implements `BaseChatModel` over a pool of heterogeneous chat
//! models (OpenAI / Anthropic / Gemini / Ollama / ...), picking one per call
//! according to a [`RoutingStrategy`] and falling back to the next model when
//! the chosen one fails.
//!
//! Besides plain error fallback, the router supports three operational
//! controls (B13, 0.22.4):
//!
//! - **Latency-driven routing**: [`RoutingStrategy::LeastLatency`] sorts by
//!   the per-model EMA latency, and [`RoutingStrategy::LatencyWeighted`]
//!   draws the primary model with weights derived from the same observed
//!   latencies. Both warm themselves on real call timings.
//! - **Per-model rate limiting**: attach a [`ModelRateLimit`] to a slot and
//!   saturated models queue callers FIFO (bounded, with a wait timeout); a
//!   slot that cannot be admitted is skipped and the next candidate tried.
//! - **Budget circuit breaker**: attach a shared [`RouterBudget`]; calls
//!   whose projected spend would cross the cap skip the paid slot and roll
//!   over to free fallbacks, and measured usage latches the breaker.
//!
//! # Example
//! ```
//! use lc_core::router_llm::{RouterLLM, RoutingStrategy};
//!
//! // Empty primary-first fallback router. Register real models (OpenAIChat,
//! // AnthropicChat, ...) via the `with_model` / `with_fallbacks` builders;
//! // the router then tries them in order, falling back on error.
//! let _router = RouterLLM::new(RoutingStrategy::Fallback);
//! ```
//!
//! Operational controls on a populated router:
//!
//! ```ignore
//! // paid primary limited to 30 starts/min and 4 concurrent, free local
//! // fallback, whole router trips at a 5 USD cumulative spend:
//! let router = RouterLLM::new(RoutingStrategy::Fallback)
//!     .with_priced_model(paid_model, ModelPrice::new(2.5, 10.0))
//!     .with_last_rate_limit(ModelRateLimit::per_minute(30).with_max_concurrent(4))
//!     .with_model(local_model)
//!     .with_budget(RouterBudget::with_cost_and_token_limits(5.0, 1_000_000));
//! ```
//!
//! Each provider declares its own error type (`OpenAIError`, `AnthropicError`,
//! ...), so `RouterLLM` cannot hold them behind a single `dyn BaseChatModel`.
//! Instead it wraps every model in a `ModelAdapter` that converts the
//! model's native error into the unified [`RouterError`].

mod budget;
mod rate;

pub use budget::{BudgetExceeded, BudgetKind, RouterBudget};
pub use rate::{ModelRateLimit, RateLimitReason};

use crate::cost::ModelPrice;
use crate::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use crate::model_registry::ModelRegistry;
use crate::runnables::Runnable;
use crate::RunnableConfig;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_schema::Message;
use std::fmt::{self, Display, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(test)]
mod tests;

/// Unified error for [`RouterLLM`].
///
/// Aggregates heterogeneous provider errors behind `Box<dyn Error>` so a
/// single router can mix providers whose native error types differ.
#[derive(Debug)]
#[non_exhaustive]
pub enum RouterError {
    /// No models were configured on the router.
    Empty,
    /// Every candidate model was tried and all failed.
    /// `tried` is the number of models attempted; `last` is the final error.
    AllFailed {
        /// The number of models attempted.
        tried: usize,
        /// The final error from the last attempted model.
        last: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A single model failed (wrapped when propagating from an adapter).
    Model {
        /// The name of the model that failed.
        model: String,
        /// The underlying provider error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The slot's per-model rate limiter could not admit the call in time.
    /// The router treats this like a model failure and tries the next slot.
    RateLimited {
        /// The name of the rate-limited model.
        model: String,
        /// Whether the queue was full or the wait timed out.
        reason: RateLimitReason,
    },
    /// The router budget would have been exceeded by the projected call.
    /// The router skips that slot (free slots remain reachable) and tries
    /// the next candidate; this surfaces as the final error only when every
    /// candidate was skipped.
    BudgetExceeded(
        /// Snapshot of the exceeded dimension, used amount and limit.
        BudgetExceeded,
    ),
}

impl Display for RouterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RouterError::Empty => write!(f, "no models configured in router"),
            RouterError::AllFailed { tried, last } => {
                write!(f, "all {} models failed; last error: {}", tried, last)
            }
            RouterError::Model { model, source } => {
                write!(f, "model '{}' error: {}", model, source)
            }
            RouterError::RateLimited { model, reason } => {
                write!(f, "model '{}' rate limited: {}", model, reason)
            }
            RouterError::BudgetExceeded(exceeded) => {
                write!(f, "router budget skip — {exceeded}")
            }
        }
    }
}

impl std::error::Error for RouterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RouterError::AllFailed { last, .. } => Some(last.as_ref()),
            RouterError::Model { source, .. } => Some(source.as_ref()),
            RouterError::BudgetExceeded(exceeded) => Some(exceeded),
            RouterError::Empty | RouterError::RateLimited { .. } => None,
        }
    }
}

/// Strategy for selecting which model a [`RouterLLM`] tries first.
///
/// Regardless of strategy, a failed model triggers fallback to the next
/// candidate in the derived order until one succeeds or all fail.
pub enum RoutingStrategy {
    /// Always try models in registration order (primary-first). Use with
    /// [`RouterLLM::with_fallbacks`] for classic primary + backups semantics.
    Fallback,
    /// Rotate the starting index across calls so traffic spreads evenly.
    RoundRobin,
    /// Prefer the model with the lowest recent latency (exponential moving
    /// average updated after each call).
    LeastLatency,
    /// Pick the primary model by a weighted draw over observed latency:
    /// slot weight is `(1 / latency_ms).powf(beta)`, so `beta = 1.0` makes a
    /// 100 ms model ten times as likely to lead as a 1000 ms one. Models
    /// never tried are optimistically given the best observed latency so
    /// they still receive exploration traffic; when nothing has been tried
    /// all weights are equal. Remaining slots follow the draw as a fallback
    /// chain sorted by weight. The draw is a deterministic SplitMix64
    /// sequence seeded per router (no `rand` dependency).
    LatencyWeighted(f64),
    /// Prefer the model with the lowest configured cost.
    LowestCost,
    /// Pick the primary index from a user-supplied closure over the input
    /// text; remaining models are tried in registration order as fallback.
    InputDirected(Arc<dyn Fn(&str) -> usize + Send + Sync>),
}

/// A chat model pool with routing and fallback.
///
/// See the [module docs](self) for design rationale.
pub struct RouterLLM {
    name: String,
    slots: Vec<ModelSlot>,
    strategy: RoutingStrategy,
    /// Round-robin cursor.
    counter: AtomicUsize,
    /// Deterministic PRNG state driving [`RoutingStrategy::LatencyWeighted`].
    rng: AtomicU64,
    /// Optional model catalog (B3): prices `with_model_as`-keyed slots for
    /// [`RoutingStrategy::LowestCost`] without repeating numbers at the call site.
    registry: Option<Arc<ModelRegistry>>,
    /// Optional shared spend/token circuit breaker (B13).
    budget: Option<Arc<RouterBudget>>,
}

impl RouterLLM {
    /// Create an empty router with the given strategy. Add models via the
    /// `with_*` builder methods.
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            name: "router".to_string(),
            slots: Vec::new(),
            strategy,
            counter: AtomicUsize::new(0),
            rng: AtomicU64::new(0x9E37_79B9_7F4A_7C15),
            registry: None,
            budget: None,
        }
    }

    /// Attaches a shared [`RouterBudget`]. Every subsequent call projects its
    /// prompt spend against the remaining budget, skips paid slots once the
    /// breaker is tripped (free slots stay reachable as fallback), and
    /// records the model-reported token usage after a successful call.
    pub fn with_budget(mut self, budget: Arc<RouterBudget>) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Attaches a model registry. Keyed slots ([`RouterLLM::with_model_as`])
    /// then take their [`RoutingStrategy::LowestCost`] weight from the
    /// registry's blended per-1K price. An explicit cost from
    /// [`RouterLLM::with_cost`] always wins; a key the registry cannot resolve
    /// sorts last (treated as infinitely expensive).
    pub fn with_registry(mut self, registry: Arc<ModelRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set a human-readable name returned by `model_name`.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Register a model.
    pub fn with_model<M>(mut self, model: M) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        self.slots
            .push(ModelSlot::new(Box::new(ModelAdapter(model)), None));
        self
    }

    /// Register a model with a relative cost used by [`RoutingStrategy::LowestCost`].
    pub fn with_cost<M>(mut self, model: M, cost: f64) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        self.slots
            .push(ModelSlot::new(Box::new(ModelAdapter(model)), Some(cost)));
        self
    }

    /// Register a model keyed as `"<provider>/<model-id>"` in the attached
    /// [`ModelRegistry`] (see [`RouterLLM::with_registry`]). Under
    /// [`RoutingStrategy::LowestCost`] the slot is weighted by the registry's
    /// blended per-1K price; other strategies ignore the key.
    pub fn with_model_as<M, K>(mut self, model: M, registry_key: K) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
        K: Into<String>,
    {
        self.slots
            .push(ModelSlot::new(Box::new(ModelAdapter(model)), None).with_key(registry_key));
        self
    }

    /// Register a model with an explicit per-token [`ModelPrice`]. The price
    /// both participates in [`RoutingStrategy::LowestCost`] (blended) and
    /// prices projected/actual spend for an attached [`RouterBudget`].
    pub fn with_priced_model<M>(mut self, model: M, price: ModelPrice) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        self.slots
            .push(ModelSlot::new(Box::new(ModelAdapter(model)), None).with_price(Some(price)));
        self
    }

    /// Register a model behind a per-model [`ModelRateLimit`] (B13). Saturated
    /// slots queue callers FIFO; when the queue is full or the configured
    /// wait timeout elapses, the call skips to the next candidate model.
    pub fn with_model_rate_limited<M>(mut self, model: M, limit: ModelRateLimit) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        self.slots.push(
            ModelSlot::new(Box::new(ModelAdapter(model)), None)
                .with_gate(rate::ModelGate::new(&limit)),
        );
        self
    }

    /// Attaches a [`ModelPrice`] to the most recently registered slot.
    ///
    /// Use to combine registration styles, e.g. a keyed slot whose registry
    /// entry lacks pricing. A no-op with a warning when no slot exists.
    pub fn with_last_price(mut self, price: ModelPrice) -> Self {
        match self.slots.last_mut() {
            Some(slot) => slot.price = Some(price),
            None => log::warn!("with_last_price called before any model was registered"),
        }
        self
    }

    /// Attaches a [`ModelRateLimit`] to the most recently registered slot.
    ///
    /// A no-op with a warning when no slot exists.
    pub fn with_last_rate_limit(mut self, limit: ModelRateLimit) -> Self {
        match self.slots.last_mut() {
            Some(slot) => slot.gate = Some(rate::ModelGate::new(&limit)),
            None => log::warn!("with_last_rate_limit called before any model was registered"),
        }
        self
    }

    /// Convenience constructor for primary-first fallback over models of the
    /// same type. The primary is tried first; each fallback is tried in order
    /// until one succeeds.
    pub fn with_fallbacks<M>(primary: M, fallbacks: Vec<M>) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        let mut router = RouterLLM::new(RoutingStrategy::Fallback);
        router = router.with_model(primary);
        for fb in fallbacks {
            router = router.with_model(fb);
        }
        router
    }

    /// Number of registered models.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether no models are registered.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Derive the candidate ordering for one call.
    fn candidate_order(&self, input: &str) -> Vec<usize> {
        let n = self.slots.len();
        match &self.strategy {
            RoutingStrategy::Fallback => (0..n).collect(),
            RoutingStrategy::RoundRobin => {
                if n == 0 {
                    (0..n).collect()
                } else {
                    let start = self.counter.fetch_add(1, Ordering::SeqCst) % n;
                    (0..n).map(|i| (start + i) % n).collect()
                }
            }
            RoutingStrategy::LeastLatency => {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|a, b| {
                    // H4 fix: untried models (latency 0.0) sort last by using f64::MAX
                    let la = {
                        let v = self.slots[*a].latency();
                        if v == 0.0 {
                            f64::MAX
                        } else {
                            v
                        }
                    };
                    let lb = {
                        let v = self.slots[*b].latency();
                        if v == 0.0 {
                            f64::MAX
                        } else {
                            v
                        }
                    };
                    la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                });
                idx
            }
            RoutingStrategy::LatencyWeighted(beta) => {
                // Non-finite / negative beta is a configuration mistake; use a
                // linear exponent rather than poisoning ordering with NaNs.
                let beta = if beta.is_finite() && *beta >= 0.0 {
                    *beta
                } else {
                    1.0
                };
                let observed: Vec<f64> = self.slots.iter().map(|s| s.latency()).collect();
                // Untried slots optimistically assume the best seen latency
                // (equal weights when the whole pool is cold).
                let best = observed
                    .iter()
                    .copied()
                    .filter(|l| *l > 0.0)
                    .fold(f64::MAX, f64::min);
                let assumed = if best == f64::MAX { 1.0 } else { best };
                let weights: Vec<f64> = observed
                    .iter()
                    .map(|l| {
                        let latency = if *l == 0.0 { assumed } else { *l };
                        (1.0 / latency).powf(beta)
                    })
                    .collect();
                let total: f64 = weights.iter().sum();
                let target = if total > 0.0 {
                    self.next_pseudo() * total
                } else {
                    0.0
                };
                let mut primary = n.saturating_sub(1);
                let mut cumulative = 0.0;
                for (i, weight) in weights.iter().enumerate() {
                    cumulative += *weight;
                    if target < cumulative {
                        primary = i;
                        break;
                    }
                }
                // Fallback chain: remaining slots by descending weight,
                // registration order breaks ties deterministically.
                let mut rest: Vec<usize> = (0..n).filter(|&i| i != primary).collect();
                rest.sort_by(|&a, &b| {
                    weights[b]
                        .partial_cmp(&weights[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                let mut order = Vec::with_capacity(n);
                order.push(primary);
                order.extend(rest);
                order
            }
            RoutingStrategy::LowestCost => {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|a, b| {
                    let ca = self.effective_cost(&self.slots[*a]);
                    let cb = self.effective_cost(&self.slots[*b]);
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                });
                idx
            }
            RoutingStrategy::InputDirected(f) => {
                let primary = f(input);
                if primary < n {
                    let mut order = vec![primary];
                    order.extend((0..n).filter(|&i| i != primary));
                    order
                } else {
                    // Out-of-range index: fall back to registration order.
                    (0..n).collect()
                }
            }
        }
    }

    fn first_text(messages: &[Message]) -> &str {
        messages.first().map(|m| m.content.as_str()).unwrap_or("")
    }

    /// Effective LowestCost weight: explicit `with_cost` wins, then an
    /// explicit slot [`ModelPrice`], then the registry price of a keyed slot,
    /// then +∞ (unweighted slots sort last).
    fn effective_cost(&self, slot: &ModelSlot) -> f64 {
        if let Some(cost) = slot.cost {
            return cost;
        }
        if let Some(price) = self.slot_price(slot) {
            return price.blended_per_1k();
        }
        f64::MAX
    }

    /// Pricing used to measure budget spend: explicit slot price first, then
    /// the registry entry of a keyed slot, then `None` (a free / unpriced
    /// slot whose calls project zero cost).
    fn slot_price(&self, slot: &ModelSlot) -> Option<ModelPrice> {
        if slot.price.is_some() {
            return slot.price;
        }
        if let (Some(registry), Some(key)) = (&self.registry, &slot.registry_key) {
            if let Some(info) = registry.get_by_key(key) {
                return Some(info.price);
            }
        }
        None
    }

    /// tiktoken (or byte-length fallback) estimate of the pending prompt size.
    fn estimate_prompt_tokens(&self, messages: &[Message]) -> usize {
        let joined = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.get_num_tokens(&joined)
    }

    /// Pre-call budget gate for one slot. When the projected call would
    /// cross a cap, returns the exceeded dimension so the caller can skip
    /// this slot and try the next candidate.
    fn precheck_budget(
        &self,
        slot: &ModelSlot,
        estimated_tokens: usize,
    ) -> Result<(), BudgetExceeded> {
        let Some(budget) = &self.budget else {
            return Ok(());
        };
        let projected = self
            .slot_price(slot)
            .map(|price| price.cost_of(estimated_tokens, 0))
            .unwrap_or(0.0);
        budget.precheck(projected, estimated_tokens as u64)
    }

    /// Post-success accounting: real model-reported usage priced through the
    /// slot's price feeds the (potentially latching) breaker.
    fn record_usage(&self, slot: &ModelSlot, usage: &TokenUsage) {
        if let Some(budget) = &self.budget {
            let cost = self
                .slot_price(slot)
                .map(|price| price.cost_of(usage.prompt_tokens, usage.completion_tokens))
                .unwrap_or(0.0);
            budget.record(cost, usage.total_tokens as u64);
        }
    }

    /// Deterministic uniform draw in `[0, 1)` (SplitMix64), so
    /// latency-weighted routing needs neither an `rand` dependency nor
    /// process-global randomness.
    fn next_pseudo(&self) -> f64 {
        let mut z = self
            .rng
            .fetch_add(1, Ordering::SeqCst)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 53 bits → mantissa of a f64 in [0, 1).
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    async fn chat_routed(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, RouterError> {
        if self.slots.is_empty() {
            return Err(RouterError::Empty);
        }
        let order = self.candidate_order(Self::first_text(&messages));
        // H3 fix: remove ineffective Arc — just clone messages directly for each attempt
        let estimated_tokens = self.estimate_prompt_tokens(&messages);
        let mut last_err: Option<Box<dyn std::error::Error + Send + Sync>> = None;
        for &idx in &order {
            let slot = &self.slots[idx];

            // Budget circuit breaker: a projected overrun skips the slot
            // instead of spending on it; free slots project zero and stay.
            if let Err(exceeded) = self.precheck_budget(slot, estimated_tokens) {
                last_err = Some(Box::new(RouterError::BudgetExceeded(exceeded)));
                continue;
            }

            // Rate gate: queue FIFO; a saturated slot is skipped like a
            // failed model so the fallback chain keeps moving.
            let permit = match &slot.gate {
                Some(gate) => match gate.acquire(slot.model.name()).await {
                    Ok(permit) => Some(permit),
                    Err(e) => {
                        last_err = Some(Box::new(e));
                        continue;
                    }
                },
                None => None,
            };

            // Timing starts after admission: queued wait is contention, not
            // model latency, and must not distort latency-driven routing.
            let start = Instant::now();
            let res = slot.model.chat(messages.clone(), config.clone()).await;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            slot.update_latency(elapsed_ms);
            match res {
                Ok(result) => {
                    if let Some(usage) = result.token_usage.as_ref() {
                        self.record_usage(slot, usage);
                    }
                    drop(permit);
                    return Ok(result);
                }
                Err(e) => last_err = Some(Box::new(e)),
            }
        }
        Err(RouterError::AllFailed {
            tried: order.len(),
            last: last_err.unwrap_or_else(|| {
                Box::new(std::io::Error::other(
                    "candidate order produced no attempts",
                ))
            }),
        })
    }

    async fn stream_chat_routed(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, RouterError>> + Send>>, RouterError>
    {
        if self.slots.is_empty() {
            return Err(RouterError::Empty);
        }
        let order = self.candidate_order(Self::first_text(&messages));
        // H3 fix: remove ineffective Arc — just clone messages directly for each attempt
        let estimated_tokens = self.estimate_prompt_tokens(&messages);
        let mut last_err: Option<Box<dyn std::error::Error + Send + Sync>> = None;
        for &idx in &order {
            let slot = &self.slots[idx];

            if let Err(exceeded) = self.precheck_budget(slot, estimated_tokens) {
                last_err = Some(Box::new(RouterError::BudgetExceeded(exceeded)));
                continue;
            }

            // The permit must stay alive until the *body* finishes streaming,
            // not just until headers arrive; it is moved into the wrapped
            // stream below and released when the stream is dropped/done.
            let permit = match &slot.gate {
                Some(gate) => match gate.acquire(slot.model.name()).await {
                    Ok(permit) => Some(permit),
                    Err(e) => {
                        last_err = Some(Box::new(e));
                        continue;
                    }
                },
                None => None,
            };

            let start = Instant::now();
            match slot
                .model
                .stream_chat(messages.clone(), config.clone())
                .await
            {
                Ok(stream) => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    slot.update_latency(elapsed_ms);
                    let budget = self.budget.clone();
                    let price = self.slot_price(slot);
                    let guarded = async_stream::stream! {
                        let _permit = permit;
                        let mut recorded = false;
                        let mut inner = stream;
                        while let Some(item) = inner.next().await {
                            if let Ok(chunk) = &item {
                                if !recorded {
                                    if let Some(usage) = chunk.token_usage.as_ref() {
                                        if let Some(budget) = &budget {
                                            let cost = price
                                                .map(|p| {
                                                    p.cost_of(
                                                        usage.prompt_tokens,
                                                        usage.completion_tokens,
                                                    )
                                                })
                                                .unwrap_or(0.0);
                                            budget.record(cost, usage.total_tokens as u64);
                                        }
                                        recorded = true;
                                    }
                                }
                            }
                            yield item;
                        }
                    };
                    return Ok(Box::pin(guarded));
                }
                Err(e) => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    slot.update_latency(elapsed_ms);
                    last_err = Some(Box::new(e));
                }
            }
        }
        Err(RouterError::AllFailed {
            tried: order.len(),
            last: last_err.unwrap_or_else(|| {
                Box::new(std::io::Error::other(
                    "candidate order produced no attempts",
                ))
            }),
        })
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for RouterLLM {
    type Error = RouterError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat_routed(input, config).await
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for RouterLLM {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        crate::token_counter::count_tokens(text).unwrap_or_else(|e| {
            // 编码器加载失败时按字节数高估(宁可略高,不静默按 0 算导致路由/截断误判)
            log::warn!("token counting failed, falling back to byte-length estimate: {e}");
            text.len()
        })
    }

    fn with_temperature(self, _temp: f32) -> Self {
        // The router deliberately does not override temperature: each slot
        // owns its model's sampling parameters, and they cannot be mutated
        // behind a `Box<dyn RoutedModel>` trait object. Configure temperature
        // on the individual models before registering them (Q5: no silent
        // change — this is an explicit no-op, not an attempt to apply it).
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self {
        // Same rationale as `with_temperature`: per-slot max_tokens is owned
        // by each registered model and cannot be changed after boxing.
        self
    }
}

#[async_trait]
impl BaseChatModel for RouterLLM {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat_routed(messages, config).await
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        self.stream_chat_routed(messages, config).await
    }
}

// ---------------------------------------------------------------------------
// Internal: heterogeneous model adapter
// ---------------------------------------------------------------------------

/// Internal trait unifying the error type across providers.
#[async_trait]
trait RoutedModel: Send + Sync {
    /// The wrapped model's reported name (used in rate-limit errors).
    fn name(&self) -> &str;

    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, RouterError>;
    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, RouterError>> + Send>>, RouterError>;
}

/// Wraps any `BaseChatModel` whose error is `std::error::Error + Send + Sync`,
/// converting its native error into [`RouterError`].
struct ModelAdapter<M: BaseChatModel>(M);

#[async_trait]
impl<M: BaseChatModel> RoutedModel for ModelAdapter<M>
where
    M::Error: std::error::Error + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.0.model_name()
    }

    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, RouterError> {
        let name = self.0.model_name().to_string();
        self.0
            .chat(messages, config)
            .await
            .map_err(|e| RouterError::Model {
                model: name,
                source: Box::new(e),
            })
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, RouterError>> + Send>>, RouterError>
    {
        let name = self.0.model_name().to_string();
        let inner = self
            .0
            .stream_chat(messages, config)
            .await
            .map_err(|e| RouterError::Model {
                model: name.clone(),
                source: Box::new(e),
            })?;
        let mapped = inner.map(move |item| {
            item.map_err(|e| RouterError::Model {
                model: name.clone(),
                source: Box::new(e),
            })
        });
        Ok(Box::pin(mapped))
    }
}

/// One registered model plus routing metadata.
struct ModelSlot {
    model: Box<dyn RoutedModel>,
    cost: Option<f64>,
    /// `"<provider>/<id>"` key resolved against the router's [`ModelRegistry`].
    registry_key: Option<String>,
    /// Explicit per-token price used when no `with_cost` weight or registry
    /// entry applies; also prices budget spend for this slot.
    price: Option<ModelPrice>,
    /// Optional per-model rate / concurrency admission gate.
    gate: Option<Arc<rate::ModelGate>>,
    /// Exponential moving average latency in milliseconds.
    latency_ms: Mutex<f64>,
}

impl ModelSlot {
    fn new(model: Box<dyn RoutedModel>, cost: Option<f64>) -> Self {
        Self {
            model,
            cost,
            registry_key: None,
            price: None,
            gate: None,
            latency_ms: Mutex::new(0.0),
        }
    }

    fn with_key<K: Into<String>>(mut self, key: K) -> Self {
        self.registry_key = Some(key.into());
        self
    }

    fn with_price(mut self, price: Option<ModelPrice>) -> Self {
        self.price = price;
        self
    }

    fn with_gate(mut self, gate: Arc<rate::ModelGate>) -> Self {
        self.gate = Some(gate);
        self
    }

    fn latency(&self) -> f64 {
        *self.latency_ms.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn update_latency(&self, ms: f64) {
        let mut cur = self.latency_ms.lock().unwrap_or_else(|e| e.into_inner());
        if *cur == 0.0 {
            *cur = ms;
        } else {
            // EMA: weight history 0.7, new sample 0.3.
            *cur = *cur * 0.7 + ms * 0.3;
        }
    }
}
