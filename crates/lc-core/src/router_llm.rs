// src/core/router_llm.rs
//! Model routing, fallback and load balancing.
//!
//! `RouterLLM` implements `BaseChatModel` over a pool of heterogeneous chat
//! models (OpenAI / Anthropic / Gemini / Ollama / ...), picking one per call
//! according to a [`RoutingStrategy`] and falling back to the next model when
//! the chosen one fails.
//!
//! # Example
//! ```
//! use langchainrust::{RouterLLM, RoutingStrategy};
//!
//! // Empty primary-first fallback router. Register real models (OpenAIChat,
//! // AnthropicChat, ...) via the `with_model` / `with_fallbacks` builders;
//! // the router then tries them in order, falling back on error.
//! let _router = RouterLLM::new(RoutingStrategy::Fallback);
//! ```
//!
//! Each provider declares its own error type (`OpenAIError`, `AnthropicError`,
//! ...), so `RouterLLM` cannot hold them behind a single `dyn BaseChatModel`.
//! Instead it wraps every model in a `ModelAdapter` that converts the
//! model's native error into the unified [`RouterError`].

use crate::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use crate::runnables::Runnable;
use crate::RunnableConfig;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_schema::Message;
use std::fmt::{self, Display, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Unified error for [`RouterLLM`].
///
/// Aggregates heterogeneous provider errors behind `Box<dyn Error>` so a
/// single router can mix providers whose native error types differ.
#[derive(Debug)]
pub enum RouterError {
    /// No models were configured on the router.
    Empty,
    /// Every candidate model was tried and all failed.
    /// `tried` is the number of models attempted; `last` is the final error.
    AllFailed {
        tried: usize,
        last: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A single model failed (wrapped when propagating from an adapter).
    Model {
        model: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
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
        }
    }
}

impl std::error::Error for RouterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RouterError::AllFailed { last, .. } => Some(last.as_ref()),
            RouterError::Model { source, .. } => Some(source.as_ref()),
            RouterError::Empty => None,
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
        }
    }

    /// Set a human-readable name returned by `model_name`.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Register a model. Its `model_name()` is used as the slot label.
    pub fn with_model<M>(mut self, model: M) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        let label = model.model_name().to_string();
        self.slots
            .push(ModelSlot::new(label, Box::new(ModelAdapter(model)), None));
        self
    }

    /// Register a model with an explicit slot label (useful when several
    /// slots share the same underlying model name).
    pub fn with_named_model<M>(mut self, name: impl Into<String>, model: M) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        self.slots.push(ModelSlot::new(
            name.into(),
            Box::new(ModelAdapter(model)),
            None,
        ));
        self
    }

    /// Register a model with a relative cost used by [`RoutingStrategy::LowestCost`].
    pub fn with_cost<M>(mut self, model: M, cost: f64) -> Self
    where
        M: BaseChatModel + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        let label = model.model_name().to_string();
        self.slots.push(ModelSlot::new(
            label,
            Box::new(ModelAdapter(model)),
            Some(cost),
        ));
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
            RoutingStrategy::LowestCost => {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|a, b| {
                    let ca = self.slots[*a].cost.unwrap_or(f64::MAX);
                    let cb = self.slots[*b].cost.unwrap_or(f64::MAX);
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
        let mut last_err: Option<Box<dyn std::error::Error + Send + Sync>> = None;
        for &idx in &order {
            let slot = &self.slots[idx];
            let start = Instant::now();
            let res = slot.model.chat(messages.clone(), config.clone()).await;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            slot.update_latency(elapsed_ms);
            match res {
                Ok(result) => return Ok(result),
                Err(e) => last_err = Some(Box::new(e)),
            }
        }
        Err(RouterError::AllFailed {
            tried: order.len(),
            last: last_err.expect("non-empty order guarantees at least one error"),
        })
    }

    async fn stream_chat_routed(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, RouterError>> + Send>>, RouterError> {
        if self.slots.is_empty() {
            return Err(RouterError::Empty);
        }
        let order = self.candidate_order(Self::first_text(&messages));
        // H3 fix: remove ineffective Arc — just clone messages directly for each attempt
        let mut last_err: Option<Box<dyn std::error::Error + Send + Sync>> = None;
        for &idx in &order {
            let slot = &self.slots[idx];
            let start = Instant::now();
            match slot
                .model
                .stream_chat(messages.clone(), config.clone())
                .await
            {
                Ok(s) => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    slot.update_latency(elapsed_ms);
                    return Ok(s);
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
            last: last_err.expect("non-empty order guarantees at least one error"),
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
        crate::token_counter::count_tokens(text)
    }

    fn with_temperature(self, _temp: f32) -> Self {
        // Per-model temperature is owned by each slot; the router itself does
        // not override it. No-op to satisfy the trait contract.
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self {
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        self.stream_chat_routed(messages, config).await
    }
}

// ---------------------------------------------------------------------------
// Internal: heterogeneous model adapter
// ---------------------------------------------------------------------------

/// Internal trait unifying the error type across providers.
#[async_trait]
trait RoutedModel: Send + Sync {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, RouterError>;
    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, RouterError>> + Send>>, RouterError>;
}

/// Wraps any `BaseChatModel` whose error is `std::error::Error + Send + Sync`,
/// converting its native error into [`RouterError`].
struct ModelAdapter<M: BaseChatModel>(M);

#[async_trait]
impl<M: BaseChatModel> RoutedModel for ModelAdapter<M>
where
    M::Error: std::error::Error + Send + Sync + 'static,
{
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, RouterError>> + Send>>, RouterError> {
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
    #[allow(dead_code)]
    name: String,
    model: Box<dyn RoutedModel>,
    cost: Option<f64>,
    /// Exponential moving average latency in milliseconds.
    latency_ms: Mutex<f64>,
}

impl ModelSlot {
    fn new(name: String, model: Box<dyn RoutedModel>, cost: Option<f64>) -> Self {
        Self {
            name,
            model,
            cost,
            latency_ms: Mutex::new(0.0),
        }
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

#[cfg(test)]
mod tests {
    //! Internal unit tests for routing order derivation.
    //! Heterogeneous-model behavior is covered in `tests/unit/router_llm.rs`.

    use super::*;

    fn slot_at(name: &str, cost: Option<f64>, latency: f64) -> ModelSlot {
        // A minimal stand-in model is not needed to test order derivation;
        // we only exercise `candidate_order` / `latency` logic. Build slots
        // with a no-op model via a tiny helper trait impl below.
        struct Noop;
        #[async_trait]
        impl RoutedModel for Noop {
            async fn chat(
                &self,
                _m: Vec<Message>,
                _c: Option<RunnableConfig>,
            ) -> Result<LLMResult, RouterError> {
                Err(RouterError::Empty)
            }
            async fn stream_chat(
                &self,
                _m: Vec<Message>,
                _c: Option<RunnableConfig>,
            ) -> Result<Pin<Box<dyn Stream<Item = Result<String, RouterError>> + Send>>, RouterError>
            {
                Err(RouterError::Empty)
            }
        }
        let s = ModelSlot::new(name.to_string(), Box::new(Noop), cost);
        *s.latency_ms.lock().unwrap() = latency;
        s
    }

    fn router_with(slots: Vec<ModelSlot>, strategy: RoutingStrategy) -> RouterLLM {
        RouterLLM {
            name: "router".to_string(),
            slots,
            strategy,
            counter: AtomicUsize::new(0),
        }
    }

    #[test]
    fn candidate_order_fallback_is_registration_order() {
        let r = router_with(
            vec![
                slot_at("a", None, 0.0),
                slot_at("b", None, 0.0),
                slot_at("c", None, 0.0),
            ],
            RoutingStrategy::Fallback,
        );
        assert_eq!(r.candidate_order(""), vec![0, 1, 2]);
    }

    #[test]
    fn candidate_order_lowest_cost_sorts_by_cost() {
        let r = router_with(
            vec![
                slot_at("pricey", Some(10.0), 0.0), // idx 0
                slot_at("cheap", Some(1.0), 0.0),   // idx 1
            ],
            RoutingStrategy::LowestCost,
        );
        assert_eq!(r.candidate_order(""), vec![1, 0]);
    }

    #[test]
    fn candidate_order_least_latency_sorts_by_latency() {
        let r = router_with(
            vec![
                slot_at("slow", None, 80.0), // idx 0
                slot_at("fast", None, 5.0),  // idx 1
            ],
            RoutingStrategy::LeastLatency,
        );
        assert_eq!(r.candidate_order(""), vec![1, 0]);
    }

    #[test]
    fn candidate_order_input_directed_puts_primary_first() {
        let r = router_with(
            vec![slot_at("a", None, 0.0), slot_at("b", None, 0.0)],
            RoutingStrategy::InputDirected(Arc::new(|s| if s.contains("x") { 1 } else { 0 })),
        );
        assert_eq!(r.candidate_order("hello"), vec![0, 1]);
        assert_eq!(r.candidate_order("x marks"), vec![1, 0]);
    }

    #[test]
    fn candidate_order_input_directed_invalid_index_falls_back() {
        let r = router_with(
            vec![slot_at("a", None, 0.0), slot_at("b", None, 0.0)],
            RoutingStrategy::InputDirected(Arc::new(|_| 99)),
        );
        assert_eq!(r.candidate_order("hi"), vec![0, 1]);
    }

    #[test]
    fn update_latency_ema_blends_samples() {
        let s = slot_at("m", None, 0.0);
        s.update_latency(100.0);
        assert_eq!(s.latency(), 100.0); // first sample replaces 0
        s.update_latency(100.0);
        // 100 * 0.7 + 100 * 0.3 = 100
        assert_eq!(s.latency(), 100.0);
        s.update_latency(40.0);
        // 100 * 0.7 + 40 * 0.3 = 82
        assert_eq!(s.latency(), 82.0);
    }
}
