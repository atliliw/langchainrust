//! Unit + integration tests for routing order, latency weighting, per-model
//! rate limiting and the budget circuit breaker. All model traffic is
//! scripted in-process; paused Tokio time makes windows and waits instant.

use super::*;
use crate::cost::ModelPrice;
use crate::language_models::TokenUsage;
use crate::model_registry::ModelInfo;
use futures_util::StreamExt;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers: slots and routers for order-derivation tests
// ---------------------------------------------------------------------------

fn slot_at(cost: Option<f64>, latency: f64) -> ModelSlot {
    // A minimal stand-in model is not needed to test order derivation;
    // we only exercise `candidate_order` / `latency` logic. Build slots
    // with a no-op model via a tiny helper trait impl below.
    struct Noop;
    #[async_trait]
    impl RoutedModel for Noop {
        fn name(&self) -> &str {
            "noop"
        }

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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, RouterError>> + Send>>, RouterError>
        {
            Err(RouterError::Empty)
        }
    }
    let s = ModelSlot::new(Box::new(Noop), cost);
    *s.latency_ms.lock().unwrap_or_else(|e| e.into_inner()) = latency;
    s
}

fn router_with(slots: Vec<ModelSlot>, strategy: RoutingStrategy) -> RouterLLM {
    RouterLLM {
        name: "router".to_string(),
        slots,
        strategy,
        counter: AtomicUsize::new(0),
        rng: AtomicU64::new(0x9E37_79B9_7F4A_7C15),
        registry: None,
        budget: None,
    }
}

fn router_with_registry(slots: Vec<ModelSlot>, registry: Arc<ModelRegistry>) -> RouterLLM {
    RouterLLM {
        name: "router".to_string(),
        slots,
        strategy: RoutingStrategy::LowestCost,
        counter: AtomicUsize::new(0),
        rng: AtomicU64::new(0x9E37_79B9_7F4A_7C15),
        registry: Some(registry),
        budget: None,
    }
}

// ---------------------------------------------------------------------------
// Helpers: scripted BaseChatModel for end-to-end router behavior
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ScriptError(&'static str);

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "scripted model failure: {}", self.0)
    }
}

impl std::error::Error for ScriptError {}

#[derive(Clone)]
struct ScriptedModel {
    name: &'static str,
    calls: Arc<AtomicUsize>,
    delay: Duration,
    fail: bool,
    usage: Option<TokenUsage>,
}

impl ScriptedModel {
    fn new(name: &'static str, delay_ms: u64) -> Self {
        Self {
            name,
            calls: Arc::new(AtomicUsize::new(0)),
            delay: Duration::from_millis(delay_ms),
            fail: false,
            usage: None,
        }
    }

    fn failing(mut self) -> Self {
        self.fail = true;
        self
    }

    fn with_usage(mut self, prompt: usize, completion: usize) -> Self {
        self.usage = Some(TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        });
        self
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for ScriptedModel {
    type Error = ScriptError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for ScriptedModel {
    fn model_name(&self) -> &str {
        self.name
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        // Deterministic cheap estimate (provider reports real usage anyway).
        text.len() / 4
    }

    fn with_temperature(self, _temp: f32) -> Self
    where
        Self: Sized,
    {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        self
    }
}

#[async_trait]
impl BaseChatModel for ScriptedModel {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, ScriptError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        if self.fail {
            return Err(ScriptError(self.name));
        }
        Ok(LLMResult {
            content: format!("{}-reply", self.name),
            model: self.name.to_string(),
            token_usage: self.usage.clone(),
            tool_calls: None,
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, ScriptError>> + Send>>, ScriptError>
    {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let delay = self.delay;
        let usage = self.usage.clone();
        let name = self.name;
        if self.fail {
            return Err(ScriptError(name));
        }
        Ok(Box::pin(async_stream::stream! {
            tokio::time::sleep(delay).await;
            yield Ok(StreamChunk::new("hello "));
            yield Ok(StreamChunk {
                text: "world".to_string(),
                token_usage: usage,
                tool_calls: None,
            });
        }))
    }
}

async fn invoke(router: &RouterLLM) -> Result<LLMResult, RouterError> {
    router.invoke(vec![Message::human("hi")], None).await
}

// ---------------------------------------------------------------------------
// Order derivation (ported from the former inline module)
// ---------------------------------------------------------------------------

#[test]
fn candidate_order_fallback_is_registration_order() {
    let r = router_with(
        vec![slot_at(None, 0.0), slot_at(None, 0.0), slot_at(None, 0.0)],
        RoutingStrategy::Fallback,
    );
    assert_eq!(r.candidate_order(""), vec![0, 1, 2]);
}

#[test]
fn candidate_order_lowest_cost_sorts_by_cost() {
    let r = router_with(
        vec![
            slot_at(Some(10.0), 0.0), // idx 0
            slot_at(Some(1.0), 0.0),  // idx 1
        ],
        RoutingStrategy::LowestCost,
    );
    assert_eq!(r.candidate_order(""), vec![1, 0]);
}

#[test]
fn candidate_order_lowest_cost_uses_registry_prices_for_keyed_slots() {
    let registry = Arc::new(
        ModelRegistry::new()
            .with(ModelInfo::new(
                "p",
                "cheap",
                128_000,
                ModelPrice::new(0.1, 0.4), // blended 0.175
            ))
            .with(ModelInfo::new(
                "p",
                "pricey",
                128_000,
                ModelPrice::new(2.5, 10.0), // blended 4.375
            )),
    );
    let r = router_with_registry(
        vec![
            slot_at(None, 0.0).with_key("p/pricey"), // idx 0 registered first
            slot_at(None, 0.0).with_key("p/cheap"),
        ],
        registry,
    );
    assert_eq!(r.candidate_order(""), vec![1, 0]);
}

#[test]
fn lowest_cost_unresolvable_key_sorts_last_and_explicit_cost_wins() {
    let registry = Arc::new(ModelRegistry::new().with(ModelInfo::new(
        "p",
        "known",
        8_000,
        ModelPrice::new(1.0, 1.0),
    )));
    let r = router_with_registry(
        vec![
            slot_at(None, 0.0).with_key("p/missing"),    // unknown → +∞
            slot_at(None, 0.0).with_key("p/known"),      // registry 1.0
            slot_at(Some(0.1), 0.0).with_key("p/known"), // explicit beats registry
        ],
        registry,
    );
    assert_eq!(r.candidate_order(""), vec![2, 1, 0]);
}

#[test]
fn lowest_cost_without_registry_treats_keyed_slots_as_max() {
    let r = router_with(
        vec![slot_at(None, 0.0).with_key("p/a"), slot_at(Some(1.0), 0.0)],
        RoutingStrategy::LowestCost,
    );
    assert_eq!(r.candidate_order(""), vec![1, 0]);
}

#[test]
fn lowest_cost_uses_slot_price_without_cost_override() {
    let r = router_with(
        vec![
            slot_at(None, 0.0).with_price(Some(ModelPrice::new(5.0, 5.0))),
            slot_at(None, 0.0).with_price(Some(ModelPrice::new(1.0, 1.0))),
        ],
        RoutingStrategy::LowestCost,
    );
    assert_eq!(r.candidate_order(""), vec![1, 0]);
}

#[test]
fn candidate_order_least_latency_sorts_by_latency() {
    let r = router_with(
        vec![
            slot_at(None, 80.0), // idx 0
            slot_at(None, 5.0),  // idx 1
        ],
        RoutingStrategy::LeastLatency,
    );
    assert_eq!(r.candidate_order(""), vec![1, 0]);
}

#[test]
fn candidate_order_input_directed_puts_primary_first() {
    let r = router_with(
        vec![slot_at(None, 0.0), slot_at(None, 0.0)],
        RoutingStrategy::InputDirected(Arc::new(|s| if s.contains("x") { 1 } else { 0 })),
    );
    assert_eq!(r.candidate_order("hello"), vec![0, 1]);
    assert_eq!(r.candidate_order("x marks"), vec![1, 0]);
}

#[test]
fn candidate_order_input_directed_invalid_index_falls_back() {
    let r = router_with(
        vec![slot_at(None, 0.0), slot_at(None, 0.0)],
        RoutingStrategy::InputDirected(Arc::new(|_| 99)),
    );
    assert_eq!(r.candidate_order("hi"), vec![0, 1]);
}

#[test]
fn update_latency_ema_blends_samples() {
    let s = slot_at(None, 0.0);
    s.update_latency(100.0);
    assert_eq!(s.latency(), 100.0); // first sample replaces 0
    s.update_latency(100.0);
    // 100 * 0.7 + 100 * 0.3 = 100
    assert_eq!(s.latency(), 100.0);
    s.update_latency(40.0);
    // 100 * 0.7 + 40 * 0.3 = 82
    assert_eq!(s.latency(), 82.0);
}

// ---------------------------------------------------------------------------
// LatencyWeighted
// ---------------------------------------------------------------------------

#[test]
fn latency_weighted_is_uniform_on_a_cold_pool() {
    let r = router_with(
        vec![slot_at(None, 0.0), slot_at(None, 0.0)],
        RoutingStrategy::LatencyWeighted(1.0),
    );
    let mut zeros = 0usize;
    for _ in 0..200 {
        if r.candidate_order("")[0] == 0 {
            zeros += 1;
        }
    }
    // Equal weights → each slot leads about half the time (deterministic PRNG).
    assert!(
        (80..=120).contains(&zeros),
        "cold-pool share should be ~50%, got {zeros}/200"
    );
}

#[test]
fn latency_weighted_prefers_fast_slot_in_proportion_to_latency() {
    // 100 ms vs 10 ms, beta 1 → fast slot gets 10/11 ≈ 91% of primaries.
    let r = router_with(
        vec![slot_at(None, 100.0), slot_at(None, 10.0)],
        RoutingStrategy::LatencyWeighted(1.0),
    );
    let mut fast = 0usize;
    for _ in 0..400 {
        if r.candidate_order("")[0] == 1 {
            fast += 1;
        }
    }
    assert!(
        fast >= 330,
        "10x faster slot should lead ~91%, got {fast}/400"
    );
}

#[test]
fn latency_weighted_shares_shift_when_latency_data_updates() {
    let slow = slot_at(None, 50.0);
    let fast = slot_at(None, 50.0);
    let r = router_with(vec![slow, fast], RoutingStrategy::LatencyWeighted(1.0));
    let mut slot1_before = 0usize;
    for _ in 0..200 {
        if r.candidate_order("")[0] == 1 {
            slot1_before += 1;
        }
    }
    assert!((70..=130).contains(&slot1_before), "{slot1_before}");

    // Fresh latency data makes slot 1 five times faster: weights must follow.
    // Set the EMA directly (an `update_latency` sample would only blend to 38).
    *r.slots[1]
        .latency_ms
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = 10.0;
    let mut slot1_after = 0usize;
    for _ in 0..400 {
        if r.candidate_order("")[0] == 1 {
            slot1_after += 1;
        }
    }
    assert!(
        slot1_after >= 300,
        "after latency update fast slot should dominate, got {slot1_after}/400"
    );
}

#[test]
fn latency_weighted_untried_slots_explore_at_best_observed_latency() {
    // Slot 0 observed at 10 ms, slot 1 never tried → assumed 10 ms, so the
    // new model shares the lead 50/50 rather than being parked.
    let r = router_with(
        vec![slot_at(None, 10.0), slot_at(None, 0.0)],
        RoutingStrategy::LatencyWeighted(1.0),
    );
    let mut untried = 0usize;
    for _ in 0..200 {
        if r.candidate_order("")[0] == 1 {
            untried += 1;
        }
    }
    assert!((70..=130).contains(&untried), "{untried}");
}

#[test]
fn latency_weighted_fallback_chain_follows_weights() {
    // latencies 100 / 10 / 50 → weight order: idx1 > idx2 > idx0.
    let r = router_with(
        vec![
            slot_at(None, 100.0),
            slot_at(None, 10.0),
            slot_at(None, 50.0),
        ],
        RoutingStrategy::LatencyWeighted(1.0),
    );
    for _ in 0..50 {
        let order = r.candidate_order("");
        let primary = order[0];
        let mut sorted: Vec<usize> = (0..3).filter(|&i| i != primary).collect();
        sorted.sort_by(|&a, &b| {
            let wa = 1.0 / r.slots[a].latency();
            let wb = 1.0 / r.slots[b].latency();
            wb.partial_cmp(&wa).unwrap()
        });
        assert_eq!(order[1..], sorted, "tail must follow weight order");
        assert_eq!(order.len(), 3);
    }
}

#[test]
fn latency_weighted_invalid_beta_falls_back_to_linear() {
    let r = router_with(
        vec![slot_at(None, 100.0), slot_at(None, 10.0)],
        RoutingStrategy::LatencyWeighted(f64::NAN),
    );
    let order = r.candidate_order("");
    assert_eq!(order.len(), 2);
}

// ---------------------------------------------------------------------------
// Rate limiting + queue-driven fallback
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn rate_limited_primary_falls_back_then_recovers_after_window() {
    let primary = ScriptedModel::new("primary", 5);
    let fallback = ScriptedModel::new("fallback", 5);
    let limit = ModelRateLimit::per_minute(1)
        .with_window(Duration::from_millis(200))
        .with_wait_timeout(Duration::from_millis(50));
    let router = RouterLLM::new(RoutingStrategy::Fallback)
        .with_model_rate_limited(primary.clone(), limit)
        .with_model(fallback.clone());

    // First call: primary admits immediately.
    let r1 = invoke(&router).await.unwrap();
    assert_eq!(r1.content, "primary-reply");
    assert_eq!(primary.call_count(), 1);

    // Second call: rate exhausted, queue wait times out → fallback serves.
    let r2 = invoke(&router).await.unwrap();
    assert_eq!(r2.content, "fallback-reply");
    assert_eq!(primary.call_count(), 1);
    assert_eq!(fallback.call_count(), 1);

    // Window rolls over and the permit returns: primary leads again.
    tokio::time::sleep(Duration::from_millis(210)).await;
    let r3 = invoke(&router).await.unwrap();
    assert_eq!(r3.content, "primary-reply");
    assert_eq!(primary.call_count(), 2);
}

#[tokio::test(start_paused = true)]
async fn concurrency_slot_is_held_for_whole_stream_and_fallback_serves() {
    // Primary allows 1 concurrent call and waits 25 ms for a free slot;
    // while one stream is open, a second call must roll to the fallback.
    let primary = ScriptedModel::new("primary", 1).with_usage(10, 5);
    let fallback = ScriptedModel::new("fallback", 1);
    let limit = ModelRateLimit::per_minute(0)
        .with_max_concurrent(1)
        .with_wait_timeout(Duration::from_millis(25));
    let router = Arc::new(
        RouterLLM::new(RoutingStrategy::Fallback)
            .with_model_rate_limited(primary.clone(), limit)
            .with_model(fallback.clone()),
    );

    // Call A starts a stream and holds it open for ~10 ms without consuming.
    let router_a = router.clone();
    let call_a = tokio::spawn(async move {
        let mut stream = router_a
            .stream_chat(vec![Message::human("a")], None)
            .await
            .unwrap();
        // Hold the stream (and its concurrency permit) without polling.
        let first = stream.next().await;
        (stream, first)
    });
    // Let call A establish (paused time: task runs to first yield).
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;

    // Call B: primary slot saturated → timeout → fallback.
    let rb = invoke(&router).await.unwrap();
    assert_eq!(rb.content, "fallback-reply");
    assert_eq!(fallback.call_count(), 1);

    // Drain A; it still came from the primary.
    let (mut stream, first) = call_a.await.unwrap();
    assert!(first.unwrap().unwrap().text.contains("hello"));
    while stream.next().await.is_some() {}
    assert_eq!(primary.call_count(), 1);
}

// ---------------------------------------------------------------------------
// Budget circuit breaker
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn budget_trips_paid_primary_and_rolls_to_free_fallback() {
    // 1000/1000 tokens at 0.01 / 0.04 per 1K = exactly 0.05 USD per call.
    let primary = ScriptedModel::new("primary", 1).with_usage(1000, 1000);
    let fallback = ScriptedModel::new("fallback", 1);
    let budget = Arc::new(RouterBudget::with_cost_limit(0.05));
    let router = RouterLLM::new(RoutingStrategy::Fallback)
        .with_priced_model(primary.clone(), ModelPrice::new(0.01, 0.04))
        .with_model(fallback.clone())
        .with_budget(budget.clone());

    // First call fits within the cap and measures exactly to the limit.
    let r1 = invoke(&router).await.unwrap();
    assert_eq!(r1.content, "primary-reply");
    assert!(
        (budget.spent_usd() - 0.05).abs() < 1e-9,
        "{}",
        budget.spent_usd()
    );

    // Second call: any positive projection overshoots → primary skipped,
    // free fallback serves.
    let r2 = invoke(&router).await.unwrap();
    assert_eq!(r2.content, "fallback-reply");
    assert_eq!(primary.call_count(), 1);
    assert_eq!(fallback.call_count(), 1);
    assert!(budget.trips() >= 1);
}

#[tokio::test(start_paused = true)]
async fn budget_without_free_fallback_returns_budget_error() {
    let primary = ScriptedModel::new("primary", 1).with_usage(1000, 1000);
    let budget = Arc::new(RouterBudget::with_cost_limit(0.05));
    let router = RouterLLM::new(RoutingStrategy::Fallback)
        .with_priced_model(primary, ModelPrice::new(0.01, 0.04))
        .with_budget(budget);

    invoke(&router).await.unwrap();
    let err = invoke(&router).await.unwrap_err();
    match err {
        RouterError::AllFailed { last, .. } => {
            assert!(
                last.to_string().contains("budget"),
                "expected budget-exceeded final error, got: {last}"
            );
        }
        other => panic!("expected AllFailed, got {other:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn budget_records_stream_usage_from_final_chunk() {
    let primary = ScriptedModel::new("primary", 1).with_usage(1000, 1000);
    let fallback = ScriptedModel::new("fallback", 1);
    let budget = Arc::new(RouterBudget::with_cost_limit(0.05));
    let router = RouterLLM::new(RoutingStrategy::Fallback)
        .with_priced_model(primary.clone(), ModelPrice::new(0.01, 0.04))
        .with_model(fallback.clone())
        .with_budget(budget.clone());

    let mut stream = router
        .stream_chat(vec![Message::human("hi")], None)
        .await
        .unwrap();
    let mut collected = String::new();
    while let Some(chunk) = stream.next().await {
        collected.push_str(&chunk.unwrap().text);
    }
    assert_eq!(collected, "hello world");
    // Usage arrived on the final chunk and priced the breaker to its limit.
    assert!((budget.spent_usd() - 0.05).abs() < 1e-9);

    // Next chat call must already roll over to the free fallback.
    let r = invoke(&router).await.unwrap();
    assert_eq!(r.content, "fallback-reply");
    assert_eq!(primary.call_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn token_budget_blocks_even_free_models_once_exceeded() {
    // Token dimension is price-agnostic: free models are skipped too once the
    // projected prompt would cross the cap.
    let primary = ScriptedModel::new("primary", 1).with_usage(80, 20);
    let budget = Arc::new(RouterBudget::with_token_limit(100));
    let router = RouterLLM::new(RoutingStrategy::Fallback)
        .with_model(primary)
        .with_budget(budget.clone());

    invoke(&router).await.unwrap(); // records 100 tokens
    let err = invoke(&router).await.unwrap_err();
    match err {
        RouterError::AllFailed { last, .. } => {
            assert!(last.to_string().contains("token budget"), "got: {last}");
        }
        other => panic!("expected AllFailed, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Observed-latency routing end to end
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn observed_call_latencies_reorder_least_latency_routing() {
    // Slot 0 is registered first but slow; slot 1 answers quickly.
    let slow = ScriptedModel::new("slow", 80);
    let fast = ScriptedModel::new("fast", 5);
    let router = RouterLLM::new(RoutingStrategy::LeastLatency)
        .with_model(slow.clone())
        .with_model(fast.clone());

    // Cold pool: registration order → slow serves the first call.
    let r1 = invoke(&router).await.unwrap();
    assert_eq!(r1.content, "slow-reply");

    // Fast slot has never succeeded yet; give it one observed sample by
    // forcing a fallback (slow fails this once).
    // Build fresh models so counters stay interpretable.
    let slow2 = ScriptedModel::new("slow2", 80).failing();
    let fast2 = ScriptedModel::new("fast2", 5);
    let router2 = RouterLLM::new(RoutingStrategy::LeastLatency)
        .with_model(slow2.clone())
        .with_model(fast2.clone());
    let r2 = invoke(&router2).await.unwrap();
    assert_eq!(r2.content, "fast2-reply"); // slow attempted first, fell through
    assert_eq!(slow2.call_count(), 1);
    assert_eq!(fast2.call_count(), 1);

    // Now both have latency data; the next call must lead with the fast one.
    let r3 = invoke(&router2).await.unwrap();
    assert_eq!(r3.content, "fast2-reply");
    assert_eq!(fast2.call_count(), 2);
    assert_eq!(slow2.call_count(), 1, "slow slot must not be retried first");
}
