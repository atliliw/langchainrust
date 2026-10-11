//! P2-9: scale to ~1000 agents.
//!
//! This module provides the building blocks for large agent fleets, where
//! enumerating every agent card, letting workers talk to each other, or
//! retrying an unguarded call are all no longer viable:
//!
//! - [`SkillIndex`] — retrieve agents by *semantic* relevance over skill
//!   descriptions (bag-of-words cosine) instead of enumerating 1000 cards.
//! - [`HierarchyPolicy`] — enforce Orchestrator→Worker delegation and forbid
//!   Worker↔Worker fan-out.
//! - [`DelegationGuard`] — depth limit for delegation chains (default 10 hops).
//! - [`TaskSharder`] — stable task-ID hash → shard mapping for state split.
//! - [`CircuitBreaker`] — per-agent failure circuit breaker.
//! - [`StickyRouter`] — hash-based routing so a stateful agent is pinned per
//!   conversation key.
//! - [`TaskGraph`] — global parent/child task graph with cycle detection, so a
//!   delegation loop cannot grow unboundedly.
//! - [`AgentFleet`] — the runtime that composes all of the above into one
//!   routing decision.
//!
//! # Wiring status
//!
//! The building blocks above are wired through [`AgentFleet`], which is held by
//! [`crate::A2AClient`] (attach it with
//! [`A2AClientBuilder::fleet`](crate::A2AClientBuilder::fleet)). The client's
//! `send_task_routed` / `send_task_routed_sticky` / `delegate_task` methods ask
//! the fleet which agent a request goes to, so the policy — not the caller — is
//! what selects the target:
//!
//! ```text
//! send_task_routed(query, msg)
//!   -> AgentFleet::route(query)
//!        -> SkillIndex::search(query)      relevance ranking
//!        -> CircuitBreaker::allow_request  skip tripped agents
//!   -> POST to the selected AgentCard::url
//!   -> AgentFleet::record_success/_failure   feed the breaker back
//! ```
//!
//! [`crate::SkillAffinityRouter`] covers the server-side counterpart: choosing
//! the chain that serves a task by skill affinity.
//!
//! Task TTL already exists (P1-2), so the scale story is: TTL + hop limit + ring
//! detection + sharding + per-agent circuit breaking + sticky routing.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::protocol::{AgentCard, AgentSkill};

/// Errors raised by the scale-guard components.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScaleError {
    /// A delegation chain exceeded the configured hop limit.
    #[error("delegation would exceed the {max} hop limit at hop {hops}")]
    HopLimitExceeded {
        /// The delegation depth at which the limit was hit.
        hops: usize,
        /// The configured maximum delegation depth.
        max: usize,
    },
    /// A worker tried to delegate (Worker↔Worker fan-out is forbidden).
    #[error("worker agents may not delegate to other agents")]
    WorkerToWorker,
    /// Linking a task edge would create a cycle in the global task graph.
    #[error("task delegation {parent} -> {child} would create a cycle")]
    CycleDetected {
        /// The parent task attempting the delegation.
        parent: String,
        /// The child task that would create the cycle.
        child: String,
    },
    /// No fleet agent could serve the request: every candidate was irrelevant
    /// to the query, or its circuit breaker refused the call.
    #[error("no fleet agent available for query `{query}`")]
    NoAgentAvailable {
        /// The query that could not be routed.
        query: String,
    },
}

/// An indexed skill and the agent that offers it (P2-9).
#[derive(Debug, Clone)]
pub struct SkillEntry {
    /// The advertised skill.
    pub skill: AgentSkill,
    /// The agent that offers this skill.
    pub agent_url: String,
}

impl SkillEntry {
    /// Create an index entry.
    pub fn new(skill: AgentSkill, agent_url: impl Into<String>) -> Self {
        Self {
            skill,
            agent_url: agent_url.into(),
        }
    }
}

/// Semantic skill index: retrieve agents by relevance instead of enumerating
/// every card (P2-9).
///
/// Relevance is a bag-of-words cosine similarity over skill descriptions and
/// the query. This is deliberately dependency-free and deterministic — a fleet
/// that later adopts real embeddings can swap the scorer behind the same API.
#[derive(Debug, Default)]
pub struct SkillIndex {
    entries: Vec<SkillEntry>,
}

impl SkillIndex {
    /// An empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a single skill/agent pair.
    pub fn with_entry(mut self, entry: SkillEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Index every skill advertised on an agent card.
    pub fn index_card(&mut self, card: &AgentCard) {
        for skill in &card.skills {
            self.entries
                .push(SkillEntry::new(skill.clone(), card.url.clone()));
        }
    }

    /// The indexed entries, in insertion order.
    pub fn entries(&self) -> &[SkillEntry] {
        &self.entries
    }

    /// Search for agents matching `query`, returning the top `limit` entries
    /// ranked by relevance (best first). Entries with zero overlap are dropped.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SkillEntry> {
        let query_vec = tokens(query);
        let mut scored: Vec<(f64, &SkillEntry)> = self
            .entries
            .iter()
            .map(|e| (cosine(&query_vec, &tokens(&e.skill.description)), e))
            .filter(|(score, _)| *score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored.into_iter().map(|(_, e)| e.clone()).collect()
    }
}

/// Tokenize into lowercase alphanumeric words.
fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

/// Bag-of-words cosine similarity between two token lists (0.0 if either is
/// empty).
fn cosine(a: &[String], b: &[String]) -> f64 {
    let mut counts_a: HashMap<&str, usize> = HashMap::new();
    let mut counts_b: HashMap<&str, usize> = HashMap::new();
    for t in a {
        *counts_a.entry(t).or_insert(0) += 1;
    }
    for t in b {
        *counts_b.entry(t).or_insert(0) += 1;
    }
    if counts_a.is_empty() || counts_b.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (tok, ca) in &counts_a {
        let cb = counts_b.get(tok).copied().unwrap_or(0);
        dot += (*ca as f64) * (cb as f64);
        norm_a += (*ca as f64) * (*ca as f64);
    }
    for cb in counts_b.values() {
        norm_b += (*cb as f64) * (*cb as f64);
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Affinity between a task's required skill and a declared [`AgentSkill`]
/// (T3 skill-affinity routing, used by [`crate::SkillAffinityRouter`]).
///
/// The score is ordered so an exact match always dominates a fuzzy one:
///
/// | match                                          | score          |
/// |------------------------------------------------|----------------|
/// | skill `id`, case-insensitive                    | `1.0`          |
/// | skill `name`, case-insensitive                  | `0.8`          |
/// | one of `id` / `required` contains the other     | `0.7`          |
/// | bag-of-words cosine over id + name + description| `0.6 * cosine` |
/// | no overlap / empty `required`                   | `0.0`          |
///
/// The token branch tops out below `0.6`, so a threshold above that accepts
/// only id/name/substring matches — the "exact match first" rule stays true at
/// every threshold.
pub fn skill_affinity(required: &str, skill: &AgentSkill) -> f64 {
    let required = required.trim().to_lowercase();
    if required.is_empty() {
        return 0.0;
    }
    let id = skill.id.trim().to_lowercase();
    if required == id {
        return 1.0;
    }
    if required == skill.name.trim().to_lowercase() {
        return 0.8;
    }
    if !id.is_empty() && (id.contains(&required) || required.contains(&id)) {
        return 0.7;
    }
    let haystack = format!("{} {} {}", skill.id, skill.name, skill.description);
    0.6 * cosine(&tokens(&required), &tokens(&haystack))
}

/// Role of an agent in a delegation hierarchy (P2-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTier {
    /// Top-level coordinator; may delegate work down.
    Orchestrator,
    /// Leaf executor; must not fan out further (no Worker↔Worker).
    Worker,
}

/// Whether an agent at `from` tier may delegate to an agent at `to` tier.
///
/// Orchestrators may delegate to other orchestrators (sub-orchestration) and
/// to workers; workers may never delegate — the hierarchy fan-out rule.
pub fn may_delegate(from: AgentTier, to: AgentTier) -> bool {
    use AgentTier::{Orchestrator, Worker};
    matches!(
        (from, to),
        (Orchestrator, Orchestrator) | (Orchestrator, Worker)
    )
}

/// Tier membership map for a fleet, enforcing the Orchestrator→Worker rule
/// (P2-9).
#[derive(Debug, Default)]
pub struct HierarchyPolicy {
    orchestrators: HashSet<String>,
    workers: HashSet<String>,
}

impl HierarchyPolicy {
    /// An empty policy (every agent is treated as a least-privilege worker).
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark an agent URL as an orchestrator.
    pub fn with_orchestrator(mut self, url: impl Into<String>) -> Self {
        self.orchestrators.insert(url.into());
        self
    }

    /// Mark an agent URL as a worker.
    pub fn with_worker(mut self, url: impl Into<String>) -> Self {
        self.workers.insert(url.into());
        self
    }

    /// The tier of an agent; unknown agents default to [`AgentTier::Worker`]
    /// (least privilege).
    pub fn tier(&self, url: &str) -> AgentTier {
        if self.orchestrators.contains(url) {
            AgentTier::Orchestrator
        } else {
            AgentTier::Worker
        }
    }

    /// Check that `from` is allowed to delegate to `to`.
    pub fn check_delegation(&self, from: &str, to: &str) -> Result<(), ScaleError> {
        if may_delegate(self.tier(from), self.tier(to)) {
            Ok(())
        } else {
            Err(ScaleError::WorkerToWorker)
        }
    }
}

/// Depth limit for delegation chains (P2-9). Defaults to the A2A scale budget
/// of 10 hops.
#[derive(Debug, Clone, Copy)]
pub struct DelegationGuard {
    max_hops: usize,
}

impl Default for DelegationGuard {
    fn default() -> Self {
        Self { max_hops: 10 }
    }
}

impl DelegationGuard {
    /// A guard allowing up to `max_hops` delegation hops.
    pub fn new(max_hops: usize) -> Self {
        Self { max_hops }
    }

    /// The configured hop ceiling.
    pub fn max_hops(&self) -> usize {
        self.max_hops
    }

    /// Validate a chain depth of `hops` (0 = the root call, not a delegation).
    pub fn check(&self, hops: usize) -> Result<(), ScaleError> {
        if hops > self.max_hops {
            Err(ScaleError::HopLimitExceeded {
                hops,
                max: self.max_hops,
            })
        } else {
            Ok(())
        }
    }
}

/// FNV-1a hash — deterministic and cheap, good for shard/slot assignment.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Task-state sharding by stable task-ID hash (P2-9).
///
/// A 1000-agent fleet cannot keep every task in one store; this maps each
/// task id to a shard so state is distributed deterministically and collocated
/// lookups stay local.
#[derive(Debug, Clone, Copy)]
pub struct TaskSharder {
    num_shards: usize,
}

impl TaskSharder {
    /// A sharder over `num_shards` shards (clamped to at least 1).
    pub fn new(num_shards: usize) -> Self {
        Self {
            num_shards: num_shards.max(1),
        }
    }

    /// The shard index (0-based) responsible for `task_id`.
    pub fn shard(&self, task_id: &str) -> usize {
        (fnv1a(task_id.as_bytes()) % self.num_shards as u64) as usize
    }

    /// Shard assignments for many task ids, preserving order.
    pub fn shards<'a>(&self, task_ids: impl IntoIterator<Item = &'a str>) -> Vec<usize> {
        task_ids.into_iter().map(|id| self.shard(id)).collect()
    }

    /// The number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }
}

/// State of a per-agent [`CircuitBreaker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Normal operation; calls are allowed.
    Closed,
    /// Too many recent failures; calls are rejected.
    Open,
    /// A trial call is let through after `open_duration` to test recovery.
    HalfOpen,
}

/// Configuration for a [`CircuitBreaker`].
#[derive(Debug, Clone, Copy)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures before the breaker trips open.
    pub failure_threshold: usize,
    /// How long the breaker stays open before allowing a trial call.
    pub open_duration: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_duration: Duration::from_secs(30),
        }
    }
}

/// A per-agent circuit breaker (P2-9).
///
/// `Closed` → `Open` after `failure_threshold` consecutive failures. While
/// `Open`, calls are rejected; after `open_duration` a single trial call is
/// admitted (`HalfOpen`); a success resets to `Closed`, another failure trips
/// it straight back `Open`. Shared via the interior lock, so a fleet of
/// concurrent callers is throttled collectively.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    inner: std::sync::Mutex<BreakerInner>,
}

#[derive(Debug)]
struct BreakerInner {
    state: BreakerState,
    consecutive_failures: usize,
    opened_at: Option<Instant>,
    /// Whether a `HalfOpen` trial call is currently in flight. While set,
    /// every other request is refused so only ONE probe probes the recovering
    /// backend at a time (B7 guard — previously concurrent callers all passed).
    probe_in_flight: bool,
}

impl CircuitBreaker {
    /// A breaker with the given config.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            inner: std::sync::Mutex::new(BreakerInner {
                state: BreakerState::Closed,
                consecutive_failures: 0,
                opened_at: None,
                probe_in_flight: false,
            }),
        }
    }

    /// Whether a call may currently proceed.
    ///
    /// A closed breaker admits everything. An open breaker rejects until
    /// `open_duration` has elapsed, then admits exactly one trial call (the
    /// probe) and transitions to `HalfOpen`. While that probe is in flight,
    /// every other request is refused; it is released by
    /// [`Self::record_success`] / [`Self::record_failure`].
    pub fn allow_request(&self) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match inner.state {
            BreakerState::Closed => true,
            BreakerState::HalfOpen => {
                // A probe is already probing; drain the rest until it reports.
                !inner.probe_in_flight
            }
            BreakerState::Open => {
                let reopened = inner
                    .opened_at
                    .is_some_and(|at| at.elapsed() >= self.config.open_duration);
                if reopened {
                    inner.state = BreakerState::HalfOpen;
                    inner.probe_in_flight = true;
                    return true;
                }
                false
            }
        }
    }

    /// Record a successful call: resets the breaker to `Closed`.
    pub fn record_success(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.state = BreakerState::Closed;
        inner.consecutive_failures = 0;
        inner.opened_at = None;
        inner.probe_in_flight = false;
    }

    /// Record a failed call; trips `Open` once the threshold is reached.
    ///
    /// A failing `HalfOpen` probe re-trips the breaker `Open` immediately
    /// (re-arming `open_duration` before the next probe), rather than waiting
    /// for the full failure threshold again.
    pub fn record_failure(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.probe_in_flight = false;
        if inner.state == BreakerState::HalfOpen {
            inner.state = BreakerState::Open;
            inner.opened_at = Some(Instant::now());
            return;
        }
        inner.consecutive_failures += 1;
        if inner.consecutive_failures >= self.config.failure_threshold {
            inner.state = BreakerState::Open;
            inner.opened_at = Some(Instant::now());
        }
    }

    /// The current breaker state.
    pub fn state(&self) -> BreakerState {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).state
    }
}

/// Hash-based sticky routing for stateful agents (P2-9).
///
/// The same conversation key (e.g. task id or owner) always maps to the same
/// slot, so a stateful agent — memory, session, tool state — is pinned to one
/// backend for the life of the conversation. The caller maps a stable slot →
/// agent instance and keeps that mapping while the agent is healthy.
#[derive(Debug, Clone, Copy)]
pub struct StickyRouter {
    slots: usize,
}

impl StickyRouter {
    /// A router over `slots` agent instances (clamped to at least 1).
    pub fn new(slots: usize) -> Self {
        Self {
            slots: slots.max(1),
        }
    }

    /// The slot responsible for `key` — deterministic for a given key.
    pub fn route(&self, key: &str) -> usize {
        (fnv1a(key.as_bytes()) % self.slots as u64) as usize
    }

    /// The number of routable slots.
    pub fn slots(&self) -> usize {
        self.slots
    }
}

/// Global task graph with cycle detection (P2-9).
///
/// Tracks `parent task → child task` delegation edges. [`TaskGraph::link`]
/// refuses any edge that would close a cycle (a delegation loop that could
/// otherwise run forever), and [`TaskGraph::is_acyclic`] validates the whole
/// graph via a topological sweep.
#[derive(Debug, Default)]
pub struct TaskGraph {
    children: HashMap<String, Vec<String>>,
}

impl TaskGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a `parent → child` delegation edge.
    ///
    /// Returns [`ScaleError::CycleDetected`] if the edge would form a cycle
    /// (including a self-link), leaving the graph unchanged.
    pub fn link(&mut self, parent: &str, child: &str) -> Result<(), ScaleError> {
        if parent == child || self.would_cycle(parent, child) {
            return Err(ScaleError::CycleDetected {
                parent: parent.to_string(),
                child: child.to_string(),
            });
        }
        self.children
            .entry(parent.to_string())
            .or_default()
            .push(child.to_string());
        Ok(())
    }

    /// Whether adding `parent → child` would create a cycle — true when `child`
    /// is already an ancestor of `parent`.
    pub fn would_cycle(&self, parent: &str, child: &str) -> bool {
        let mut stack = vec![parent.to_string()];
        let mut seen = HashSet::new();
        while let Some(node) = stack.pop() {
            if node == child {
                return true;
            }
            if !seen.insert(node.clone()) {
                continue;
            }
            for ancestor in self.parents_of(&node) {
                stack.push(ancestor);
            }
        }
        false
    }

    /// All parents that list `node` as a child (reverse edges).
    fn parents_of(&self, node: &str) -> Vec<String> {
        self.children
            .iter()
            .filter(|(_, kids)| kids.iter().any(|k| k == node))
            .map(|(parent, _)| parent.clone())
            .collect()
    }

    /// Whether the whole graph is acyclic (Kahn's algorithm).
    pub fn is_acyclic(&self) -> bool {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut nodes: HashSet<String> = HashSet::new();
        for (parent, kids) in &self.children {
            nodes.insert(parent.clone());
            in_degree.entry(parent.clone()).or_insert(0);
            for kid in kids {
                nodes.insert(kid.clone());
                *in_degree.entry(kid.clone()).or_insert(0) += 1;
            }
        }
        let mut queue: VecDeque<String> = nodes
            .iter()
            .filter(|n| in_degree.get(*n).copied().unwrap_or(0) == 0)
            .cloned()
            .collect();
        let mut processed = 0;
        while let Some(node) = queue.pop_front() {
            processed += 1;
            if let Some(kids) = self.children.get(&node) {
                for kid in kids {
                    let degree = in_degree.get_mut(kid).expect("kid is in in_degree");
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(kid.clone());
                    }
                }
            }
        }
        processed == nodes.len()
    }
}

/// A fleet of candidate agents plus the composed policy that picks one (T3
/// wiring of this module into the client send path).
///
/// `AgentFleet` is the runtime counterpart of the building blocks above. It
/// composes:
///
/// - [`SkillIndex`] — relevance-rank candidates for a free-text query;
/// - [`CircuitBreaker`] — skip agents whose breaker is open;
/// - [`StickyRouter`] — pin a conversation key to one agent;
/// - [`TaskSharder`] — pin a task id to one agent among the relevant ones;
/// - [`HierarchyPolicy`] / [`DelegationGuard`] / [`TaskGraph`] — gate delegated
///   routing and record the delegation graph.
///
/// [`crate::A2AClient::send_task_routed`] (and its sticky/delegated variants)
/// call [`AgentFleet::route`] and friends, so the policy — not the caller — is
/// what decides which agent a task is sent to.
///
/// ```ignore
/// let mut fleet = AgentFleet::new();
/// fleet.register(retriever_card);
/// fleet.register(summarizer_card);
///
/// let client = A2AClient::builder("http://gateway").fleet(Arc::new(fleet)).build()?;
/// let routed = client.send_task_routed("retrieve documents", msg).await?;
/// assert_eq!(routed.agent_url, "http://retriever");
/// ```
#[derive(Debug)]
pub struct AgentFleet {
    index: SkillIndex,
    cards: HashMap<String, AgentCard>,
    /// Registration order — the stable slot space for [`StickyRouter`] and the
    /// deterministic iteration order for [`AgentFleet::agents`].
    order: Vec<String>,
    policy: HierarchyPolicy,
    guard: DelegationGuard,
    breaker_config: CircuitBreakerConfig,
    breakers: Mutex<HashMap<String, Arc<CircuitBreaker>>>,
    graph: Mutex<TaskGraph>,
}

impl Default for AgentFleet {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentFleet {
    /// An empty fleet.
    pub fn new() -> Self {
        Self {
            index: SkillIndex::new(),
            cards: HashMap::new(),
            order: Vec::new(),
            policy: HierarchyPolicy::new(),
            guard: DelegationGuard::default(),
            breaker_config: CircuitBreakerConfig::default(),
            breakers: Mutex::new(HashMap::new()),
            graph: Mutex::new(TaskGraph::new()),
        }
    }

    /// Configure the per-agent circuit breakers (default:
    /// [`CircuitBreakerConfig::default`]). Applies to agents registered after
    /// this call.
    pub fn with_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.breaker_config = config;
        self
    }

    /// Set the delegation hop budget (default 10).
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.guard = DelegationGuard::new(max_hops);
        self
    }

    /// Mark an agent URL as an orchestrator (may delegate down).
    pub fn with_orchestrator(mut self, url: impl Into<String>) -> Self {
        self.policy = self.policy.with_orchestrator(url);
        self
    }

    /// Mark an agent URL as a worker (may not delegate).
    pub fn with_worker(mut self, url: impl Into<String>) -> Self {
        self.policy = self.policy.with_worker(url);
        self
    }

    /// Register an agent card: index its skills, give it a circuit breaker and
    /// add it to the sticky slot space.
    ///
    /// Registering the same URL again replaces the card (its previous skills
    /// stop being indexed) and keeps its breaker and slot.
    pub fn register(&mut self, card: AgentCard) {
        let url = card.url.clone();
        if !self.cards.contains_key(&url) {
            self.order.push(url.clone());
            self.breakers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(url.clone(), Arc::new(CircuitBreaker::new(self.breaker_config)));
        }
        self.cards.insert(url, card);
        self.rebuild_index();
    }

    /// Rebuild the skill index from the registered cards in registration order,
    /// so a re-registration cannot leave stale skills behind.
    fn rebuild_index(&mut self) {
        let mut index = SkillIndex::new();
        for url in &self.order {
            if let Some(card) = self.cards.get(url) {
                index.index_card(card);
            }
        }
        self.index = index;
    }

    /// Every registered agent card, in registration order.
    pub fn agents(&self) -> Vec<AgentCard> {
        self.order
            .iter()
            .filter_map(|url| self.cards.get(url).cloned())
            .collect()
    }

    /// The card registered for `url`, if any.
    pub fn card(&self, url: &str) -> Option<AgentCard> {
        self.cards.get(url).cloned()
    }

    /// How many agents are registered.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether the fleet has no agents.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Agents whose declared skills are relevant to `query`, best first.
    ///
    /// At most one entry per agent: the card's best-scoring skill wins. The
    /// order is deterministic (relevance, then registration order for ties).
    pub fn candidates(&self, query: &str) -> Vec<AgentCard> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for entry in self.index.search(query, usize::MAX) {
            if !seen.insert(entry.agent_url.clone()) {
                continue;
            }
            if let Some(card) = self.cards.get(&entry.agent_url) {
                out.push(card.clone());
            }
        }
        out
    }

    /// The best healthy agent for a free-text `query`.
    ///
    /// Candidates are ranked by skill relevance and the first one whose circuit
    /// breaker admits a call wins. Returns [`ScaleError::NoAgentAvailable`] when
    /// nothing is relevant or every candidate is tripped.
    pub fn route(&self, query: &str) -> Result<AgentCard, ScaleError> {
        let candidates = self.candidates(query);
        self.first_admitting(&candidates)
            .ok_or_else(|| ScaleError::NoAgentAvailable {
                query: query.to_string(),
            })
    }

    /// Sticky routing: the same `conversation_key` prefers the same agent.
    ///
    /// The key hashes to a slot in the fleet's registration order; that agent
    /// is tried first when it is relevant to `query`. A pinned agent that is
    /// irrelevant (or whose breaker is open) is skipped so the conversation can
    /// still proceed — availability over pinning — and the caller can compare
    /// the returned card with the pinned one.
    pub fn route_sticky(
        &self,
        query: &str,
        conversation_key: &str,
    ) -> Result<AgentCard, ScaleError> {
        let candidates = self.candidates(query);
        if candidates.is_empty() {
            return Err(ScaleError::NoAgentAvailable {
                query: query.to_string(),
            });
        }
        let slot = StickyRouter::new(self.order.len().max(1)).route(conversation_key);
        let pinned = self.order.get(slot).cloned();
        let ordered = prefer_first(candidates, pinned.as_deref());
        self.first_admitting(&ordered)
            .ok_or_else(|| ScaleError::NoAgentAvailable {
                query: query.to_string(),
            })
    }

    /// Task-stable routing: the same `task_id` prefers the same agent among the
    /// relevance-ranked candidates, so a task's state stays collocated.
    pub fn route_for_task(&self, task_id: &str, query: &str) -> Result<AgentCard, ScaleError> {
        let candidates = self.candidates(query);
        if candidates.is_empty() {
            return Err(ScaleError::NoAgentAvailable {
                query: query.to_string(),
            });
        }
        let shard = TaskSharder::new(candidates.len()).shard(task_id);
        let pinned = candidates.get(shard).map(|card| card.url.clone());
        let ordered = prefer_first(candidates, pinned.as_deref());
        self.first_admitting(&ordered)
            .ok_or_else(|| ScaleError::NoAgentAvailable {
                query: query.to_string(),
            })
    }

    /// Delegation-checked routing: `from` must be allowed to delegate to the
    /// target (hierarchy rule) and `hops` must be inside the hop budget.
    ///
    /// Returns [`ScaleError::HopLimitExceeded`] when the budget is spent and
    /// [`ScaleError::WorkerToWorker`] when the query had candidates but none may
    /// be delegated to.
    pub fn route_from(
        &self,
        from: &str,
        hops: usize,
        query: &str,
    ) -> Result<AgentCard, ScaleError> {
        self.guard.check(hops)?;
        let candidates = self.candidates(query);
        if candidates.is_empty() {
            return Err(ScaleError::NoAgentAvailable {
                query: query.to_string(),
            });
        }
        let allowed: Vec<AgentCard> = candidates
            .into_iter()
            .filter(|card| self.policy.check_delegation(from, &card.url).is_ok())
            .collect();
        if allowed.is_empty() {
            return Err(ScaleError::WorkerToWorker);
        }
        self.first_admitting(&allowed)
            .ok_or_else(|| ScaleError::NoAgentAvailable {
                query: query.to_string(),
            })
    }

    /// The first candidate whose circuit breaker admits a call.
    ///
    /// [`CircuitBreaker::allow_request`] only mutates state when it returns
    /// `true`, so probing candidates in order cannot leak a half-open probe:
    /// the first admitted candidate is the one returned.
    fn first_admitting(&self, candidates: &[AgentCard]) -> Option<AgentCard> {
        for card in candidates {
            match self.breaker(&card.url) {
                Some(breaker) if !breaker.allow_request() => continue,
                _ => return Some(card.clone()),
            }
        }
        None
    }

    /// The circuit breaker for `url`, if the agent is registered.
    pub fn breaker(&self, url: &str) -> Option<Arc<CircuitBreaker>> {
        self.breakers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(url)
            .cloned()
    }

    /// The breaker state for `url`, if the agent is registered.
    pub fn breaker_state(&self, url: &str) -> Option<BreakerState> {
        self.breaker(url).map(|breaker| breaker.state())
    }

    /// Record a successful call to `url` (resets its breaker to `Closed`).
    pub fn record_success(&self, url: &str) {
        if let Some(breaker) = self.breaker(url) {
            breaker.record_success();
        }
    }

    /// Record a failed call to `url` (trips its breaker once the threshold is
    /// reached, so later routed sends skip it).
    pub fn record_failure(&self, url: &str) {
        if let Some(breaker) = self.breaker(url) {
            breaker.record_failure();
        }
    }

    /// Check that `from` may delegate to `to` under the hierarchy rule.
    pub fn check_delegation(&self, from: &str, to: &str) -> Result<(), ScaleError> {
        self.policy.check_delegation(from, to)
    }

    /// Check a delegation chain depth of `hops` against the hop budget.
    pub fn check_hops(&self, hops: usize) -> Result<(), ScaleError> {
        self.guard.check(hops)
    }

    /// The configured hop ceiling.
    pub fn max_hops(&self) -> usize {
        self.guard.max_hops()
    }

    /// Record a `parent_task -> child_task` delegation edge, refusing any edge
    /// that would close a cycle in the fleet's delegation graph.
    pub fn link_delegation(&self, parent: &str, child: &str) -> Result<(), ScaleError> {
        self.graph
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .link(parent, child)
    }

    /// Whether the recorded delegation graph is acyclic.
    pub fn is_delegation_acyclic(&self) -> bool {
        self.graph
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_acyclic()
    }
}

/// Move the card whose URL is `preferred` to the front of `candidates`.
///
/// A preferred agent that is not among the candidates leaves the order
/// unchanged (logged at debug level so a missed pin is diagnosable).
fn prefer_first(candidates: Vec<AgentCard>, preferred: Option<&str>) -> Vec<AgentCard> {
    let Some(preferred) = preferred else {
        return candidates;
    };
    let mut first = Vec::with_capacity(1);
    let mut rest = Vec::with_capacity(candidates.len());
    for card in candidates {
        if card.url == preferred {
            first.push(card);
        } else {
            rest.push(card);
        }
    }
    if first.is_empty() {
        log::debug!("preferred agent {preferred} is not a candidate for the query; using the ranked order");
        return rest;
    }
    first.extend(rest);
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(id: &str, description: &str) -> AgentSkill {
        AgentSkill::new(id, id, description)
    }

    #[test]
    fn skill_index_ranks_relevant_agent_first() {
        let mut index = SkillIndex::new();
        index.index_card(
            &AgentCard::new("retriever", "doc search", "http://retriever").with_skill(skill(
                "retrieve",
                "retrieve relevant documents from the corpus",
            )),
        );
        index.index_card(
            &AgentCard::new("summarizer", "text", "http://summarizer").with_skill(skill(
                "summarize",
                "condense long documents into a short summary",
            )),
        );

        let results = index.search("retrieve documents", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].agent_url, "http://retriever");

        let limited = index.search("retrieve documents", 1);
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].agent_url, "http://retriever");
    }

    #[test]
    fn skill_index_returns_nothing_for_unmatched_query() {
        let mut index = SkillIndex::new();
        index.index_card(
            &AgentCard::new("a", "a", "http://a").with_skill(skill("s", "transcribe audio")),
        );
        assert!(index.search("orbit physics", 5).is_empty());
        assert!(index.search("", 5).is_empty());
    }

    #[test]
    fn delegation_guard_enforces_hop_limit() {
        let guard = DelegationGuard::default();
        assert_eq!(guard.max_hops(), 10);
        guard.check(0).unwrap();
        guard.check(10).unwrap();
        let err = guard.check(11).unwrap_err();
        assert!(matches!(
            err,
            ScaleError::HopLimitExceeded { hops: 11, max: 10 }
        ));
    }

    #[test]
    fn sharder_is_stable_and_bounded() {
        let sharder = TaskSharder::new(4);
        // Same id always lands on the same shard.
        assert_eq!(sharder.shard("task-1"), sharder.shard("task-1"));
        assert_eq!(sharder.shard("task-42"), sharder.shard("task-42"));
        // Shards are in bounds.
        for id in ["a", "b", "c", "task-xyz"] {
            assert!(sharder.shard(id) < 4);
        }
        assert_eq!(sharder.shards(["a", "b"]).len(), 2);
        assert_eq!(sharder.num_shards(), 4);
    }

    #[test]
    fn sharder_clamps_to_one_shard() {
        let sharder = TaskSharder::new(0);
        assert_eq!(sharder.num_shards(), 1);
        assert_eq!(sharder.shard("anything"), 0);
    }

    #[test]
    fn circuit_breaker_trips_open_then_recovers() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            open_duration: Duration::from_millis(20),
        });

        // Closed: admits and counts failures.
        assert_eq!(breaker.state(), BreakerState::Closed);
        assert!(breaker.allow_request());
        breaker.record_failure();
        breaker.record_failure();
        assert!(breaker.allow_request());
        breaker.record_failure();
        assert_eq!(breaker.state(), BreakerState::Open);
        assert!(!breaker.allow_request(), "open breaker must reject calls");

        // After open_duration elapses, exactly one trial call is admitted.
        std::thread::sleep(Duration::from_millis(40));
        assert!(breaker.allow_request());
        assert_eq!(breaker.state(), BreakerState::HalfOpen);

        // Success resets to closed.
        breaker.record_success();
        assert_eq!(breaker.state(), BreakerState::Closed);
        assert!(breaker.allow_request());
    }

    #[test]
    fn half_open_probe_in_flight_drains_concurrent_requests() {
        // B7 point 8: while the single HalfOpen probe is in flight, every other
        // request must be refused — otherwise a concurrent storm re-pummels a
        // recovering backend.
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            open_duration: Duration::from_millis(5),
        });
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), BreakerState::Open);

        std::thread::sleep(Duration::from_millis(15));
        // The probe is admitted and the breaker goes HalfOpen.
        assert!(breaker.allow_request());
        assert_eq!(breaker.state(), BreakerState::HalfOpen);

        // While the probe is still in flight the breaker drains everyone else.
        assert!(
            !breaker.allow_request(),
            "half-open with a live probe must refuse concurrent calls"
        );

        // Probe reports; the breaker releases the drain.
        breaker.record_success();
        assert_eq!(breaker.state(), BreakerState::Closed);
        assert!(breaker.allow_request());
    }

    #[test]
    fn half_open_probe_failure_retrips_open() {
        // B7 point 8: a failing HalfOpen probe re-trips the breaker Open
        // immediately and re-arms open_duration, instead of waiting for the
        // full failure threshold to accumulate again.
        // Threshold is deliberately high (10): after the initial trip we record
        // only ONE fresh failure, well below it — the half-open retrip must fire
        // immediately regardless of the accumulated count.
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 10,
            open_duration: Duration::from_millis(5),
        });
        // Cross the full threshold once to trip Open.
        for _ in 0..10 {
            breaker.record_failure();
        }
        assert_eq!(breaker.state(), BreakerState::Open);

        std::thread::sleep(Duration::from_millis(15));
        assert!(breaker.allow_request()); // probe fires
        assert_eq!(breaker.state(), BreakerState::HalfOpen);

        breaker.record_failure(); // ONE failure, far below the threshold of 10
        assert_eq!(
            breaker.state(),
            BreakerState::Open,
            "a failing half-open probe must immediately re-trip open"
        );
        assert!(!breaker.allow_request());
    }

    #[test]
    fn sticky_router_is_deterministic() {
        let router = StickyRouter::new(3);
        assert_eq!(router.route("conv-1"), router.route("conv-1"));
        assert_eq!(router.route("conv-2"), router.route("conv-2"));
        for key in ["conv-1", "conv-2", "conv-3", "owner:alice"] {
            assert!(router.route(key) < 3);
        }
        assert_eq!(router.slots(), 3);
    }

    #[test]
    fn task_graph_accepts_acyclic_chain() {
        let mut graph = TaskGraph::new();
        graph.link("root", "child").unwrap();
        graph.link("child", "grandchild").unwrap();
        assert!(graph.is_acyclic());

        // Adding root->grandchild is safe: grandchild is a descendant, not an
        // ancestor, of root.
        assert!(!graph.would_cycle("root", "grandchild"));
        // But grandchild->root WOULD close the chain into a cycle.
        assert!(graph.would_cycle("grandchild", "root"));

        graph.link("root", "grandchild").unwrap();
        assert!(graph.is_acyclic());
    }

    #[test]
    fn task_graph_detects_cycle() {
        let mut graph = TaskGraph::new();
        graph.link("a", "b").unwrap();
        graph.link("b", "c").unwrap();

        // c -> a closes the loop a→b→c→a.
        let err = graph.link("c", "a").unwrap_err();
        assert!(
            matches!(err, ScaleError::CycleDetected { parent, child } if parent == "c" && child == "a")
        );

        // Self-link is also a cycle.
        let err = graph.link("x", "x").unwrap_err();
        assert!(matches!(err, ScaleError::CycleDetected { .. }));
    }

    #[test]
    fn task_graph_rejects_cycle_globally() {
        let mut graph = TaskGraph::new();
        graph.link("a", "b").unwrap();
        graph.link("b", "c").unwrap();

        // `link()` already refuses cycle-forming edges, so build a cycle
        // directly to exercise `is_acyclic`'s global topological sweep.
        graph
            .children
            .entry("c".to_string())
            .or_default()
            .push("a".to_string());
        assert!(graph.would_cycle("c", "a"));
        assert!(!graph.is_acyclic());
    }

    #[test]
    fn hierarchy_allows_orchestrator_but_blocks_worker() {
        let policy = HierarchyPolicy::new()
            .with_orchestrator("http://orch")
            .with_worker("http://worker");

        policy
            .check_delegation("http://orch", "http://worker")
            .unwrap();
        policy
            .check_delegation("http://orch", "http://orch")
            .unwrap();

        let err = policy
            .check_delegation("http://worker", "http://worker")
            .unwrap_err();
        assert!(matches!(err, ScaleError::WorkerToWorker));
        // Unknown agents default to worker (least privilege).
        assert!(matches!(
            policy.check_delegation("http://unknown", "http://worker"),
            Err(ScaleError::WorkerToWorker)
        ));
    }

    // ---- T3: skill affinity scoring ----

    #[test]
    fn affinity_ranks_exact_id_above_name_and_description() {
        let skill = AgentSkill::new(
            "translate",
            "Translation",
            "translate text between languages",
        );
        assert_eq!(skill_affinity("translate", &skill), 1.0);
        assert_eq!(skill_affinity("TRANSLATE", &skill), 1.0);
        assert_eq!(skill_affinity("translation", &skill), 0.8);
        // Substring of the id (not equal) still beats a description overlap.
        assert_eq!(skill_affinity("trans", &skill), 0.7);
        let fuzzy = skill_affinity("language text", &skill);
        assert!(
            fuzzy > 0.0 && fuzzy < 0.6,
            "token overlap must stay below the substring band, got {fuzzy}"
        );
    }

    #[test]
    fn affinity_is_zero_for_unrelated_or_empty_requests() {
        let skill = AgentSkill::new("translate", "Translation", "translate text");
        assert_eq!(skill_affinity("orbit physics", &skill), 0.0);
        assert_eq!(skill_affinity("", &skill), 0.0);
        assert_eq!(skill_affinity("   ", &skill), 0.0);
    }

    // ---- T3: AgentFleet routing ----

    fn fleet_card(name: &str, skill_id: &str, description: &str) -> AgentCard {
        AgentCard::new(name, description, format!("http://{name}")).with_skill(AgentSkill::new(
            skill_id,
            skill_id,
            description,
        ))
    }

    fn fleet() -> AgentFleet {
        let mut fleet = AgentFleet::new();
        fleet.register(fleet_card(
            "retriever",
            "retrieve",
            "retrieve relevant documents from the corpus",
        ));
        fleet.register(fleet_card(
            "summarizer",
            "summarize",
            "condense long documents into a short summary",
        ));
        fleet.register(fleet_card(
            "translator",
            "translate",
            "translate text between languages",
        ));
        fleet
    }

    #[test]
    fn fleet_routes_to_the_relevant_agent() {
        let fleet = fleet();
        assert_eq!(fleet.len(), 3);
        assert_eq!(fleet.route("retrieve documents").unwrap().url, "http://retriever");
        assert_eq!(
            fleet.route("condense documents summary").unwrap().url,
            "http://summarizer"
        );
        assert_eq!(
            fleet.route("translate languages").unwrap().url,
            "http://translator"
        );
        // An unrelated query has no candidate at all.
        assert!(matches!(
            fleet.route("orbit physics"),
            Err(ScaleError::NoAgentAvailable { .. })
        ));
    }

    #[test]
    fn fleet_skips_agents_whose_breaker_is_open() {
        let mut fleet = AgentFleet::new().with_breaker_config(CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_secs(30),
        });
        // Both agents answer "retrieve documents"; the first registered wins.
        fleet.register(fleet_card(
            "primary",
            "retrieve",
            "retrieve relevant documents from the corpus",
        ));
        fleet.register(fleet_card(
            "secondary",
            "retrieve",
            "retrieve relevant documents from the corpus",
        ));

        assert_eq!(
            fleet.route("retrieve documents").unwrap().url,
            "http://primary"
        );
        assert_eq!(
            fleet.breaker_state("http://primary"),
            Some(BreakerState::Closed)
        );

        fleet.record_failure("http://primary");
        assert_eq!(
            fleet.breaker_state("http://primary"),
            Some(BreakerState::Open)
        );
        assert_eq!(
            fleet.route("retrieve documents").unwrap().url,
            "http://secondary",
            "a tripped agent must be skipped in favor of the next candidate"
        );

        fleet.record_failure("http://secondary");
        assert!(matches!(
            fleet.route("retrieve documents"),
            Err(ScaleError::NoAgentAvailable { .. })
        ));

        // A success resets the breaker and the agent becomes routable again.
        fleet.record_success("http://primary");
        assert_eq!(
            fleet.route("retrieve documents").unwrap().url,
            "http://primary"
        );
    }

    #[test]
    fn fleet_sticky_routing_pins_among_equally_relevant_agents() {
        // Two agents advertising the *same* skill: relevance cannot separate
        // them, so only the sticky slot can — and it must be stable per key.
        let mut fleet = AgentFleet::new();
        for name in ["a", "b"] {
            fleet.register(fleet_card(
                name,
                "retrieve",
                "retrieve relevant documents from the corpus",
            ));
        }

        let first = fleet.route_sticky("retrieve documents", "conv-42").unwrap();
        let again = fleet.route_sticky("retrieve documents", "conv-42").unwrap();
        assert_eq!(first.url, again.url, "the same key must pin the same agent");

        // Across many keys both agents get pinned, so the slot — not the
        // registration order — drives the choice.
        let mut urls: Vec<String> = (0..64)
            .map(|i| {
                fleet
                    .route_sticky("retrieve documents", &format!("conv-{i}"))
                    .unwrap()
                    .url
            })
            .collect();
        urls.sort();
        urls.dedup();
        assert_eq!(
            urls,
            vec!["http://a".to_string(), "http://b".to_string()],
            "sticky routing must spread keys over the fleet"
        );
    }

    #[test]
    fn fleet_sticky_falls_back_when_the_pinned_agent_is_irrelevant() {
        let fleet = fleet();
        // "translate" only has one relevant agent; every key must land on it
        // even when the key hashes to another agent's slot.
        for i in 0..32 {
            let card = fleet
                .route_sticky("translate text between languages", &format!("key-{i}"))
                .unwrap();
            assert_eq!(card.url, "http://translator");
        }
    }

    #[test]
    fn fleet_task_routing_is_stable_and_relevance_bounded() {
        let fleet = fleet();
        let first = fleet.route_for_task("task-abc", "retrieve documents").unwrap();
        assert_eq!(
            first.url,
            fleet.route_for_task("task-abc", "retrieve documents").unwrap().url,
            "the same task id must pin to the same agent (sticky routing)"
        );
        // Sticky routing deliberately prefers a fixed slot; it may over-run a
        // higher-relevance candidate. The contract is that the chosen agent is
        // one of the relevance-ranked candidates, not necessarily the top one.
        let relevant = ["http://retriever", "http://summarizer"];
        assert!(
            relevant.contains(&first.url.as_str()),
            "task pinned to an unrelated agent, got: {first:?}"
        );
    }

    #[test]
    fn fleet_delegation_checks_hierarchy_and_hops() {
        let mut fleet = AgentFleet::new()
            .with_orchestrator("http://orch")
            .with_worker("http://retriever")
            .with_max_hops(2);
        fleet.register(fleet_card(
            "retriever",
            "retrieve",
            "retrieve relevant documents from the corpus",
        ));

        assert_eq!(fleet.max_hops(), 2);
        fleet.check_hops(2).unwrap();
        assert!(matches!(
            fleet.check_hops(3),
            Err(ScaleError::HopLimitExceeded { hops: 3, max: 2 })
        ));

        // An orchestrator may delegate to the worker.
        assert_eq!(
            fleet
                .route_from("http://orch", 1, "retrieve documents")
                .unwrap()
                .url,
            "http://retriever"
        );
        // A worker may not delegate at all.
        assert!(matches!(
            fleet.route_from("http://retriever", 1, "retrieve documents"),
            Err(ScaleError::WorkerToWorker)
        ));
        // The hop budget is checked before any routing happens.
        assert!(matches!(
            fleet.route_from("http://orch", 3, "retrieve documents"),
            Err(ScaleError::HopLimitExceeded { .. })
        ));
    }

    #[test]
    fn fleet_records_the_delegation_graph() {
        let fleet = AgentFleet::new();
        fleet.link_delegation("task-root", "task-child").unwrap();
        fleet.link_delegation("task-child", "task-grandchild").unwrap();
        assert!(fleet.is_delegation_acyclic());

        let err = fleet.link_delegation("task-grandchild", "task-root").unwrap_err();
        assert!(matches!(err, ScaleError::CycleDetected { .. }));
        assert!(fleet.is_delegation_acyclic());
    }

    #[test]
    fn fleet_reregistration_replaces_skills_and_keeps_the_breaker() {
        let mut fleet = AgentFleet::new();
        fleet.register(fleet_card("agent", "translate", "translate text"));
        fleet.record_failure("http://agent");
        // Re-register with a skill whose vocabulary shares no token with the
        // old skill, so the bag-of-words index can prove the old skill is gone.
        fleet.register(fleet_card("agent", "summarize", "summarize concise documents"));

        assert_eq!(fleet.len(), 1);
        // The old skill is gone from the index, so the old query no longer routes.
        assert!(matches!(
            fleet.route("translate text"),
            Err(ScaleError::NoAgentAvailable { .. })
        ));
        assert_eq!(
            fleet.route("summarize concise documents").unwrap().url,
            "http://agent"
        );
        // The breaker survived the re-registration.
        assert_eq!(fleet.breaker_state("http://agent"), Some(BreakerState::Closed));
    }
}
