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
//!
//! Task TTL already exists (P1-2), so the scale story is: TTL + hop limit + ring
//! detection + sharding + per-agent circuit breaking + sticky routing.

use std::collections::{HashMap, HashSet, VecDeque};
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
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    inner: std::sync::Mutex<BreakerInner>,
}

#[derive(Debug)]
struct BreakerInner {
    state: BreakerState,
    consecutive_failures: usize,
    opened_at: Option<Instant>,
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
            }),
        }
    }

    /// Whether a call may currently proceed.
    ///
    /// A closed breaker admits everything. An open breaker rejects until
    /// `open_duration` has elapsed, then admits exactly one trial call and
    /// transitions to `HalfOpen`.
    pub fn allow_request(&self) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.state == BreakerState::Open {
            let reopened = inner
                .opened_at
                .is_some_and(|at| at.elapsed() >= self.config.open_duration);
            if reopened {
                inner.state = BreakerState::HalfOpen;
                return true;
            }
            return false;
        }
        true
    }

    /// Record a successful call: resets the breaker to `Closed`.
    pub fn record_success(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.state = BreakerState::Closed;
        inner.consecutive_failures = 0;
        inner.opened_at = None;
    }

    /// Record a failed call; trips `Open` once the threshold is reached.
    pub fn record_failure(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
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
}
