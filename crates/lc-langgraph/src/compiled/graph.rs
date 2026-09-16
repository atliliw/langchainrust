// crates/lc-langgraph/src/compiled/graph.rs
//! CompiledGraph struct definition, constructors, getters, and runtime injection

use super::types::{DynamicTask, GraphExecution, GraphInvocation, PendingInterrupt};
use crate::checkpointer::{CheckpointInfo, Checkpointer};
use crate::edge::{ConditionalEdge, GraphEdge};
use crate::errors::{GraphError, GraphResult};
use crate::node::GraphNode;
use crate::state::{Reducer, StateSchema};
use crate::subgraph::SubgraphNode;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::RwLock;

/// CompiledGraph - Ready-to-execute graph
///
/// Created from StateGraph.compile(). Handles:
/// - State management and updates
/// - Edge routing (fixed and conditional)
/// - Execution with recursion limits
/// - Checkpointing for persistence
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct CompiledGraph<S: StateSchema> {
    pub(super) nodes: HashMap<String, Arc<dyn GraphNode<S>>>,
    pub(super) edges: Vec<GraphEdge>,
    pub(super) entry_point: String,
    pub(super) default_reducer: Arc<dyn Reducer<S>>,
    pub(super) conditional_routers: HashMap<String, Arc<dyn ConditionalEdge<S>>>,
    pub(super) checkpointer: Option<Arc<Mutex<dyn Checkpointer<S> + Send>>>,
    pub(super) recursion_limit: usize,
    pub(super) interrupt_before: Vec<String>,
    pub(super) interrupt_after: Vec<String>,

    pub(super) runtime_nodes: Arc<RwLock<HashMap<String, Arc<dyn GraphNode<S>>>>>,
    pub(super) runtime_edges: Arc<RwLock<Vec<GraphEdge>>>,
    pub(super) runtime_conditional_routers:
        Arc<RwLock<HashMap<String, Arc<dyn ConditionalEdge<S>>>>>,
    pub(super) task_inbox: Arc<Mutex<Vec<DynamicTask>>>,
}

impl<S: StateSchema> std::fmt::Debug for CompiledGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges)
            .field("entry_point", &self.entry_point)
            .field("recursion_limit", &self.recursion_limit)
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
            .finish()
    }
}

impl<S: StateSchema> CompiledGraph<S> {
    pub(crate) fn new(
        nodes: HashMap<String, Arc<dyn GraphNode<S>>>,
        edges: Vec<GraphEdge>,
        entry_point: String,
        default_reducer: Arc<dyn Reducer<S>>,
    ) -> Self {
        Self {
            nodes,
            edges,
            entry_point,
            default_reducer,
            conditional_routers: HashMap::new(),
            checkpointer: None,
            recursion_limit: 25,
            interrupt_before: Vec::new(),
            interrupt_after: Vec::new(),
            runtime_nodes: Arc::new(RwLock::new(HashMap::new())),
            runtime_edges: Arc::new(RwLock::new(Vec::new())),
            runtime_conditional_routers: Arc::new(RwLock::new(HashMap::new())),
            task_inbox: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn add_router(&mut self, name: String, router: Arc<dyn ConditionalEdge<S>>) {
        self.conditional_routers.insert(name, router);
    }

    /// Attach a checkpointer to this graph for state persistence.
    pub fn with_checkpointer<C: Checkpointer<S> + 'static>(mut self, checkpointer: C) -> Self {
        self.checkpointer = Some(Arc::new(Mutex::new(checkpointer)));
        self
    }

    /// Set the maximum recursion depth before execution aborts.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Set the node names before which execution should be interrupted.
    pub fn with_interrupt_before(mut self, nodes: Vec<String>) -> Self {
        self.interrupt_before = nodes;
        self
    }

    /// Set the node names after which execution should be interrupted.
    pub fn with_interrupt_after(mut self, nodes: Vec<String>) -> Self {
        self.interrupt_after = nodes;
        self
    }

    /// Return the names of all static nodes registered in the graph.
    pub fn node_names(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Return the graph's fixed edges.
    pub fn get_edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Return the entry point node name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Return the configured recursion limit.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Return the nodes that interrupt execution before running.
    pub fn interrupt_before(&self) -> &[String] {
        &self.interrupt_before
    }

    /// Return the nodes that interrupt execution after running.
    pub fn interrupt_after(&self) -> &[String] {
        &self.interrupt_after
    }

    /// Get the last checkpoint state (for interrupt recovery), together with the
    /// recursion budget consumed up to that checkpoint.
    ///
    /// H5: no longer guesses "most recent" from `list()`'s HashMap order + `ids.last()`;
    /// instead the checkpointer's `last()` returns the truly newest checkpoint by
    /// (timestamp, seq).
    pub async fn last_checkpoint_state(&self) -> Option<(S, usize)> {
        if let Some(ref cp) = self.checkpointer {
            let guard = cp.lock().await;
            guard.last().await.ok()?
        } else {
            None
        }
    }

    /// Return a locked handle to the attached checkpointer, or an error when none
    /// is attached (state history / time-travel need one) or it is contended.
    async fn checkpointer_locked(
        &self,
    ) -> GraphResult<tokio::sync::MutexGuard<'_, dyn Checkpointer<S> + Send>> {
        let Some(cp) = &self.checkpointer else {
            return Err(GraphError::CheckpointError(
                "state history / time-travel require a checkpointer (attach one with with_checkpointer)"
                    .to_string(),
            ));
        };
        Ok(cp.lock().await)
    }

    /// State history: every step's checkpoint snapshot, oldest first. This is the
    /// "agent 当时为什么这么决策" inspection primitive — each snapshot exposes the
    /// state, ordering keys, and recursion budget at that point in the run.
    pub async fn get_state_history(&self) -> GraphResult<Vec<CheckpointInfo<S>>> {
        let guard = self.checkpointer_locked().await?;
        guard.snapshots().await
    }

    /// Time-travel: fork a fresh execution timeline from an old checkpoint.
    ///
    /// - `checkpoint_id`: the snapshot to branch from (see [`get_state_history`](Self::get_state_history)).
    /// - `continue_at_node`: the node the forked run resumes at — normally the
    ///   `next_node` recorded in the history step immediately after that snapshot.
    /// - `override_state`: optionally replace the snapshot state with new input
    ///   ("改输入、从那条线分叉重跑") before running forward.
    ///
    /// The fork is seeded as a NEW checkpoint so the original history is untouched
    /// (its own lineage), continues counting against the snapshot's recursion
    /// budget, and — because it starts at `continue_at_node` and only runs
    /// FORWARD via `invoke_from_node` — never replays side effects emitted before
    /// the fork point.
    pub async fn fork_from(
        &self,
        checkpoint_id: &str,
        continue_at_node: impl Into<String>,
        override_state: Option<S>,
    ) -> GraphResult<GraphInvocation<S>> {
        let snapshot = {
            let guard = self.checkpointer_locked().await?;
            guard
                .snapshots()
                .await?
                .into_iter()
                .find(|s| s.id == checkpoint_id)
                .ok_or_else(|| {
                    GraphError::CheckpointError(format!("Checkpoint '{checkpoint_id}' not found"))
                })?
        };
        let base_state = override_state.unwrap_or(snapshot.state);

        // Seed a fresh checkpoint so the fork lives in its own lineage.
        let guard = self.checkpointer_locked().await?;
        guard.save(&base_state, snapshot.recursion_count).await?;
        drop(guard);

        self.invoke_from_node_with_count(
            continue_at_node.into(),
            base_state,
            snapshot.recursion_count,
        )
        .await
    }

    /// Create a resume execution context (from the last checkpoint)
    /// interrupted_node may be "node_name" or "after_node_name"
    pub async fn create_resume_execution(
        &self,
        interrupted_node: &str,
    ) -> Option<GraphExecution<S>> {
        let (state, recursion_count) = self.last_checkpoint_state().await?;
        let current = interrupted_node
            .strip_prefix("after_")
            .unwrap_or(interrupted_node);
        Some(GraphExecution {
            state,
            current_node: current.to_string(),
            steps: Vec::new(),
            // M6: resume carries over the recursion budget already consumed before the
            // interrupt, instead of restarting from zero — otherwise repeated
            // interrupt→resume could bypass recursion_limit indefinitely.
            recursion_count,
            interrupted_at: interrupted_node.to_string(),
            pending_interrupt: None,
        })
    }

    /// Resume from a runtime interrupt at `interrupted_node`, feeding the
    /// human's `value` back into that node.
    ///
    /// The interrupted node is re-entered (LangGraph-style) and the value is
    /// injected into its `NodeConfig.metadata` under
    /// [`crate::node::INTERRUPT_RESUME_KEY`], so an [`crate::node::InterruptibleNode`]
    /// closure can branch on `Some(decision)` to continue past the suspension
    /// without re-running side effects. Requires a checkpointer with the
    /// checkpoint that was persisted when the interrupt fired.
    pub async fn resume_with_value(
        &self,
        interrupted_node: &str,
        value: serde_json::Value,
    ) -> GraphResult<GraphInvocation<S>> {
        let mut execution = self
            .create_resume_execution(interrupted_node)
            .await
            .ok_or_else(|| {
                GraphError::ResumeError(format!(
                    "cannot resume '{}': no checkpoint to resume from (attach a checkpointer)",
                    interrupted_node
                ))
            })?;
        execution.pending_interrupt = Some(PendingInterrupt {
            id: interrupted_node.to_string(),
            value,
        });
        self.invoke_with_execution(execution).await
    }

    pub(super) async fn get_node(&self, name: &str) -> GraphResult<Arc<dyn GraphNode<S>>> {
        if let Some(node) = self.nodes.get(name) {
            return Ok(node.clone());
        }
        if let Some(node) = self.runtime_nodes.read().await.get(name) {
            return Ok(node.clone());
        }
        Err(GraphError::ExecutionError(format!(
            "Node '{}' not found",
            name
        )))
    }

    /// Submit a new task for dynamic planning mid-execution.
    ///
    /// Returns an error instead of silently dropping the task when the inbox
    /// lock is contended (previously a failed `try_lock` swallowed it, so a task
    /// could vanish without any signal).
    pub fn submit_task(&self, description: impl Into<String>) -> Result<(), GraphError> {
        let mut inbox = self.task_inbox.try_lock().map_err(|_| {
            GraphError::ExecutionError(
                "dynamic task inbox is busy; task refused (not silently dropped)".to_string(),
            )
        })?;
        inbox.push(DynamicTask {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
        });
        Ok(())
    }

    /// Inject a runtime node directly
    pub async fn inject_node(&self, name: &str, node: Arc<dyn GraphNode<S>>) -> GraphResult<()> {
        self.runtime_nodes
            .write()
            .await
            .insert(name.to_string(), node);
        Ok(())
    }

    /// Inject a runtime edge directly
    pub async fn inject_edge(&self, source: &str, target: &str) -> GraphResult<()> {
        self.runtime_edges
            .write()
            .await
            .push(GraphEdge::fixed(source, target));
        Ok(())
    }

    /// Inject a subgraph as a runtime node
    pub async fn inject_subgraph<SubS: StateSchema + 'static>(
        &self,
        name: &str,
        subgraph: CompiledGraph<SubS>,
        input_mapper: impl Fn(&S) -> SubS + Send + Sync + 'static,
        output_mapper: impl Fn(&SubS, &mut S) + Send + Sync + 'static,
    ) -> GraphResult<()> {
        let node = SubgraphNode::new(name, subgraph, input_mapper, output_mapper);
        self.runtime_nodes
            .write()
            .await
            .insert(name.to_string(), Arc::new(node));
        Ok(())
    }
}
