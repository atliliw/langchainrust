// crates/lc-langgraph/src/compiled/types.rs
//! Public types for graph execution results and streaming events

use crate::edge::GraphEdge;
use crate::node::GraphNode;
use crate::state::{StateSchema, StateUpdate};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

/// GraphInvocation - Result of graph execution
#[derive(Debug)]
pub struct GraphInvocation<S: StateSchema> {
    /// The final state after execution.
    pub final_state: S,
    /// Execution steps recorded during the run.
    pub steps: Vec<ExecutionStep>,
    /// Number of recursive steps consumed during execution.
    pub recursion_count: usize,
}

impl<S: StateSchema> GraphInvocation<S> {
    /// Return the final state.
    pub fn state(&self) -> &S {
        &self.final_state
    }

    /// Return the recorded execution steps.
    pub fn steps(&self) -> &[ExecutionStep] {
        &self.steps
    }
}

/// ExecutionStep - Single step in execution history
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    /// A regular node execution step.
    Node {
        /// Name of the executed node.
        name: String,
        /// Metadata returned by the node execution.
        metadata: HashMap<String, JsonValue>,
    },
    /// A checkpoint was saved during execution.
    Checkpoint {
        /// The saved checkpoint id.
        id: String,
        /// The next node scheduled to run after the checkpoint.
        next_node: String,
    },
    /// A parallel branch execution step.
    ParallelNode {
        /// Name of the parallel branch.
        branch: String,
        /// Metadata returned by the branch execution.
        metadata: HashMap<String, JsonValue>,
    },
}

impl ExecutionStep {
    /// Construct a `Node` execution step.
    pub fn node(name: impl Into<String>, metadata: HashMap<String, JsonValue>) -> Self {
        Self::Node {
            name: name.into(),
            metadata,
        }
    }

    /// Construct a `Checkpoint` execution step.
    pub fn checkpoint(id: impl Into<String>, next_node: impl Into<String>) -> Self {
        Self::Checkpoint {
            id: id.into(),
            next_node: next_node.into(),
        }
    }

    /// Construct a `ParallelNode` execution step.
    pub fn parallel_node(branch: impl Into<String>, metadata: HashMap<String, JsonValue>) -> Self {
        Self::ParallelNode {
            branch: branch.into(),
            metadata,
        }
    }
}

/// ParallelBranch - Result of a parallel execution branch
#[derive(Debug, Clone)]
pub struct ParallelBranch<S: StateSchema> {
    /// Name of the parallel branch.
    pub name: String,
    /// Final state produced by the branch.
    pub final_state: S,
    /// Execution steps recorded in the branch.
    pub steps: Vec<ExecutionStep>,
}

/// ParallelInvocation - Result of graph execution with parallel branches
#[derive(Debug)]
pub struct ParallelInvocation<S: StateSchema> {
    /// The final merged state after execution.
    pub final_state: S,
    /// Execution steps recorded during the run.
    pub steps: Vec<ExecutionStep>,
    /// Number of recursive steps consumed during execution.
    pub recursion_count: usize,
    /// Captured results of each parallel branch.
    pub parallel_branches: Vec<ParallelBranch<S>>,
}

impl<S: StateSchema> ParallelInvocation<S> {
    /// Return the final merged state.
    pub fn state(&self) -> &S {
        &self.final_state
    }

    /// Return the captured parallel branch results.
    pub fn branches(&self) -> &[ParallelBranch<S>] {
        &self.parallel_branches
    }
}

/// StreamEvent - Event for streaming execution
#[derive(Debug, Clone)]
pub enum StreamEvent<S: StateSchema> {
    /// Execution started with the initial state.
    Start(S),
    /// A node was entered with the current state.
    EnterNode(String, S),
    /// A node completed with its state update.
    NodeComplete(String, StateUpdate<S>),
    /// The state was updated after applying a reducer.
    StateUpdate(S),
    /// Execution finished with the final state.
    End(S),
}

impl<S: StateSchema> StreamEvent<S> {
    /// Construct a `Start` stream event.
    pub fn start(state: S) -> Self {
        Self::Start(state)
    }

    /// Construct an `EnterNode` stream event.
    pub fn enter_node(name: impl Into<String>, state: S) -> Self {
        Self::EnterNode(name.into(), state)
    }

    /// Construct a `NodeComplete` stream event.
    pub fn node_complete(name: impl Into<String>, update: StateUpdate<S>) -> Self {
        Self::NodeComplete(name.into(), update)
    }

    /// Construct a `StateUpdate` stream event.
    pub fn state_update(state: S) -> Self {
        Self::StateUpdate(state)
    }

    /// Construct an `End` stream event.
    pub fn end(state: S) -> Self {
        Self::End(state)
    }
}

/// GraphExecution - State for interrupted execution that can be resumed
#[derive(Debug, Clone)]
pub struct GraphExecution<S: StateSchema> {
    /// The state at the point of interruption.
    pub state: S,
    /// The node where execution was interrupted.
    pub current_node: String,
    /// Execution steps recorded so far.
    pub steps: Vec<ExecutionStep>,
    /// Recursion budget consumed up to the interruption.
    pub recursion_count: usize,
    /// Where execution was interrupted ("node" or "after_node").
    pub interrupted_at: String,
}

impl<S: StateSchema> GraphExecution<S> {
    /// Create a new execution context for resuming from an interruption.
    pub fn new(
        state: S,
        current_node: impl Into<String>,
        interrupted_at: impl Into<String>,
    ) -> Self {
        Self {
            state,
            current_node: current_node.into(),
            steps: Vec::new(),
            recursion_count: 0,
            interrupted_at: interrupted_at.into(),
        }
    }

    /// Return the current state.
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Return the interruption point.
    pub fn interrupted_at(&self) -> &str {
        &self.interrupted_at
    }
}

// ── Dynamic extension types ──

/// A task submitted externally for dynamic planning mid-execution.
#[derive(Debug, Clone)]
pub struct DynamicTask {
    /// Unique id of the task.
    pub id: String,
    /// Description of the task to plan for.
    pub description: String,
}

/// Result of dynamic planning: nodes and edges to inject into a running graph.
pub struct DynamicInjection<S: StateSchema> {
    /// Nodes to inject into the runtime, keyed by name.
    pub nodes: Vec<(String, Arc<dyn GraphNode<S>>)>,
    /// Edges to inject into the runtime.
    pub edges: Vec<GraphEdge>,
}

/// Planner trait for converting external tasks into graph nodes/edges at runtime.
#[async_trait]
pub trait DynamicPlanner<S: StateSchema>: Send + Sync {
    /// Given pending tasks and current state, produce nodes/edges to inject.
    async fn plan(
        &self,
        tasks: &[DynamicTask],
        current_state: &S,
    ) -> Result<DynamicInjection<S>, String>;
}
