// src/langgraph/compiled/types.rs
//! Public types for graph execution results and streaming events

use super::super::edge::GraphEdge;
use super::super::node::GraphNode;
use super::super::state::{StateSchema, StateUpdate};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

/// GraphInvocation - Result of graph execution
#[derive(Debug)]
pub struct GraphInvocation<S: StateSchema> {
    pub final_state: S,
    pub steps: Vec<ExecutionStep>,
    pub recursion_count: usize,
}

impl<S: StateSchema> GraphInvocation<S> {
    pub fn state(&self) -> &S {
        &self.final_state
    }

    pub fn steps(&self) -> &[ExecutionStep] {
        &self.steps
    }
}

/// ExecutionStep - Single step in execution history
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    Node {
        name: String,
        metadata: HashMap<String, JsonValue>,
    },
    Checkpoint {
        id: String,
        next_node: String,
    },
    ParallelNode {
        branch: String,
        metadata: HashMap<String, JsonValue>,
    },
}

impl ExecutionStep {
    pub fn node(name: String, metadata: HashMap<String, JsonValue>) -> Self {
        Self::Node { name, metadata }
    }

    pub fn checkpoint(id: String, next_node: String) -> Self {
        Self::Checkpoint { id, next_node }
    }

    pub fn parallel_node(branch: String, metadata: HashMap<String, JsonValue>) -> Self {
        Self::ParallelNode { branch, metadata }
    }
}

/// ParallelBranch - Result of a parallel execution branch
#[derive(Debug, Clone)]
pub struct ParallelBranch<S: StateSchema> {
    pub name: String,
    pub final_state: S,
    pub steps: Vec<ExecutionStep>,
}

/// ParallelInvocation - Result of graph execution with parallel branches
#[derive(Debug)]
pub struct ParallelInvocation<S: StateSchema> {
    pub final_state: S,
    pub steps: Vec<ExecutionStep>,
    pub recursion_count: usize,
    pub parallel_branches: Vec<ParallelBranch<S>>,
}

impl<S: StateSchema> ParallelInvocation<S> {
    pub fn state(&self) -> &S {
        &self.final_state
    }

    pub fn branches(&self) -> &[ParallelBranch<S>] {
        &self.parallel_branches
    }
}

/// StreamEvent - Event for streaming execution
#[derive(Debug, Clone)]
pub enum StreamEvent<S: StateSchema> {
    Start(S),
    EnterNode(String, S),
    NodeComplete(String, StateUpdate<S>),
    StateUpdate(S),
    End(S),
}

impl<S: StateSchema> StreamEvent<S> {
    pub fn start(state: S) -> Self {
        Self::Start(state)
    }

    pub fn enter_node(name: String, state: S) -> Self {
        Self::EnterNode(name, state)
    }

    pub fn node_complete(name: String, update: StateUpdate<S>) -> Self {
        Self::NodeComplete(name, update)
    }

    pub fn state_update(state: S) -> Self {
        Self::StateUpdate(state)
    }

    pub fn end(state: S) -> Self {
        Self::End(state)
    }
}

/// GraphExecution - State for interrupted execution that can be resumed
#[derive(Debug, Clone)]
pub struct GraphExecution<S: StateSchema> {
    pub state: S,
    pub current_node: String,
    pub steps: Vec<ExecutionStep>,
    pub recursion_count: usize,
    pub interrupted_at: String,
}

impl<S: StateSchema> GraphExecution<S> {
    pub fn new(state: S, current_node: String, interrupted_at: String) -> Self {
        Self {
            state,
            current_node,
            steps: Vec::new(),
            recursion_count: 0,
            interrupted_at,
        }
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn interrupted_at(&self) -> &str {
        &self.interrupted_at
    }
}

// ── Dynamic extension types ──

/// A task submitted externally for dynamic planning mid-execution.
#[derive(Debug, Clone)]
pub struct DynamicTask {
    pub id: String,
    pub description: String,
}

/// Result of dynamic planning: nodes and edges to inject into a running graph.
pub struct DynamicInjection<S: StateSchema> {
    pub nodes: Vec<(String, Arc<dyn GraphNode<S>>)>,
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
