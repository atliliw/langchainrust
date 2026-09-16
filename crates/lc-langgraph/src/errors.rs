// crates/lc-langgraph/src/errors.rs

use thiserror::Error;

/// Errors that can occur during graph construction and execution.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum GraphError {
    /// A graph validation check failed.
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// An error occurred while executing the graph.
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// An error occurred while routing between nodes.
    #[error("Routing error: {0}")]
    RoutingError(String),

    /// The recursion limit was reached during execution.
    #[error("Recursion limit reached: {0}")]
    RecursionLimitReached(usize),

    /// A node reported an error.
    #[error("Node error: {0}")]
    NodeError(String),

    /// A checkpoint operation failed.
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    /// Optimistic-concurrency conflict while editing a checkpoint
    /// (`Checkpointer::update_state`): the stored version no longer matches
    /// the version the caller expected.
    #[error(
        "Checkpoint version conflict on '{checkpoint_id}': expected {expected}, found {actual}"
    )]
    CheckpointVersionConflict {
        /// Checkpoint the edit targeted.
        checkpoint_id: String,
        /// Version the edit was based on.
        expected: u64,
        /// Version currently stored.
        actual: u64,
    },

    /// A state update or merge failed.
    #[error("State error: {0}")]
    StateError(String),

    /// Execution was interrupted at the named node.
    #[error("Execution interrupted: {0}")]
    ExecutionInterrupted(String),

    /// Resuming execution failed.
    #[error("Resume error: {0}")]
    ResumeError(String),

    /// The graph contains a cycle with no path to `END`.
    #[error("Graph contains infinite cycle: {0}")]
    InfiniteCycleError(String),

    /// A node is unreachable from the entry point.
    #[error("Orphan node detected: {0}")]
    OrphanNodeError(String),

    /// Two edges duplicate the same source-target pair.
    #[error("Duplicate edge: {0}")]
    DuplicateEdgeError(String),

    /// A routing function returned a key with no matching target.
    #[error("Missing route target: {0}")]
    MissingRouteTargetError(String),

    /// An unexpected runtime error occurred.
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// A node requested a runtime (human-in-the-loop) interrupt. This is a
    /// *control-flow signal* produced by a node so it can suspend mid-execution.
    /// The compiled graph converts it into a [`GraphError::DynamicInterrupt`]
    /// (after persisting a checkpoint) before it reaches the graph caller.
    #[error("node requested a runtime interrupt")]
    InterruptRequest {
        /// Payload describing the interrupt (the question to ask, context, ...).
        payload: serde_json::Value,
    },

    /// Execution was suspended by a node's runtime interrupt. The checkpointer
    /// (when attached) has already persisted the run's state, so a process
    /// restart can resume. Feed a human's decision back in with
    /// [`crate::compiled::graph::CompiledGraph::resume_with_value`].
    #[error("node '{node}' suspended execution with a runtime interrupt")]
    DynamicInterrupt {
        /// The node that requested the interrupt (re-entered on resume).
        node: String,
        /// The payload the node published when it suspended.
        payload: serde_json::Value,
    },
}

/// Convenience result type for graph operations.
pub type GraphResult<T> = Result<T, GraphError>;
