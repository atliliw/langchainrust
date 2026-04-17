// src/langgraph/errors.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Routing error: {0}")]
    RoutingError(String),

    #[error("Recursion limit reached: {0}")]
    RecursionLimitReached(usize),

    #[error("Node error: {0}")]
    NodeError(String),

    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    #[error("State error: {0}")]
    StateError(String),

    #[error("Execution interrupted: {0}")]
    ExecutionInterrupted(String),

    #[error("Resume error: {0}")]
    ResumeError(String),

    #[error("Graph contains infinite cycle: {0}")]
    InfiniteCycleError(String),

    #[error("Orphan node detected: {0}")]
    OrphanNodeError(String),

    #[error("Duplicate edge: {0}")]
    DuplicateEdgeError(String),

    #[error("Missing route target: {0}")]
    MissingRouteTargetError(String),
}

pub type GraphResult<T> = Result<T, GraphError>;
