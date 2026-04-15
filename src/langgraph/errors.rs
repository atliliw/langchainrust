// src/langgraph/errors.rs
//! Error types for LangGraph

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
}

pub type GraphResult<T> = Result<T, GraphError>;
