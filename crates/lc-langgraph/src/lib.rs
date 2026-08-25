#![warn(missing_docs)]
// crates/lc-langgraph/src/lib.rs
//! LangGraph - Graph-based orchestration framework for building stateful LLM applications
//!
//! LangGraph provides a low-level graph orchestration framework that enables:
//! - Stateful, long-running agent workflows
//! - Conditional routing and branching
//! - Subgraph composition
//! - Cycle support for iterative processes
//!
//! # Core Concepts
//!
//! - **StateGraph**: Main graph class that manages State, Nodes, and Edges
//! - **Node**: Execution unit that takes state and produces state updates
//! - **Edge**: Transition between nodes (fixed or conditional)
//! - **State**: Data structure that flows through the graph
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_langgraph::{StateGraph, GraphNode, StateSchema, START, END};
//! use serde::{Deserialize, Serialize};
//!
//! // Define state
//! #[derive(Serialize, Deserialize, Clone)]
//! struct MyState {
//!     messages: Vec<String>,
//!     count: usize,
//! }
//!
//! impl StateSchema for MyState {}
//!
//! // Create graph
//! let mut graph = StateGraph::<MyState>::new();
//!
//! // Add nodes
//! graph.add_node("process", |state: MyState| {
//!     MyState {
//!         messages: state.messages.clone(),
//!         count: state.count + 1,
//!     }
//! });
//!
//! // Add edges
//! graph.add_edge(START, "process");
//! graph.add_edge("process", END);
//!
//! // Compile and run
//! let compiled = graph.compile();
//! let result = compiled.invoke(MyState { messages: vec![], count: 0 }).await?;
//! ```

pub mod checkpointer;
pub mod compiled;
pub mod edge;
/// Graph error types and result aliases.
pub mod errors;
pub mod graph;
pub mod node;
pub mod persistence;
pub mod state;
pub mod subgraph;

// Re-export core types
pub use checkpointer::{
    CheckpointData, Checkpointer, FileCheckpointer, MemoryCheckpointer,
    ThreadSafeMemoryCheckpointer,
};
pub use compiled::{
    CompiledGraph, DynamicInjection, DynamicPlanner, DynamicTask, ExecutionStep, GraphExecution,
    GraphInvocation, ParallelBranch, ParallelInvocation, StreamEvent,
};
pub use edge::{AsyncFunctionRouter, ConditionalEdge, EdgeTarget, FunctionRouter, GraphEdge};
pub use errors::{GraphError, GraphResult};
pub use graph::{GraphBuilder, StateGraph, END, START};
pub use node::{AsyncFn, AsyncNode, GraphNode, NodeConfig, NodeResult};
pub use persistence::{
    EdgeDefinition, EdgeType, FilePersistence, GraphDefinition, GraphPersistence,
    MemoryPersistence, NodeDefinition, NodeType, PersistenceError, RouterDefinition,
};
pub use state::{
    AgentState, AppendMessagesReducer, AppendReducer, AppendStepsReducer, MessageEntry,
    MessageRole, Reducer, ReplaceReducer, StateSchema, StateUpdate, StepEntry,
};
pub use subgraph::{SubgraphBuilder, SubgraphNode};

#[cfg(feature = "mongodb-persistence")]
pub use persistence::{MongoConfig, MongoPersistence};
