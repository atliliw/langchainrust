// src/langgraph/mod.rs
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
//! use langchainrust::langgraph::{StateGraph, GraphNode, StateSchema, START, END};
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

pub mod state;
pub mod node;
pub mod edge;
pub mod graph;
pub mod compiled;
pub mod errors;
pub mod checkpointer;
pub mod subgraph;
pub mod persistence;

// Re-export core types
pub use state::{StateSchema, StateUpdate, Reducer, ReplaceReducer, AppendReducer, AppendMessagesReducer, AppendStepsReducer, AgentState, MessageEntry, MessageRole, StepEntry};
pub use node::{GraphNode, NodeResult, NodeConfig, AsyncNode, AsyncFn};
pub use edge::{GraphEdge, ConditionalEdge, EdgeTarget, FunctionRouter, AsyncFunctionRouter};
pub use graph::{StateGraph, GraphBuilder, START, END};
pub use compiled::{CompiledGraph, GraphInvocation, StreamEvent, ExecutionStep, GraphExecution, ParallelInvocation, ParallelBranch};
pub use errors::{GraphError, GraphResult};
pub use checkpointer::{Checkpointer, MemoryCheckpointer, ThreadSafeMemoryCheckpointer, FileCheckpointer, CheckpointData};
pub use subgraph::{SubgraphNode, SubgraphBuilder};
pub use persistence::{
    GraphPersistence, GraphDefinition, NodeDefinition, EdgeDefinition,
    NodeType, EdgeType, RouterDefinition, MemoryPersistence, FilePersistence,
};

#[cfg(feature = "mongodb-persistence")]
pub use persistence::{MongoPersistence, MongoConfig};