# LangGraph

LangGraph provides graph-based orchestration for stateful, long-running LLM workflows with conditional routing, subgraphs, parallel execution, and human-in-the-loop.

## Core Concepts

| Concept | Type | Description |
|---------|------|-------------|
| `StateGraph<S>` | Builder | Define nodes, edges, and reducers on state `S` |
| `GraphBuilder<S>` | Builder | Fluent (consuming) variant of `StateGraph` |
| `CompiledGraph<S>` | Runtime | Compiled, executable graph |
| `GraphNode<S>` | Trait | Node execution: `async fn execute(&self, state, config) -> NodeResult<S>` |
| `ConditionalEdge<S>` | Trait | Dynamic routing: `async fn route(&self, state) -> String` |
| `Reducer<S>` | Trait | State merge strategy: `fn reduce(&self, current, update) -> S` |
| `Checkpointer<S>` | Trait | State persistence: `save`, `load`, `list`, `delete` |
| `SubgraphNode<S, SubS>` | Node | Embed a `CompiledGraph<SubS>` inside a parent graph |
| `StreamEvent<S>` | Enum | `Start`, `EnterNode`, `NodeComplete`, `StateUpdate`, `End` |

## Basic Graph

```rust
use langchainrust::{
    StateGraph, GraphBuilder, StateSchema, StateUpdate,
    AgentState, START, END,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MyState { messages: Vec<String>, count: usize }
impl StateSchema for MyState {}

let mut graph = StateGraph::<MyState>::new();
graph.add_node_fn("process", |state: MyState| {
    Ok(StateUpdate::full(MyState {
        messages: state.messages.clone(),
        count: state.count + 1,
    }))
});
graph.add_edge(START, "process");
graph.add_edge("process", END);

let compiled = graph.compile()?;
let result = compiled.invoke(MyState { messages: vec![], count: 0 }).await?;
```

## Conditional Routing & Human-in-the-Loop

```rust
use langchainrust::{GraphBuilder, FunctionRouter, AgentState, StateUpdate, START, END};
use std::collections::HashMap;

let compiled = GraphBuilder::<AgentState>::new()
    .add_node_fn("entry", |s| Ok(StateUpdate::full(s.clone())))
    .add_node_fn("short", |s| { /* ... */ })
    .add_node_fn("long", |s| { /* ... */ })
    .add_node_fn("review", |s| { /* ... */ })
    .add_edge(START, "entry")
    .add_conditional_edges("entry", "router",
        HashMap::from([("short".into(), "short".into()), ("long".into(), "long".into())]),
        None)
    .add_edge("short", "review")
    .add_edge("long", "review")
    .add_edge("review", END)
    .set_conditional_router("router", FunctionRouter::new(|s: &AgentState| {
        if s.input.len() < 10 { "short" } else { "long" }.to_string()
    }))
    .compile()?
    .with_interrupt_before(vec!["review".to_string()]);  // HITL

// First invoke returns ExecutionInterrupted
// Resume with: compiled.resume(execution).await?
```

## Checkpointer & Subgraphs

```rust
use langchainrust::{ThreadSafeMemoryCheckpointer, SubgraphBuilder, START, END};

// Checkpointer for state persistence
let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
let compiled = graph.compile()?.with_checkpointer(checkpointer);

// Subgraph composition
let subgraph = GraphBuilder::<AgentState>::new()
    .add_node_fn("sub_process", |s| Ok(StateUpdate::full(s.clone())))
    .add_edge(START, "sub_process")
    .add_edge("sub_process", END)
    .compile()?;

let parent = GraphBuilder::<AgentState>::new()
    .add_subgraph_same_state("subworkflow", subgraph)
    .add_edge(START, "subworkflow")
    .add_edge("subworkflow", END)
    .compile()?;
```

## Parallel & Streaming

```rust
// Fan-out / fan-in
graph.add_fan_out("entry", vec!["branch_a".into(), "branch_b".into()]);
graph.add_fan_in(vec!["branch_a".into(), "branch_b".into()], "merge");

// Streaming
let stream = compiled.stream(state);
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::EnterNode(name, state) => { /* ... */ }
        StreamEvent::NodeComplete(name, update) => { /* ... */ }
        StreamEvent::End(final_state) => { /* ... */ }
        _ => {}
    }
}

// Visualization
println!("{}", compiled.visualize_ascii());
println!("{}", compiled.visualize_mermaid());
```
