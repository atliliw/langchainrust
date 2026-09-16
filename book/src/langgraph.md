# LangGraph

LangGraph provides graph-based orchestration for stateful, long-running LLM workflows with conditional routing, subgraphs, parallel execution, dynamic in-node interrupt/resume human-in-the-loop, and checkpoint state history with fork-based time travel.

## Core Concepts

| Concept | Type | Description |
|---------|------|-------------|
| `StateGraph<S>` | Builder | Define nodes, edges, and reducers on state `S` |
| `GraphBuilder<S>` | Builder | Fluent (consuming) variant of `StateGraph` |
| `CompiledGraph<S>` | Runtime | Compiled, executable graph |
| `GraphNode<S>` | Trait | Node execution: `async fn execute(&self, state, config) -> NodeResult<S>` |
| `ConditionalEdge<S>` | Trait | Dynamic routing: `async fn route(&self, state) -> String` |
| `Reducer<S>` | Trait | State merge strategy: `fn reduce(&self, current, update) -> S` |
| `Checkpointer<S>` | Trait | State persistence: `save`, `load`, `list`, `delete`, `snapshots` |
| `SubgraphNode<S, SubS>` | Node | Embed a `CompiledGraph<SubS>` inside a parent graph |
| `InterruptibleNode<S>` | Node | Resume-aware node: the closure receives `resume: Option<&Value>` (`None` on first pass, `Some(decision)` after a human answers) |
| `CheckpointInfo<S>` | Record | One history entry: `id`, `timestamp`, `seq`, `recursion_count`, `state` |
| `StreamEvent<S>` | Enum | `Start`, `EnterNode`, `NodeComplete`, `StateUpdate`, `NodeInterrupt`, `Resumed`, `End` |

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
// Resume with: compiled.create_resume_execution("review") ... compiled.resume(execution).await?
```

The static list fixes stopping points *before launch*. When the decision depends on
runtime data ("this action would spend money — ask a human"), use a dynamic in-node
interrupt instead.

## Dynamic interrupt() + resume (human-in-the-loop)

`InterruptibleNode` wraps a closure that is called **twice** per interrupt. On the
first pass `resume` is `None`; the node does its pre-side-effect work and suspends
itself by returning `GraphError::InterruptRequest { payload }`. After a human
answers, `resume_with_value` re-enters the same node with the answer and the closure
runs again with `resume = Some(decision)` — branch on it so the side effect that was
approved happens once, on the resumed pass only.

```rust
use langchainrust::{
    GraphBuilder, InterruptibleNode, AgentState, StateUpdate,
    GraphError, GraphNode, START, END, ThreadSafeMemoryCheckpointer,
};
use serde_json::json;

let approve = InterruptibleNode::new("charge", |state, resume| {
    let output = state.output.clone().unwrap_or_default();
    Box::pin(async move {
        match resume {
            // First pass: suspend, hand the approval context to the outside world.
            None => Err(GraphError::InterruptRequest {
                payload: json!({ "kind": "tool_approval", "command": output }),
            }),
            // Resumed with the human's decision: perform the side effect once.
            Some(decision) => {
                let mut next = state.clone();
                next.set_output(format!("decision={decision}, charged"));
                Ok(StateUpdate::full(next))
            }
        }
    })
});

let compiled = GraphBuilder::<AgentState>::new()
    .add_node(approve)                       // InterruptibleNode implements GraphNode
    .add_edge(START, "charge")
    .add_edge("charge", END)
    .compile()?
    // A checkpointer is mandatory: the interrupt payload is persisted into the
    // checkpoint, so even a process restart can resume instead of replaying.
    .with_checkpointer(ThreadSafeMemoryCheckpointer::new());

// First run suspends. The error carries the node name + payload.
let err = compiled.invoke(AgentState::new("charge $99")).await.unwrap_err();
// GraphError::DynamicInterrupt { node, payload } surfaces the suspended node.

// A human answers; the SAME node is re-entered with the value. Nothing before the
// interrupt is replayed, and the recursion budget carries over.
let invocation = compiled
    .resume_with_value("charge", json!({ "approved": true }))
    .await?;
println!("{}", invocation.final_state.output.as_deref().unwrap_or(""));
```

Key semantics:

- **One persistence system.** The interrupt payload lives in the graph checkpoint
  (memory / file / SQLite / Postgres / Redis). A brand-new process over the same
  checkpoint directory can call `resume_with_value`.
- **Side effects run once.** Keep any first-pass work that must not repeat inside
  the `None` branch; the `Some(decision)` branch is the only place effects fire.
- **Cascading approvals.** Resuming one node can immediately suspend at the next
  interrupt — each resume returns the next `DynamicInterrupt` until the graph
  reaches `END`.
- **Streaming.** A suspended node emits `StreamEvent::NodeInterrupt(name, payload)`
  and the resumed pass emits `StreamEvent::Resumed(name, decision)`; pre-interrupt
  events are not replayed.

When agents run on the graph, `lc_agents::graph_approval::ApprovalGate` builds
exactly this pattern: an agent tool approval is an interrupt payload, and
`ApprovalDecision` (`Allow` / `Deny { reason }` / `Modify { arguments, note }`) is
the resume value — Deny skips the tool with no side effect.

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

## State History & Time Travel

Every checkpointed step is also a *snapshot*. `get_state_history` lists them, oldest
first, and `fork_from` branches the run off any one of them.

```rust
use langchainrust::CheckpointInfo;

// Run a graph with a checkpointer, then inspect where it has been.
let history: Vec<CheckpointInfo<AgentState>> = compiled.get_state_history()?;
for cp in &history {
    println!("seq={} ts={} recursion={} id={}", cp.seq, cp.timestamp, cp.recursion_count, cp.id);
    let state_at_that_point: &AgentState = &cp.state;
}

// Replay a "what if": branch off an old snapshot and re-run forward from a
// chosen node. Pass Some(state) to edit history (e.g. fix a bad tool output);
// None replays the snapshot's state as-is.
let forked = compiled
    .fork_from(&history[2].id, "analyze", Some(corrected_state))
    .await?;
println!("forked answer: {}", forked.final_state.output.as_deref().unwrap_or(""));
```

The fork is a **new checkpoint lineage** — it never mutates the original run, so you
can keep the production timeline and explore alternatives side by side. Execution is
forward-only from the chosen node: nodes *before* the fork point do not run again
(their side effects are assumed already applied). If you need a true re-execution
including those effects, fork from an earlier snapshot. Forks compose with interrupts:
if the branched path suspends, resume it with `resume_with_value` exactly like any
other run.

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
