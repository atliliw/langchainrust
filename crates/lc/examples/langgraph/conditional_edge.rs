//! Conditional routing example
//!
//! Shows a LangGraph conditional edge: route to the short / long node by input length
//! (no API key required).
//!
//! # Run
//! ```bash
//! cargo run --example langgraph_conditional_edge
//! ```

use langchainrust::{AgentState, FunctionRouter, StateGraph, StateUpdate, END, START};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("entry", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("short", |state| {
        let mut s = state.clone();
        s.set_output("short path".to_string());
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("long", |state| {
        let mut s = state.clone();
        s.set_output("long path".to_string());
        Ok(StateUpdate::full(s))
    });

    graph.add_edge(START, "entry");
    let targets = HashMap::from([
        ("short".to_string(), "short".to_string()),
        ("long".to_string(), "long".to_string()),
    ]);
    graph.add_conditional_edges("entry", "router", targets, None);
    graph.add_edge("short", END);
    graph.add_edge("long", END);

    let router = FunctionRouter::new(|state: &AgentState| {
        if state.input.len() < 10 {
            "short"
        } else {
            "long"
        }
        .to_string()
    });
    graph.set_conditional_router("router", router);

    let compiled = graph.compile()?;

    for input in ["hi", "this is a very long input"] {
        let result = compiled.invoke(AgentState::new(input.to_string())).await?;
        println!("input {:?} -> {:?}", input, result.final_state.output);
    }
    Ok(())
}
