//! Basic graph workflow example
//!
//! Shows a linear LangGraph: START -> greet -> reply -> END (no API key required).
//!
//! # Run
//! ```bash
//! cargo run --example langgraph_basic_graph
//! ```

use langchainrust::{AgentState, GraphBuilder, StateUpdate, END, START};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("greet", |state: &AgentState| {
            Ok(StateUpdate::full(AgentState::new(format!(
                "Hello:{}",
                state.input
            ))))
        })
        .add_node_fn("reply", |state: &AgentState| {
            let mut s = state.clone();
            s.set_output(format!("Reply:{}", state.input));
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "greet")
        .add_edge("greet", "reply")
        .add_edge("reply", END)
        .compile()?;

    let result = compiled
        .invoke(AgentState::new("world".to_string()))
        .await?;
    println!("input:  {}", result.final_state.input);
    println!("output: {:?}", result.final_state.output);
    Ok(())
}
