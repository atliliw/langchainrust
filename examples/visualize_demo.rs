use langchainrust::{AgentState, GraphBuilder, StateUpdate, END, START};

fn main() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("step1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step2", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step3", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile()
        .unwrap();

    println!("=== ASCII 可视化 ===");
    println!("{}", compiled.visualize_ascii());

    println!("=== Mermaid 可视化 ===");
    println!("{}", compiled.visualize_mermaid());

    println!("=== JSON 可视化 ===");
    println!(
        "{}",
        serde_json::to_string_pretty(&compiled.visualize_json()).unwrap()
    );
}
