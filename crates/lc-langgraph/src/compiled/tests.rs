// crates/lc-langgraph/src/compiled/tests.rs
//! Tests for CompiledGraph

#[cfg(test)]
mod tests {
    use crate::graph::{GraphBuilder, END, START};
    use crate::state::{AgentState, StateUpdate};

    #[tokio::test]
    async fn test_simple_linear_graph() {
        let compiled = GraphBuilder::<AgentState>::new()
            .add_node_fn("step1", |state| {
                Ok(StateUpdate::full(AgentState::new(state.input.clone())))
            })
            .add_node_fn("step2", |state| {
                let mut new_state = state.clone();
                new_state.set_output("done".to_string());
                Ok(StateUpdate::full(new_state))
            })
            .add_edge(START, "step1")
            .add_edge("step1", "step2")
            .add_edge("step2", END)
            .compile()
            .unwrap();

        let input = AgentState::new("test input".to_string());
        let result = compiled.invoke(input).await.unwrap();

        assert!(result.final_state.output.is_some());
        assert_eq!(result.recursion_count, 2);
    }

    #[tokio::test]
    async fn test_stream_execution() {
        let compiled = GraphBuilder::<AgentState>::new()
            .add_node_fn("process", |state| Ok(StateUpdate::full(state.clone())))
            .add_edge(START, "process")
            .add_edge("process", END)
            .compile()
            .unwrap();

        let input = AgentState::new("test".to_string());
        let events = compiled.stream(input).await.unwrap();

        assert!(!events.is_empty());
    }
}
