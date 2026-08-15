// crates/lc-langgraph/src/compiled/tests.rs
//! Tests for CompiledGraph

use crate::errors::GraphError;
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
    let events = compiled.stream_collected(input).await.unwrap();

    assert!(!events.is_empty());
}

/// Builds a linear chain of `n` nodes (START -> n1 -> ... -> n{n} -> END).
fn chain_of(n: usize) -> crate::compiled::CompiledGraph<AgentState> {
    // `GraphBuilder` here is the consuming builder (`mut self -> Self`).
    let mut builder = GraphBuilder::<AgentState>::new();
    for i in 1..=n {
        let name = format!("n{}", i);
        builder = builder.add_node_fn(name.clone(), |state| Ok(StateUpdate::full(state.clone())));
        if i == 1 {
            builder = builder.add_edge(START, name.clone());
        }
        if i == n {
            builder = builder.add_edge(name, END);
        }
    }
    // Link consecutive nodes
    for i in 1..n {
        builder = builder.add_edge(format!("n{}", i), format!("n{}", i + 1));
    }
    builder.compile().unwrap()
}

#[tokio::test]
async fn test_recursion_limit_exact_fit_not_misreported() {
    // Q3 off-by-one: a chain using exactly `limit` steps and then reaching END
    // must succeed. The old `count >= limit` post-check fired at count == limit,
    // making the effective max `limit - 1`.
    let compiled = chain_of(3).with_recursion_limit(3);

    let result = compiled
        .invoke(AgentState::new("x".to_string()))
        .await
        .unwrap();
    assert_eq!(result.recursion_count, 3);
}

#[tokio::test]
async fn test_recursion_limit_exceeded_errors() {
    // A chain longer than the limit must error rather than silently truncate.
    let compiled = chain_of(3).with_recursion_limit(2);

    let err = compiled
        .invoke(AgentState::new("x".to_string()))
        .await
        .unwrap_err();
    assert!(matches!(err, GraphError::RecursionLimitReached(2)));
}

#[tokio::test]
async fn test_invoke_from_node_enforces_recursion_limit() {
    // Q3: invoke_from_node previously had no limit check at all — a loop longer
    // than the limit silently truncated. It must now return RecursionLimitReached.
    let compiled = chain_of(3).with_recursion_limit(1);

    let err = compiled
        .invoke_from_node("n1".to_string(), AgentState::new("x".to_string()))
        .await
        .unwrap_err();
    assert!(matches!(err, GraphError::RecursionLimitReached(1)));
}

#[tokio::test]
async fn test_stream_reports_recursion_limit_hit() {
    // Q3: stream() previously emitted a silent `end` when the limit was hit.
    // It must now surface RecursionLimitReached instead.
    let compiled = chain_of(3).with_recursion_limit(1);

    let events = compiled
        .stream_collected(AgentState::new("x".to_string()))
        .await;

    let err = events.unwrap_err();
    assert!(matches!(err, GraphError::RecursionLimitReached(1)));
}
