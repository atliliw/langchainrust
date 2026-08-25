// crates/lc-langgraph/src/compiled/tests.rs
//! Tests for CompiledGraph

use crate::checkpointer::ThreadSafeMemoryCheckpointer;
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

#[tokio::test]
async fn test_resume_preserves_recursion_budget() {
    // M6: 中断后 resume 必须沿用中断前已消耗的递归预算,而不是清零重来。
    // 否则反复 interrupt→resume 可以无限绕过 recursion_limit。
    let compiled = chain_of(4)
        .with_recursion_limit(3)
        .with_interrupt_before(vec!["n3".to_string()])
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

    // 第一次运行:执行 n1、n2 后在 n3 前中断。
    let err = compiled
        .invoke(AgentState::new("x".to_string()))
        .await
        .unwrap_err();
    assert!(matches!(err, GraphError::ExecutionInterrupted(ref node) if node == "n3"));

    // 中断时已执行 2 步 → 最近 checkpoint 记录的递归预算应为 2。
    let execution = compiled.create_resume_execution("n3").await.expect(
        "should be able to build a resume execution from the latest checkpoint after interruption",
    );
    assert_eq!(
        execution.recursion_count, 2,
        "M6: resume must carry over the already-consumed recursion budget"
    );

    // limit=3 且已消耗 2 → 续跑执行 n3 后即触顶,报 RecursionLimitReached。
    // 若预算被错误清零,续跑会跑完 n3、n4 并"成功"——正是 M6 要堵住的洞。
    let err = compiled.resume(execution).await.unwrap_err();
    assert!(matches!(err, GraphError::RecursionLimitReached(3)));

    // 预算充足时,同样的中断→续跑应当完整跑完剩余节点。
    let compiled_ok = chain_of(4)
        .with_recursion_limit(10)
        .with_interrupt_before(vec!["n3".to_string()])
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());
    let _ = compiled_ok
        .invoke(AgentState::new("x".to_string()))
        .await
        .unwrap_err();
    let execution = compiled_ok
        .create_resume_execution("n3")
        .await
        .expect("build resume execution context");
    assert_eq!(execution.recursion_count, 2);
    let result = compiled_ok.resume(execution).await.unwrap();
    assert_eq!(
        result.recursion_count, 4,
        "resume should fully execute n3, n4"
    );
}
