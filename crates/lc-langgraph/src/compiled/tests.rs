// crates/lc-langgraph/src/compiled/tests.rs
//! Tests for CompiledGraph

use crate::checkpointer::ThreadSafeMemoryCheckpointer;
use crate::compiled::types::{DynamicInjection, DynamicPlanner, DynamicTask};
use crate::errors::GraphError;
use crate::graph::{GraphBuilder, END, START};
use crate::node::NodeResult;
use crate::state::{AgentState, StateUpdate};
use async_trait::async_trait;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

/// A no-op planner so `invoke_dynamic` can run over a purely static fan-out graph.
struct NoopPlanner;

#[async_trait]
impl DynamicPlanner<AgentState> for NoopPlanner {
    async fn plan(
        &self,
        _tasks: &[DynamicTask],
        _state: &AgentState,
    ) -> Result<DynamicInjection<AgentState>, String> {
        Ok(DynamicInjection {
            nodes: Vec::new(),
            edges: Vec::new(),
        })
    }
}

#[tokio::test]
async fn test_invoke_dynamic_runs_all_fanout_branches() {
    // A8 regression: `invoke_dynamic` used to route through `find_next_node`, which
    // returns only `targets[0]` for a FanOut edge — so b2 (and any state/effect it
    // produced) was silently dropped. Now it must run every branch and merge.
    let ran = Arc::new(AtomicUsize::new(0));

    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("main", |_| {
            Ok(StateUpdate::full(AgentState::new("x".to_string())))
        })
        .add_node_fn("b1", {
            let ran = ran.clone();
            move |_| {
                ran.fetch_add(1, Ordering::SeqCst);
                Ok(StateUpdate::unchanged())
            }
        })
        .add_node_fn("b2", {
            let ran = ran.clone();
            move |_| {
                ran.fetch_add(1, Ordering::SeqCst);
                Ok(StateUpdate::unchanged())
            }
        })
        .add_node_fn("join", |state| {
            let mut s = state.clone();
            s.set_output("done".to_string());
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "main")
        .add_fan_out("main", vec!["b1".to_string(), "b2".to_string()])
        .add_fan_in(vec!["b1".to_string(), "b2".to_string()], "join")
        .add_edge("join", END)
        .compile()
        .unwrap();

    let result = compiled
        .invoke_dynamic(AgentState::new("input".to_string()), &NoopPlanner)
        .await
        .unwrap();

    assert_eq!(
        ran.load(Ordering::SeqCst),
        2,
        "both fan-out branches must have executed, not just targets[0]"
    );
    assert_eq!(result.final_state.output.as_deref(), Some("done"));
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
    // M6: resume after an interrupt must carry over the already-consumed recursion budget,
    // not restart from zero. Otherwise repeated interrupt→resume can bypass recursion_limit.
    let compiled = chain_of(4)
        .with_recursion_limit(3)
        .with_interrupt_before(vec!["n3".to_string()])
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

    // First run: executes n1, n2, then interrupts before n3.
    let err = compiled
        .invoke(AgentState::new("x".to_string()))
        .await
        .unwrap_err();
    assert!(matches!(err, GraphError::ExecutionInterrupted(ref node) if node == "n3"));

    // 2 steps were consumed at interruption → the latest checkpoint records budget 2.
    let execution = compiled.create_resume_execution("n3", None).await.expect(
        "should be able to build a resume execution from the latest checkpoint after interruption",
    );
    assert_eq!(
        execution.recursion_count, 2,
        "M6: resume must carry over the already-consumed recursion budget"
    );

    // limit=3 with 2 already consumed → resume hits the cap right after n3, so
    // RecursionLimitReached is reported. If the budget were wrongly zeroed, the resume
    // would run n3, n4 and "succeed" — exactly the hole M6 closes.
    let err = compiled.resume(execution).await.unwrap_err();
    assert!(matches!(err, GraphError::RecursionLimitReached(3)));

    // With a sufficient budget, the same interrupt→resume should run all remaining nodes.
    let compiled_ok = chain_of(4)
        .with_recursion_limit(10)
        .with_interrupt_before(vec!["n3".to_string()])
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());
    let _ = compiled_ok
        .invoke(AgentState::new("x".to_string()))
        .await
        .unwrap_err();
    let execution = compiled_ok
        .create_resume_execution("n3", None)
        .await
        .expect("build resume execution context");
    assert_eq!(execution.recursion_count, 2);
    let result = compiled_ok.resume(execution).await.unwrap();
    assert_eq!(
        result.recursion_count, 4,
        "resume should fully execute n3, n4"
    );
}

/// v0.25.0 #1: a node suspends itself mid-execution with a runtime interrupt; a
/// human's decision is fed back so the SAME node re-enters and continues (no
/// re-execution of the interrupt path). State + recursion budget survive via
/// the checkpointer, so `resume_with_value` restores the run from disk.
#[tokio::test]
async fn test_dynamic_interrupt_and_resume_with_value() {
    use crate::node::InterruptibleNode;
    use std::pin::Pin;

    // `side_effects` counts how many times the pre-suspension work runs. On the
    // resume pass `resume` is `Some(_)`, so the closure skips re-asking and
    // finalizes with the decision — proving the interrupt is not re-triggered.
    let side_effects = Arc::new(AtomicUsize::new(0));
    let seen_effects = side_effects.clone();

    let compiled = GraphBuilder::<AgentState>::new()
        .add_node(InterruptibleNode::new("ask", move |_state, resume| {
            // Clone the effect counter into the future: a `Fn` closure may be
            // called more than once, so it cannot own the `Arc` across the
            // `async move` block.
            let effects = seen_effects.clone();
            // Own the resume value (the future must be `'static`, so it cannot
            // borrow the closure's `&Option<&Value>` argument).
            let resume_owned = resume.cloned();
            Box::pin(async move {
                if let Some(decision) = resume_owned.as_ref() {
                    // Human answered — continue past the suspension.
                    let mut new_state = AgentState::new("resumed");
                    new_state.set_output(format!("approved={}", decision));
                    Ok(StateUpdate::full(new_state))
                } else {
                    // First pass — suspend and ask the human.
                    effects.fetch_add(1, Ordering::SeqCst);
                    Err(GraphError::InterruptRequest {
                        payload: serde_json::json!({
                            "question": "proceed?",
                            "kind": "human_approval",
                        }),
                    })
                }
            }) as Pin<Box<dyn Future<Output = NodeResult<AgentState>> + Send>>
        }))
        .add_node_fn("finish", |state| {
            let mut new_state = state.clone();
            new_state.set_output(format!(
                "final:{}",
                state.output.clone().unwrap_or_default()
            ));
            Ok(StateUpdate::full(new_state))
        })
        .add_edge(START, "ask")
        .add_edge("ask", "finish")
        .add_edge("finish", END)
        .compile()
        .unwrap()
        .with_recursion_limit(10)
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

    // First invoke suspends at "ask".
    let err = compiled
        .invoke(AgentState::new("x".to_string()))
        .await
        .unwrap_err();
    let (node, payload) = match err {
        GraphError::DynamicInterrupt { node, payload } => (node, payload),
        other => panic!("expected DynamicInterrupt, got {other:?}"),
    };
    assert_eq!(node, "ask");
    assert_eq!(payload["kind"], "human_approval");
    // The interrupt-pass side effect ran exactly once.
    assert_eq!(side_effects.load(Ordering::SeqCst), 1);

    // Resume with the human's approval; the interrupted node re-enters with it.
    let inv = compiled
        .resume_with_value(&node, serde_json::json!({ "approved": true }))
        .await
        .unwrap();
    let out = inv.final_state.output.as_deref().unwrap();
    assert!(out.starts_with("final:approved="), "got {out}");
    // The interrupt path must NOT have run again on resume.
    assert_eq!(side_effects.load(Ordering::SeqCst), 1);
}

/// H7: `resume_from_interrupt` fixes the resume to the checkpoint stamped in the
/// payload (`__checkpoint_id`), NOT `last()`. A second invoke writes NEWER
/// checkpoints whose state drifts from the interrupted run; resuming with the
/// FIRST payload must re-enter from the first run's state ("in=x"), proving the
/// stamped checkpoint (not `last()`, which now points at the second run's "in=y")
/// drives the resume.
#[tokio::test]
async fn test_resume_from_interrupt_prefers_stamped_checkpoint_over_last() {
    use crate::node::InterruptibleNode;
    use std::pin::Pin;

    let compiled = GraphBuilder::<AgentState>::new()
        .add_node(InterruptibleNode::new(
            "ask",
            |state: &AgentState, resume| {
                let r = resume.cloned();
                let input = state.input.clone();
                Box::pin(async move {
                    if let Some(d) = r.as_ref() {
                        let mut ns = AgentState::new(format!("resumed:{input}"));
                        ns.set_output(format!("approved={d};in={input}"));
                        Ok(StateUpdate::full(ns))
                    } else {
                        Err(GraphError::InterruptRequest {
                            payload: serde_json::json!({ "kind": "human" }),
                        })
                    }
                }) as Pin<Box<dyn Future<Output = NodeResult<AgentState>> + Send>>
            },
        ))
        .add_node_fn("finish", |s| {
            let mut ns = s.clone();
            ns.set_output(format!("final:{}", s.output.clone().unwrap_or_default()));
            Ok(StateUpdate::full(ns))
        })
        .add_edge(START, "ask")
        .add_edge("ask", "finish")
        .add_edge("finish", END)
        .compile()
        .unwrap()
        .with_recursion_limit(10)
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

    // First run suspends at "ask" with input "x".
    let err = compiled.invoke(AgentState::new("x")).await.unwrap_err();
    let (node, payload1) = match err {
        GraphError::DynamicInterrupt { node, payload } => (node, payload),
        other => panic!("expected DynamicInterrupt, got {other:?}"),
    };
    assert_eq!(node, "ask");
    let id1 = payload1["__checkpoint_id"]
        .as_str()
        .expect("payload must stamp __checkpoint_id")
        .to_string();

    // A second run writes NEWER checkpoints (input "y"), so `last()` no longer
    // points at the first run's interrupt save.
    let err2 = compiled.invoke(AgentState::new("y")).await.unwrap_err();
    let (_, payload2) = match err2 {
        GraphError::DynamicInterrupt { node: _, payload } => ((), payload),
        other => panic!("expected DynamicInterrupt, got {other:?}"),
    };
    let id2 = payload2["__checkpoint_id"]
        .as_str()
        .expect("second payload must stamp __checkpoint_id")
        .to_string();
    assert_ne!(
        id1, id2,
        "second invoke must write a distinct (newer) checkpoint than the first"
    );

    // Resuming with the FIRST payload must resolve the FIRST checkpoint
    // (state.input == "x"), not `last()` (which now holds the "y" run's state).
    let inv = compiled
        .resume_from_interrupt(&node, &payload1, serde_json::json!("yes"))
        .await
        .unwrap();
    let out = inv.final_state.output.as_deref().unwrap();
    assert!(out.contains("in=x"), "expected first-run state, got {out}");
    assert!(
        !out.contains("resumed:y"),
        "must not use last() state, got {out}"
    );
}

/// H8: the successor of an `after_` interrupt gets its `interrupt_before`
/// re-checked (matching `invoke`), even though it is the *first* node of the
/// resume run. Previously the `first_node` exemption suppressed the before-check
/// on the successor, so the two execution entries behaved differently.
#[tokio::test]
async fn test_after_interrupt_resume_rechecks_interrupt_before_on_successor() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("a", |s| {
            let mut ns = s.clone();
            ns.set_output("ran_a".to_string());
            Ok(StateUpdate::full(ns))
        })
        .add_node_fn("b", |s| {
            let mut ns = s.clone();
            ns.set_output("ran_b".to_string());
            Ok(StateUpdate::full(ns))
        })
        .add_node_fn("c", |s| {
            let mut ns = s.clone();
            ns.set_output("ran_c".to_string());
            Ok(StateUpdate::full(ns))
        })
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("c", END)
        .compile()
        .unwrap()
        .with_recursion_limit(10)
        .with_interrupt_after(vec!["a".to_string()])
        .with_interrupt_before(vec!["b".to_string()])
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

    // First invoke runs `a`, then suspends `after_a`.
    let err = compiled.invoke(AgentState::new("x")).await.unwrap_err();
    assert!(
        matches!(&err, GraphError::ExecutionInterrupted(n) if n.as_str() == "after_a"),
        "expected after_a interrupt, got {err:?}"
    );

    // Resuming the after_ interrupt advances to the successor `b` — a node that
    // has never run, so its `interrupt_before` must fire.
    let execution = compiled
        .create_resume_execution("after_a", None)
        .await
        .expect("resume context");
    let err = compiled.invoke_with_execution(execution).await.unwrap_err();
    assert!(
        matches!(&err, GraphError::ExecutionInterrupted(n) if n.as_str() == "b"),
        "successor b must be before-interrupted on resume, got {err:?}"
    );
}

/// v0.25.0 #1 (step 5, durable): a runtime interrupt persisted via the
/// file-backed checkpointer survives a full process/GOTRESS restart — a brand-new
/// graph over the same checkpoint directory (no in-memory carryover) resumes the
/// interrupted node from disk with the human's decision.
#[tokio::test]
async fn test_dynamic_interrupt_survives_restart_via_file_checkpointer() {
    use crate::checkpointer::FileCheckpointer;
    use crate::node::InterruptibleNode;
    use std::pin::Pin;

    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_path_buf();

    let build_into_file = |path: std::path::PathBuf| {
        GraphBuilder::<AgentState>::new()
            .add_node(InterruptibleNode::new("ask", |_state, resume| {
                let r = resume.cloned();
                Box::pin(async move {
                    if let Some(d) = r.as_ref() {
                        let mut ns = AgentState::new("resumed");
                        ns.set_output(format!("approved={}", d));
                        Ok(StateUpdate::full(ns))
                    } else {
                        Err(GraphError::InterruptRequest {
                            payload: serde_json::json!({ "q": "go?" }),
                        })
                    }
                }) as Pin<Box<dyn Future<Output = NodeResult<AgentState>> + Send>>
            }))
            .add_node_fn("finish", |s| {
                let mut ns = s.clone();
                ns.set_output(format!("final:{}", s.output.clone().unwrap_or_default()));
                Ok(StateUpdate::full(ns))
            })
            .add_edge(START, "ask")
            .add_edge("ask", "finish")
            .add_edge("finish", END)
            .compile()
            .unwrap()
            .with_recursion_limit(10)
            .with_checkpointer(FileCheckpointer::new(path).unwrap())
    };

    // First "process": interrupt and drop the graph (simulating a crash before resume).
    {
        let compiled = build_into_file(dir_path.clone());
        let err = compiled.invoke(AgentState::new("x")).await.unwrap_err();
        match err {
            GraphError::DynamicInterrupt { node, .. } => assert_eq!(node, "ask"),
            other => panic!("expected DynamicInterrupt, got {other:?}"),
        }
    }

    // Second "process": brand-new graph over the same checkpoint directory.
    let compiled2 = build_into_file(dir_path);
    let inv = compiled2
        .resume_with_value("ask", serde_json::json!("ok"))
        .await
        .unwrap();
    let out = inv.final_state.output.as_deref().unwrap();
    assert!(out.starts_with("final:approved="), "got {out}");
}

// ── #3: state history / fork / time-travel ──

/// Linear n1 -> n2 -> n3 graph where each node bumps its own side-effect
/// counter and stamps its name into the output. The counters expose whether a
/// node actually re-ran after a fork (the time-travel side-effect invariant).
fn marking_chain(
    c1: Arc<AtomicUsize>,
    c2: Arc<AtomicUsize>,
    c3: Arc<AtomicUsize>,
) -> crate::compiled::CompiledGraph<AgentState> {
    GraphBuilder::<AgentState>::new()
        .add_node_fn("n1", move |state| {
            c1.fetch_add(1, Ordering::SeqCst);
            let mut s = state.clone();
            s.set_output("n1");
            Ok(StateUpdate::full(s))
        })
        .add_node_fn("n2", move |state| {
            c2.fetch_add(1, Ordering::SeqCst);
            let mut s = state.clone();
            s.set_output("n2");
            Ok(StateUpdate::full(s))
        })
        .add_node_fn("n3", move |state| {
            c3.fetch_add(1, Ordering::SeqCst);
            let mut s = state.clone();
            s.set_output("n3");
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "n1")
        .add_edge("n1", "n2")
        .add_edge("n2", "n3")
        .add_edge("n3", END)
        .compile()
        .unwrap()
        .with_recursion_limit(10)
}

#[tokio::test]
async fn test_state_history_lists_every_step_oldest_first() {
    let z = || Arc::new(AtomicUsize::new(0));
    let compiled = marking_chain(z(), z(), z())
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());
    compiled.invoke(AgentState::new("x")).await.unwrap();

    let history = compiled.get_state_history().await.unwrap();
    // pre-run snapshot + one after each of n1/n2/n3.
    assert_eq!(history.len(), 4);
    // Recursion budget grows monotonically; the oldest snapshot is pre-run.
    assert_eq!(
        history
            .iter()
            .map(|s| s.recursion_count)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(history[0].recursion_count, 0);
    assert!(history[0].state.output.is_none());
    // The snapshot after n1 captures n1's state (the "why did it decide that" evidence).
    assert_eq!(history[1].state.output.as_deref(), Some("n1"));
    assert_eq!(history[3].state.output.as_deref(), Some("n3"));
    // Ordering keys are non-decreasing (seq breaks same-second ties).
    for w in history.windows(2) {
        assert!(
            (w[0].timestamp, w[0].seq) <= (w[1].timestamp, w[1].seq),
            "history must be ordered by (timestamp, seq)"
        );
    }
}

#[tokio::test]
async fn test_fork_from_snapshot_runs_forward_without_replaying_side_effects() {
    let (c1, c2, c3) = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let compiled = marking_chain(c1.clone(), c2.clone(), c3.clone())
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());
    compiled.invoke(AgentState::new("x")).await.unwrap();
    for c in [&c1, &c2, &c3] {
        assert_eq!(c.load(Ordering::SeqCst), 1);
    }

    // Fork from the snapshot captured right after n1, resuming at n2.
    let after_n1 = compiled.get_state_history().await.unwrap()[1].id.clone();
    let fork = compiled.fork_from(&after_n1, "n2", None).await.unwrap();

    // Forward run reached n3, continuing the recursion budget (started at 1).
    assert_eq!(fork.final_state.output.as_deref(), Some("n3"));
    assert_eq!(fork.recursion_count, 3);
    // The pre-fork node n1 must NOT re-run; n2/n3 re-run on the fork timeline.
    assert_eq!(
        c1.load(Ordering::SeqCst),
        1,
        "n1 side effect must not replay"
    );
    assert_eq!(c2.load(Ordering::SeqCst), 2, "n2 runs on the fork");
    assert_eq!(c3.load(Ordering::SeqCst), 2, "n3 runs on the fork");

    // A fork seeds ONE new checkpoint; it never rewrites the original lineage.
    assert_eq!(
        compiled.get_state_history().await.unwrap().len(),
        5,
        "fork appends a lineage seed, does not rewrite history"
    );
}

#[tokio::test]
async fn test_fork_from_snapshot_with_overridden_input() {
    // "回到任意旧快照、改输入、从那条线分叉重跑": the echo node surfaces the
    // input the forked timeline actually ran with.
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("n1", |state| {
            let mut s = state.clone();
            s.set_output("n1");
            Ok(StateUpdate::full(s))
        })
        .add_node_fn("echo", |state| {
            let mut s = state.clone();
            s.set_output(format!("echo:{}", state.input));
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "n1")
        .add_edge("n1", "echo")
        .add_edge("echo", END)
        .compile()
        .unwrap()
        .with_recursion_limit(10)
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());
    compiled.invoke(AgentState::new("original")).await.unwrap();

    let after_n1 = compiled.get_state_history().await.unwrap()[1].id.clone();
    let fork = compiled
        .fork_from(&after_n1, "echo", Some(AgentState::new("rewritten")))
        .await
        .unwrap();
    assert_eq!(
        fork.final_state.output.as_deref(),
        Some("echo:rewritten"),
        "the forked timeline must run with the overridden input, not the original"
    );
}

#[tokio::test]
async fn test_state_history_and_fork_require_checkpointer() {
    let compiled = chain_of(2);
    assert!(compiled.get_state_history().await.is_err());
    assert!(compiled.fork_from("anything", "n1", None).await.is_err());
}

/// B4: a time-travel fork seeds its timeline from a snapshot, and that seed —
/// the newest checkpoint in history — is stamped with the `parent_id` of the
/// snapshot it branched from, while the original lineage's checkpoints remain
/// parent-free.
#[tokio::test]
async fn test_fork_seed_records_parent_lineage() {
    let compiled = marking_chain(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    )
    .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());
    compiled.invoke(AgentState::new("x")).await.unwrap();

    let history = compiled.get_state_history().await.unwrap();
    let after_n1 = history[1].id.clone();

    compiled.fork_from(&after_n1, "n2", None).await.unwrap();

    let history = compiled.get_state_history().await.unwrap();
    assert_eq!(history.len(), 5, "fork appends one lineage seed");
    let seed = history.last().unwrap();
    assert_ne!(seed.id, after_n1);
    assert_eq!(
        seed.parent.as_deref(),
        Some(after_n1.as_str()),
        "fork seed must record the snapshot it branched from"
    );
    // The original lineage's snapshots (and the seed's own save) are roots.
    for s in history.iter().take(history.len() - 1) {
        assert_eq!(s.parent, None, "original lineage must be parent-free");
    }
}

/// B4: a runtime interrupt stamps the exact checkpoint to resume from in the
/// payload (`__checkpoint_id`), and `resume_from_checkpoint` resumes that
/// specific checkpoint by id inside its thread (instead of "last checkpoint").
#[tokio::test]
async fn test_resume_from_checkpoint_by_id_and_thread() {
    use crate::checkpointer::DEFAULT_THREAD;
    use crate::node::InterruptibleNode;
    use std::pin::Pin;

    let compiled = GraphBuilder::<AgentState>::new()
        .add_node(InterruptibleNode::new("ask", |_state, resume| {
            let resume_owned = resume.cloned();
            Box::pin(async move {
                if let Some(decision) = resume_owned.as_ref() {
                    let mut s = AgentState::new("resume");
                    s.set_output(format!("decision={}", decision));
                    Ok(StateUpdate::full(s))
                } else {
                    Err(GraphError::InterruptRequest {
                        payload: serde_json::json!({ "kind": "human_approval" }),
                    })
                }
            }) as Pin<Box<dyn Future<Output = NodeResult<AgentState>> + Send>>
        }))
        .add_node_fn("finish", |state| {
            let mut s = state.clone();
            s.set_output(format!(
                "final:{}",
                state.output.clone().unwrap_or_default()
            ));
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "ask")
        .add_edge("ask", "finish")
        .add_edge("finish", END)
        .compile()
        .unwrap()
        .with_recursion_limit(10)
        .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

    let err = compiled.invoke(AgentState::new("x")).await.unwrap_err();
    let (node, payload) = match err {
        GraphError::DynamicInterrupt { node, payload } => (node, payload),
        other => panic!("expected DynamicInterrupt, got {other:?}"),
    };
    assert_eq!(node, "ask");
    assert_eq!(payload["kind"], "human_approval");
    let checkpoint_id = payload["__checkpoint_id"]
        .as_str()
        .expect("interrupt payload must stamp the checkpoint id")
        .to_string();

    let inv = compiled
        .resume_from_checkpoint(
            DEFAULT_THREAD,
            &checkpoint_id,
            &node,
            serde_json::json!("yes"),
        )
        .await
        .unwrap();
    assert_eq!(
        inv.final_state.output.as_deref(),
        Some("final:decision=\"yes\"")
    );
}

/// B4: two field reducers registered via `set_reducer` compose deterministically
/// — each patches only its own field, so over a multi-node run BOTH accumulating
/// fields survive (neither is clobbered by the other reducer running "last"),
/// independent of registration order.
#[tokio::test]
async fn test_two_field_reducers_patch_independently() {
    use crate::state::{AppendMessagesReducer, AppendStepsReducer, MessageEntry, StepEntry};

    let mut graph = crate::graph::StateGraph::<AgentState>::new();
    // Register "steps" first to prove registration order is not load-bearing.
    graph.set_reducer("steps", Arc::new(AppendStepsReducer));
    graph.set_reducer("messages", Arc::new(AppendMessagesReducer));

    // Each node returns ONLY its own new message + step (not the full history),
    // the shape the append reducers expect.
    graph.add_node_fn("n1", |_state| {
        let mut s = AgentState::new("n1");
        s.messages = vec![MessageEntry::ai("m1".to_string())];
        s.steps = vec![StepEntry::new("act1", "obs1")];
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("n2", |_state| {
        let mut s = AgentState::new("n2");
        s.messages = vec![MessageEntry::ai("m2".to_string())];
        s.steps = vec![StepEntry::new("act2", "obs2")];
        Ok(StateUpdate::full(s))
    });
    graph.add_edge(START, "n1");
    graph.add_edge("n1", "n2");
    graph.add_edge("n2", END);

    let compiled = graph.compile().unwrap().with_recursion_limit(10);
    let final_state = compiled
        .invoke(AgentState::new("in"))
        .await
        .unwrap()
        .final_state;

    // 1 initial human message + m1 + m2, and step1 + step2 — proving the two
    // field reducers composed without one dropping the other's contribution.
    assert_eq!(
        final_state.messages.len(),
        3,
        "got {:?}",
        final_state.messages
    );
    assert_eq!(final_state.steps.len(), 2, "got {:?}", final_state.steps);
    assert_eq!(
        final_state
            .messages
            .iter()
            .filter(|m| m.role == crate::state::MessageRole::AI)
            .count(),
        2,
        "both AI messages must be accumulated"
    );
    assert_eq!(
        final_state
            .steps
            .iter()
            .map(|s| s.action.as_str())
            .collect::<Vec<_>>(),
        vec!["act1", "act2"]
    );
}
