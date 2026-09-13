//! B2 gate (0.22.4): durable interrupt/resume, cross-process state
//! consistency, time travel, and storage-side OCC — all through the SQLite
//! checkpointer against a real database file.
//!
//! Compiled/run only with the `checkpoint-sqlite` feature:
//!
//! ```text
//! cargo test -p lc-langgraph --features checkpoint-sqlite --test checkpoint_durable_resume
//! ```
#![cfg(feature = "checkpoint-sqlite")]

use lc_langgraph::Checkpointer;
use lc_langgraph::{
    AgentState, CompiledGraph, GraphBuilder, GraphError, SqliteCheckpointer, StateUpdate, END,
    START,
};

const THREAD: &str = "durable-resume-thread";

/// Four identity-ish nodes that each stamp `output` with their own name, so a
/// checkpointed state reveals exactly how far the run had progressed.
fn stamped_chain() -> CompiledGraph<AgentState> {
    let mut builder = GraphBuilder::<AgentState>::new();
    for i in 1..=4usize {
        let name = format!("n{i}");
        builder = builder.add_node_fn(name.clone(), move |state: &AgentState| {
            let mut next = state.clone();
            next.set_output(format!("n{i}"));
            Ok(StateUpdate::full(next))
        });
        if i == 1 {
            builder = builder.add_edge(START, name.clone());
        }
        if i == 4 {
            builder = builder.add_edge(name, END);
        }
    }
    for i in 1..4usize {
        builder = builder.add_edge(format!("n{i}"), format!("n{}", i + 1));
    }
    builder.compile().expect("chain compiles")
}

#[tokio::test]
async fn durable_interrupt_resume_time_travel_and_occ() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("graph.db");

    // --- "Process 1": run up to the interrupt before n3, then exit. ---------
    let cp_process1 = SqliteCheckpointer::<AgentState>::new(&db_path, THREAD).expect("open sqlite");
    let compiled = stamped_chain()
        .with_recursion_limit(10)
        .with_interrupt_before(vec!["n3".to_string()])
        .with_checkpointer(cp_process1);

    let err = compiled
        .invoke(AgentState::new("payload".to_string()))
        .await
        .unwrap_err();
    assert!(
        matches!(err, GraphError::ExecutionInterrupted(ref node) if node == "n3"),
        "run must interrupt before n3, got {err:?}"
    );
    drop(compiled);

    // Independent connections over the same file observe the durable history:
    // one checkpoint at run start, then one after every completed node.
    let reader = SqliteCheckpointer::<AgentState>::new(&db_path, THREAD).unwrap();
    let history = reader.list().await.unwrap();
    assert_eq!(
        history.len(),
        3,
        "run-start checkpoint plus checkpoints after n1 and n2 are durable"
    );

    // Time travel: the ordered history reopens the state at every past step.
    assert_eq!(
        reader.load(&history[0]).await.unwrap().output.as_deref(),
        None,
        "the run-start checkpoint holds the state before any node ran"
    );
    assert_eq!(
        reader.load(&history[1]).await.unwrap().output.as_deref(),
        Some("n1")
    );
    assert_eq!(
        reader.load(&history[2]).await.unwrap().output.as_deref(),
        Some("n2")
    );
    let (last_state, recursion) = reader.last().await.unwrap().unwrap();
    assert_eq!(last_state.output.as_deref(), Some("n2"));
    assert_eq!(recursion, 2);

    // Storage-side OCC across two separate connections: two editors that both
    // based their edit on version 1 cannot both win.
    let editor_a = SqliteCheckpointer::<AgentState>::new(&db_path, THREAD).unwrap();
    let editor_b = SqliteCheckpointer::<AgentState>::new(&db_path, THREAD).unwrap();
    let mut forked = reader.load(&history[1]).await.unwrap();
    forked.set_output("edited-by-a".to_string());
    let new_version = editor_a
        .update_state(&history[1], &forked, 1)
        .await
        .expect("first editor wins");
    assert_eq!(new_version, 2);
    let conflict = editor_b
        .update_state(&history[1], &forked, 1)
        .await
        .unwrap_err();
    assert!(
        matches!(
            conflict,
            GraphError::CheckpointVersionConflict {
                expected: 1,
                actual: 2,
                ..
            }
        ),
        "stale edit from a second connection must conflict, got {conflict:?}"
    );

    // --- "Process 2": a freshly compiled graph resumes from the SQLite file. -
    let cp_process2 =
        SqliteCheckpointer::<AgentState>::new(&db_path, THREAD).expect("reopen sqlite");
    let resumed_graph = stamped_chain().with_checkpointer(cp_process2);
    let execution = resumed_graph
        .create_resume_execution("n3")
        .await
        .expect("resume execution builds from the durable checkpoint");
    assert_eq!(
        execution.recursion_count, 2,
        "budget carries across processes"
    );
    assert_eq!(execution.state.output.as_deref(), Some("n2"));

    let result = resumed_graph
        .resume(execution)
        .await
        .expect("resume completes");
    assert_eq!(result.recursion_count, 4);
    assert_eq!(result.final_state.output.as_deref(), Some("n4"));
    assert_eq!(result.final_state.input, "payload");

    // Time-travel replay/fork: branch a brand-new run off the older n1 state.
    let n1_state = reader.load(&history[1]).await.unwrap();
    let forked_run = stamped_chain()
        .with_recursion_limit(10)
        .invoke_from_node("n2".to_string(), n1_state)
        .await
        .expect("replay from an old checkpoint");
    assert_eq!(forked_run.final_state.output.as_deref(), Some("n4"));
}
