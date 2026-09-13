//! B2 (0.22.4) — durable checkpoints: interrupt/resume across restarts,
//! time travel, and optimistic state edits with `SqliteCheckpointer`.
//!
//! Run with:
//!
//! ```text
//! cargo run -p lc-langgraph --features checkpoint-sqlite --example durable_checkpoint
//! ```
//!
//! The same code works against Postgres or Redis by swapping the checkpointer
//! (`PostgresCheckpointer::connect` / `RedisCheckpointer::connect`); the graph
//! and resume APIs are backend-agnostic.

use lc_langgraph::{
    AgentState, Checkpointer, GraphBuilder, GraphError, SqliteCheckpointer, StateUpdate, END, START,
};

const THREAD_ID: &str = "example-thread-001";

fn build_graph(
    db_path: &std::path::Path,
) -> Result<lc_langgraph::CompiledGraph<AgentState>, Box<dyn std::error::Error>> {
    let checkpointer = SqliteCheckpointer::<AgentState>::new(db_path, THREAD_ID)?;
    let mut builder = GraphBuilder::<AgentState>::new();
    for i in 1..=3usize {
        let name = format!("n{i}");
        builder = builder.add_node_fn(name.clone(), move |state: &AgentState| {
            let mut next = state.clone();
            next.set_output(format!("reached n{i}"));
            Ok(StateUpdate::full(next))
        });
        if i == 1 {
            builder = builder.add_edge(START, name.clone());
        }
        if i == 3 {
            builder = builder.add_edge(name, END);
        }
    }
    for i in 1..3usize {
        builder = builder.add_edge(format!("n{i}"), format!("n{}", i + 1));
    }
    Ok(builder
        .compile()?
        // Pause the run right before n3: this is where a human approval, a
        // process restart, or a deployment cut-over could happen.
        .with_interrupt_before(vec!["n3".to_string()])
        .with_recursion_limit(10)
        .with_checkpointer(checkpointer))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::temp_dir().join("lc_durable_checkpoint_example.db");
    // Start from a clean file so repeated runs print the same story.
    let _ = std::fs::remove_file(&db_path);

    // ---- Phase 1: run until the interrupt ----------------------------------
    let graph = build_graph(&db_path)?;
    match graph
        .invoke(AgentState::new("durable payload".to_string()))
        .await
    {
        Err(GraphError::ExecutionInterrupted(node)) => {
            println!("phase 1: interrupted before '{node}' — process may exit now");
        }
        other => panic!("expected an interrupt, got {other:?}"),
    }
    drop(graph);

    // ---- Phase 2: reopen the database (as a new process would) -------------
    let cp = SqliteCheckpointer::<AgentState>::new(&db_path, THREAD_ID)?;
    let history = cp.list().await?;
    println!(
        "phase 2: {} durable checkpoint(s) for thread '{THREAD_ID}'",
        history.len()
    );

    // Time travel: inspect any previous state without changing "the present".
    if let Some(first) = history.first() {
        let old = cp.load(first).await?;
        println!(
            "time travel: oldest checkpoint input='{}', output={old:?}",
            old.input
        );
    }

    // Optimistic state edit: branch history starts at version 1.
    if let Some(target) = history.get(1) {
        match cp.update_state(target, &cp.load(target).await?, 1).await {
            Ok(version) => println!("update_state: checkpoint now at version {version}"),
            Err(GraphError::CheckpointVersionConflict {
                expected,
                actual,
                ..
            }) => println!("update_state: conflict (expected v{expected}, actual v{actual}) — reload and retry"),
            Err(other) => return Err(other.into()),
        }
    }

    // ---- Phase 3: a freshly built graph resumes from the durable state -----
    let resumed_graph = build_graph(&db_path)?;
    let execution = resumed_graph
        .create_resume_execution("n3")
        .await
        .expect("a checkpoint to resume from");
    println!(
        "phase 3: resuming with {} recursion step(s) already consumed",
        execution.recursion_count
    );
    let result = resumed_graph.resume(execution).await?;
    println!(
        "phase 3: resumed run finished — output='{}', total steps={}",
        result.final_state.output.as_deref().unwrap_or("<none>"),
        result.recursion_count
    );

    Ok(())
}
