//! Lock-in test for stream-cancellation behavior (0.20.0 A-H2).
//!
//! A-H2: dropping the stream returned by `AgentExecutor::stream` must stop the
//! background agent loop. Before the fix, the loop kept running after the consumer
//! hung up — executing further plans and tool calls (and burning LLM tokens) for a
//! listener that is gone. This test drops the stream while a tool is in flight and
//! asserts the loop stops at the next boundary instead of planning again.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_agents::types::{AgentAction, AgentOutput, AgentStep, ToolInput};
use lc_agents::{AgentError, AgentExecutor, AgentStreamEvent, BaseAgent};
use lc_core::tools::{BaseTool, ToolError};

/// Tool that signals `started` as soon as it enters `run`, then blocks until
/// `release` becomes true. Lets the test pin the agent loop deterministically inside
/// a tool call before dropping the stream.
struct GatedTool {
    started: tokio::sync::watch::Sender<bool>,
    release: tokio::sync::watch::Sender<bool>,
}

#[async_trait]
impl BaseTool for GatedTool {
    fn name(&self) -> &str {
        "gate"
    }
    fn description(&self) -> &str {
        "blocks until released"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        let _ = self.started.send(true);
        let mut rx = self.release.subscribe();
        while !*rx.borrow() {
            if rx.changed().await.is_err() {
                break;
            }
        }
        Ok("gated-ok".to_string())
    }
}

/// Always plans one `gate` action and never finishes on its own. If the agent loop
/// survives a dropped stream and plans again, `plan_calls` reaches 2 — the bug.
struct SingleCallAgent {
    plan_calls: Arc<AtomicUsize>,
}

#[async_trait]
impl BaseAgent for SingleCallAgent {
    async fn plan(
        &self,
        _intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        self.plan_calls.fetch_add(1, Ordering::SeqCst);
        Ok(AgentOutput::Action(AgentAction {
            tool: "gate".to_string(),
            tool_input: ToolInput::Object {
                value: serde_json::json!({}),
            },
            log: "call_gate".to_string(),
        }))
    }
}

/// 0.20.0 A-H2: dropping the stream cancels the background agent loop.
#[tokio::test]
async fn dropped_stream_stops_background_agent_loop() {
    let (started_tx, mut started_rx) = tokio::sync::watch::channel(false);
    let (release_tx, _release_rx) = tokio::sync::watch::channel(false);
    let plan_calls = Arc::new(AtomicUsize::new(0));

    let executor = AgentExecutor::new(
        Arc::new(SingleCallAgent {
            plan_calls: plan_calls.clone(),
        }),
        vec![Arc::new(GatedTool {
            started: started_tx,
            release: release_tx.clone(),
        })],
    );

    // Reader owns the stream: consume up to the first ToolStart (the loop is about to
    // run the gated tool), then wait for the drop order and return — dropping the stream.
    let stream = executor.stream("go".to_string());
    let (drop_tx, drop_rx) = tokio::sync::oneshot::channel::<()>();
    let reader = tokio::spawn(async move {
        let mut stream = stream;
        while let Some(ev) = stream.next().await {
            if matches!(ev, Ok(AgentStreamEvent::ToolStart { .. })) {
                break;
            }
        }
        let _ = drop_rx.await;
    });

    // The tool must actually be running before we drop, so the loop is deterministically
    // blocked inside `run` — not between steps where the timing could race.
    tokio::time::timeout(Duration::from_secs(5), async {
        while !*started_rx.borrow() {
            if started_rx.changed().await.is_err() {
                break;
            }
        }
    })
    .await
    .expect("gated tool should start within 5s");
    assert_eq!(
        plan_calls.load(Ordering::SeqCst),
        1,
        "one plan before the drop"
    );

    // Drop the stream (reader returns, wrapper sends the cancel signal).
    let _ = drop_tx.send(());
    tokio::time::timeout(Duration::from_secs(5), reader)
        .await
        .expect("reader should finish after dropping the stream")
        .expect("reader task should not panic");

    // Release the in-flight tool. With the fix the loop finishes the tool, then hits
    // the cancellation check and stops; without it, it plans again.
    let _ = release_tx.send(true);
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(
        plan_calls.load(Ordering::SeqCst),
        1,
        "dropped stream must stop the agent loop — plan must not run again"
    );
}
