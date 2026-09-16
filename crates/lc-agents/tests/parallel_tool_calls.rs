//! N3 (v0.24.0) lock-in: multiple tool calls in ONE model response ("一步多调")
//! execute concurrently on both the `invoke` and `stream` paths, and the
//! observations attach in the order the model emitted the actions — not the
//! order the tools happen to finish.
//!
//! The capability already existed (`AgentOutput::Actions` →
//! `execute_tools_parallel` / `execute_tools_parallel_for_stream`,
//! `join_all` + concurrency semaphore); N3 downgraded to a documentation
//! clarification plus this regression test, which proves the calls actually
//! overlap rather than merely both being present.
//!
//! Overlap is proven with a two-party [`Barrier`] shared by the tools: each
//! tool only returns once BOTH have started. A serialized executor deadlocks
//! on the first call — the outer `tokio::time::timeout` turns that into a fast
//! failure instead of a hung test. No network is involved.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lc_agents::types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
use lc_agents::{AgentError, AgentExecutor, AgentStreamEvent, BaseAgent};
use lc_core::tools::{BaseTool, ToolError};
use tokio::sync::Barrier;

/// Tool that only completes once every tool sharing the barrier has started.
struct BarrierTool {
    name: &'static str,
    output: &'static str,
    barrier: Arc<Barrier>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl BaseTool for BarrierTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "N3 barrier synchronization tool"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Releases only when both calls in the batch are in flight.
        self.barrier.wait().await;
        Ok(self.output.to_string())
    }
}

/// Round 1 emits two tool calls in one batch (emission order `a`, `b`);
/// round 2 verifies the observations attached in that same order and finishes.
struct ParallelOnceAgent;

#[async_trait]
impl BaseAgent for ParallelOnceAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
        _config: Option<&lc_core::runnables::RunnableConfig>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Actions(vec![
                AgentAction {
                    tool: "a".to_string(),
                    tool_input: ToolInput::String {
                        value: "1".to_string(),
                    },
                    log: "call-a".to_string(),
                },
                AgentAction {
                    tool: "b".to_string(),
                    tool_input: ToolInput::String {
                        value: "2".to_string(),
                    },
                    log: "call-b".to_string(),
                },
            ]));
        }

        assert_eq!(
            intermediate_steps.len(),
            2,
            "batch must attach as two steps"
        );
        assert_eq!(intermediate_steps[0].action.tool, "a");
        assert_eq!(intermediate_steps[0].observation, "A-out");
        assert_eq!(intermediate_steps[1].action.tool, "b");
        assert_eq!(intermediate_steps[1].observation, "B-out");
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!(
                "{}|{}",
                intermediate_steps[0].observation, intermediate_steps[1].observation
            ),
            String::new(),
        )))
    }
}

fn harness() -> (AgentExecutor, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let barrier = Arc::new(Barrier::new(2));
    let calls_a = Arc::new(AtomicUsize::new(0));
    let calls_b = Arc::new(AtomicUsize::new(0));
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(BarrierTool {
            name: "a",
            output: "A-out",
            barrier: barrier.clone(),
            calls: calls_a.clone(),
        }),
        Arc::new(BarrierTool {
            name: "b",
            output: "B-out",
            barrier: barrier.clone(),
            calls: calls_b.clone(),
        }),
    ];
    // Above the batch size so permits alone never serialize the pair.
    let executor = AgentExecutor::new(Arc::new(ParallelOnceAgent), tools).with_max_concurrency(8);
    (executor, calls_a, calls_b)
}

/// invoke: both calls in the batch run concurrently (barrier) and observations
/// attach in action-emission order.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invoke_runs_multiple_tool_calls_concurrently_in_action_order() {
    let (executor, calls_a, calls_b) = harness();
    let out = tokio::time::timeout(Duration::from_secs(15), executor.invoke("go".to_string()))
        .await
        .expect("serialized execution would deadlock on the shared barrier")
        .expect("parallel invoke should finish successfully");

    assert_eq!(out, "A-out|B-out");
    assert_eq!(
        calls_a.load(Ordering::SeqCst),
        1,
        "tool a runs exactly once"
    );
    assert_eq!(
        calls_b.load(Ordering::SeqCst),
        1,
        "tool b runs exactly once"
    );
}

/// stream: same overlap guarantee; ToolEnd events arrive in action order.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_runs_multiple_tool_calls_concurrently_in_action_order() {
    use futures_util::StreamExt;

    let (executor, calls_a, calls_b) = harness();
    let mut stream = executor.stream("go".to_string());
    let mut tool_ends: Vec<String> = Vec::new();

    let collect = async {
        while let Some(event) = stream.next().await {
            match event {
                Ok(AgentStreamEvent::ToolEnd { output, .. }) => tool_ends.push(output),
                Ok(AgentStreamEvent::FinalAnswer { content }) => {
                    return Ok::<_, AgentError>(content)
                }
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(String::new())
    };

    let out = tokio::time::timeout(Duration::from_secs(15), collect)
        .await
        .expect("serialized execution would deadlock on the shared barrier")
        .expect("parallel stream should finish successfully");

    assert_eq!(out, "A-out|B-out");
    assert_eq!(
        tool_ends,
        vec!["A-out".to_string(), "B-out".to_string()],
        "ToolEnd order must follow action order, not completion order"
    );
    assert_eq!(calls_a.load(Ordering::SeqCst), 1);
    assert_eq!(calls_b.load(Ordering::SeqCst), 1);
}
