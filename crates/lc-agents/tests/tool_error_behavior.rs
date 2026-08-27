//! Lock-in tests for tool-execution-error behavior differences (0.18.2, MEDIUM-2).
//!
//! Design decision: the sequential `invoke` path **hard-stops** on tool failure (returns
//! `Err(ToolExecutionError)`); the streaming path (single tool + parallel) converts a tool
//! failure into an **observation** fed back to the agent, which recovers on its own.
//! This file locks in both paths' existing semantics to prevent silent flips — reverting
//! the sequential path back to observation, or the streaming path back to an Error event
//! that aborts the stream, would both be regressions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use lc_agents::types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
use lc_agents::{AgentError, AgentExecutor, AgentStreamEvent, BaseAgent};
use lc_core::tools::{BaseTool, ToolError};

/// Tool that always fails: every call returns `ExecutionFailed` (since 0.18.2,
/// `tool_error_observation` converts it into an observation; `ToolError` has no `Message`
/// variant — `ExecutionFailed` is the right choice).
struct FailingTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl BaseTool for FailingTool {
    fn name(&self) -> &str {
        "failing"
    }
    fn description(&self) -> &str {
        "always fails"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(ToolError::ExecutionFailed("boom".to_string()))
    }
}

/// Calls the failing tool on the first round, then finishes (verifies it recovers from the
/// error observation).
struct FailThenFinishAgent;

#[async_trait]
impl BaseAgent for FailThenFinishAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Action(AgentAction {
                tool: "failing".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"x": 1}),
                },
                log: "call_fail".to_string(),
            }));
        }
        Ok(AgentOutput::Finish(AgentFinish::new(
            "done".to_string(),
            String::new(),
        )))
    }
}

fn failing_harness() -> (AgentExecutor, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tool = FailingTool {
        calls: calls.clone(),
    };
    let executor = AgentExecutor::new(Arc::new(FailThenFinishAgent), vec![Arc::new(tool)]);
    (executor, calls)
}

/// Streaming: a single-tool failure is converted into an observation and fed back, the
/// stream continues and finishes normally (it no longer aborts the stream).
#[tokio::test]
async fn stream_tool_error_continues() {
    use futures_util::StreamExt;

    let (executor, calls) = failing_harness();
    let mut stream = executor.stream("go".to_string());

    let mut events: Vec<Result<AgentStreamEvent, AgentError>> = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    assert_eq!(calls.load(Ordering::SeqCst), 1, "必败工具应执行 1 次");

    // No Error events and no Err items anywhere — before 0.18.2 a single-tool failure
    // emitted an Error event and aborted the stream; fixed.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Err(_) | Ok(AgentStreamEvent::Error { .. }))),
        "流式单工具失败应转 observation 继续,got: {:?}",
        events
    );

    // Find the ToolEnd, whose output is the error observation (same format as the
    // parallel path).
    let tool_end = events.iter().find_map(|e| match e {
        Ok(AgentStreamEvent::ToolEnd { output, .. }) => Some(output.clone()),
        _ => None,
    });
    let tool_end = tool_end.expect("应发 ToolEnd 事件");
    assert!(
        tool_end.contains("[Tool execution error"),
        "ToolEnd 应为错误 observation,got: {tool_end}"
    );

    // The stream continues on to FinalAnswer.
    assert!(
        matches!(
            events.last(),
            Some(Ok(AgentStreamEvent::FinalAnswer { content })) if content == "done"
        ),
        "流应继续到 FinalAnswer,got: {:?}",
        events.last()
    );
}

/// Sequential invoke: tool failure hard-stops — returns `Err(ToolExecutionError)`, not an
/// observation. Locks in the sequential-path semantics against a silent flip to
/// observation.
#[tokio::test]
async fn sequential_tool_error_stays_hard() {
    let (executor, calls) = failing_harness();
    let err = executor.invoke("go".to_string()).await.unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        matches!(err, AgentError::ToolExecutionError(_)),
        "顺序路径工具失败应硬停返回 Err(ToolExecutionError),got: {err:?}"
    );
}
