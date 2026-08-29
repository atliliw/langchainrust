//! Lock-in tests for tool-execution-error behavior (0.18.2 MEDIUM-2 → 0.20.0 S3.1).
//!
//! 0.20.0 S3.1 design decision: **all four** tool-execution paths (invoke/stream ×
//! single/parallel) convert a tool failure into an **observation** fed back to the agent,
//! which recovers on its own. Before S3.1 the sequential `invoke` single-tool path
//! hard-stopped (returned `Err(ToolExecutionError)`); the streaming paths (single +
//! parallel) and the parallel `invoke` path were already soft. This file locks in the
//! unified semantics to prevent a silent flip back to hard-fail on any path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use lc_agents::types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
use lc_agents::{AgentError, AgentExecutor, AgentStreamEvent, BaseAgent};
use lc_core::tools::{BaseTool, ToolError};

/// Tool that always fails: every call returns `ExecutionFailed`.
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

/// Tool that always succeeds, returning a fixed value.
struct SucceedTool;

#[async_trait]
impl BaseTool for SucceedTool {
    fn name(&self) -> &str {
        "succeed"
    }
    fn description(&self) -> &str {
        "always succeeds"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        Ok("ok-value".to_string())
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

/// Round 1: parallel [failing, succeed]; round 2: Finish reporting whether the error
/// observation and the success result both landed in `intermediate_steps`.
struct ParallelFailThenFinishAgent;

#[async_trait]
impl BaseAgent for ParallelFailThenFinishAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Actions(vec![
                AgentAction {
                    tool: "failing".to_string(),
                    tool_input: ToolInput::Object {
                        value: serde_json::json!({"x": 1}),
                    },
                    log: "call_fail".to_string(),
                },
                AgentAction {
                    tool: "succeed".to_string(),
                    tool_input: ToolInput::Object {
                        value: serde_json::json!({"y": 2}),
                    },
                    log: "call_ok".to_string(),
                },
            ]));
        }
        let saw_err = intermediate_steps
            .iter()
            .any(|s| s.observation.contains("[Tool execution error"));
        let saw_ok = intermediate_steps
            .iter()
            .any(|s| s.observation.contains("ok-value"));
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!("done err={saw_err} ok={saw_ok}"),
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

fn parallel_failing_harness() -> (AgentExecutor, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let executor = AgentExecutor::new(
        Arc::new(ParallelFailThenFinishAgent),
        vec![
            Arc::new(FailingTool {
                calls: calls.clone(),
            }),
            Arc::new(SucceedTool),
        ],
    );
    (executor, calls)
}

/// Streaming single-tool: a failure is converted into an observation and fed back, the
/// stream continues and finishes normally.
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

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Err(_) | Ok(AgentStreamEvent::Error { .. }))),
        "流式单工具失败应转 observation 继续,got: {:?}",
        events
    );

    let tool_end = events.iter().find_map(|e| match e {
        Ok(AgentStreamEvent::ToolEnd { output, .. }) => Some(output.clone()),
        _ => None,
    });
    let tool_end = tool_end.expect("应发 ToolEnd 事件");
    assert!(
        tool_end.contains("[Tool execution error"),
        "ToolEnd 应为错误 observation,got: {tool_end}"
    );

    assert!(
        matches!(
            events.last(),
            Some(Ok(AgentStreamEvent::FinalAnswer { content })) if content == "done"
        ),
        "流应继续到 FinalAnswer,got: {:?}",
        events.last()
    );
}

/// Sequential invoke single-tool: a failure is converted into an observation and fed back;
/// the loop continues and finishes normally. (0.20.0 S3.1 inverted the pre-S3.1
/// hard-stop semantics — this was the last hard-fail path.)
#[tokio::test]
async fn sequential_tool_error_becomes_observation() {
    let (executor, calls) = failing_harness();
    let out = executor
        .invoke("go".to_string())
        .await
        .expect("顺序 invoke 单工具失败应转 observation 继续跑,got Err");
    assert_eq!(out, "done", "顺序 invoke 应恢复到最终答案");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "必败工具应执行 1 次");
}

/// Parallel invoke: a failing tool among the batch converts to an observation while the
/// succeeding tool's result survives; the loop continues and finishes normally.
#[tokio::test]
async fn invoke_parallel_tool_error_becomes_observation() {
    let (executor, calls) = parallel_failing_harness();
    let out = executor
        .invoke("go".to_string())
        .await
        .expect("并行 invoke 工具错误应转 observation 继续跑,got Err");
    assert_eq!(
        out, "done err=true ok=true",
        "应同时保留错误 observation 与成功结果,got: {out}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "必败工具应执行 1 次");
}

/// 0.20.0 S3.1: all four tool-execution paths (invoke/stream × single/parallel) soft-fail
/// identically — a tool error becomes an observation, no Err / Error event escapes, and
/// the loop reaches the final answer.
#[tokio::test]
async fn all_four_tool_error_paths_soft_fail_consistently() {
    use futures_util::StreamExt;

    let expect_no_error = |events: &[Result<AgentStreamEvent, AgentError>]| {
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Err(_) | Ok(AgentStreamEvent::Error { .. }))),
            "路径不应泄漏 Err/Error 事件,got: {:?}",
            events
        );
    };

    // invoke-single
    {
        let (executor, calls) = failing_harness();
        let out = executor
            .invoke("go".to_string())
            .await
            .expect("invoke 单工具失败应软失败");
        assert_eq!(out, "done");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    // invoke-parallel
    {
        let (executor, calls) = parallel_failing_harness();
        let out = executor
            .invoke("go".to_string())
            .await
            .expect("invoke 并行失败应软失败");
        assert_eq!(out, "done err=true ok=true", "{out}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    // stream-single
    {
        let (executor, calls) = failing_harness();
        let mut stream = executor.stream("go".to_string());
        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(ev);
        }
        expect_no_error(&events);
        assert!(
            matches!(events.last(), Some(Ok(AgentStreamEvent::FinalAnswer { content })) if content == "done"),
            "{:?}",
            events.last()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    // stream-parallel
    {
        let (executor, calls) = parallel_failing_harness();
        let mut stream = executor.stream("go".to_string());
        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(ev);
        }
        expect_no_error(&events);
        assert!(
            matches!(events.last(), Some(Ok(AgentStreamEvent::FinalAnswer { content })) if content == "done err=true ok=true"),
            "{:?}",
            events.last()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

/// Tool that returns `ControlAbort` (e.g. a handoff-cycle / depth-guard stop) — a
/// framework guardrail, NOT an execution failure (A-H1).
struct AbortTool;

#[async_trait]
impl BaseTool for AbortTool {
    fn name(&self) -> &str {
        "abort"
    }
    fn description(&self) -> &str {
        "always control-aborts"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        Err(ToolError::ControlAbort(
            "handoff cycle detected".to_string(),
        ))
    }
}

/// Calls `abort` once; round 2 (never reached after a hard stop) finishes.
struct CallAbortAgent;

#[async_trait]
impl BaseAgent for CallAbortAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Action(AgentAction {
                tool: "abort".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"x": 1}),
                },
                log: "call_abort".to_string(),
            }));
        }
        Ok(AgentOutput::Finish(AgentFinish::new(
            "done".to_string(),
            String::new(),
        )))
    }
}

/// Round 1: parallel `[abort, succeed]`; round 2 (never reached) finishes.
struct ParallelAbortAgent;

#[async_trait]
impl BaseAgent for ParallelAbortAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Actions(vec![
                AgentAction {
                    tool: "abort".to_string(),
                    tool_input: ToolInput::Object {
                        value: serde_json::json!({"x": 1}),
                    },
                    log: "call_abort".to_string(),
                },
                AgentAction {
                    tool: "succeed".to_string(),
                    tool_input: ToolInput::Object {
                        value: serde_json::json!({"y": 2}),
                    },
                    log: "call_ok".to_string(),
                },
            ]));
        }
        Ok(AgentOutput::Finish(AgentFinish::new(
            "done".to_string(),
            String::new(),
        )))
    }
}

fn abort_harness() -> AgentExecutor {
    AgentExecutor::new(
        Arc::new(CallAbortAgent),
        vec![Arc::new(AbortTool), Arc::new(SucceedTool)],
    )
}

fn parallel_abort_harness() -> AgentExecutor {
    AgentExecutor::new(
        Arc::new(ParallelAbortAgent),
        vec![Arc::new(AbortTool), Arc::new(SucceedTool)],
    )
}

/// 0.20.0 A-H1: `ControlAbort` (handoff cycle / depth guard) is a framework guardrail,
/// not an execution failure — it must stay HARD on all four paths. Before A-H1 the
/// streaming paths softened it into an observation, so a handoff-cycle guard would feed
/// a new plan every round and run forever instead of stopping. Execution failures stay
/// soft (locked by the tests above); this locks in the control-abort exception.
#[tokio::test]
async fn control_abort_stays_hard_on_all_paths() {
    use futures_util::StreamExt;

    // invoke-single: hard Err escapes.
    {
        let executor = abort_harness();
        let err = executor
            .invoke("go".to_string())
            .await
            .expect_err("invoke 单工具 ControlAbort 应硬失败,got Ok");
        assert!(
            err.to_string().contains("aborted"),
            "错误应包含 'aborted',got: {err}"
        );
    }
    // invoke-parallel: hard Err escapes, the whole batch aborts.
    {
        let executor = parallel_abort_harness();
        let err = executor
            .invoke("go".to_string())
            .await
            .expect_err("invoke 并行 ControlAbort 应硬失败,got Ok");
        assert!(
            err.to_string().contains("aborted"),
            "错误应包含 'aborted',got: {err}"
        );
    }
    // stream-single: Error event, no FinalAnswer, no softened ToolEnd.
    {
        let executor = abort_harness();
        let mut stream = executor.stream("go".to_string());
        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(ev);
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(_) | Ok(AgentStreamEvent::Error { .. }))),
            "流式单工具 ControlAbort 应发 Error 事件,got: {:?}",
            events
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(AgentStreamEvent::FinalAnswer { .. }))),
            "ControlAbort 不应到达 FinalAnswer,got: {:?}",
            events
        );
        assert!(
            !events.iter().any(|e| matches!(
                e,
                Ok(AgentStreamEvent::ToolEnd { name, .. }) if name == "abort"
            )),
            "ControlAbort 不应软化成 ToolEnd observation,got: {:?}",
            events
        );
    }
    // stream-parallel: Error event, no FinalAnswer.
    {
        let executor = parallel_abort_harness();
        let mut stream = executor.stream("go".to_string());
        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(ev);
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(_) | Ok(AgentStreamEvent::Error { .. }))),
            "流式并行 ControlAbort 应发 Error 事件,got: {:?}",
            events
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(AgentStreamEvent::FinalAnswer { .. }))),
            "ControlAbort 不应到达 FinalAnswer,got: {:?}",
            events
        );
    }
}

/// Round 1: parallel `[ghost (never registered), succeed]`; round 2: Finish reporting
/// whether the not-found observation and the success result both landed in
/// `intermediate_steps` (0.20.0 A-H3).
struct ParallelGhostThenFinishAgent;

#[async_trait]
impl BaseAgent for ParallelGhostThenFinishAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Actions(vec![
                AgentAction {
                    tool: "ghost".to_string(),
                    tool_input: ToolInput::Object {
                        value: serde_json::json!({"x": 1}),
                    },
                    log: "call_ghost".to_string(),
                },
                AgentAction {
                    tool: "succeed".to_string(),
                    tool_input: ToolInput::Object {
                        value: serde_json::json!({"y": 2}),
                    },
                    log: "call_ok".to_string(),
                },
            ]));
        }
        let saw_not_found = intermediate_steps
            .iter()
            .any(|s| s.observation.contains("[Tool not found"));
        let saw_ok = intermediate_steps
            .iter()
            .any(|s| s.observation.contains("ok-value"));
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!("done notfound={saw_not_found} ok={saw_ok}"),
            String::new(),
        )))
    }
}

/// 0.20.0 A-H3: a hallucinated / unregistered tool name in a parallel batch becomes an
/// observation instead of aborting the whole batch — the successful sibling tools' results
/// survive and the agent recovers. Before A-H3, `Err(ToolNotFound) => return Err(e)`
/// discarded the partial batch and killed the whole invoke.
#[tokio::test]
async fn invoke_parallel_hallucinated_tool_keeps_batch() {
    let executor = AgentExecutor::new(
        Arc::new(ParallelGhostThenFinishAgent),
        // Only `succeed` is registered; `ghost` must surface as ToolNotFound at runtime.
        vec![Arc::new(SucceedTool)],
    );
    let out = executor
        .invoke("go".to_string())
        .await
        .expect("并行 batch 中未注册工具名应转 observation 继续跑,got Err");
    assert_eq!(
        out, "done notfound=true ok=true",
        "应同时保留 not-found observation 与同批成功结果,got: {out}"
    );
}
