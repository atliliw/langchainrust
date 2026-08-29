// lc-agents/src/executor/tools.rs
//! Tool execution helpers shared by the streaming and non-streaming paths.

use super::AgentError;
use crate::types::{AgentAction, ToolInput};
use lc_core::tools::{BaseTool, ToolError};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Tool-**execution** error → observation text, fed back to the loop so the agent can
/// recover on its own. 0.20.0 S3.1 unified all four execution paths (invoke/stream ×
/// single/parallel) to this soft-fail semantics — the sequential `invoke` single-tool
/// path previously hard-failed upward and no longer does.
///
/// Only `AgentError::ToolExecutionError` (the tool ran and failed) is routed here.
/// Framework guardrails that reject a call *before* execution — `ToolNotFound`, tool
/// permission policy, hook `Reject`, and `ToolError::ControlAbort` (e.g. the handoff
/// cycle / depth guard) — are **not** soft-failed: the agent cannot recover from them
/// by re-planning, so they propagate hard.
pub(crate) fn tool_error_observation(err: &AgentError) -> String {
    format!("[Tool execution error: {err}]")
}

/// Executes a tool with an optional timeout.
///
/// With `Some(d)`, the tool call is cancelled (and errors) if it exceeds `d`.
/// Shared by both the non-streaming and streaming execution paths.
pub(crate) async fn run_tool_with_timeout(
    tool: &Arc<dyn BaseTool>,
    input: String,
    timeout: Option<Duration>,
) -> Result<String, ToolError> {
    let fut = tool.run(input);
    match timeout {
        Some(d) => match tokio::time::timeout(d, fut).await {
            Ok(result) => result,
            Err(_) => Err(ToolError::Timeout(d.as_secs())),
        },
        None => fut.await,
    }
}

/// Helper: execute a single tool for streaming (no RunTree dependency).
pub(crate) async fn execute_tool_for_stream(
    tools: &[Arc<dyn BaseTool>],
    action: &AgentAction,
    timeout: Option<Duration>,
) -> Result<String, AgentError> {
    let tool = tools
        .iter()
        .find(|t| t.name() == action.tool)
        .ok_or_else(|| AgentError::ToolNotFound(action.tool.clone()))?;

    let input_str = match &action.tool_input {
        ToolInput::String { value: s } => s.clone(),
        ToolInput::Object { value: v } => serde_json::to_string(v)
            .map_err(|e| AgentError::Other(format!("Failed to serialize tool input: {}", e)))?,
    };

    run_tool_with_timeout(tool, input_str, timeout)
        .await
        .map_err(|e| match e {
            // 0.20.0 A-H1: keep `ControlAbort` (handoff cycle / depth guard) distinct
            // from a plain execution failure, mirroring `execute_tool_inner`. The
            // streaming caller must be able to tell "the agent cannot recover, stop"
            // apart from "the tool ran and failed, re-plan".
            ToolError::ControlAbort(msg) => AgentError::Other(format!("Tool call aborted: {msg}")),
            other => AgentError::ToolExecutionError(other.to_string()),
        })
}

/// Helper: execute multiple tools in parallel for streaming.
///
/// Concurrency is capped at `max_concurrency` via a local semaphore.
///
/// 0.20.0 A-H1: mirrors the non-streaming parallel path — only a tool-**execution**
/// error becomes an observation; a framework guardrail (`ToolNotFound` /
/// `ControlAbort` / input serialization) in any one tool propagates hard as `Err` so
/// the caller ends the stream instead of feeding a re-plan loop that cannot recover.
pub(crate) async fn execute_tools_parallel_for_stream(
    tools: &[Arc<dyn BaseTool>],
    actions: &[AgentAction],
    timeout: Option<Duration>,
    max_concurrency: usize,
) -> Result<Vec<String>, AgentError> {
    use futures_util::future::join_all;

    let sem = Arc::new(Semaphore::new(max_concurrency));
    let futures = actions.iter().map(|action| {
        let sem = sem.clone();
        async move {
            let _permit = sem
                .acquire_owned()
                .await
                .map_err(|e| AgentError::Other(format!("concurrency semaphore closed: {e}")))?;
            execute_tool_for_stream(tools, action, timeout).await
        }
    });

    let results = join_all(futures).await;

    let mut observations = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(output) => observations.push(output),
            Err(e @ AgentError::ToolExecutionError(_)) => {
                observations.push(tool_error_observation(&e))
            }
            Err(e) => return Err(e),
        }
    }
    Ok(observations)
}
