// lc-agents/src/executor/tools.rs
//! Tool execution helpers shared by the streaming and non-streaming paths.

use super::AgentError;
use crate::types::{AgentAction, ToolInput};
use lc_core::tools::{BaseTool, ToolError};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

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
        .map_err(|e| AgentError::ToolExecutionError(e.to_string()))
}

/// Helper: execute multiple tools in parallel for streaming.
///
/// Concurrency is capped at `max_concurrency` via a local semaphore.
pub(crate) async fn execute_tools_parallel_for_stream(
    tools: &[Arc<dyn BaseTool>],
    actions: &[AgentAction],
    timeout: Option<Duration>,
    max_concurrency: usize,
) -> Vec<String> {
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

    results
        .into_iter()
        .map(|result| result.unwrap_or_else(|e| format!("[Tool execution error: {}]", e)))
        .collect()
}
