// lc-agents/src/executor/tools.rs
//! Tool execution helpers shared by the streaming and non-streaming paths.

use super::AgentError;
use lc_core::tools::{BaseTool, ToolError};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// A11: build an O(1) name → tool index from a tool list.
///
/// The prior lookups were `tools.iter().find(|t| t.name() == name)` — O(n) on every
/// tool call. `entry().or_insert_with` preserves the original "first match wins"
/// semantics exactly: the linear scan returned the *first* tool whose name matched,
/// and `or_insert_with` keeps the first-inserted entry when names collide.
pub(crate) fn index_tools(tools: &[Arc<dyn BaseTool>]) -> HashMap<String, Arc<dyn BaseTool>> {
    let mut map: HashMap<String, Arc<dyn BaseTool>> = HashMap::with_capacity(tools.len());
    for tool in tools {
        map.entry(tool.name().to_string())
            .or_insert_with(|| tool.clone());
    }
    map
}

/// A1 Spotlighting — open marker wrapping untrusted tool output.
///
/// Mirrors `lc_guardrails`' `DEFAULT_OPEN_MARKER`/`DEFAULT_CLOSE_MARKER` and its escaping
/// (a backslash-prefixed close tag) so output wrapped here parse back cleanly through
/// [`lc_guardrails::spotlighting::unwrap`]. `lc-agents` cannot import `lc-guardrails`
/// (the dependency points the other way), so the constants are mirrored rather than reused.
pub(crate) const UNTRUSTED_OPEN: &str = "<untrusted_data>";
pub(crate) const UNTRUSTED_CLOSE: &str = "</untrusted_data>";

/// A1: wraps untrusted tool output in the data delimiters, escaping any embedded close
/// marker so a hostile tool result cannot forge a premature `</untrusted_data>`.
pub(crate) fn wrap_tool_output(output: &str) -> String {
    let escaped = output.replace(UNTRUSTED_CLOSE, "\\</untrusted_data>");
    format!("{UNTRUSTED_OPEN}{escaped}{UNTRUSTED_CLOSE}")
}

/// Tool-**execution** error → observation text, fed back to the loop so the agent can
/// recover on its own. 0.20.0 S3.1 unified all four execution paths (invoke/stream ×
/// single/parallel) to this soft-fail semantics — the sequential `invoke` single-tool
/// path previously hard-failed upward and no longer does.
///
/// Only `AgentError::ToolExecutionError` (the tool ran and failed) is routed here.
/// A hallucinated / unregistered `ToolNotFound` is also **soft** (a recoverable
/// observation reflecting a tool that never executed — stage-G G1), matching the
/// nailed 0.20.0 A-H3 parallel semantics so the sequential and parallel paths cannot
/// diverge. Framework guardrails that reject a call *before* execution — tool
/// permission policy, hook `Reject`, and `ToolError::ControlAbort` (e.g. the handoff
/// cycle / depth guard) — are **not** soft-failed: the agent cannot recover from them
/// by re-planning, so they propagate hard.
pub(crate) fn tool_error_observation(err: &AgentError) -> String {
    format!("[Tool execution error: {err}]")
}

/// stage-G G6: shared formatting for an approval-denial observation. All four
/// execution paths (invoke/stream × single/parallel) produce byte-identical
/// text so the non-execution predicate below stays reliable.
pub(crate) fn denied_observation(reason: &str) -> String {
    format!("[DENIED by approval: {reason}]")
}

/// stage-G G6: a soft observation reflecting a tool that **never executed** —
/// an approval `Deny` or a hallucinated / unregistered name. Such calls must
/// not consume the `max_tool_calls` budget (only executed tools count); the
/// loops decrement `metrics.tool_calls` for these before continuing.
pub(crate) fn is_non_execution_observation(obs: &str) -> bool {
    obs.starts_with("[DENIED by approval:") || obs.starts_with("[Tool not found: ")
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Minimal named tool for the A11 index tests.
    struct NamedTool(&'static str);

    #[async_trait]
    impl BaseTool for NamedTool {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "A11 index test tool"
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(input)
        }
    }

    fn named(name: &'static str) -> Arc<dyn BaseTool> {
        Arc::new(NamedTool(name))
    }

    #[test]
    fn index_resolves_each_tool_by_name() {
        let tools: Vec<Arc<dyn BaseTool>> = vec![named("alpha"), named("beta"), named("gamma")];
        let idx = index_tools(&tools);
        assert!(idx.contains_key("alpha"));
        assert!(idx.contains_key("beta"));
        assert!(idx.contains_key("gamma"));
        assert!(!idx.contains_key("delta"));
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn index_preserves_first_match_on_name_collision() {
        // The old linear `find` returned the *first* tool whose name matched; the
        // index uses `or_insert_with`, so the first-inserted entry must win too —
        // otherwise the O(1) optimization would silently change which tool runs.
        let first = named("dup");
        let second = named("dup");
        let tools: Vec<Arc<dyn BaseTool>> = vec![first.clone(), second.clone()];
        let idx = index_tools(&tools);
        let resolved = idx.get("dup").unwrap();
        assert!(
            Arc::ptr_eq(resolved, &first),
            "first-inserted tool must win on name collision"
        );
    }
}
