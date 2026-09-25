// lc-agents/src/executor/tool_gate.rs
//! B5: `ToolGate` — the single tool-execution chain shared by the `invoke` and
//! `stream` paths.
//!
//! Before B5 the hooks → approval → final-name resolution → execution → callback
//! chain lived only in [`super::agent_loop::AgentExecutor::execute_tool_inner`]
//! (invoke), while the stream path ran tools through a stripped-down helper
//! (`execute_tool_for_stream`) that skipped hooks / approval / callbacks / tool
//! RunTree. This gate converges both paths onto one implementation so their
//! behavior cannot drift.
//!
//! Order (same for invoke and stream):
//!
//! ```text
//!   on_before_tool_call (hooks) → approval gate (allow/modify/deny)
//!   → resolve FINAL tool by name → tool permission policy → Rule of Two
//!   → run tool (timeout, RunTree child + callbacks) → on_after_tool_call → spotlight
//! ```
//!
//! The tool is resolved **after** hooks/approval so a hook `Modify` that renames
//! the tool executes the renamed tool, and the args / metadata all use that final
//! tool (fixes the pre-B5 "old tool object + new params" bug).
//!
//! Framework **hard** errors (hook `Reject`, policy rejection,
//! `ToolError::ControlAbort`) abort a parallel batch; a hallucinated /
//! unregistered tool name (`ToolNotFound`) and tool **execution** failures
//! (`ToolExecutionError`) remain soft observations (`[Tool not found: …]` /
//! `[Tool execution error: …]`), and approval **`Deny`** is clamped to
//! `[DENIED by approval: …]` — all letting the model re-plan rather than cancel
//! safe siblings. This mirrors the pre-B5 S3.1 semantics (guardrails abort,
//! execution errors soft-fail) and the 0.20.0 A-H3 rule that a hallucinated name
//! keeps its batch (stage-G G1 unifies the single path to match). When a hard
//! error does abort a batch, observations already collected from completed
//! siblings are preserved in `AgentError::BatchAborted::partial` (stage-G G3).

use super::engine::AgentExecutor;
use super::tools::{denied_observation, run_tool_with_timeout, tool_error_observation, wrap_tool_output};
use super::AgentError;
use crate::approval::{ApprovalDecision, ApprovalHandler};
use crate::hooks::{AgentHook, ToolCallAction, ToolCallContext, ToolResultContext};
use crate::policy::ToolPolicy;
use crate::resume::{PendingApproval, ResumeStore};
use crate::types::{AgentAction, ToolInput};
use lc_callbacks::{
    semconv::{GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_DESCRIPTION},
    CallbackManager, RunTree, RunType,
};
use lc_core::tools::{BaseTool, ToolError};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// The outcome of a single tool-call chain, distinguishing a normal observation
/// from an approval **Deny**. Both mean "the tool did not produce output"; a Deny
/// is surfaced as a soft `[DENIED by approval: …]` observation on every path
/// (invoke, stream, and parallel batch) so the model can re-plan around it,
/// rather than aborting the run or a batch of safe siblings.
#[derive(Debug)]
pub(crate) enum ToolOutcome {
    /// Normal terminal observation (tool output, hook `Skip`).
    Done(String),
    /// Approval handler returned `Deny`.
    Denied { reason: String },
}

/// Checkpoint context for cross-process resume. Carries references to the
/// caller-built snapshot and the store; the gate persists it just before the
/// approval gate and clears it once the decision lands (invoke-only — the
/// stream path passes `None`).
pub(crate) struct ResumeCtx<'a> {
    /// Checkpoint template pre-filled with loop context.
    pub(crate) pending: &'a PendingApproval,
    /// Checkpoint storage.
    pub(crate) store: &'a Arc<dyn ResumeStore>,
}

/// B5: the per-slot output of a parallel-batch task — `(emission index, outcome)`.
/// The index reorders observations back into action order; the inner `Result` is
/// the tool's own outcome (its `AgentError`), the outer one is the task wrapper.
type BatchJoinOutput = Result<(usize, Result<String, AgentError>), AgentError>;

/// Applies an approval decision to `tool_ctx`; returns `Some(reason)` for Deny.
fn apply_approval_decision(
    decision: ApprovalDecision,
    tool_ctx: &mut ToolCallContext,
) -> Option<String> {
    match decision {
        ApprovalDecision::Allow => None,
        ApprovalDecision::Deny { reason } => {
            log::info!(
                target: "lc_agents::approval",
                "tool_call denied by approval handler name={} reason={}",
                tool_ctx.name,
                reason
            );
            Some(reason)
        }
        ApprovalDecision::Modify { arguments, note } => {
            log::info!(
                target: "lc_agents::approval",
                "tool_call arguments modified by approval handler name={} note={}",
                tool_ctx.name,
                note
            );
            tool_ctx.arguments = arguments;
            None
        }
    }
}

/// The shared tool-execution gate.
///
/// `Clone` shares cheaply-reusable pieces (the tool index lives behind an `Arc`),
/// so it can be copied into `'static` stream tasks and into per-peer join tasks.
#[derive(Clone)]
pub(crate) struct ToolGate {
    /// O(1) name → tool index (A11).
    tools: Arc<HashMap<String, Arc<dyn BaseTool>>>,
    hooks: Vec<Arc<dyn AgentHook>>,
    approval: Option<Arc<dyn ApprovalHandler>>,
    policy: Option<ToolPolicy>,
    callbacks: Option<Arc<CallbackManager>>,
    rule_of_two: bool,
    spotlight: bool,
    timeout: Option<Duration>,
    concurrency: Arc<Semaphore>,
}

impl ToolGate {
    /// Builds a gate from the executor's shared configuration.
    pub(crate) fn from_executor(ex: &AgentExecutor) -> Self {
        Self::from_parts(
            ex.tools_by_name.clone(),
            ex.hooks.clone(),
            ex.approval.clone(),
            ex.tool_policy.clone(),
            ex.callbacks.clone(),
            ex.rule_of_two,
            ex.spotlight_tool_output,
            ex.tool_timeout,
            ex.concurrency_sem.clone(),
        )
    }

    /// Explicit construction (used by the stream task, which holds the executor's
    /// pieces as owned locals rather than an `AgentExecutor`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        tools: HashMap<String, Arc<dyn BaseTool>>,
        hooks: Vec<Arc<dyn AgentHook>>,
        approval: Option<Arc<dyn ApprovalHandler>>,
        policy: Option<ToolPolicy>,
        callbacks: Option<Arc<CallbackManager>>,
        rule_of_two: bool,
        spotlight: bool,
        timeout: Option<Duration>,
        concurrency: Arc<Semaphore>,
    ) -> Self {
        Self {
            tools: Arc::new(tools),
            hooks,
            approval,
            policy,
            callbacks,
            rule_of_two,
            spotlight,
            timeout,
            concurrency,
        }
    }

    /// stage-G G2: clear the cross-process resume checkpoint — but only once this
    /// tool-call chain has reached a terminal (the tool actually ran, or the call
    /// resolved to a final observation / hard error). The checkpoint must survive
    /// while the tool is executing so a crash mid-run can resume the model's
    /// in-flight step instead of silently dropping it.
    ///
    /// The pre-G2 code cleared the checkpoint immediately after the approval gate
    /// and *before* `run_tool_with_timeout` — a crash between that clear and the
    /// tool's completion left the store empty and the step the model was about to
    /// execute lost (at-most-once degenerated to zero-times). The clear here runs
    /// only on the way out of every terminal of [`Self::execute_one`].
    async fn clear_resume(resume: Option<&ResumeCtx<'_>>) {
        if let Some(ctx) = resume {
            if let Err(e) = ctx.store.clear_pending().await {
                log::warn!(
                    target: "lc_agents::resume",
                    "failed to clear pending approval: {}",
                    e
                );
            }
        }
    }

    /// Runs one tool through the full chain. Returns a `ToolOutcome` so the caller
    /// can distinguish an approval `Deny` from a normal observation.
    pub(crate) async fn execute_one(
        &self,
        action: &AgentAction,
        root_run: &RunTree,
        resume: Option<&ResumeCtx<'_>>,
        pre_decided: Option<ApprovalDecision>,
    ) -> Result<ToolOutcome, AgentError> {
        // B5: resolve the tool_id up front. An explicit provider id wins; otherwise
        // the gate stamps a framework uuid so observability / resume never persist
        // an empty id.
        let tool_id = match &action.tool_call_id {
            Some(id) if !id.is_empty() => id.clone(),
            _ => uuid::Uuid::new_v4().to_string(),
        };

        let mut tool_ctx = ToolCallContext {
            name: action.tool.clone(),
            arguments: match &action.tool_input {
                ToolInput::String { value: s } => serde_json::from_str::<serde_json::Value>(s)
                    .unwrap_or(serde_json::Value::String(s.clone())),
                ToolInput::Object { value: v } => v.clone(),
            },
            tool_id,
        };

        // 1. Sync hooks — may rename the tool and/or rewrite arguments.
        for hook in &self.hooks {
            match hook.on_before_tool_call(&mut tool_ctx) {
                ToolCallAction::Continue => {}
                ToolCallAction::Modify { name, arguments } => {
                    tool_ctx.name = name;
                    tool_ctx.arguments = arguments;
                }
                ToolCallAction::Reject { reason } => {
                    return Err(AgentError::Other(format!(
                        "Tool call rejected by hook: {}",
                        reason
                    )));
                }
                ToolCallAction::Skip => {
                    return Ok(ToolOutcome::Done("[Skipped by hook]".to_string()));
                }
            }
        }

        // 2. Approval gate (§4.2) + cross-process resume. Deny is the only path that
        //    yields `ToolOutcome::Denied`.
        let deny_reason = if let Some(pre) = pre_decided {
            apply_approval_decision(pre, &mut tool_ctx)
        } else if let Some(handler) = &self.approval {
            if let Some(ctx) = resume {
                let mut pending = ctx.pending.clone();
                pending.tool_name = tool_ctx.name.clone();
                pending.arguments = tool_ctx.arguments.clone();
                pending.tool_id = tool_ctx.tool_id.clone();
                if let Err(e) = ctx.store.save_pending(&pending).await {
                    log::warn!(
                        target: "lc_agents::resume",
                        "failed to persist pending approval: {}",
                        e
                    );
                }
            }
            apply_approval_decision(handler.approve(&tool_ctx).await, &mut tool_ctx)
        } else {
            None
        };

        // After the tool call chain reaches each terminal — an execution error
        // (below), a pre-execution refusal, or a completed run — the checkpoint is
        // cleared. Crucially the clear happens on the way out, never before the tool
        // runs: a crash while the tool is executing must leave the checkpoint in the
        // store so resume can re-attempt the step (stage-G G2 fixes the old
        // clear-before-execute that dropped it entirely in a crash).
        if let Some(reason) = deny_reason {
            Self::clear_resume(resume).await;
            return Ok(ToolOutcome::Denied { reason });
        }

        // 3. Resolve the FINAL tool by the (possibly hook-modified) name.
        let tool_name = tool_ctx.name.clone();
        let tool = match self.tools.get(&tool_name) {
            Some(t) => t,
            None => {
                Self::clear_resume(resume).await;
                return Err(AgentError::ToolNotFound(tool_name.clone()));
            }
        };

        // 4. Tool permission policy (P2-9) — on the final name.
        if let Some(policy) = &self.policy {
            if let Err(e) = policy.check(&tool_name) {
                Self::clear_resume(resume).await;
                return Err(e);
            }
        }

        // 5. A2 Rule of Two — on the final tool.
        if self.rule_of_two && tool.risk().count_armed() >= 3 {
            Self::clear_resume(resume).await;
            log::warn!(
                target: "lc_agents::rule_of_two",
                "blocked high-risk tool '{}' (risk={}/3)",
                tool_name,
                tool.risk().count_armed()
            );
            return Ok(ToolOutcome::Done(
                "[BLOCKED by Rule of Two: tool declares untrusted-input + sensitive-access + state-changing]"
                    .to_string(),
            ));
        }

        let input_for_tool = serde_json::to_string(&tool_ctx.arguments)
            .unwrap_or_else(|_| tool_ctx.arguments.to_string());

        // 6. Tool-level RunTree + chain callbacks (T10 metadata: description + id).
        let mut tool_run = root_run.create_child(
            &tool_name,
            RunType::Tool,
            json!({"input": input_for_tool.clone()}),
        );
        tool_run = tool_run.with_metadata(GEN_AI_TOOL_DESCRIPTION, json!(tool.description()));
        tool_run = tool_run.with_metadata(GEN_AI_TOOL_CALL_ID, json!(tool_ctx.tool_id.clone()));

        if let Some(ref callbacks) = self.callbacks {
            for handler in callbacks.handlers() {
                handler
                    .on_tool_start(&tool_run, &tool_name, &input_for_tool)
                    .await;
            }
        }

        let started = Instant::now();
        let result = run_tool_with_timeout(tool, input_for_tool.clone(), self.timeout).await;
        // The tool has now run (or its timeout fired): the checkpoint's job is done,
        // this step's fate is fixed regardless of success or error, so clear it.
        Self::clear_resume(resume).await;
        let tool_duration_ms = started.elapsed().as_millis();
        let trace = root_run
            .trace_id
            .map(|id| id.to_string())
            .unwrap_or_default();

        match result {
            Ok(output) => {
                log::info!(
                    target: "lc_agents::audit",
                    "tool_call trace_id={} name={} input={} duration_ms={} outcome=ok",
                    trace,
                    tool_name,
                    input_for_tool,
                    tool_duration_ms
                );
                tool_run.end(json!({"output": output.clone()}));
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_tool_end(&tool_run, &output).await;
                    }
                }

                // 7. on_after_tool_call — carries the same tool_id that was stamped
                //    on the start (T10).
                let mut result_ctx = ToolResultContext {
                    name: tool_name,
                    result: output.clone(),
                    tool_id: tool_ctx.tool_id,
                };
                for hook in &self.hooks {
                    if let Err(e) = hook.on_after_tool_call(&mut result_ctx) {
                        log::warn!("Hook on_after_tool_call error: {}", e);
                    }
                }

                // A1 spotlighting.
                let observed = if self.spotlight {
                    wrap_tool_output(&result_ctx.result)
                } else {
                    result_ctx.result
                };
                Ok(ToolOutcome::Done(observed))
            }
            Err(e) => {
                log::info!(
                    target: "lc_agents::audit",
                    "tool_call trace_id={} name={} input={} duration_ms={} outcome=error:{}",
                    trace,
                    tool_name,
                    input_for_tool,
                    tool_duration_ms,
                    e
                );
                tool_run.end_with_error(e.to_string());
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_tool_error(&tool_run, &e.to_string()).await;
                    }
                }
                // Framework control-abort is a refusal to execute (hard); any other
                // ToolError is an execution failure (soft observation).
                match e {
                    ToolError::ControlAbort(msg) => {
                        Err(AgentError::Other(format!("Tool call aborted: {msg}")))
                    }
                    other => Err(AgentError::ToolExecutionError(other.to_string())),
                }
            }
        }
    }

    /// Executes a parallel batch through the gate.
    ///
    /// - Tool **execution** failures (`ToolExecutionError`), unregistered names
    ///   (`ToolNotFound`), and approval **`Deny`** become ordered observations so
    ///   sibling results survive and the model can re-plan around them.
    /// - Framework hard errors — hook `Reject`, policy rejection,
    ///   `ControlAbort` — **abort the whole batch** (`JoinSet::abort_all`, B5): the
    ///   agent cannot re-plan around them, so siblings are cancelled rather than run
    ///   pointlessly. The hard error is returned.
    ///
    /// Concurrency is capped by the gate's shared semaphore.
    pub(crate) async fn execute_many(
        &self,
        actions: &[AgentAction],
        root_run: &RunTree,
    ) -> Result<Vec<String>, AgentError> {
        let sem = self.concurrency.clone();
        // Shared parent for the spawned tasks. `RunTree::create_child` takes `&self`,
        // so a shared Arc (cloned snapshot — the parent is not mutated mid-batch) lets
        // each task build its own tool child-run while staying `'static` (JoinSet).
        let shared_root = Arc::new(root_run.clone());
        let mut set: tokio::task::JoinSet<BatchJoinOutput> = tokio::task::JoinSet::new();
        for (idx, action) in actions.iter().enumerate() {
            let gate = self.clone();
            let sem = sem.clone();
            let action = action.clone();
            let shared_root = shared_root.clone();
            set.spawn(async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .map_err(|e| AgentError::Other(format!("concurrency semaphore closed: {e}")))?;
                let (idx, r) = (
                    idx,
                    match gate.execute_one(&action, &shared_root, None, None).await {
                        Ok(ToolOutcome::Done(text)) => Ok(text),
                        // Deny is a soft observation (unified with the single-invoke
                        // path): the model gets `[DENIED ...]` back to re-plan around,
                        // not a batch-aborting error.
                        Ok(ToolOutcome::Denied { reason }) => Ok(denied_observation(&reason)),
                        Err(e) => Err(e),
                    },
                );
                Ok((idx, r))
            });
        }

        let mut results: Vec<(usize, Result<String, AgentError>)> =
            Vec::with_capacity(actions.len());
        while let Some(outcome) = set.join_next().await {
            let (idx, r) = match outcome {
                Ok(Ok(pair)) => pair,
                Ok(Err(join_reason)) => {
                    return Err(AgentError::Other(format!(
                        "parallel tool task join failure: {join_reason}"
                    )))
                }
                Err(panic) => {
                    return Err(AgentError::Other(format!(
                        "parallel tool task panicked: {panic}"
                    )))
                }
            };
            // A framework hard error (hook Reject / policy / ControlAbort) aborts
            // the whole batch — (Approval `Deny` is NOT hard: it is unified into a
            // soft observation above, so denials surface to the model instead of
            // cancelling safe siblings.) Sibling observations already collected are
            // preserved in the returned error (stage-G G3) rather than discarded.
            match &r {
                Err(AgentError::ToolExecutionError(_)) | Err(AgentError::ToolNotFound(_)) => {
                    results.push((idx, r));
                }
                Err(_) => {
                    set.abort_all();
                    // stage-G G3: don't silently discard already-completed siblings;
                    // carry their observations so the caller can surface them.
                    let partial = results
                        .iter()
                        .filter_map(|(_, rr)| match rr {
                            Ok(o) => Some(o.clone()),
                            Err(e @ AgentError::ToolExecutionError(_)) => {
                                Some(tool_error_observation(e))
                            }
                            Err(AgentError::ToolNotFound(name)) => {
                                Some(format!("[Tool not found: {name}]"))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    return Err(AgentError::BatchAborted {
                        cause: Box::new(r.unwrap_err()),
                        partial,
                    });
                }
                Ok(_) => results.push((idx, r)),
            }
        }

        // Reassemble in the action order the model emitted.
        results.sort_by_key(|(i, _)| *i);
        results
            .into_iter()
            .map(|(_, r)| match r {
                Ok(o) => Ok(o),
                Err(e @ AgentError::ToolExecutionError(_)) => Ok(tool_error_observation(&e)),
                Err(AgentError::ToolNotFound(name)) => Ok(format!("[Tool not found: {name}]")),
                Err(e) => Err(e),
            })
            .collect()
    }
}
