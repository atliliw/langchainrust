// lc-agents/src/executor/agent_loop.rs
//! `AgentExecutor`'s decision loop: the main agent loop plus sequential/parallel tool
//! execution.
//!
//! Works alongside `executor.rs` (struct + builder + invoke/stream entry points) and
//! `plan.rs` (cached planning).

use super::budget::{budget_iteration_gate, budget_token_gate, budget_tool_gate};
use super::engine::AgentExecutor;
use super::tools::{run_tool_with_timeout, tool_error_observation};
use super::AgentError;
use crate::approval::ApprovalDecision;
use crate::hooks::{ToolCallAction, ToolCallContext, ToolResultContext};
use crate::metrics::AgentMetrics;
use crate::resume::{PendingApproval, ResumeStore};
use crate::types::{AgentAction, AgentOutput, AgentStep, ToolInput};
use lc_callbacks::{RunTree, RunType};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Cross-process resume (§4.2): the checkpoint context needed for a single tool call.
///
/// Constructed by the agent loop in the Action branch: `tool_name` / `arguments` /
/// `tool_id` start as placeholders, then `execute_tool_inner` fills in the **final values**
/// the approval saw once the synchronous hooks finish, and persists them; the checkpoint
/// is cleared once the approval decision lands. The parallel tool path
/// (`execute_tools_parallel`) does not build one — concurrent multi-tool approvals never
/// persist, so checkpoints cannot overwrite each other.
pub(crate) struct ResumeContext<'a> {
    /// Checkpoint template pre-filled with loop context (inputs / steps / iteration /
    /// budget accumulation / trace).
    pending: &'a PendingApproval,
    /// Checkpoint storage (persist before / clear after approval).
    store: &'a Arc<dyn ResumeStore>,
}

/// Applies an approval decision to `tool_ctx`.
///
/// Returns `Some(reason)` for **Deny** (aborts execution; the rejection observation is fed
/// back to the loop); `None` for Allow / Modify (execution continues). Modify overwrites
/// `tool_ctx.arguments`.
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

impl AgentExecutor {
    /// Runs the agent loop from scratch.
    ///
    /// Accumulates `metrics` (LLM calls, tool calls, token usage) as it goes.
    pub(crate) async fn run_agent_loop(
        &self,
        inputs: HashMap<String, String>,
        intermediate_steps: Vec<AgentStep>,
        root_run: &mut RunTree,
        metrics: &mut AgentMetrics,
    ) -> Result<String, AgentError> {
        self.run_agent_loop_from(inputs, intermediate_steps, 0, root_run, metrics)
            .await
    }

    /// Runs the agent loop starting at a given iteration.
    ///
    /// Cross-process resume (§4.2) uses this to continue from a pending iteration: the
    /// iteration / tool-call budgets keep counting from the checkpoint's accumulated
    /// amounts, and already-completed intermediate steps are not replayed.
    pub(crate) async fn run_agent_loop_from(
        &self,
        inputs: HashMap<String, String>,
        mut intermediate_steps: Vec<AgentStep>,
        start_iteration: usize,
        root_run: &mut RunTree,
        metrics: &mut AgentMetrics,
    ) -> Result<String, AgentError> {
        // Budget gate (§4.2): start the loop timer, used by the max_duration /
        // max_iterations checks.
        let loop_start = Instant::now();

        for iteration in start_iteration..self.max_iterations {
            // Budget gate: iteration-level (iteration count + wall-clock). Off by default
            // (returns None immediately when the config is None).
            if let Some(err) = budget_iteration_gate(
                self.budget.as_ref(),
                self.max_iterations,
                iteration,
                loop_start,
            ) {
                return Err(err);
            }

            if self.verbose {
                log::info!("=== Iteration {} ===", iteration + 1);
            }

            let output = self
                .plan_cached(&intermediate_steps, &inputs, metrics)
                .await?;

            // Budget gate: cumulative tokens after an LLM call; hard-stops when the limit
            // is exceeded.
            if let Some(err) = budget_token_gate(self.budget.as_ref(), metrics) {
                return Err(err);
            }

            match output {
                AgentOutput::Finish(finish) => {
                    if self.verbose {
                        log::info!("Final answer: {:?}", finish.return_values);
                    }
                    return Ok(finish.output().unwrap_or("").to_string());
                }

                AgentOutput::Action(action) => {
                    metrics.tool_calls += 1;
                    if self.verbose {
                        log::info!("Action: {}({})", action.tool, action.tool_input);
                    }

                    // Budget gate: check cumulative call count and wall-clock before the
                    // tool runs.
                    if let Some(err) = budget_tool_gate(self.budget.as_ref(), metrics, loop_start) {
                        return Err(err);
                    }

                    // Cross-process resume (§4.2): build the checkpoint context (only when
                    // a store is configured). Carries a snapshot of the loop context only;
                    // tool_name / arguments / tool_id are filled in by execute_tool_inner
                    // with the final values the approval sees, once the sync hooks finish.
                    // `inputs` / `intermediate_steps` are cloned as snapshots so resume
                    // continues from this batch of intermediate steps without replaying
                    // already-completed tool calls.
                    let pending = PendingApproval {
                        tool_name: action.tool.clone(),
                        arguments: serde_json::Value::Null,
                        tool_id: String::new(),
                        inputs: inputs.clone(),
                        steps: intermediate_steps.clone(),
                        iteration,
                        tool_calls_consumed: metrics.tool_calls,
                        tokens_consumed: metrics.total_tokens,
                        trace_id: root_run.trace_id.map(|id| id.to_string()),
                    };
                    let resume_ctx = self.resume_store.as_ref().map(|store| ResumeContext {
                        pending: &pending,
                        store,
                    });

                    let observation = self
                        .execute_tool_inner(&action, root_run, resume_ctx.as_ref(), None)
                        .await?;

                    if self.verbose {
                        log::info!("Observation: {}", observation);
                    }

                    intermediate_steps.push(AgentStep::new(action, observation));
                }

                AgentOutput::Actions(actions) => {
                    metrics.tool_calls += actions.len();
                    if self.verbose {
                        log::info!("Parallel actions: {} count", actions.len());
                        for action in &actions {
                            log::info!("  - {}({})", action.tool, action.tool_input);
                        }
                    }

                    // Budget gate: check cumulative call count and wall-clock before the
                    // tool runs.
                    if let Some(err) = budget_tool_gate(self.budget.as_ref(), metrics, loop_start) {
                        return Err(err);
                    }

                    let observations = self.execute_tools_parallel(&actions, root_run).await?;

                    if self.verbose {
                        for (i, obs) in observations.iter().enumerate() {
                            log::info!("Observation {}: {}", i + 1, obs);
                        }
                    }

                    for (action, observation) in actions.into_iter().zip(observations.into_iter()) {
                        intermediate_steps.push(AgentStep::new(action, observation));
                    }
                }
            }
        }

        // Reaching the iteration cap returns a placeholder string. It must not be only
        // verbose-visible: the caller cannot distinguish a real answer from the
        // placeholder, so log it explicitly at warning level.
        log::warn!(
            "agent reached max iterations {} without returning a final answer; returning a placeholder result (not the real final answer)",
            self.max_iterations
        );

        let finish = self.agent.return_stopped_response(&intermediate_steps);
        Ok(finish.output().unwrap_or("").to_string())
    }

    /// Executes multiple tools in parallel.
    ///
    /// Collects successful results and reports failures as error observations
    /// rather than discarding partial results when one tool fails.
    /// Concurrency is capped by the executor's global `concurrency_sem`.
    async fn execute_tools_parallel(
        &self,
        actions: &[AgentAction],
        root_run: &RunTree,
    ) -> Result<Vec<String>, AgentError> {
        use futures_util::future::join_all;

        let sem = self.concurrency_sem.clone();
        let futures = actions.iter().map(|action| {
            let sem = sem.clone();
            async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .map_err(|e| AgentError::Other(format!("concurrency semaphore closed: {e}")))?;
                self.execute_tool(action, root_run).await
            }
        });

        let results = join_all(futures).await;

        let mut observations = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok(output) => observations.push(output),
                Err(e) => observations.push(tool_error_observation(&e)),
            }
        }
        Ok(observations)
    }

    /// Executes a single tool (no resume context, no pre-decided approval).
    async fn execute_tool(
        &self,
        action: &AgentAction,
        root_run: &RunTree,
    ) -> Result<String, AgentError> {
        self.execute_tool_inner(action, root_run, None, None).await
    }

    /// Executes a single tool with optional cross-process resume integration.
    ///
    /// - `resume_ctx`: when non-None, the checkpoint (including the final
    ///   `tool_name` / `arguments` after synchronous-hook mutation) is persisted
    ///   **before** entering the approval gate to await approval, and cleared once the
    ///   decision **lands**. The parallel tool path (`execute_tools_parallel`) passes
    ///   `None` so concurrent multi-tool approvals never persist and cannot overwrite
    ///   each other's checkpoints.
    /// - `pre_decided`: when non-None, the approval handler is skipped and the given
    ///   decision is used directly (cross-process resume injects the decision without
    ///   re-running approval; `resume_ctx` should then be `None` — the checkpoint was
    ///   already claimed in [`AgentExecutor::resume`](crate::executor::AgentExecutor::resume)).
    pub(crate) async fn execute_tool_inner(
        &self,
        action: &AgentAction,
        root_run: &RunTree,
        resume_ctx: Option<&ResumeContext<'_>>,
        pre_decided: Option<ApprovalDecision>,
    ) -> Result<String, AgentError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == action.tool)
            .ok_or_else(|| AgentError::ToolNotFound(action.tool.clone()))?;

        let _input_str = match &action.tool_input {
            ToolInput::String { value: s } => s.clone(),
            ToolInput::Object { value: v } => serde_json::to_string(v)
                .map_err(|e| AgentError::Other(format!("Failed to serialize tool input: {}", e)))?,
        };

        // Run hooks: on_before_tool_call
        let mut tool_ctx = ToolCallContext {
            name: action.tool.clone(),
            arguments: match &action.tool_input {
                ToolInput::String { value: s } => {
                    // If the string is valid JSON, parse it as a Value to avoid
                    // double-encoding when serde_json::to_string() is called later.
                    // Otherwise wrap it as Value::String.
                    serde_json::from_str::<serde_json::Value>(s)
                        .unwrap_or(serde_json::Value::String(s.clone()))
                }
                ToolInput::Object { value: v } => v.clone(),
            },
            tool_id: String::new(),
        };

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
                    return Ok("[Skipped by hook]".to_string());
                }
            }
        }

        // Approval gate (§4.2) + cross-process resume (§4.2): after the sync hooks,
        // before the actual execution.
        // - Normal invoke: no pre_decided, goes through the handler; persists the
        //   checkpoint before approval, clears it once the decision lands.
        // - Resume: injects pre_decided, no persist/clear (the checkpoint was already
        //   claimed in resume()).
        // Deny is isomorphic with ToolCallAction::Skip — the rejection is fed back as an
        // observation; the tool does not run and the loop is not interrupted; the next
        // plan round sees the rejection observation and adjusts on its own.
        let deny_reason: Option<String> = if let Some(pre) = pre_decided {
            apply_approval_decision(pre, &mut tool_ctx)
        } else if let Some(handler) = &self.approval {
            // Cross-process resume: persist before approval. The sync hooks have already
            // run, so this is the final value the approval sees.
            if let Some(ctx) = resume_ctx {
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

        // The approval decision has landed: clear the checkpoint (Allow / Modify continue
        // execution; Deny returns the rejection observation).
        if let Some(ctx) = resume_ctx {
            if let Err(e) = ctx.store.clear_pending().await {
                log::warn!(
                    target: "lc_agents::resume",
                    "failed to clear pending approval: {}",
                    e
                );
            }
        }

        if let Some(reason) = deny_reason {
            return Ok(format!("[DENIED by approval: {reason}]"));
        }

        let tool_name = tool_ctx.name.clone();

        // P2-9: tool permission policy (permission tiering + sandbox gate). Allows when
        // unconfigured.
        if let Some(policy) = &self.tool_policy {
            policy.check(&tool_name)?;
        }

        let input_for_tool = serde_json::to_string(&tool_ctx.arguments)
            .unwrap_or_else(|_| tool_ctx.arguments.to_string());

        let mut tool_run = root_run.create_child(
            &tool_name,
            RunType::Tool,
            json!({"input": input_for_tool.clone()}),
        );

        if let Some(ref callbacks) = self.callbacks {
            for handler in callbacks.handlers() {
                handler
                    .on_tool_start(&tool_run, &tool_name, &input_for_tool)
                    .await;
            }
        }

        let tool_started = std::time::Instant::now();
        let result = run_tool_with_timeout(tool, input_for_tool.clone(), self.tool_timeout).await;
        let tool_duration_ms = tool_started.elapsed().as_millis();
        let trace = root_run
            .trace_id
            .map(|id| id.to_string())
            .unwrap_or_default();

        match result {
            Ok(output) => {
                // P1-6: tool call audit log with trace/input/duration/outcome.
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

                // Run hooks: on_after_tool_call
                let mut result_ctx = ToolResultContext {
                    name: tool_name,
                    result: output.clone(),
                    tool_id: String::new(),
                };
                for hook in &self.hooks {
                    if let Err(e) = hook.on_after_tool_call(&mut result_ctx) {
                        log::warn!("Hook on_after_tool_call error: {}", e);
                    }
                }

                Ok(result_ctx.result)
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
                Err(AgentError::ToolExecutionError(e.to_string()))
            }
        }
    }
}
