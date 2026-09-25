// lc-agents/src/executor/agent_loop.rs
//! `AgentExecutor`'s decision loop: the main agent loop plus sequential/parallel tool
//! execution.
//!
//! Works alongside `executor.rs` (struct + builder + invoke/stream entry points) and
//! `plan.rs` (cached planning).

use super::budget::{budget_cost_gate, budget_iteration_gate, budget_token_gate, budget_tool_gate};
use super::engine::{AgentExecutor, MaxIterationsPolicy};
use super::tool_gate::{ResumeCtx, ToolGate, ToolOutcome};
use super::tools::{denied_observation, is_non_execution_observation, tool_error_observation};
use super::AgentError;
use crate::approval::ApprovalDecision;
use crate::metrics::AgentMetrics;
use crate::resume::PendingApproval;
use crate::types::{AgentAction, AgentOutput, AgentStep, ToolInput};
use lc_callbacks::RunTree;
use lc_core::runnables::RunnableConfig;
use std::collections::HashMap;
use std::time::Instant;

//
// The checkpoint context + approval-decision application live in `super::tool_gate`
// (`ResumeCtx` / `apply_approval_decision`) so the stream path shares them.

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
        config: Option<&RunnableConfig>,
    ) -> Result<String, AgentError> {
        self.run_agent_loop_from(inputs, intermediate_steps, 0, root_run, metrics, config)
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
        // A18: per-round planning config (callbacks + trace linkage).
        config: Option<&RunnableConfig>,
    ) -> Result<String, AgentError> {
        // Budget gate (§4.2): start the loop timer, used by the max_duration /
        // max_iterations checks.
        let loop_start = Instant::now();

        // stage-G G7: capture the shared `CostTracker`'s spend as this run's
        // baseline. `max_cost_usd` then caps THIS run's incremental spend, not the
        // tracker's lifetime cumulative total — a prior / concurrent run sharing the
        // same `Arc` can no longer trip another run's budget.
        let cost_baseline = match &self.cost_tracker {
            Some(tracker) => tracker.total_cost_usd().await,
            None => 0.0,
        };

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

            // 0.21.0 S6.1: context compaction (off by default). Checked before
            // every plan round; drops the oldest whole steps (action + pair
            // observation stay together) when the trigger fires. The same
            // semantics run in the streaming path — the two paths cannot diverge.
            if let Some(config) = &self.compaction {
                let tokens = metrics.total_tokens.unwrap_or(0);
                let (kept, dropped) = config.compact(&intermediate_steps, tokens);
                if dropped > 0 {
                    log::info!(
                        target: "lc_agents::compaction",
                        "compacted {} of {} steps ({} remain)",
                        dropped,
                        dropped + kept.len(),
                        kept.len()
                    );
                    intermediate_steps = kept;
                    metrics.compactions += 1;
                }
            }

            let output = self
                .plan_cached(&intermediate_steps, &inputs, metrics, config)
                .await?;

            // Budget gate: cumulative tokens after an LLM call; hard-stops when the limit
            // is exceeded.
            if let Some(err) = budget_token_gate(self.budget.as_ref(), metrics) {
                return Err(err);
            }

            // B3 (0.22.4): cumulative USD spend gate after the LLM call. Reads the
            // shared CostTracker — the same Arc the tracking LLM records into, so the
            // spend reflects the call that just returned. No tracker → no measurement
            // → the limit cannot trip. stage-G G7: gated on this run's *incremental*
            // spend (current total minus the run-start baseline), so a shared tracker
            // never lets one run's cumulative spend trip another run's budget.
            if let Some(tracker) = &self.cost_tracker {
                let incremental = (tracker.total_cost_usd().await - cost_baseline).max(0.0);
                if let Some(err) = budget_cost_gate(self.budget.as_ref(), incremental) {
                    return Err(err);
                }
            }

            match output {
                AgentOutput::Finish(finish) => {
                    if self.verbose {
                        log::info!("Final answer: {:?}", finish.return_values);
                    }
                    return Ok(finish.output().unwrap_or("").to_string());
                }

                AgentOutput::Action(action) => {
                    // 0.22.0 audit fix (H-A5): the ReAct parse-repair pseudo-tool is
                    // not a real tool — never executed. Its input is fed back as the
                    // observation so the model can re-emit in the correct format;
                    // the agent hard-fails if it fails to parse twice in a row.
                    if action.tool == crate::react::agent::PARSE_ERROR_TOOL {
                        let observation = match &action.tool_input {
                            ToolInput::String { value } => value.clone(),
                            ToolInput::Object { value } => value.to_string(),
                        };
                        if self.verbose {
                            log::info!("Parse repair observation: {}", observation);
                        }
                        intermediate_steps.push(AgentStep::new(action, observation));
                        continue;
                    }

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
                    let resume_ctx = self.resume_store.as_ref().map(|store| ResumeCtx {
                        pending: &pending,
                        store,
                    });

                    // 0.20.0 S3.1:工具**执行**错误(工具真的跑了、失败返回 ToolError)
                    // 转 observation 喂回循环,agent 可自救——四条执行路径(顺序/并行 ×
                    // invoke/stream)一致。框架级守卫拒绝(权限策略 / hook 拒绝 /
                    // ControlAbort 交接环与深度中止)不是执行失败,仍
                    // 硬失败上抛:agent 无法靠重规划绕过它们,软化成 observation 会让
                    // 策略拒绝、预算配额与交接环检测形同虚设。
                    // stage-G G1:造化出的 / 未注册工具名 (`ToolNotFound`) 是**软**
                    // observation(工具从未执行,model 可重规划),与 0.20.0 A-H3 并行
                    // 路径钉死的语义一致——同一输入在顺序/并行两条路径不再一硬一软。
                    let observation = match self
                        .execute_tool_inner(&action, root_run, resume_ctx.as_ref(), None)
                        .await
                    {
                        Ok(obs) => obs,
                        Err(e @ AgentError::ToolExecutionError(_)) => tool_error_observation(&e),
                        Err(AgentError::ToolNotFound(name)) => {
                            format!("[Tool not found: {name}]")
                        }
                        Err(e) => return Err(e),
                    };

                    // stage-G G6: a denied / unregistered call never executed, so it
                    // must not consume the `max_tool_calls` budget — undo the
                    // pre-increment above (only executed tools count).
                    if is_non_execution_observation(&observation) {
                        metrics.tool_calls = metrics.tool_calls.saturating_sub(1);
                    }

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

                    // stage-G G6: denied / not-found calls in a batch never executed —
                    // undo their share of the pre-increment so they don't consume the
                    // `max_tool_calls` budget (only executed tools count).
                    let never_executed = observations
                        .iter()
                        .filter(|o| is_non_execution_observation(o))
                        .count();
                    metrics.tool_calls =
                        metrics.tool_calls.saturating_sub(never_executed);

                    if self.verbose {
                        for (i, obs) in observations.iter().enumerate() {
                            log::info!("Observation {}: {}", i + 1, obs);
                        }
                    }

                    // zip 第二参数本身收 IntoIterator,不必显式 .into_iter()
                    // (stable clippy::useless_conversion)。
                    for (action, observation) in actions.into_iter().zip(observations) {
                        intermediate_steps.push(AgentStep::new(action, observation));
                    }
                }
            }
        }

        // 0.22.0 C4 fix: the iteration cap is a failure, not a silent
        // placeholder. Default policy fails the run with `MaxIterationsReached`
        // so callers (PlanExecute included) can distinguish "did not converge"
        // from a real answer; `MaxIterationsPolicy::Placeholder` restores the
        // legacy ≤ 0.21.x behavior.
        log::warn!(
            "agent reached max iterations {} without returning a final answer (policy: {:?})",
            self.max_iterations,
            self.on_max_iterations
        );
        if self.on_max_iterations == MaxIterationsPolicy::Error {
            return Err(AgentError::MaxIterationsReached);
        }

        let finish = self.agent.return_stopped_response(&intermediate_steps);
        Ok(finish.output().unwrap_or("").to_string())
    }

    /// Executes multiple tools in parallel.
    ///
    /// Collects successful results and reports failures as error observations
    /// rather than discarding partial results when one tool fails. A tool that
    /// ran and failed (`ToolExecutionError`) or that was never registered
    /// (`ToolNotFound` — e.g. an LLM hallucinated name, 0.20.0 A-H3) becomes an
    /// observation, so the batch's other results survive and the loop can
    /// recover. Non-recoverable framework guardrails — approval `Deny`, hook
    /// `Reject`, permission policy, `ControlAbort` — abort the whole batch hard
    /// (B5): siblings are cancelled via `JoinSet::abort_all`, matching S3.1
    /// semantics. Concurrency is capped by the executor's global
    /// `concurrency_sem`.
    async fn execute_tools_parallel(
        &self,
        actions: &[AgentAction],
        root_run: &RunTree,
    ) -> Result<Vec<String>, AgentError> {
        let gate = ToolGate::from_executor(self);
        gate.execute_many(actions, root_run).await
    }

    /// Executes a single tool with optional cross-process resume integration.
    ///
    /// Delegates to the shared [`ToolGate`] (the same chain the stream path uses),
    /// translating an approval **Deny** into the `[DENIED by approval: …]`
    /// observation the loop feeds back so the model can re-plan.
    ///
    /// - `resume_ctx`: when non-None, the checkpoint (including the final
    ///   `tool_name` / `arguments` / `tool_id` after synchronous-hook mutation) is
    ///   persisted **before** the approval gate and cleared once the decision lands.
    ///   The parallel path (`execute_tools_parallel` → `execute_many`) passes `None` so
    ///   concurrent multi-tool approvals never persist and cannot overwrite each other.
    /// - `pre_decided`: when non-None, the approval handler is skipped and the given
    ///   decision is used directly (cross-process resume).
    pub(crate) async fn execute_tool_inner(
        &self,
        action: &AgentAction,
        root_run: &RunTree,
        resume_ctx: Option<&ResumeCtx<'_>>,
        pre_decided: Option<ApprovalDecision>,
    ) -> Result<String, AgentError> {
        let gate = ToolGate::from_executor(self);
        match gate
            .execute_one(action, root_run, resume_ctx, pre_decided)
            .await?
        {
            ToolOutcome::Done(result) => Ok(result),
            ToolOutcome::Denied { reason } => Ok(denied_observation(&reason)),
        }
    }
}
