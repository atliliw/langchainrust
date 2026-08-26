// lc-agents/src/executor/agent_loop.rs
//! `AgentExecutor` 的决策循环:主 agent loop + 工具顺序/并行执行。
//!
//! 与 `executor.rs`(结构体 + 构建器 + invoke/stream 入口)、`plan.rs`(缓存规划)配合。

use super::budget::BudgetExceeded;
use super::engine::AgentExecutor;
use super::tools::run_tool_with_timeout;
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

/// 跨进程 resume(§4.2):单次工具调用所需的挂起点上下文。
///
/// 由 agent loop 在 Action 分支构造:`tool_name` / `arguments` / `tool_id` 先占位,
/// `execute_tool_inner` 在同步 hook 跑完后用审批看到的**最终值**填充并落盘;
/// 审批决定落地后清除。并行工具路径(`execute_tools_parallel`)不构造它 ——
/// 多工具并发审批互不落盘,避免互相覆盖挂起点。
pub(crate) struct ResumeContext<'a> {
    /// 已填好 loop 上下文的挂起点模板(输入 / 步骤 / 迭代 / 预算累计 / trace)。
    pending: &'a PendingApproval,
    /// 挂起点存储(审批前后落盘 / 清除)。
    store: &'a Arc<dyn ResumeStore>,
}

/// 把审批决定落到 `tool_ctx`。
///
/// 返回 `Some(reason)` 表示 **Deny**(中止执行,拒绝 observation 喂回循环);
/// `None` 表示 Allow / Modify(继续执行)。Modify 会覆盖 `tool_ctx.arguments`。
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
    /// 跨进程 resume(§4.2)用它从挂起迭代续跑:迭代预算 / 工具次数预算从挂起点
    /// 累计量继续计数,已完成的中间步骤不重放。
    pub(crate) async fn run_agent_loop_from(
        &self,
        inputs: HashMap<String, String>,
        mut intermediate_steps: Vec<AgentStep>,
        start_iteration: usize,
        root_run: &mut RunTree,
        metrics: &mut AgentMetrics,
    ) -> Result<String, AgentError> {
        // 预算门(§4.2):循环起表,供 max_duration / max_iterations 检查。
        let loop_start = Instant::now();

        for iteration in start_iteration..self.max_iterations {
            // 预算门:迭代级(迭代次数 + 总时长)。默认关(None 时立即返回 None)。
            if let Some(err) = self.budget_iteration_gate(iteration, loop_start) {
                return Err(err);
            }

            if self.verbose {
                log::info!("=== Iteration {} ===", iteration + 1);
            }

            let output = self
                .plan_cached(&intermediate_steps, &inputs, metrics)
                .await?;

            // 预算门:LLM 调用后累计 token,超限硬停。
            if let Some(err) = self.budget_token_gate(metrics) {
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

                    // 预算门:工具调用前检查累计次数与总时长。
                    if let Some(err) = self.budget_tool_gate(metrics, loop_start) {
                        return Err(err);
                    }

                    // 跨进程 resume(§4.2):构造挂起点上下文(仅当配置了 store)。
                    // 只带 loop 上下文快照;tool_name / arguments / tool_id 由
                    // execute_tool_inner 在同步 hook 跑完后填审批看到的最终值。
                    // `inputs` / `intermediate_steps` 克隆快照,resume 时从这批
                    // 中间步骤续跑,不重放已完成的工具调用。
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

                    // 预算门:工具调用前检查累计次数与总时长。
                    if let Some(err) = self.budget_tool_gate(metrics, loop_start) {
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

        // 达到迭代上限返回占位串,不能只靠 verbose 可见:调用方无法区分真实答案与占位,
        // 这里记 error 级日志显式暴露
        log::warn!(
            "agent reached max iterations {} without returning a final answer; returning a placeholder result (not the real final answer)",
            self.max_iterations
        );

        let finish = self.agent.return_stopped_response(&intermediate_steps);
        Ok(finish.output().unwrap_or("").to_string())
    }

    /// 预算门(§4.2):迭代级检查(迭代次数 + 总时长)。超限 → 返回错误。
    ///
    /// `max_iterations` 取 `min(self.max_iterations, budget.max_iterations)`
    /// 作为有效上限 —— budget 比默认更紧时超出即硬停;比默认更松时由
    /// `self.max_iterations` 兜底,不改变原有占位返回路径。
    fn budget_iteration_gate(&self, iteration: usize, loop_start: Instant) -> Option<AgentError> {
        let budget = self.budget.as_ref()?;
        if let Some(limit) = budget.max_iterations {
            let effective = limit.min(self.max_iterations);
            if iteration >= effective {
                return Some(AgentError::BudgetExceeded(BudgetExceeded::Iterations {
                    limit: effective,
                }));
            }
        }
        if let Some(limit) = budget.max_duration {
            let elapsed = loop_start.elapsed();
            if elapsed >= limit {
                return Some(AgentError::BudgetExceeded(BudgetExceeded::Duration {
                    limit,
                    elapsed,
                }));
            }
        }
        None
    }

    /// 预算门(§4.2):LLM 调用后累计 token 检查。agent 不上报 token 时不生效。
    fn budget_token_gate(&self, metrics: &AgentMetrics) -> Option<AgentError> {
        let budget = self.budget.as_ref()?;
        let limit = budget.max_tokens?;
        let actual = metrics.total_tokens.unwrap_or(0);
        if actual >= limit {
            return Some(AgentError::BudgetExceeded(BudgetExceeded::Tokens {
                limit,
                actual,
            }));
        }
        None
    }

    /// 预算门(§4.2):工具调用前检查累计次数与总时长。
    ///
    /// `metrics.tool_calls` 已先自增,故用 `> limit` 判定 —— 允许恰好 `limit`
    /// 次工具执行,第 `limit + 1` 次触发硬停。
    fn budget_tool_gate(&self, metrics: &AgentMetrics, loop_start: Instant) -> Option<AgentError> {
        let budget = self.budget.as_ref()?;
        if let Some(limit) = budget.max_tool_calls {
            if metrics.tool_calls > limit {
                return Some(AgentError::BudgetExceeded(BudgetExceeded::ToolCalls {
                    limit,
                    actual: metrics.tool_calls,
                }));
            }
        }
        if let Some(limit) = budget.max_duration {
            let elapsed = loop_start.elapsed();
            if elapsed >= limit {
                return Some(AgentError::BudgetExceeded(BudgetExceeded::Duration {
                    limit,
                    elapsed,
                }));
            }
        }
        None
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
                Err(e) => observations.push(format!("[Tool execution error: {}]", e)),
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
    /// - `resume_ctx`:非 None 时,进入人审门等待审批**之前**把挂起点(含同步
    ///   hook 修改后的最终 `tool_name` / `arguments`)落盘;审批决定**落地后**
    ///   清除。并行工具路径(`execute_tools_parallel`)传 `None`,多工具并发审批
    ///   互不落盘,避免互相覆盖挂起点。
    /// - `pre_decided`:非 None 时跳过审批 handler,直接用给定决定(跨进程 resume
    ///   注入决定,不重跑审批;此时 `resume_ctx` 应为 None —— 挂起点已在
    ///   [`AgentExecutor::resume`](crate::executor::AgentExecutor::resume) 里认领)。
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

        // 人审门(§4.2) + 跨进程 resume(§4.2):同步 hook 之后、实际执行之前。
        // - 正常 invoke:无 pre_decided,走 handler;审批前落盘、决定落地后清除。
        // - resume:注入 pre_decided,不再落盘/清除(挂起点已在 resume() 认领)。
        // Deny 与 ToolCallAction::Skip 同构 —— 以 observation 喂回循环,不执行
        // 工具也不中断运行,下一轮 plan 看到拒绝观察后自行调整。
        let deny_reason: Option<String> = if let Some(pre) = pre_decided {
            apply_approval_decision(pre, &mut tool_ctx)
        } else if let Some(handler) = &self.approval {
            // 跨进程 resume:审批前落盘。同步 hook 已跑完,这里是审批看到的最终值。
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

        // 审批决定已落地:清除挂起点(Allow / Modify 继续执行;Deny 拒绝观察返回)。
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

        // P2-9: 工具权限策略(权限分级 + 沙箱门禁)。未配置则放行。
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
